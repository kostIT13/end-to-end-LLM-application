# end-to-end RAG Service

Production-ready **Retrieval-Augmented Generation** (RAG) сервис с гибридным поиском, переранжированием, верификацией цитат и защитой от prompt injection.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HTTP Client                                    │
│                     POST /api/v1/base/ask                                   │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                           FastAPI (src/main.py)                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Security Filter (injection.py)                   │   │
│  │  Regex-проверка на prompt injection: "игнорируй", "forget", "act as"  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                      RAG Pipeline (generation/pipeline.py)                   │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  Hybrid      │    │   BM25       │    │  Reranker    │    │   LLM     │  │
│  │  Search      ├────►  (rank-      ├────►  (Ollama     ├────►  Client   │  │
│  │  (RRF-fuse)  │    │   bm25)      │    │   LLM)       │    │  (OpenAI  │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    │   API)    │  │
│         │                  │                    │            └─────┬─────┘  │
│         │    ┌─────────────▼────────────────────┘                  │        │
│         │    │              Chunk Retrieval                        │        │
│         │    │  ┌─────────────────┐  ┌──────────────────────┐     │        │
│         └────┼──► Dense Search   │  │  BM25 Keyword Search │     │        │
│              │  │ (pgvector HNSW) │  │  (BM25Okapi)         │     │        │
│              │  └────────┬────────┘  └──────────┬───────────┘     │        │
│              └───────────┼──────────────────────┘                 │        │
│                          │ RRF Fusion (k=60)                      │        │
│                          ▼                                        │        │
│              ┌──────────────────────┐                             │        │
│              │  Citation Verifier   │◄────────────────────────────┘        │
│              │  (verify_citations)  │                                      │
│              └──────────────────────┘                                      │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                         PostgreSQL + pgvector                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Таблица chunks: id, doc_id, text, embedding(VECTOR(1536)), meta    │   │
│  │  Индекс: HNSW (cosine similarity)                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                         Ollama (локальные LLM)                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  • nomic-embed-text  — эмбеддинг документов и запросов               │   │
│  │  • gemma-4-31b / llama3.2:1b — генерация ответа (fallback)          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Компоненты pipeline'а

| Этап | Компонент | Файл |
|------|-----------|------|
| 1 | **Security Filter** | [`src/core/security/injection.py`](src/core/security/injection.py) |
| 2 | **Dense Retrieval** (pgvector HNSW, cosine distance) | [`src/services/document_chunks/repository.py`](src/services/document_chunks/repository.py) |
| 3 | **BM25 Keyword Retrieval** (BM25Okapi) | [`src/services/rag/bm25_retrieval.py`](src/services/rag/bm25_retrieval.py) |
| 4 | **Hybrid Fusion** (RRF, k=60) | [`src/services/rag/hybrid.py`](src/services/rag/hybrid.py) |
| 5 | **Reranker** (Ollama LLM, опционально) | [`src/services/rag/reranker.py`](src/services/rag/reranker.py) |
| 6 | **LLM Generation** (OpenAI-совместимый API) | [`src/services/llm_client.py`](src/services/llm_client.py) |
| 7 | **Citation Verification** (точное совпадение текста) | [`src/services/generation/verifier.py`](src/services/generation/verifier.py) |

---

## Быстрый старт

### 1. Предварительные требования

- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/install/) (v2)
- 8+ GB RAM (рекомендуется 16 GB для работы LLM)

### 2. Конфигурация

Скопируйте `.env.example` в `.env` и отредактируйте:

```bash
cp .env.example .env
```

Минимальная конфигурация для локального запуска с Ollama:

```env
LOG_LEVEL=INFO

# Ollama (локально)
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
EMBED_DIMENSION=768

# OpenAI-совместимый LLM (можно тоже через Ollama)
LLM_BASE_URL=http://ollama:11434/v1
LLM_API_KEY=ollama
LLM_MODEL_PRIMARY=gemma-4-31b
LLM_MODEL_CHEAP=llama3.2:1b

# PostgreSQL + pgvector
PGVECTOR_URL=postgresql+asyncpg://rag:ragpass@postgres:5432/ragdb
PGVECTOR_DB=ragdb
PGVECTOR_HOST=postgres
PGVECTOR_PASSWORD=ragpass
PGVECTOR_USER=rag
PGVECTOR_PORT=5432
```

### 3. Запуск

```bash
docker compose up -d
```

Сервис будет доступен по адресу: **http://localhost:8000**

Swagger UI: **http://localhost:8000/docs**

### 4. Загрузка модели эмбеддингов в Ollama

```bash
docker exec -it end_to_end_ollama ollama pull nomic-embed-text
```

### 5. Индексация корпуса документов

Поместите `.txt` файлы в `data/corpus/` и выполните:

```bash
docker exec -it end_to_end_service python -m src.services.indexing --corpus-dir /app/data/corpus
```

### 6. Проверка здоровья

```bash
curl http://localhost:8000/api/v1/base/health
# {"status": "ok", "service": "RAG API"}
```

---

## API

### `POST /api/v1/base/ask` — задать вопрос

```bash
curl -X POST http://localhost:8000/api/v1/base/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Какова пропускная способность магистрали?",
    "top_k": 5,
    "use_hybrid_search": true,
    "use_reranker": false
  }'
```

**Параметры запроса** ([`AskRequest`](src/api/base_schemas.py)):

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `question` | string | — | Текст вопроса (1–2000 символов) |
| `temperature` | float | 0.3 | Температура LLM (0.0–2.0) |
| `max_tokens` | int | 500 | Максимум токенов в ответе |
| `top_k` | int | 5 | Количество извлекаемых чанков (1–20) |
| `use_hybrid_search` | bool | true | Гибридный поиск (dense + BM25) |
| `use_reranker` | bool | false | Использовать reranker |
| `skip_security_check` | bool | false | Отключить проверку injection |

**Ответ** ([`RAGResponse`](src/api/pipeline_schemas.py)):

```json
{
  "answer": "Пропускная способность магистрали составляет 120 000 баррелей в сутки.",
  "sources": [
    {
      "chunk_id": "uuid-...",
      "doc_id": "doc_001",
      "text": "Магистраль рассчитана на 120 000 баррелей в сутки...",
      "score": 0.89,
      "fused_score": 0.042
    }
  ],
  "citations": [
    {"doc_id": "doc_001", "quote": "120 000 баррелей в сутки"}
  ],
  "has_valid_citations": true,
  "model_used": "gemma-4-31b",
  "processing_time_ms": 2345.67
}
```

### `POST /api/v1/base/ask/stream` — стриминг ответа

То же, что и `/ask`, но ответ приходит токен за токеном (`text/plain`).

---

## Дизайн-решения

### Почему chunk_size = 500 слов?

В [`fixed_chunker`](src/services/rag/chunker.py) используется разбиение по словам (не по токенам) с размером окна **500 слов** и перекрытием **50 слов**.

- **500 слов** ≈ ~650 токенов — золотая середина: достаточно контекста для ответа на вопрос, но не слишком много для точного semantic search.
- **Перекрытие 50 слов** (10%) гарантирует, что релевантный фрагмент не "разрежется" границей чанка.
- Разбиение по словам (не по предложениям) выбрано для простоты и предсказуемости; для русского языка это даёт стабильные результаты без dependency hell NLP-пайплайнов.

### Почему nomic-embed-text?

- **768-мерные эмбеддинги** — хороший баланс между качеством и производительностью (против 1536 у OpenAI/text-embedding-3-small).
- Работает локально через Ollama, без внешних API.
- Поддерживает префиксы `"query"` / `"passage"` для asymmetric retrieval — в [`embedder.py`](src/services/rag/embedder.py) это используется явно.

> **Важно:** В модели [`DocumentChunks`](src/core/database/models.py) размерность вектора указана как `Vector(1536)`, что не соответствует 768-мерному `nomic-embed-text`. При использовании этой модели необходимо исправить миграцию на `Vector(768)`.

### Почему HNSW, а не IVFFlat?

- **HNSW** даёт лучшее качество поиска (higher recall) за счёт графовой структуры.
- Не требует перестроения индекса после вставки новых данных (в отличие от IVFFlat, которому нужен `CREATE INDEX` заново).
- Минус: больше памяти, но для корпуса среднего размера (до 1M векторов) это некритично.

### Почему RRF (Reciprocal Rank Fusion)?

- Простой, детерминированный и эффективный метод объединения dense + sparse результатов.
- Параметр `k=60` — стандартное значение из литературы, которое хорошо работает на практике.
- Альтернатива — обученный reranker поверх fusion, но RRF не требует данных для обучения.

### Почему reranker опционален?

- Реренкер через Ollama LLM ([`reranker.py`](src/services/rag/reranker.py)) добавляет ~1-3 секунды к latency.
- Для многих запросов топ-5 из RRF уже содержит релевантный чанк на первой позиции.
- Реренкер полезен для сложных, многосоставных запросов, где dense + BM25 могут ошибаться.

### Почему citation verification?

- LLM склонна к галлюцинациям, даже имея контекст.
- [`verifier.py`](src/services/generation/verifier.py) проверяет, что каждая цитата **дословно** присутствует в исходном чанке (с нормализацией пробелов и регистра).
- Если цитата не найдена — она помечается как `invalid_citations`, а `has_valid_citations` становится `false`.
- Это даёт пользователю прозрачность: он видит, какие части ответа подтверждены документами.

### Почему LLMClient с fallback?

- [`LLMClient`](src/services/llm_client.py) использует OpenAI-совместимый клиент (поддерживает Ollama, OpenAI, Together, vLLM, etc.).
- **Retry-логика**: при RateLimitError / APITimeoutError — exponential backoff (2^attempt, max 10s).
- **Fallback**: если primary модель исчерпала retry, запрос уходит на `LLM_MODEL_CHEAP` (например, `llama3.2:1b`).
- Это обеспечивает отказоустойчивость: сервис ответит даже при перегрузке основной модели.

### Почему защита от prompt injection на regex?

- [`injection.py`](src/core/security/injection.py) использует набор regex-паттернов на русском и английском.
- Покрывает типовые атаки: "ignore all instructions", "forget everything", "ты теперь", "новые инструкции".
- Для production рекомендуется добавить второй уровень — LLM-as-a-judge (проверка через отдельную модель).
- Регулярки быстрые (O(1) по времени) и не добавляют latency.

---

## Eval-система

В директории [`eval/`](eval/) находится система оценки качества RAG-пайплайна:

- [`golden.jsonl`](eval/golden.jsonl) — golden-датасет с эталонными парами вопрос-ответ.
- [`metrics.py`](eval/metrics.py) — метрики качества (точность цитирования, полнота ответа).
- [`start_golden.py`](eval/start_golden.py) — скрипт прогона golden-тестов.

```bash
docker exec -it end_to_end_service python -m eval.start_golden
```

---

## Структура проекта

```
├── docker-compose.yml          # Оркестрация: postgres + ollama + service
├── Dockerfile                  # Python 3.12-slim + uv
├── pyproject.toml              # Зависимости проекта
├── .env.example                # Шаблон конфигурации
├── alembic/                    # Миграции БД
│   └── versions/
│       └── ab05de0e1dd6_create_document_chunks_table.py
├── data/
│   └── corpus/                 # .txt файлы для индексации
├── eval/
│   ├── golden.jsonl            # Golden-датасет
│   ├── metrics.py              # Метрики
│   └── start_golden.py         # Запуск eval
└── src/
    ├── main.py                 # FastAPI app + lifespan
    ├── api/
    │   ├── base_schemas.py     # AskRequest
    │   ├── dependencies.py     # DI для LLMClient, DocumentChunksService
    │   ├── endpoints.py        # /health, /ask, /ask/stream
    │   └── pipeline_schemas.py # RAGResponse, CitationModel
    ├── core/
    │   ├── config.py           # Pydantic Settings
    │   ├── logging_settings.py # Loguru-конфигурация
    │   ├── database/
    │   │   ├── base.py         # SQLAlchemy DeclarativeBase
    │   │   ├── db.py           # Engine + session factory
    │   │   └── models.py       # DocumentChunks (pgvector)
    │   └── security/
    │       └── injection.py    # Prompt injection detection
    └── services/
        ├── indexing.py         # CLI-индексатор корпуса
        ├── llm_client.py       # OpenAI-клиент с retry + fallback
        ├── document_chunks/
        │   ├── base.py                 # Абстрактный репозиторий
        │   ├── repository.py           # SQLAlchemy-реализация
        │   └── document_chunks_service.py  # Бизнес-логика
        ├── generation/
        │   ├── pipeline.py     # answer_question + stream
        │   ├── prompts.py      # RAG_SYSTEM_PROMPT + builder
        │   └── verifier.py     # Citation verification
        └── rag/
            ├── chunker.py      # fixed_chunker (500 слов, overlap 50)
            ├── embedder.py     # nomic-embed-text через Ollama
            ├── bm25_retrieval.py # BM25Okapi с persist в pickle
            ├── hybrid.py       # RRF fusion (k=60)
            └── reranker.py     # LLM-based reranker
```

---

## Зависимости

| Пакет | Назначение |
|-------|-----------|
| `fastapi` + `uvicorn` | HTTP-сервер |
| `sqlalchemy` + `asyncpg` | Асинхронная работа с PostgreSQL |
| `pgvector` | Векторный поиск (HNSW, cosine) |
| `alembic` | Миграции схемы БД |
| `ollama` | Локальные LLM и эмбеддинги |
| `openai` | OpenAI-совместимый клиент |
| `rank-bm25` | BM25-ранжирование |
| `httpx` | HTTP-клиент для Ollama |
| `numpy` | Работа с эмбеддингами |
| `pydantic` + `pydantic-settings` | Валидация схем и конфигурации |
| `loguru` | Логирование |
| `tenacity` | Retry-логика (опционально) |

---

## Лицензия

MIT