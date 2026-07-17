FROM python:3.12-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
RUN pip install uv

COPY pyproject.toml ./

# Устанавливаем зависимости через uv с зеркалом Tencent
RUN uv pip install --system --no-cache-dir \
    --index-url https://mirrors.cloud.tencent.com/pypi/simple \
    -e .

COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]