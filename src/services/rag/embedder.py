import httpx
import numpy as np
from typing import List
from loguru import logger
from functools import lru_cache
from src.core.config import settings


@lru_cache(maxsize=1)
def get_embed_model() -> str:
    return settings.OLLAMA_EMBED_MODEL


async def embed_texts(texts: List[str], prefix: str = "") -> np.ndarray:
    if not texts:
        return np.zeros((0, settings.EMBED_DIMENSION), dtype=np.float32)
    
    if prefix:
        texts = [f"{prefix} {t}" for t in texts]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        responses = []
        for text in texts:
            try:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                    json={
                        "model": settings.OLLAMA_EMBED_MODEL,
                        "prompt": text,
                    }
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding", [])
                if embedding:
                    responses.append(embedding)
                else:
                    logger.warning("Empty embedding for text: {}", text[:50])
                    responses.append([0.0] * settings.EMBED_DIMENSION)
            except Exception as e:
                logger.error("Ollama embedding error: {}", e)
                responses.append([0.0] * settings.EMBED_DIMENSION)
    
    return np.asarray(responses, dtype=np.float32)


async def embed_query(text: str) -> List[float]:
    text = text.strip()
    if not text:
        raise ValueError("Empty query text")
    
    embeddings = await embed_texts([text], prefix="query")
    return embeddings[0].tolist() 


async def embed_passages(texts: List[str]) -> np.ndarray:
    """Создает эмбеддинги для документов. Возвращает np.ndarray."""
    if not texts:
        return np.zeros((0, settings.EMBED_DIMENSION), dtype=np.float32)
    
    return await embed_texts(texts, prefix="passage")