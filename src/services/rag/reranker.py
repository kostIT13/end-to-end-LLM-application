import httpx
import asyncio
from typing import List, Dict, Any
from loguru import logger
from src.core.config import settings


async def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        return chunks
    
    try:
        reranked = await _rerank_with_ollama(query, chunks)
        
        for i, chunk in enumerate(reranked[:top_k]):
            chunk["rerank_score"] = 1.0 - (i / len(reranked))
        
        return reranked[:top_k]
        
    except Exception as e:
        logger.error("Reranking failed: {}, falling back to original order", e)
        return chunks[:top_k]


async def _rerank_with_ollama(
    query: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(f"[{i+1}] {chunk['text'][:500]}...")
    
    prompt = f"""Оцени релевантность каждого документа запросу от 1 до 10.
Запрос: {query}

Документы:
{chr(10).join(documents)}

Ответь только числами через запятую в порядке документов, например: 8, 5, 9, 3
"""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.LLM_MODEL_CHEAP or "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.0,
            }
        )
        data = response.json()
        text = data.get("response", "")
        
    try:
        scores = [float(s.strip()) for s in text.split(",") if s.strip()]
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored]
    except Exception:
        logger.warning("Failed to parse reranker scores, returning original order")
        return chunks