import pytest
from src.services.rag.reranker import rerank_chunks


@pytest.mark.asyncio
async def test_rerank_empty_chunks():
    result = await rerank_chunks("query", [])
    assert result == []


@pytest.mark.asyncio
async def test_rerank_less_than_top_k():
    chunks = [{"chunk_id": "1", "text": "test"}]
    result = await rerank_chunks("query", chunks, top_k=5)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "1"