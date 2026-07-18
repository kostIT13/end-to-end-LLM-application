from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from src.services.rag.embedder import embed_query
from src.services.rag.bm25_retrieval import BM25Index
from src.services.document_chunks.repository import SQLAlchemyDocumentChunksRepository


def rrf_fuse(rankings: List[List[str]], k: int = 60) -> List[tuple[str, float]]:
    if k <= 0:
        raise ValueError("k must be positive")
    
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


async def hybrid_search(
    session: AsyncSession,
    query: str,
    bm25: Optional[BM25Index] = None,
    top_k_dense: int = 10,
    top_k_bm25: int = 10,
    top_k_final: int = 5,
) -> List[Dict[str, Any]]:
    query_embedding = embed_query(query)
    
    repo = SQLAlchemyDocumentChunksRepository(session)
    dense_results = await repo.search_dense(
        query_embedding.tolist(),
        k=top_k_dense
    )
    
    if bm25 is None:
        return dense_results[:top_k_final]
    
    bm25_results = await bm25.search(query, top_k_bm25)
    
    dense_ranking = [str(r["chunk_id"]) for r in dense_results]
    bm25_ranking = [str(r["chunk_id"]) for r in bm25_results]
    
    fused = rrf_fuse([dense_ranking, bm25_ranking])
    
    by_id: dict[str, dict] = {}
    for r in dense_results + bm25_results:
        key = str(r["chunk_id"])
        if key not in by_id:
            by_id[key] = r.copy()
        else:
            if "score" in r:
                by_id[key]["bm25_score"] = r["score"]
    
    top_final = []
    for cid_str, fused_score in fused[:top_k_final]:
        chunk = by_id.get(cid_str, {})
        if chunk:
            chunk["fused_score"] = fused_score
            top_final.append(chunk)
    
    logger.debug("Hybrid search: dense={}, bm25={}, final={}",
                 len(dense_results), len(bm25_results), len(top_final))
    
    return top_final