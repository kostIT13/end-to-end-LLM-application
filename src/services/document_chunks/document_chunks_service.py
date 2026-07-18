from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
import numpy as np
from src.services.document_chunks.repository import SQLAlchemyDocumentChunksRepository
from src.services.rag.chunker import fixed_chunker
from src.services.rag.embedder import embed_passages, embed_query
from src.services.rag.bm25_retrieval import BM25Index
from src.services.rag.hybrid import hybrid_search
from src.core.config import settings


class DocumentChunksService:
    def __init__(self, repository: SQLAlchemyDocumentChunksRepository, session: AsyncSession):
        self.repository = repository
        self.session = session
        self._bm25: Optional[BM25Index] = None
    
    async def index_document(
        self,
        doc_id: str,
        text: str,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> int:
        chunks_text = fixed_chunker(text, chunk_size, chunk_overlap)
        
        if not chunks_text:
            logger.warning("No chunks generated for doc {}", doc_id)
            return 0
        
        logger.debug("Generated {} chunks for doc {}", len(chunks_text), doc_id)
        
        embeddings = await embed_passages(chunks_text)
        
        chunks_to_insert = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks_text, embeddings)):
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            chunks_to_insert.append({
                "doc_id": doc_id,
                "text": chunk_text,
                "embedding": embedding_list,
                "meta": {
                    "source": source,
                    "chunk_index": i,
                    **(metadata or {})
                }
            })
        
        await self.repository.upsert_chunks(chunks_to_insert)
        
        await self._rebuild_bm25()
        
        logger.info("Indexed {} chunks for doc {}", len(chunks_to_insert), doc_id)
        return len(chunks_to_insert)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        if use_hybrid:
            bm25 = await self._get_bm25()
            results = await hybrid_search(
                session=self.session,
                query=query,
                bm25=bm25,
                top_k_final=top_k,
            )
        else:
            query_embedding = await embed_query(query)
            results = await self.repository.search_dense(
                query_embedding.tolist(),
                k=top_k
            )
        
        return results

    async def _get_bm25(self) -> Optional[BM25Index]:
        if self._bm25 is not None:
            return self._bm25
        
        try:
            self._bm25 = BM25Index.load()
            logger.info("BM25 index loaded from disk")
            return self._bm25
        except FileNotFoundError:
            logger.warning("BM25 index not found. Run indexing first.")
            return None
    
    async def _rebuild_bm25(self) -> None:
        all_chunks = await self.repository.get_all_chunks()
        if not all_chunks:
            logger.warning("No chunks to build BM25 index")
            return
        
        self._bm25 = BM25Index(all_chunks)
        self._bm25.save()
        logger.info("BM25 index rebuilt with {} chunks", len(all_chunks))
    
    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return await self.repository.get_chunk_by_id(chunk_id)
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        result = await self.repository.delete_chunk(chunk_id)
        if result:
            await self._rebuild_bm25()
        return result
    
    async def count(self) -> int:
        return await self.repository.count_chunks()
    
    async def get_all_chunks(self) -> List[Dict[str, Any]]:
        return await self.repository.get_all_chunks()
    
    async def clear_all_chunks(self) -> None:
        await self.repository.clear_all_chunks()
        self._bm25 = None
        logger.warning("All chunks cleared")