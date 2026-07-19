from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from loguru import logger
from src.services.document_chunks.base import DocumentChunksRepository
from src.core.database.models import DocumentChunks


class SQLAlchemyDocumentChunksRepository(DocumentChunksRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def search_dense(
        self,
        query_embedding: List[float],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        
        stmt = text("""
            SELECT id, doc_id, text, meta,
                   1 - (embedding <=> :query_embedding) AS score
            FROM chunks
            ORDER BY embedding <=> :query_embedding
            LIMIT :limit
        """)
        
        result = await self.session.execute(
            stmt,
            {
                "query_embedding": embedding_str,
                "limit": k
            }
        )
        
        rows = result.all()
        return [
            {
                "chunk_id": row.id,
                "doc_id": row.doc_id,
                "text": row.text,
                "meta": row.meta,
                "score": float(row.score)
            }
            for row in rows
        ]

    async def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            
            db_chunk = DocumentChunks(
                doc_id=str(chunk["doc_id"]),
                text=chunk["text"],
                embedding=embedding,
                meta=chunk.get("meta", {})
            )
            self.session.add(db_chunk)
        
        await self.session.commit()
        logger.info("Upserted {} chunks", len(chunks))

    async def get_all_chunks(self) -> List[Dict[str, Any]]:
        from sqlalchemy import select
        
        stmt = select(
            DocumentChunks.id,
            DocumentChunks.doc_id,
            DocumentChunks.text
        ).order_by(DocumentChunks.id)
        
        result = await self.session.execute(stmt)
        rows = result.all()
        
        return [
            {
                "chunk_id": row.id,
                "doc_id": row.doc_id,
                "text": row.text
            }
            for row in rows
        ]

    async def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any] | None:
        from sqlalchemy import select
        
        stmt = select(DocumentChunks).where(DocumentChunks.id == chunk_id)
        result = await self.session.execute(stmt)
        chunk = result.scalar_one_or_none()
        
        if not chunk:
            return None
        
        return {
            "chunk_id": chunk.id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "meta": chunk.meta
        }

    async def delete_chunk(self, chunk_id: str) -> bool:
        from sqlalchemy import select
        
        stmt = select(DocumentChunks).where(DocumentChunks.id == chunk_id)
        result = await self.session.execute(stmt)
        chunk = result.scalar_one_or_none()
        
        if not chunk:
            return False
        
        await self.session.delete(chunk)
        await self.session.commit()
        logger.info("Deleted chunk {}", chunk_id)
        return True
    
    async def count_chunks(self) -> int:
        from sqlalchemy import select, func
        
        stmt = select(func.count()).select_from(DocumentChunks)
        result = await self.session.execute(stmt)
        return result.scalar_one()
    
    async def clear_all_chunks(self) -> None:
        from sqlalchemy import delete
        
        stmt = delete(DocumentChunks)
        await self.session.execute(stmt)
        await self.session.commit()
        logger.warning("All chunks deleted")