from src.core.database.base import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, ForeignKey, Text, Index
import uuid
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from typing import Optional


class DocumentChunks(Base):
    __tablename__= "chunks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    doc_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(1536), nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=lambda: {})

    __table_args__ = (
        Index(
            "idx_chunks_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )