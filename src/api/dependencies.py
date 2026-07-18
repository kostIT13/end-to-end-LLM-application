from src.services.llm_client import LLMClient
from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database.db import get_db
from src.services.document_chunks.document_chunks_service import DocumentChunksService


async def get_llm_client():
    return LLMClient()

LLMClientDependency = Annotated[LLMClient, Depends(get_llm_client)]


async def get_chunk_service(session: AsyncSession = Depends(get_db)):
    return DocumentChunksService(session)

DocumentChunksServiceDependency = Annotated[DocumentChunksService, Depends(get_chunk_service)]