import argparse
import asyncio
from pathlib import Path
from loguru import logger
from src.core.config import settings
from src.services.rag.chunker import fixed_chunker
from src.services.rag.embedder import embed_passages
from src.services.rag.bm25_retrieval import BM25Index
from src.core.database.db import get_db
from src.services.document_chunks.repository import SQLAlchemyDocumentChunksRepository
from src.services.document_chunks.document_chunks_service import DocumentChunksService


async def index_corpus(corpus_dir: Path) -> None:
    if not corpus_dir.exists():
        logger.error("Corpus directory not found: {}", corpus_dir)
        return
    
    async for session in get_db():
        repo = SQLAlchemyDocumentChunksRepository(session)
        service = DocumentChunksService(repo, session)
        
        txt_files = sorted(corpus_dir.glob("*.txt"))
        if not txt_files:
            logger.warning("No .txt files found in {}", corpus_dir)
            return
        
        logger.info("Found {} files to index", len(txt_files))
        
        total_chunks = 0
        for file_path in txt_files:
            try:
                text = file_path.read_text(encoding="utf-8")
                if not text.strip():
                    logger.warning("Empty file: {}", file_path.name)
                    continue
                
                logger.info("Indexing: {} ({} chars)", file_path.name, len(text))
                
                doc_id = file_path.stem  
                num_chunks = await service.index_document(
                    doc_id=doc_id,
                    text=text,
                    source=file_path.name,
                    metadata={
                        "file_name": file_path.name,
                        "file_size": file_path.stat().st_size,
                    }
                )
                total_chunks += num_chunks
                
            except Exception as e:
                logger.error("Failed to index {}: {}", file_path.name, e)
        
        logger.info("Indexing complete: {} chunks inserted", total_chunks)
        
        logger.info("Rebuilding BM25 index...")
        await service._rebuild_bm25()
        logger.info("BM25 index saved")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Index document corpus for RAG")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=settings.corpus_dir or Path("data/corpus"),
        help="Path to corpus directory with .txt files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing (overwrite existing chunks)"
    )
    args = parser.parse_args()
    
    logger.info("Starting indexing...")
    logger.info("Corpus dir: {}", args.corpus_dir)
    logger.info("Embedding model: {}", settings.OLLAMA_EMBED_MODEL)
    
    await index_corpus(args.corpus_dir)


if __name__ == "__main__":
    asyncio.run(main())