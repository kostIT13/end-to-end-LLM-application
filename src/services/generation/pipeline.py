import json
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
from src.services.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from src.services.generation.verifier import Citation, verify_citations_from_chunks
from src.services.llm_client import LLMClient
from src.services.document_chunks.document_chunks_service import DocumentChunksService
from src.api.pipeline_schemas import RAGResponse
from src.core.config import settings


async def answer_question(
    query: str,
    llm_client: LLMClient,
    chunk_service: DocumentChunksService,
    top_k: int = 5,
    use_hybrid: bool = True,
) -> RAGResponse:
    chunks = await chunk_service.search(
        query=query,
        top_k=top_k,
        use_hybrid=use_hybrid,
    )
    
    if not chunks:
        logger.warning("No relevant chunks found for query: {}", query[:50])
        return RAGResponse(
            answer="Недостаточно данных в предоставленных документах.",
            sources=[],
            citations=[],
            has_valid_citations=True,
            model_used=settings.LLM_MODEL_PRIMARY,
        )
    
    logger.info("Retrieved {} chunks for query", len(chunks))
    
    user_prompt = build_rag_user_prompt(query, chunks)
    
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        raw_answer = await llm_client.chat(
            messages=messages,
            temperature=0.1,
        )
        
        parsed = json.loads(raw_answer)
        answer_text = parsed.get("answer", raw_answer)
        citations_raw = parsed.get("citations", [])
        
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: {}", e)
        return RAGResponse(
            answer=raw_answer,
            sources=chunks,
            citations=[],
            has_valid_citations=True,
            model_used=settings.LLM_MODEL_PRIMARY,
        )
    
    citations = [
        Citation(doc_id=c["doc_id"], quote=c["quote"])
        for c in citations_raw
        if isinstance(c, dict) and "doc_id" in c and "quote" in c
    ]
    
    all_valid = True
    invalid_citations = []
    
    if citations:
        all_valid, invalid = await verify_citations_from_chunks(
            citations=citations,
            chunks=chunks,
        )
        invalid_citations = [
            {"doc_id": c.doc_id, "quote": c.quote}
            for c in invalid
        ]
        
        if not all_valid:
            logger.warning("Found {} invalid citations", len(invalid))
            valid_keys = {(c.doc_id, c.quote) for c in citations if c not in invalid}
            citations_raw = [
                c for c in citations_raw
                if (c.get("doc_id"), c.get("quote")) in valid_keys
            ]
    
    return RAGResponse(
        answer=answer_text,
        sources=chunks,
        citations=citations_raw,
        has_valid_citations=all_valid,
        invalid_citations=invalid_citations if not all_valid else None,
        model_used=settings.LLM_MODEL_PRIMARY,
    )