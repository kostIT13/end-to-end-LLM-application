import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from src.core.logging_settings import logger
from src.services.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from src.services.generation.verifier import Citation, verify_citations_from_chunks
from src.services.llm_client import LLMClient
from src.services.document_chunks.document_chunks_service import DocumentChunksService
from src.api.pipeline_schemas import RAGResponse, CitationModel
from src.core.config import settings


async def answer_question(
    query: str,
    llm_client: LLMClient,
    chunk_service: DocumentChunksService,
    top_k: int = 5,
    use_hybrid: bool = True,
    use_reranker: bool = False,
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
    invalid_citation_models = []

    if citations:
        all_valid, invalid = await verify_citations_from_chunks(
            citations=citations,
            chunks=chunks,
        )
        invalid_citation_models = [
            CitationModel(doc_id=c.doc_id, quote=c.quote)
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
        citations=[CitationModel(doc_id=c.get("doc_id", 0), quote=c.get("quote", "")) for c in citations_raw],
        has_valid_citations=all_valid,
        invalid_citations=invalid_citation_models,
        model_used=settings.LLM_MODEL_PRIMARY,
    )


async def answer_question_stream(
    query: str,
    llm_client: LLMClient,
    chunk_service: DocumentChunksService,
    top_k: int = 5,
    use_hybrid: bool = True,
    use_reranker: bool = False,
) -> AsyncGenerator[str, None]:
    chunks = await chunk_service.search(
        query=query,
        top_k=top_k,
        use_hybrid=use_hybrid,
    )

    if not chunks:
        yield "Недостаточно данных в предоставленных документах."
        return

    logger.info("Retrieved {} chunks for streaming", len(chunks))

    user_prompt = build_rag_user_prompt(query, chunks)

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        async for token in llm_client.chat_stream(messages, temperature=0.1):
            yield token
    except Exception as e:
        logger.error("Streaming error: {}", e)
        yield f"\n\nError: {str(e)}"