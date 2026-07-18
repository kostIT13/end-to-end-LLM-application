from fastapi import APIRouter, HTTPException, status, Depends
from src.api.pipeline_schemas import RAGResponse
from src.api.base_schemas import AskRequest
from src.core.security.injection import check_prompt_security
from src.services.generation.pipeline import answer_question
import time
from src.api.dependencies import LLMClientDependency, DocumentChunksServiceDependency
from loguru import logger


router = APIRouter(prefix="/api/v1/base", tags=["Main API"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/ask", response_model=RAGResponse)
async def ask(
    req: AskRequest,
    llm_client: LLMClientDependency,
    chunk_service: DocumentChunksServiceDependency,
) -> RAGResponse:
    start_time = time.time()
    
    if not req.skip_security_check:
        is_safe, warning = await check_prompt_security(
            text=req.question,
            llm_client=llm_client,
            skip_check=req.skip_security_check,
        )
        if not is_safe:
            logger.warning("Injection detected: {}", req.question[:50])
            return RAGResponse(
                answer="I cannot process this request due to security concerns.",
                sources=[],
                citations=[],
                injection_detected=True,
                security_warning=warning,
                model_used="security_filter",
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    try:
        response = await answer_question(
            query=req.question,
            llm_client=llm_client,
            chunk_service=chunk_service,
            top_k=req.top_k or 5,
            use_hybrid=req.use_hybrid_search,
        )
        
        response.processing_time_ms = (time.time() - start_time) * 1000
        response.conversation_id = req.conversation_id
        
        response.injection_detected = False
        
        return response
        
    except Exception as e:
        logger.error("RAG pipeline failed: {}", e, exc_info=True)
        return RAGResponse(
            answer="An error occurred while processing your request.",
            sources=[],
            citations=[],
            model_used="error",
            processing_time_ms=(time.time() - start_time) * 1000,
        )