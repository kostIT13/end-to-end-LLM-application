from fastapi import APIRouter, HTTPException, status
from src.api.pipeline_schemas import RAGResponse
from src.api.base_schemas import AskRequest


router = APIRouter(prefix="/api/v1/base", tags=["Main API"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/ask", response_model=RAGResponse)
async def ask(req: AskRequest) -> RAGResponse:
    pass