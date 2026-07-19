from pydantic import BaseModel, Field
from typing import Optional


class CitationModel(BaseModel):
    doc_id: int
    quote: str = Field(..., min_length=1, description="Дословная цитата из документа")


class RAGAnswer(BaseModel):
    answer: str = Field(..., description="Текст ответа на русском языке")
    citations: list[CitationModel] = Field(
        default_factory=list,
        description="Список цитат из документов, на которые опирается ответ",
    )


class RAGResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    sources: list[dict] = Field(default_factory=list)
    verified: bool = True
    has_valid_citations: bool = True
    invalid_citations: list[CitationModel] = Field(default_factory=list)
    model_used: str = "unknown"
    processing_time_ms: float = 0.0
    conversation_id: Optional[str] = None
    injection_detected: bool = False
    security_warning: Optional[str] = None