from pydantic import BaseModel, Field


class CitationModel(BaseModel):
    doc_id: int 
    quote: str = Field(..., min_length=1)


class RAGAnswer(BaseModel):
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)


class RAGResponse(BaseModel):
    answer: str 
    citations: list[CitationModel]
    verified: bool
    invalid_citations: list[CitationModel] = Field(default_factory=list)