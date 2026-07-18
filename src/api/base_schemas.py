from pydantic import BaseModel, Field
from typing import Optional


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(
        None,
        description="ID диалога для отслеживания контекста"
    )
    temperature: Optional[float] = Field(0.3, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(500, ge=100, le=4000)
    top_k: Optional[int] = Field(5, ge=1, le=20)
    use_hybrid_search: bool = True
    skip_security_check: bool = False
    use_reranker: Optional[bool] = Field(
        False,
        description="Использовать reranker для улучшения результатов"
    )
    stream: Optional[bool] = Field(
        False,
        description="Использовать стриминг ответа"
    )