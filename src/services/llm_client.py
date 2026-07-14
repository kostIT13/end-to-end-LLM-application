from src.config import settings
from openai import OpenAI


class LLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.request_timeout_s    
        )
    
        

