from src.config import settings
from openai import AsyncOpenAI, APITimeoutError, RateLimitError
from typing import TypeVar, Type
from pydantic import BaseModel
from loguru import logger
import asyncio


T = TypeVar("T", bound = BaseModel)
RETRYABLE = (RateLimitError, APITimeoutError)


class LLMClient:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.request_timeout_s    
        )
    
    async def _call(self, model: str, messages: list[dict], **kwargs) -> str:
        kwargs.pop('temperature', None)
        kwargs.pop('max_tokens', None)

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            llm_temperature=settings.llm_temperature, 
            llm_max_tokens=settings.llm_max_tokens,
            **kwargs
        ) 
        return response.choices[0].message.content or ""
        
    async def _try_with_retries(self, model: str, messages: list[dict], **kwargs) -> tuple[str, Exception | None]:
        last_exc = None

        for attempt in range(settings.max_retries):
            try:
                return await self._call(model=model, messages=messages, **kwargs)
            except RETRYABLE as e:
                last_exc = e 
                wait = min(2 ** attempt, 10)
                logger.warning(
                    "retry {n}/{m} on {exc}, sleep {s}s",
                    n = attempt + 1, m=settings.max_retries, exc=type(e).__name__, s=wait
                )
                await asyncio.sleep(wait)
        return "", last_exc
    
    async def _fallback(self, messages: list[dict], **kwargs) -> str:
        logger.warning("primary exhausted, falling back to cheap model")
        try:
            return await self._call(model=settings.llm_model_cheap, messages=messages, **kwargs)
        except Exception as e:
            logger.error("fallback also failed: {exc}", exc=type(e).__name__)
            raise 
    
    async def chat(self, messages: list[dict], model: str | None = None, **kwargs) -> str:
        target = model or settings.llm_model_primary
        result, last_exc = await self._try_with_retries(target, messages, **kwargs)
        if last_exc is None:
            return result 
        
        if target != settings.llm_model_cheap:
            return await self._fallback(messages, **kwargs)
        
        raise last_exc

async def parse_structured(self, llm: LLMClient, user_prompt: str, schema: Type[T], max_retries: int = 3, system_prompt: str | None = None) -> T:
    pass
