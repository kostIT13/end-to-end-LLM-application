from src.core.config import settings
from openai import AsyncOpenAI, APITimeoutError, RateLimitError
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError
from src.core.logging_settings import logger
import asyncio
import json


T = TypeVar("T", bound = BaseModel)
RETRYABLE = (RateLimitError, APITimeoutError)


class LLMClient:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            timeout=settings.REQUES_TIMEOUT_S    
        )
    
    async def _call(self, model: str, messages: list[dict], **kwargs) -> str:
        kwargs.pop('temperature', None)
        kwargs.pop('max_tokens', None)

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            llm_temperature=settings.LLM_TEMPERATURE, 
            llm_max_tokens=settings.LLM_MAX_TOKENS,
            **kwargs
        ) 
        return response.choices[0].message.content or ""
        
    async def _try_with_retries(self, model: str, messages: list[dict], **kwargs) -> tuple[str, Exception | None]:
        last_exc = None

        for attempt in range(settings.MAX_RETRIES):
            try:
                return await self._call(model=model, messages=messages, **kwargs)
            except RETRYABLE as e:
                last_exc = e 
                wait = min(2 ** attempt, 10)
                logger.warning(
                    "retry {n}/{m} on {exc}, sleep {s}s",
                    n = attempt + 1, m=settings.MAX_RETRIES, exc=type(e).__name__, s=wait
                )
                await asyncio.sleep(wait)
        return "", last_exc
    
    async def _fallback(self, messages: list[dict], **kwargs) -> str:
        logger.warning("primary exhausted, falling back to cheap model")
        try:
            return await self._call(model=settings.LLM_MODEL_CHEAP, messages=messages, **kwargs)
        except Exception as e:
            logger.error("fallback also failed: {exc}", exc=type(e).__name__)
            raise 
    
    async def chat(self, messages: list[dict], model: str | None = None, **kwargs) -> str:
        target = model or settings.LLM_MODEL_PRIMARY
        result, last_exc = await self._try_with_retries(target, messages, **kwargs)
        if last_exc is None:
            return result 
        
        if target != settings.LLM_MODEL_CHEAP:
            return await self._fallback(messages, **kwargs)
        
        raise last_exc

async def parse_structured(llm: LLMClient, user_prompt: str, schema: Type[T], max_retries: int = 3, system_prompt: str | None = None) -> T:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    default_system = (
        f"Respond strictly as JSON conforming to the schema:\n{schema_json}\n"
        f"No text before or after the JSON."
    )
    messages: list[dict] = [
        {"role": "system", "content": system_prompt or default_system},
        {"role": "user", "content": user_prompt},    
    ]
    last_exc: Exception | None = None 
    for attempt in range(max_retries):
        raw = await llm.chat(messages=messages)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_exc = e 
            logger.warning("attempt {n}: invalid JSON: {err}", n=attempt + 1, err=e)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Invalid JSON: {e}. Return a valid object that mathes the schema."})
            continue

        try:
            return schema.model_validate(data)
        except ValidationError as e:
            last_exc = e 
            logger.warning("attempt {n}: schema mismatch: {err}", n=attempt+1, err=e)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"JSON does not match the schema: {e}. Fix it."})
            continue

    if last_exc:
        raise last_exc

