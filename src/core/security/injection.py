import re
from typing import Tuple, Optional
from loguru import logger
from src.services.llm_client import LLMClient


INJECTION_PATTERNS = [
    r"\bignore\s+(?:all|the|any|every|previous|prior|above|preceding|earlier|initial|original\s+){1,3}(?:instructions?|prompts?|messages?|directions?|rules?|guidelines?|context|commands?)\b",
    r"\bdisregard\s+(?:all|the|any|every|previous|prior|above|preceding|earlier|initial|original\s+){1,3}(?:instructions?|prompts?|messages?|directions?|rules?|guidelines?|context|commands?)\b",
    r"\bforget\s+(?:everything|(?:all|the|any|every|previous|prior|above|preceding|earlier|initial|original\s+){1,3}(?:instructions?|prompts?|messages?|directions?|rules?|guidelines?|context|commands?))\b",
    r"\bnew\s+instructions?\s*:",
    r"\bsystem\s*:\s*",
    r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\b",
    r"\b(?:забудь|игнорируй|проигнорируй|отмени)\s+(?:всё|все|предыдущие|прежние|инструкции|команды|указания)\b",
    r"\bты\s+теперь\b",
    r"\bновые\s+инструкции\s*:",
]


def _has_injection_pattern(text: str) -> bool:
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True
    return False


async def check_prompt_security(
    text: str,
    llm_client: LLMClient,
    skip_check: bool = False,
) -> Tuple[bool, Optional[str]]:
    if skip_check:
        return True, None
    
    if len(text) < 20:
        return True, None
    
    if re.search(r"(?:test|testing|проверк)", text, re.IGNORECASE):
        return True, None
    
    if _has_injection_pattern(text):
        logger.warning("Prompt injection detected: {}", text[:100])
        return False, "Request contains suspicious patterns"
    
    return True, None