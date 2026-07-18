import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from loguru import logger


@dataclass(frozen=True)
class Citation:
    doc_id: int
    quote: str


def _normalize(s: str, normalize_ws: bool = True, case_sensitive: bool = False) -> str:
    if normalize_ws:
        s = re.sub(r"\s+", " ", s).strip()
    if not case_sensitive:
        s = s.lower()
    return s


async def verify_citations(
    citations: List[Citation],
    context: Dict[int, str],
) -> Tuple[bool, List[Citation]]:
    invalid: List[Citation] = []
    
    for c in citations:
        if c.doc_id not in context:
            logger.warning("Citation refers to unknown doc_id: {}", c.doc_id)
            invalid.append(c)
            continue
        
        quote_norm = _normalize(c.quote)
        if not quote_norm:
            logger.warning("Citation is empty for doc_id: {}", c.doc_id)
            invalid.append(c)
            continue
        
        ctx_norm = _normalize(context[c.doc_id])
        if quote_norm not in ctx_norm:
            logger.warning("Citation not found in doc_id {}: {}", c.doc_id, c.quote[:50])
            invalid.append(c)
    
    all_valid = len(invalid) == 0
    logger.debug("Citation verification: valid={}, invalid={}", 
                 len(citations) - len(invalid), len(invalid))
    
    return all_valid, invalid


async def verify_citations_from_chunks(
    citations: List[Citation],
    chunks: List[Dict[str, Any]],
) -> Tuple[bool, List[Citation]]:
    context: Dict[int, str] = {}
    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        text = chunk.get("text", "")
        if doc_id is not None:
            if doc_id in context:
                context[doc_id] += " " + text
            else:
                context[doc_id] = text
    
    return await verify_citations(citations, context)


async def filter_valid_citations(
    citations: List[Citation],
    chunks: List[Dict[str, Any]],
) -> List[Citation]:
    _, invalid = await verify_citations_from_chunks(citations, chunks)
    invalid_set = {(c.doc_id, _normalize(c.quote)) for c in invalid}
    
    return [
        c for c in citations
        if (c.doc_id, _normalize(c.quote)) not in invalid_set
    ]