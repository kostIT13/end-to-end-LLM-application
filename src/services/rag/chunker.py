from typing import List


def fixed_chunker(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
        i += step
    
    return chunks