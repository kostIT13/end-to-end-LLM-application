from abc import ABC, abstractmethod
from typing import List


class DocumentChunksRepository(ABC):
    @abstractmethod
    async def search_dense(self, embedding: list[float], k: int) -> List[dict]:
        raise NotImplementedError
    
    @abstractmethod
    async def upsert_chunks(self, chunks: list[dict]) -> None:
        raise NotImplementedError