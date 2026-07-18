import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

BM25_PATH = Path("data/bm25_index.pkl")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    async def search(self, query: str, k: int) -> list[dict]:
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: -x[1])[:k]
        return [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "text": c["text"],
                "score": float(s),
            }
            for c, s in ranked
        ]

    def save(self, path: Path = BM25_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "tokenized": self.tokenized}, f)

    @classmethod
    def load(cls, path: Path = BM25_PATH) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls.__new__(cls)
        inst.chunks = data["chunks"]
        inst.tokenized = data["tokenized"]
        inst.bm25 = BM25Okapi(inst.tokenized)
        return inst