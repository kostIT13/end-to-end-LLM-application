import pytest
from src.services.rag.chunker import fixed_chunker


def test_fixed_chunker_basic_split():
    text = "word " * 1000
    chunks = fixed_chunker(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)


def test_fixed_chunker_no_overlap():
    text = "word " * 50
    chunks = fixed_chunker(text, chunk_size=10, overlap=0)
    assert len(chunks) == 5
    assert all(len(c.split()) == 10 for c in chunks)


def test_fixed_chunker_smaller_than_chunk():
    text = "hello world test"
    chunks = fixed_chunker(text, chunk_size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == text