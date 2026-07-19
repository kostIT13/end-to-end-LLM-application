import pytest
from src.services.generation.verifier import Citation, verify_citations, verify_citations_from_chunks, filter_valid_citations


@pytest.mark.asyncio
async def test_verify_citations_all_valid():
    citations = [Citation(doc_id=1, quote="hello world")]
    context = {1: "hello world and more"}
    all_valid, invalid = await verify_citations(citations, context)
    assert all_valid is True
    assert len(invalid) == 0


@pytest.mark.asyncio
async def test_verify_citations_invalid_quote():
    citations = [Citation(doc_id=1, quote="not in text")]
    context = {1: "completely different text"}
    all_valid, invalid = await verify_citations(citations, context)
    assert all_valid is False
    assert len(invalid) == 1


@pytest.mark.asyncio
async def test_verify_citations_unknown_doc():
    citations = [Citation(doc_id=99, quote="some text")]
    context = {1: "some text"}
    all_valid, invalid = await verify_citations(citations, context)
    assert all_valid is False
    assert len(invalid) == 1


@pytest.mark.asyncio
async def test_verify_citations_from_chunks():
    citations = [Citation(doc_id=1, quote="machine learning")]
    chunks = [{"doc_id": 1, "text": "machine learning is great"}]
    all_valid, invalid = await verify_citations_from_chunks(citations, chunks)
    assert all_valid is True


@pytest.mark.asyncio
async def test_filter_valid_citations():
    citations = [
        Citation(doc_id=1, quote="valid quote"),
        Citation(doc_id=2, quote="invalid quote"),
    ]
    chunks = [
        {"doc_id": 1, "text": "valid quote here"},
        {"doc_id": 2, "text": "something else"},
    ]
    filtered = await filter_valid_citations(citations, chunks)
    assert len(filtered) == 1
    assert filtered[0].doc_id == 1