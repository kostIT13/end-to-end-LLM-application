import pytest
from src.api.pipeline_schemas import RAGAnswer, CitationModel, RAGResponse


def test_rag_answer_valid():
    data = {"answer": "test answer", "citations": [{"doc_id": 1, "quote": "test"}]}
    obj = RAGAnswer(**data)
    assert obj.answer == "test answer"
    assert len(obj.citations) == 1


def test_rag_answer_empty_citations():
    data = {"answer": "test"}
    obj = RAGAnswer(**data)
    assert obj.citations == []


def test_rag_answer_invalid_missing_answer():
    with pytest.raises(Exception):
        RAGAnswer(**{})


def test_citation_model_valid():
    c = CitationModel(doc_id=5, quote="exact quote")
    assert c.doc_id == 5
    assert c.quote == "exact quote"


def test_citation_model_empty_quote():
    with pytest.raises(Exception):
        CitationModel(doc_id=1, quote="")


def test_rag_response_defaults():
    r = RAGResponse(answer="test", citations=[])
    assert r.has_valid_citations is True
    assert r.injection_detected is False
    assert r.model_used == "unknown"