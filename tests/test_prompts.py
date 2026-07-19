from src.services.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt


def test_rag_system_prompt_contains_key_rules():
    assert "Опирайся ТОЛЬКО" in RAG_SYSTEM_PROMPT
    assert "external_content" in RAG_SYSTEM_PROMPT


def test_build_rag_user_prompt_includes_query():
    chunks = [{"doc_id": "1", "text": "some content"}]
    prompt = build_rag_user_prompt("test query", chunks)
    assert "test query" in prompt


def test_build_rag_user_prompt_includes_chunks():
    chunks = [
        {"doc_id": "1", "text": "first chunk"},
        {"doc_id": "2", "text": "second chunk"},
    ]
    prompt = build_rag_user_prompt("query", chunks)
    assert "first chunk" in prompt
    assert "second chunk" in prompt
    assert "external_content" in prompt