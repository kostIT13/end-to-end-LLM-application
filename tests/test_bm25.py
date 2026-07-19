from src.services.rag.bm25_retrieval import tokenize


def test_tokenize_russian():
    tokens = tokenize("Привет мир! Как дела?")
    assert "привет" in tokens
    assert "мир" in tokens
    assert "дела" in tokens


def test_tokenize_english():
    tokens = tokenize("Hello World! How are you?")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_special_chars():
    tokens = tokenize("test@#$%^&*()")
    assert tokens == ["test"]