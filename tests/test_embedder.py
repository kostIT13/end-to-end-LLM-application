import numpy as np
from src.services.rag.embedder import get_embed_model


def test_get_embed_model_returns_string():
    model = get_embed_model()
    assert isinstance(model, str)
    assert len(model) > 0


def test_get_embed_model_is_cached():
    model1 = get_embed_model()
    model2 = get_embed_model()
    assert model1 is model2