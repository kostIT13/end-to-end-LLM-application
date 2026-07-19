import pytest
from src.services.rag.hybrid import rrf_fuse


def test_rrf_fuse_single_ranking():
    rankings = [["a", "b", "c"]]
    result = rrf_fuse(rankings, k=60)
    assert len(result) == 3
    assert result[0][0] == "a"


def test_rrf_fuse_two_rankings():
    rankings = [["a", "b", "c"], ["b", "a", "c"]]
    result = rrf_fuse(rankings, k=60)
    assert result[0][0] == "a"
    assert result[1][0] == "b"


def test_rrf_fuse_k_must_be_positive():
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=0)