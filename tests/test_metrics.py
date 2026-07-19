import pytest
from eval.metrics import recall_top_k, mrr, bootstrap_ci, evaluate_with_ci


def test_recall_top_k_perfect():
    predicted = [{"doc_id": "1"}, {"doc_id": "2"}, {"doc_id": "3"}]
    ground_truth = [{"doc_id": "1"}, {"doc_id": "2"}]
    assert recall_top_k(predicted, ground_truth, k=3) == 1.0


def test_recall_top_k_partial():
    predicted = [{"doc_id": "1"}, {"doc_id": "4"}, {"doc_id": "5"}]
    ground_truth = [{"doc_id": "1"}, {"doc_id": "2"}, {"doc_id": "3"}]
    assert recall_top_k(predicted, ground_truth, k=3) == pytest.approx(1 / 3)


def test_recall_top_k_empty_ground_truth():
    predicted = [{"doc_id": "1"}]
    ground_truth = []
    assert recall_top_k(predicted, ground_truth, k=5) == 0.0


def test_mrr_first_is_relevant():
    predicted = [{"doc_id": "1"}, {"doc_id": "2"}]
    ground_truth = [{"doc_id": "1"}]
    assert mrr(predicted, ground_truth) == 1.0


def test_mrr_second_is_relevant():
    predicted = [{"doc_id": "3"}, {"doc_id": "1"}]
    ground_truth = [{"doc_id": "1"}]
    assert mrr(predicted, ground_truth) == 0.5


def test_mrr_no_relevant():
    predicted = [{"doc_id": "1"}, {"doc_id": "2"}]
    ground_truth = [{"doc_id": "3"}]
    assert mrr(predicted, ground_truth) == 0.0


def test_bootstrap_ci_returns_tuple():
    scores = [0.8, 0.9, 0.7, 0.85, 0.95]
    lower, upper = bootstrap_ci(scores, n_samples=100, seed=42)
    assert 0.0 <= lower <= upper <= 1.0


def test_evaluate_with_ci_passes():
    scores = [0.9, 0.85, 0.95, 0.88, 0.92]
    avg, lower, upper, passed = evaluate_with_ci(scores, threshold=0.7, n_samples=100, seed=42)
    assert passed is True
    assert lower >= 0.7