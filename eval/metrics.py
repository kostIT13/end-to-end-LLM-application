from typing import List, Dict, Any, Optional, Tuple
import random


def recall_top_k(predicted: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], k: int, key: str = "doc_id") -> float:
    if not ground_truth:
        return 0.0

    predicted_ids = [item.get(key) for item in predicted[:k] if item.get(key) is not None]
    ground_truth_ids = [item.get(key) for item in ground_truth if item.get(key) is not None]

    if not ground_truth_ids:
        return 0.0

    predicted_set = set(predicted_ids)
    ground_truth_set = set(ground_truth_ids)

    correct = len(predicted_set & ground_truth_set)
    total = len(ground_truth_set)

    return round(correct / total, 6)


def mrr(predicted: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], key: str = "doc_id") -> float:
    if not predicted or not ground_truth:
        return 0.0

    predicted_ids = [item.get(key) for item in predicted if item.get(key) is not None]
    ground_truth_ids = [item.get(key) for item in ground_truth if item.get(key) is not None]

    if not predicted_ids or not ground_truth_ids:
        return 0.0

    predicted_unique = []
    seen = set()
    for item_id in predicted_ids:
        if item_id not in seen:
            seen.add(item_id)
            predicted_unique.append(item_id)

    ground_truth_set = set(ground_truth_ids)

    for rank, item_id in enumerate(predicted_unique, start=1):
        if item_id in ground_truth_set:
            return round(1.0 / rank, 6)

    return 0.0


def bootstrap_ci( scores: List[float], n_samples: int = 1000, ci: float = 0.95, seed: Optional[int] = None,) -> Tuple[float, float]:
    if len(scores) < 2:
        raise ValueError("Need at least 2 scores")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    rng = random.Random(seed) if seed is not None else random.Random()
    
    means = []
    n = len(scores)
    for _ in range(n_samples):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    
    alpha = 1 - ci
    lower_idx = int((alpha / 2) * n_samples)
    upper_idx = max(0, min(n_samples - 1, int((1 - alpha / 2) * n_samples) - 1))
    
    return means[lower_idx], means[upper_idx]


def evaluate_with_ci(per_query_recalls: List[float], threshold: float = 0.7, ci: float = 0.95, n_samples: int = 1000, seed: Optional[int] = None,) -> Tuple[float, float, float, bool]:
    avg_recall = sum(per_query_recalls) / len(per_query_recalls)
    lower_ci, upper_ci = bootstrap_ci(per_query_recalls, n_samples, ci, seed)
    passed = lower_ci >= threshold
    
    return avg_recall, lower_ci, upper_ci, passed