"""
Custom RAG Evaluation Metrics
==============================

Metrics that are computed directly (no LLM-as-judge needed):
  - MRR (Mean Reciprocal Rank)
  - Hit Rate @k
  - Latency (P50 / P95)
  - Token Cost

These complement RAGAS LLM-based metrics (faithfulness, context precision, etc.).
"""

import time
import numpy as np
from typing import Callable, Dict, List, Optional, Any


# =============================================================================
# RETRIEVAL METRICS
# =============================================================================


def compute_mrr(
    search_results: List[Dict],
    ground_truth_context: str,
    similarity_threshold: float = 0.7,
) -> float:
    """
    Compute Mean Reciprocal Rank for a single query.

    Checks each retrieved document against the ground truth context.
    A match is found when the ground truth text overlaps significantly
    with a retrieved chunk (substring containment or high overlap).

    Args:
        search_results: List of dicts from Retriever.search(), each with 'document' key.
        ground_truth_context: The source chunk that contains the correct answer.
        similarity_threshold: Minimum fraction of ground truth words found in a chunk
                              to consider it a match.

    Returns:
        Reciprocal rank (1/rank) or 0.0 if no match found.
    """
    gt_words = set(ground_truth_context.lower().split())
    if not gt_words:
        return 0.0

    for rank, result in enumerate(search_results, start=1):
        doc_text = result.get("document", "").lower()
        doc_words = set(doc_text.split())

        # Check overlap: what fraction of ground truth words appear in this chunk
        overlap = len(gt_words & doc_words) / len(gt_words)
        if overlap >= similarity_threshold:
            return 1.0 / rank

        # Also check direct substring containment (for exact matches)
        if ground_truth_context[:200].lower() in doc_text:
            return 1.0 / rank

    return 0.0


def compute_hit_rate(
    search_results: List[Dict],
    ground_truth_context: str,
    k: int = 5,
    similarity_threshold: float = 0.7,
) -> float:
    """
    Compute Hit Rate @k for a single query.

    Returns 1.0 if the ground truth document is found in the top-k results, else 0.0.

    Args:
        search_results: List of dicts from Retriever.search().
        ground_truth_context: The source chunk containing the correct answer.
        k: Only consider the top-k results.
        similarity_threshold: Word overlap threshold for matching.

    Returns:
        1.0 (hit) or 0.0 (miss).
    """
    top_k = search_results[:k]
    mrr = compute_mrr(top_k, ground_truth_context, similarity_threshold)
    return 1.0 if mrr > 0.0 else 0.0


def compute_mrr_dataset(
    all_results: List[List[Dict]],
    all_ground_truths: List[str],
) -> float:
    """
    Compute average MRR across an entire eval dataset.

    Args:
        all_results: List of search results per query.
        all_ground_truths: Ground truth context per query.

    Returns:
        Mean MRR score.
    """
    scores = [
        compute_mrr(results, gt)
        for results, gt in zip(all_results, all_ground_truths)
    ]
    return float(np.mean(scores)) if scores else 0.0


def compute_hit_rate_dataset(
    all_results: List[List[Dict]],
    all_ground_truths: List[str],
    k: int = 5,
) -> float:
    """
    Compute average Hit Rate @k across an entire eval dataset.

    Args:
        all_results: List of search results per query.
        all_ground_truths: Ground truth context per query.
        k: Only consider top-k results.

    Returns:
        Mean Hit Rate.
    """
    scores = [
        compute_hit_rate(results, gt, k=k)
        for results, gt in zip(all_results, all_ground_truths)
    ]
    return float(np.mean(scores)) if scores else 0.0


# =============================================================================
# LATENCY METRICS
# =============================================================================


def measure_latency(
    fn: Callable,
    args: tuple = (),
    kwargs: dict = None,
    n_runs: int = 1,
) -> Dict[str, float]:
    """
    Measure execution latency of a function over n_runs.

    Args:
        fn: The function to time (e.g., generate_answer).
        args: Positional arguments to pass.
        kwargs: Keyword arguments to pass.
        n_runs: Number of times to run for percentile accuracy.

    Returns:
        Dict with 'p50_ms', 'p95_ms', 'mean_ms', 'min_ms', 'max_ms'.
    """
    if kwargs is None:
        kwargs = {}

    timings = []
    result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        timings.append(elapsed)

    arr = np.array(timings)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n_runs": n_runs,
        "last_result": result,
    }


# =============================================================================
# TOKEN COST METRICS
# =============================================================================


def extract_token_usage(openai_response) -> Dict[str, int]:
    """
    Extract token usage from an OpenAI chat completion response.

    Args:
        openai_response: The response object from client.chat.completions.create()

    Returns:
        Dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'.
    """
    usage = getattr(openai_response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def compute_avg_token_cost(token_usages: List[Dict[str, int]]) -> Dict[str, float]:
    """
    Compute average token usage across multiple queries.

    Args:
        token_usages: List of dicts from extract_token_usage().

    Returns:
        Dict with average prompt, completion, and total tokens.
    """
    if not token_usages:
        return {"avg_prompt_tokens": 0, "avg_completion_tokens": 0, "avg_total_tokens": 0}

    return {
        "avg_prompt_tokens": float(np.mean([u["prompt_tokens"] for u in token_usages])),
        "avg_completion_tokens": float(np.mean([u["completion_tokens"] for u in token_usages])),
        "avg_total_tokens": float(np.mean([u["total_tokens"] for u in token_usages])),
    }
