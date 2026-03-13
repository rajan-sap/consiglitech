"""
Unit tests — RAG Evaluation Metrics

Pure-computation tests for MRR, Hit Rate, Latency, and Token Cost.
These run instantly with no external dependencies.
"""

import time
import pytest

pytestmark = pytest.mark.unit


# ── MRR ──────────────────────────────────────────────────────────────────────

class TestComputeMRR:

    def _mrr(self, results, gt, threshold=0.7):
        from evaluation.metrics import compute_mrr
        return compute_mrr(results, gt, similarity_threshold=threshold)

    def test_perfect_match_rank_1(self, sample_search_results):
        """Ground truth at rank 1 → MRR = 1.0."""
        gt = "BMW Group achieved total revenues of EUR 155.5 billion in fiscal year 2023"
        assert self._mrr(sample_search_results, gt) == 1.0

    def test_match_rank_2(self, sample_search_results):
        """Ground truth matching rank 2 → MRR = 0.5."""
        gt = "Ford Motor Company reported total revenue of USD 176.2 billion"
        assert self._mrr(sample_search_results, gt) == 0.5

    def test_no_match(self, sample_search_results):
        """Completely unrelated ground truth → MRR = 0.0."""
        gt = "Apple Inc reported record services revenue in Q4 2023"
        assert self._mrr(sample_search_results, gt) == 0.0

    def test_empty_results(self):
        assert self._mrr([], "any text") == 0.0

    def test_empty_ground_truth(self, sample_search_results):
        assert self._mrr(sample_search_results, "") == 0.0


# ── Hit Rate ─────────────────────────────────────────────────────────────────

class TestComputeHitRate:

    def _hit(self, results, gt, k=5):
        from evaluation.metrics import compute_hit_rate
        return compute_hit_rate(results, gt, k=k)

    def test_hit_at_k1(self, sample_search_results):
        gt = "BMW Group achieved total revenues of EUR 155.5 billion in fiscal year 2023"
        assert self._hit(sample_search_results, gt, k=1) == 1.0

    def test_miss_at_k1_hit_at_k3(self, sample_search_results):
        gt = "Tesla total revenues were USD 96.8 billion"
        assert self._hit(sample_search_results, gt, k=1) == 0.0
        assert self._hit(sample_search_results, gt, k=3) == 1.0

    def test_empty_results(self):
        assert self._hit([], "any text", k=5) == 0.0

    def test_k_larger_than_results(self, sample_search_results):
        gt = "BMW Group achieved total revenues"
        assert self._hit(sample_search_results, gt, k=100) == 1.0


# ── Dataset-level metrics ────────────────────────────────────────────────────

class TestDatasetMetrics:

    def test_mrr_dataset_average(self, sample_search_results):
        from evaluation.metrics import compute_mrr_dataset
        # Two queries: one perfect hit (rank 1), one miss
        all_results = [sample_search_results, sample_search_results]
        all_gts = [
            "BMW Group achieved total revenues of EUR 155.5 billion in fiscal year 2023",
            "Apple Inc services revenue",
        ]
        avg = compute_mrr_dataset(all_results, all_gts)
        assert avg == pytest.approx(0.5, abs=0.01)  # (1.0 + 0.0) / 2

    def test_hit_rate_dataset(self, sample_search_results):
        from evaluation.metrics import compute_hit_rate_dataset
        all_results = [sample_search_results, sample_search_results]
        all_gts = [
            "BMW Group achieved total revenues of EUR 155.5 billion in fiscal year 2023",
            "Apple nonsense",
        ]
        avg = compute_hit_rate_dataset(all_results, all_gts, k=5)
        assert avg == pytest.approx(0.5, abs=0.01)


# ── Latency ──────────────────────────────────────────────────────────────────

class TestMeasureLatency:

    def test_returns_expected_keys(self):
        from evaluation.metrics import measure_latency
        stats = measure_latency(lambda: "ok", n_runs=2)
        for key in ("p50_ms", "p95_ms", "mean_ms", "min_ms", "max_ms", "n_runs"):
            assert key in stats

    def test_captures_result(self):
        from evaluation.metrics import measure_latency
        stats = measure_latency(lambda: 42, n_runs=1)
        assert stats["last_result"] == 42

    def test_timing_is_reasonable(self):
        from evaluation.metrics import measure_latency
        stats = measure_latency(lambda: time.sleep(0.05), n_runs=1)
        assert stats["mean_ms"] >= 40  # at least ~40ms


# ── Token Cost ───────────────────────────────────────────────────────────────

class TestTokenCost:

    def test_extract_from_none_usage(self):
        """If response has no 'usage' attr, return zeros."""
        from evaluation.metrics import extract_token_usage

        class FakeResponse:
            pass

        result = extract_token_usage(FakeResponse())
        assert result["total_tokens"] == 0

    def test_average_token_cost(self):
        from evaluation.metrics import compute_avg_token_cost
        usages = [
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        ]
        avg = compute_avg_token_cost(usages)
        assert avg["avg_total_tokens"] == pytest.approx(225.0)

    def test_empty_usages(self):
        from evaluation.metrics import compute_avg_token_cost
        avg = compute_avg_token_cost([])
        assert avg["avg_total_tokens"] == 0
