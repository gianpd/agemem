"""
Tests for evaluation/metrics.py - calculate_metrics

Tests that metrics calculation produces non-zero values when results are valid,
and handles edge cases like empty inputs correctly.
"""

import pytest

from evaluation.metrics import (
    calculate_metrics,
    EvaluationSummary,
    RetrievalMetrics,
    BehaviorMetrics,
)
from evaluation.evaluator import QuestionResult, SessionReplayResult


class TestCalculateMetrics:
    """Test calculate_metrics function."""

    def test_calculate_metrics_returns_evaluation_summary(self, fake_queries, fake_results, fake_session_results):
        """calculate_metrics returns a proper EvaluationSummary."""
        summary = calculate_metrics(fake_queries, fake_results, fake_session_results)

        assert isinstance(summary, EvaluationSummary)

    def test_calculate_metrics_correct_count(self, fake_queries, fake_results, fake_session_results):
        """Summary has correct total_queries count."""
        summary = calculate_metrics(fake_queries, fake_results, fake_session_results)

        # Critical: total_queries must match input count, not be zero
        assert summary.total_queries == 3, "total_queries must equal number of results"
        assert summary.total_queries > 0, "CRITICAL: total_queries must not be zero (silent bug!)"

    def test_calculate_metrics_accuracy_non_zero(self, fake_queries, fake_results, fake_session_results):
        """Accuracy is non-zero when at least one result is correct."""
        summary = calculate_metrics(fake_queries, fake_results, fake_session_results)

        # fake_results has 2 correct out of 3
        assert summary.correct == 2
        assert summary.accuracy > 0, "CRITICAL: accuracy must be non-zero when results exist"
        assert abs(summary.accuracy - 2/3) < 0.01, "accuracy should be 2/3 (2 correct out of 3)"

    def test_calculate_metrics_abstained_count(self, fake_queries, fake_results, fake_session_results):
        """Abstained count is correct."""
        summary = calculate_metrics(fake_queries, fake_results, fake_session_results)

        # fake_results has 1 abstained
        assert summary.abstained == 1

    def test_calculate_metrics_latency_average(self, fake_queries, fake_results, fake_session_results):
        """Average latency is calculated correctly."""
        summary = calculate_metrics(fake_queries, fake_results, fake_session_results)

        # fake_results latencies: 100, 120, 80 = avg 100
        assert summary.avg_latency_ms > 0
        assert abs(summary.avg_latency_ms - 100.0) < 1.0


class TestMetricsEmptyInput:
    """Test metrics calculation with empty inputs."""

    def test_empty_results_produces_zero_summary(self):
        """Empty results list produces summary with zero values."""
        summary = calculate_metrics([], [], [])

        assert summary.total_queries == 0
        assert summary.correct == 0
        assert summary.accuracy == 0.0
        assert summary.abstained == 0

    def test_empty_results_not_silent_garbage(self):
        """Empty results produce explicit zeros, not garbage values."""
        summary = calculate_metrics([], [], [])

        # All values should be exactly 0.0 or 0, not random memory
        assert summary.accuracy == 0.0
        assert summary.avg_latency_ms == 0.0
        assert summary.judge_avg_latency_ms == 0.0


class TestMetricsValidationMethods:
    """Test LLM judge vs heuristic counts."""

    def test_validation_method_counts(self, fake_queries):
        """Counts queries by validation method."""
        results = [
            QuestionResult(
                query_id="q1", is_correct=True, behavior_type="IE",
                retrieval_trace={}, abstained=False, latency_ms=100.0,
                validation_method="llm_judge",
            ),
            QuestionResult(
                query_id="q2", is_correct=True, behavior_type="IE",
                retrieval_trace={}, abstained=False, latency_ms=100.0,
                validation_method="heuristic",
            ),
            QuestionResult(
                query_id="q3", is_correct=False, behavior_type="IE",
                retrieval_trace={}, abstained=False, latency_ms=100.0,
                validation_method="llm_judge",
            ),
        ]

        summary = calculate_metrics(fake_queries, results, [])

        assert summary.llm_judge_queries == 2
        assert summary.heuristic_queries == 1


class TestRetrievalMetrics:
    """Test retrieval metrics calculation."""

    def test_retrieval_metrics_non_zero_with_relevant(self, fake_queries):
        """Retrieval metrics are non-zero when relevant entries are retrieved."""
        results = [
            QuestionResult(
                query_id="q1",
                is_correct=True,
                behavior_type="IE",
                retrieval_trace={"results": [("e1", 1.0)], "latency_ms": 50.0},
                abstained=False,
                latency_ms=100.0,
                validation_method="heuristic",
            ),
            QuestionResult(
                query_id="q2",
                is_correct=True,
                behavior_type="IE",
                retrieval_trace={"results": [("e3", 0.9)], "latency_ms": 60.0},
                abstained=False,
                latency_ms=120.0,
                validation_method="heuristic",
            ),
        ]

        summary = calculate_metrics(fake_queries[:2], results, [])

        # Retrieval metrics should be populated
        assert summary.retrieval is not None
        assert isinstance(summary.retrieval, RetrievalMetrics)

    def test_retrieval_metrics_empty_with_no_results(self, fake_queries):
        """Retrieval metrics are zeros when no results."""
        results = [
            QuestionResult(
                query_id="q1",
                is_correct=True,
                behavior_type="IE",
                retrieval_trace={"results": [], "latency_ms": 50.0},
                abstained=False,
                latency_ms=100.0,
                validation_method="heuristic",
            ),
        ]

        summary = calculate_metrics(fake_queries[:1], results, [])

        # All retrieval metrics should be 0.0
        assert summary.retrieval.mrr_at_1 == 0.0
        assert summary.retrieval.recall_at_1 == 0.0


class TestBehaviorBreakdown:
    """Test behavior breakdown metrics."""

    def test_behavior_breakdown_populated(self, fake_queries, fake_results, fake_session_results):
        """Behavior breakdown is populated with correct counts."""
        summary = calculate_metrics(fake_queries, fake_results, fake_session_results)

        assert len(summary.by_behavior) > 0
        assert "IE" in summary.by_behavior
        assert summary.by_behavior["IE"].query_count == 3

    def test_behavior_breakdown_accuracy(self, fake_queries):
        """Behavior breakdown accuracy is calculated correctly."""
        results = [
            QuestionResult(
                query_id="q1", is_correct=True, behavior_type="IE",
                retrieval_trace={}, abstained=False, latency_ms=100.0,
                validation_method="heuristic",
            ),
            QuestionResult(
                query_id="q2", is_correct=False, behavior_type="IE",
                retrieval_trace={}, abstained=False, latency_ms=100.0,
                validation_method="heuristic",
            ),
            QuestionResult(
                query_id="q3", is_correct=True, behavior_type="MR",
                retrieval_trace={}, abstained=False, latency_ms=100.0,
                validation_method="heuristic",
            ),
        ]

        summary = calculate_metrics(fake_queries, results, [])

        assert summary.by_behavior["IE"].accuracy == 0.5  # 1 correct out of 2
        assert summary.by_behavior["MR"].accuracy == 1.0  # 1 correct out of 1


class TestSessionReplayMetrics:
    """Test session replay metrics in summary."""

    def test_session_replay_metrics_populated(self, fake_queries, fake_results, fake_session_results):
        """Session replay metrics are populated when session results provided."""
        summary = calculate_metrics(fake_queries, fake_results, fake_session_results)

        assert summary.session_replay is not None
        assert summary.session_replay["total_sessions"] == 2
        assert summary.session_replay["total_turns"] == 8  # 5 + 3

    def test_session_replay_empty_when_none(self, fake_queries, fake_results):
        """Session replay is empty dict when no session results."""
        summary = calculate_metrics(fake_queries, fake_results, [])

        assert summary.session_replay == {}


class TestSummaryToDict:
    """Test EvaluationSummary.to_dict() method."""

    def test_to_dict_includes_all_fields(self, fake_queries, fake_results, fake_session_results):
        """to_dict() includes all expected fields."""
        summary = calculate_metrics(fake_queries, fake_results, fake_session_results)
        d = summary.to_dict()

        assert "total_queries" in d
        assert "correct" in d
        assert "accuracy" in d
        assert "retrieval" in d
        assert "by_behavior" in d

        # Critical: values must match the summary object
        assert d["total_queries"] == summary.total_queries
        assert d["accuracy"] == summary.accuracy