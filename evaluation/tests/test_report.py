"""
Tests for evaluation/report.py - ReportGenerator

Tests that report generation produces non-empty output with expected content,
and that metric values in reports match the input summary.
"""

import json
import pytest
from pathlib import Path

from evaluation.report import ReportGenerator
from evaluation.metrics import EvaluationSummary, RetrievalMetrics, BehaviorMetrics
from evaluation.batch_checkpoint import CheckpointState, BatchProgress


class TestReportGeneratorMarkdown:
    """Test markdown report generation."""

    def test_generate_markdown_returns_path(self, fake_summary, tmp_path: Path):
        """generate_markdown returns a valid Path."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_markdown(fake_summary, "test_session")

        assert isinstance(path, Path)
        assert path.exists()
        assert path.name == "test_session_report.md"

    def test_generate_markdown_non_empty(self, fake_summary, tmp_path: Path):
        """Markdown report is non-empty."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_markdown(fake_summary, "test_session")

        content = path.read_text(encoding="utf-8")
        assert len(content) > 0, "CRITICAL: Report must not be empty"

    def test_generate_markdown_contains_accuracy(self, fake_summary, tmp_path: Path):
        """Markdown report contains the word 'accuracy' or equivalent metric."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_markdown(fake_summary, "test_session")

        content = path.read_text(encoding="utf-8").lower()
        assert "accuracy" in content, "Report must contain accuracy metric"

    def test_generate_markdown_contains_session_id(self, fake_summary, tmp_path: Path):
        """Markdown report contains the session ID."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_markdown(fake_summary, "my_session_123")

        content = path.read_text(encoding="utf-8")
        assert "my_session_123" in content

    def test_generate_markdown_metrics_not_zero_with_valid_input(self, tmp_path: Path):
        """When summary has non-zero values, report shows non-zero metrics."""
        summary = EvaluationSummary(
            total_queries=10,
            correct=8,
            accuracy=0.8,
            abstained=2,
            avg_latency_ms=150.0,
            llm_judge_queries=5,
            heuristic_queries=5,
            judge_avg_latency_ms=200.0,
            retrieval=RetrievalMetrics(),
            by_behavior={},
            session_replay={},
        )

        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_markdown(summary, "test")

        content = path.read_text(encoding="utf-8")

        # Must show the actual non-zero values
        assert "10" in content  # total_queries
        assert "8" in content   # correct
        # 80% accuracy should appear (0.8 formatted)
        assert "80" in content or "0.8" in content


class TestReportGeneratorJSON:
    """Test JSON report generation."""

    def test_generate_json_returns_path(self, fake_summary, tmp_path: Path):
        """generate_json returns a valid Path."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_json(fake_summary, "test_session")

        assert isinstance(path, Path)
        assert path.exists()
        assert path.name == "test_session_metrics.json"

    def test_generate_json_valid_json(self, fake_summary, tmp_path: Path):
        """JSON report is valid JSON."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_json(fake_summary, "test_session")

        content = path.read_text(encoding="utf-8")
        data = json.loads(content)  # Will raise if invalid

        assert isinstance(data, dict)

    def test_generate_json_contains_summary(self, fake_summary, tmp_path: Path):
        """JSON report contains summary with correct values."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_json(fake_summary, "test_session")

        content = path.read_text(encoding="utf-8")
        data = json.loads(content)

        assert "summary" in data
        assert data["summary"]["total_queries"] == fake_summary.total_queries
        assert data["summary"]["correct"] == fake_summary.correct

    def test_generate_json_metrics_not_zero_with_valid_input(self, tmp_path: Path):
        """JSON summary metrics are not zero when input has non-zero values."""
        summary = EvaluationSummary(
            total_queries=100,
            correct=75,
            accuracy=0.75,
            abstained=10,
            avg_latency_ms=200.0,
            llm_judge_queries=50,
            heuristic_queries=50,
            judge_avg_latency_ms=300.0,
            retrieval=RetrievalMetrics(),
            by_behavior={},
            session_replay={},
        )

        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_json(summary, "test")

        content = path.read_text(encoding="utf-8")
        data = json.loads(content)

        # Critical: values must match input, not be zero
        assert data["summary"]["total_queries"] == 100
        assert data["summary"]["correct"] == 75
        assert data["summary"]["accuracy"] == 0.75


class TestPartialReport:
    """Test partial/in-progress report generation."""

    def test_generate_partial_markdown_returns_path(self, fake_summary, fake_checkpoint_state, tmp_path: Path):
        """generate_partial_markdown returns a valid Path."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_partial_markdown(fake_summary, "test_session", fake_checkpoint_state)

        assert isinstance(path, Path)
        assert path.exists()
        assert "partial" in path.name

    def test_generate_partial_markdown_shows_progress(self, fake_summary, fake_checkpoint_state, tmp_path: Path):
        """Partial markdown shows progress information."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_partial_markdown(fake_summary, "test_session", fake_checkpoint_state)

        content = path.read_text(encoding="utf-8")

        assert "PARTIAL" in content
        assert "15" in content  # completed_interactions
        assert "30" in content  # total_interactions

    def test_generate_partial_json_contains_checkpoint(self, fake_summary, fake_checkpoint_state, tmp_path: Path):
        """Partial JSON contains checkpoint state."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_partial_json(fake_summary, "test_session", fake_checkpoint_state)

        content = path.read_text(encoding="utf-8")
        data = json.loads(content)

        assert "checkpoint" in data
        assert data["checkpoint"]["session_id"] == "test_session_001"
        assert "progress" in data


class TestReportContent:
    """Test report content structure."""

    def test_report_includes_all_sections(self, fake_summary, tmp_path: Path):
        """Full report includes all expected sections."""
        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_markdown(fake_summary, "test")

        content = path.read_text(encoding="utf-8")

        # Should have these sections
        assert "## Summary" in content
        assert "## LLM-as-Judge Statistics" in content
        assert "## Retrieval Metrics" in content
        assert "## Behavior Breakdown" in content

    def test_report_includes_retrieval_metrics(self, tmp_path: Path):
        """Report includes retrieval metric values."""
        retrieval = RetrievalMetrics(
            mrr_at_1=0.5,
            mrr_at_5=0.6,
            precision_at_1=0.7,
            recall_at_10=0.8,
        )
        summary = EvaluationSummary(
            total_queries=10,
            correct=5,
            accuracy=0.5,
            retrieval=retrieval,
            by_behavior={},
            session_replay={},
        )

        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_markdown(summary, "test")

        content = path.read_text(encoding="utf-8")

        # Should contain retrieval metric names
        assert "mrr_at_1" in content or "MRR" in content.lower()

    def test_report_includes_behavior_breakdown(self, tmp_path: Path):
        """Report includes behavior breakdown."""
        summary = EvaluationSummary(
            total_queries=10,
            correct=5,
            accuracy=0.5,
            retrieval=RetrievalMetrics(),
            by_behavior={
                "IE": BehaviorMetrics(behavior="IE", query_count=7, accuracy=0.57),
                "MR": BehaviorMetrics(behavior="MR", query_count=3, accuracy=0.33),
            },
            session_replay={},
        )

        reporter = ReportGenerator(tmp_path)
        path = reporter.generate_markdown(summary, "test")

        content = path.read_text(encoding="utf-8")

        assert "IE" in content
        assert "MR" in content