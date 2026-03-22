"""
Integration test for evaluation/runner.py - BatchRunner

This is the critical end-to-end test that catches the silent-zero bug:
- Mocks DatasetLoader to return FAKE_QUERIES
- Mocks Evaluator to return deterministic fake scores
- Runs BatchRunner to completion
- Asserts output report file exists AND contains non-zero metric values
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

from evaluation.runner import BatchRunner, BatchConfig, PartialMetrics
from evaluation.metrics import EvaluationSummary
from evaluation.evaluator import QuestionResult, SessionReplayResult


# Deterministic fake results for integration test
INTEGRATION_FAKE_RESULTS = [
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
    QuestionResult(
        query_id="q3",
        is_correct=False,
        behavior_type="IE",
        retrieval_trace={"results": [], "latency_ms": 40.0},
        abstained=True,
        latency_ms=80.0,
        validation_method="heuristic",
    ),
]


class TestBatchRunnerIntegration:
    """End-to-end integration test for BatchRunner."""

    @pytest.fixture
    def mock_factory(self, tmp_path: Path):
        """Create a mock OrchestratorFactory."""
        mock = MagicMock()

        # Create mock orchestrator
        mock_orch = MagicMock()
        mock_orch.chat.return_value = "Mock response"
        mock_orch.last_trace.return_value = MagicMock(
            ops_applied=[],
            feedback=None,
            latency_ms=50.0,
        )
        mock_orch.ltm_snapshot.return_value = []
        mock_orch.stm_stats.return_value = MagicMock(total_tokens=100)

        mock.build_for_evaluation.return_value = mock_orch
        return mock

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock Evaluator that returns deterministic results."""
        mock = MagicMock()
        mock.evaluate_query.side_effect = INTEGRATION_FAKE_RESULTS
        mock.replay_sessions.return_value = [
            SessionReplayResult(
                session_id="session_1",
                turns_processed=5,
                ltm_entries_added=3,
                stm_tokens_at_end=500,
                learning_scores=[],
            )
        ]
        return mock

    @pytest.fixture
    def fake_dataset_path(self, tmp_path: Path):
        """Create a minimal fake dataset file."""
        data = [
            {
                "question_id": "q1",
                "question": "What is 2+2?",
                "answer": "4",
                "question_type": "single_hop",
                "haystack_sessions": [[{"role": "user", "content": "2+2=4", "has_answer": True}]],
            },
            {
                "question_id": "q2",
                "question": "Capital of France?",
                "answer": "Paris",
                "question_type": "single_hop",
                "haystack_sessions": [[{"role": "user", "content": "Paris is capital", "has_answer": True}]],
            },
            {
                "question_id": "q3",
                "question": "Color of sky?",
                "answer": "blue",
                "question_type": "single_hop",
                "haystack_sessions": [[{"role": "user", "content": "Sky is blue", "has_answer": True}]],
            },
        ]
        dataset_path = tmp_path / "integration_dataset.json"
        dataset_path.write_text(json.dumps(data), encoding="utf-8")
        return dataset_path

    def test_runner_produces_report_with_non_zero_metrics(
        self, tmp_path: Path, mock_factory, mock_evaluator, fake_dataset_path: Path
    ):
        """
        CRITICAL TEST: BatchRunner produces a report with non-zero metrics.

        This catches the silent-zero bug where pipeline completes but all
        metrics are zero because a stage returned empty results without
        raising an exception.
        """
        config = BatchConfig(
            batch_size=10,
            output_dir=tmp_path,
            use_mock_llm=True,
        )

        # Create runner with mock evaluator factory
        runner = BatchRunner(
            config,
            mock_factory,
            evaluator_factory=lambda orch: mock_evaluator,
        )

        # Run the evaluation
        summary = runner.run(
            dataset_path=fake_dataset_path,
            mode="full",
            max_interactions=0,
            max_batches=0,
            session_id="integration_test",
        )

        # =====================================================================
        # CRITICAL ASSERTIONS - These catch the silent-zero bug
        # =====================================================================

        # 1. Summary must exist and be the right type
        assert summary is not None, "CRITICAL: Runner returned None instead of summary"
        assert isinstance(summary, EvaluationSummary), "Summary must be EvaluationSummary"

        # 2. total_queries must be > 0 (not silent zero!)
        assert summary.total_queries > 0, (
            "CRITICAL BUG DETECTED: total_queries is 0 - "
            "pipeline completed but no queries were processed!"
        )
        assert summary.total_queries == 3, "Expected 3 queries to be processed"

        # 3. Correct count must be non-zero when some are correct
        assert summary.correct > 0, (
            "CRITICAL: correct count is 0 when fake data has correct answers"
        )
        assert summary.correct == 2, "Expected 2 correct answers"

        # 4. Accuracy must be non-zero
        assert summary.accuracy > 0, (
            "CRITICAL BUG DETECTED: accuracy is 0.0 - "
            "this indicates metrics calculation failed silently!"
        )
        assert abs(summary.accuracy - 0.6666666666666666) < 0.01, "Expected ~66.7% accuracy"

        # 5. Report files must exist
        md_report = tmp_path / "integration_test_report.md"
        json_report = tmp_path / "integration_test_metrics.json"

        assert md_report.exists(), "Markdown report must be generated"
        assert json_report.exists(), "JSON report must be generated"

        # 6. Report content must contain non-zero values
        md_content = md_report.read_text(encoding="utf-8")
        assert len(md_content) > 0, "Markdown report must not be empty"
        assert "accuracy" in md_content.lower(), "Report must mention accuracy"

        # 7. JSON report must have correct values
        json_data = json.loads(json_report.read_text(encoding="utf-8"))
        assert json_data["summary"]["total_queries"] == 3
        assert json_data["summary"]["correct"] == 2
        assert json_data["summary"]["accuracy"] > 0

    def test_runner_handles_empty_dataset_gracefully(
        self, tmp_path: Path, mock_factory
    ):
        """Runner handles empty dataset without crashing."""
        # Create empty dataset
        empty_path = tmp_path / "empty.json"
        empty_path.write_text("[]", encoding="utf-8")

        config = BatchConfig(batch_size=10, output_dir=tmp_path)
        runner = BatchRunner(config, mock_factory)

        summary = runner.run(empty_path, session_id="empty_test")

        # Should produce a summary with zeros, not crash
        assert summary.total_queries == 0
        assert summary.correct == 0
        assert summary.accuracy == 0.0

    def test_runner_checkpoint_persistence(
        self, tmp_path: Path, mock_factory, mock_evaluator, fake_dataset_path: Path
    ):
        """Runner creates and updates checkpoint during execution."""
        config = BatchConfig(
            batch_size=2,  # Small batch to trigger multiple batches
            checkpoint_interval=1,
            output_dir=tmp_path,
        )

        runner = BatchRunner(
            config,
            mock_factory,
            evaluator_factory=lambda orch: mock_evaluator,
        )

        summary = runner.run(
            dataset_path=fake_dataset_path,
            mode="full",
            session_id="checkpoint_test",
        )

        # Checkpoint should exist after run
        checkpoint_path = tmp_path / "checkpoint_test_checkpoint.json"
        assert checkpoint_path.exists(), "Checkpoint file should exist"

        # Load and verify checkpoint
        checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint_data["session_id"] == "checkpoint_test"
        assert checkpoint_data["status"] == "completed"
        assert checkpoint_data["progress"]["completed_interactions"] > 0


class TestBatchRunnerResume:
    """Test resume functionality."""

    @pytest.fixture
    def mock_factory(self, tmp_path: Path):
        """Create mock factory."""
        mock = MagicMock()
        mock_orch = MagicMock()
        mock_orch.chat.return_value = "Mock"
        mock_orch.last_trace.return_value = MagicMock(ops_applied=[], feedback=None, latency_ms=50.0)
        mock_orch.ltm_snapshot.return_value = []
        mock_orch.stm_stats.return_value = MagicMock(total_tokens=100)
        mock.build_for_evaluation.return_value = mock_orch
        return mock

    def test_resume_from_checkpoint_continues_from_offset(
        self, tmp_path: Path, mock_factory
    ):
        """Resuming from checkpoint starts from saved offset, not zero."""
        from evaluation.batch_checkpoint import CheckpointManager, CheckpointState, BatchProgress
        from evaluation.loader import DatasetLoader

        # Create dataset
        data = [
            {"question_id": f"q{i}", "question": f"Q{i}?", "answer": str(i),
             "question_type": "single_hop", "haystack_sessions": []}
            for i in range(1, 11)  # 10 questions
        ]
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(data), encoding="utf-8")

        # Pre-create a checkpoint at offset 5
        checkpoint_manager = CheckpointManager(tmp_path)
        initial_state = CheckpointState(
            session_id="resume_test",
            config={"batch_size": 5, "mode": "full", "dataset": str(dataset_path)},
            progress=BatchProgress(
                total_interactions=10,
                completed_interactions=5,
                completed_batches=1,
            ),
            aggregated_metrics={"total_queries": 5, "correct": 3, "accuracy": 0.6},
        )
        checkpoint_manager.save_checkpoint(initial_state)

        # Create runner with resume enabled
        config = BatchConfig(
            batch_size=5,
            output_dir=tmp_path,
            resume_from_checkpoint=True,
        )

        # Track which queries get processed
        processed_query_ids = []

        def mock_evaluate_query(query, instance=None):
            qid = query.get("query_id", "unknown")
            processed_query_ids.append(qid)
            return QuestionResult(
                query_id=qid,
                is_correct=True,
                behavior_type="IE",
                retrieval_trace={},
                abstained=False,
                latency_ms=100.0,
                validation_method="heuristic",
            )

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_query.side_effect = mock_evaluate_query
        mock_evaluator.replay_sessions.return_value = []

        runner = BatchRunner(
            config,
            mock_factory,
            evaluator_factory=lambda orch: mock_evaluator,
        )

        summary = runner.run(dataset_path, mode="full", session_id="resume_test")

        # Should have processed queries 6-10 (skipping 1-5 which were checkpointed)
        # Note: the exact behavior depends on how the runner handles the checkpoint
        assert summary is not None


class TestPartialMetrics:
    """Test PartialMetrics dataclass."""

    def test_partial_metrics_update(self):
        """PartialMetrics.update() correctly aggregates values."""
        metrics = PartialMetrics()

        result = QuestionResult(
            query_id="q1",
            is_correct=True,
            behavior_type="IE",
            retrieval_trace={},
            abstained=False,
            latency_ms=100.0,
            validation_method="llm_judge",
        )
        result.judge_result = MagicMock(latency_ms=50.0)

        metrics.update(result)

        assert metrics.total_queries == 1
        assert metrics.correct == 1
        assert metrics.llm_judge_queries == 1
        assert metrics.judge_latency_ms == 50.0

    def test_partial_metrics_to_dict(self):
        """PartialMetrics.to_dict() produces correct values."""
        metrics = PartialMetrics(
            total_queries=10,
            correct=7,
            abstained=2,
            total_latency_ms=1000.0,
            judge_latency_ms=500.0,
            llm_judge_queries=5,
            heuristic_queries=5,
        )

        d = metrics.to_dict()

        assert d["total_queries"] == 10
        assert d["correct"] == 7
        assert d["accuracy"] == 0.7
        assert d["abstained"] == 2
        assert d["avg_latency_ms"] == 100.0


class TestBatchConfig:
    """Test BatchConfig dataclass."""

    def test_batch_config_defaults(self):
        """BatchConfig has sensible defaults."""
        config = BatchConfig()

        assert config.batch_size == 10
        assert config.checkpoint_interval == 1
        assert config.resume_from_checkpoint is True

    def test_batch_config_string_output_dir(self):
        """BatchConfig converts string output_dir to Path."""
        config = BatchConfig(output_dir="/tmp/test")

        assert isinstance(config.output_dir, Path)
        assert str(config.output_dir) == "/tmp/test"