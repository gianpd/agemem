"""
Shared fixtures for evaluation tests.

Source Code Analysis (from evaluation/*.py):
============================================

loader.py - DatasetLoader:
  - load(path, subset=None, limit=0) -> (entries, queries, raw_data)
  - entries: list[dict] with keys: content, entry_id, tags
  - queries: list[dict] with keys: query_id, query_text, relevant_entry_ids,
             relevant_content, query_type, expected_answer

evaluator.py - Evaluator:
  - Dependencies: Orchestrator, LLMJudge (optional)
  - evaluate_query(query: dict, instance: dict) -> QuestionResult
  - QuestionResult fields: query_id, is_correct, behavior_type, retrieval_trace,
                           abstained, latency_ms, judge_result, validation_method

metrics.py - calculate_metrics:
  - calculate_metrics(queries, question_results, session_results) -> EvaluationSummary
  - EvaluationSummary fields: total_queries, correct, accuracy, abstained, avg_latency_ms,
                              llm_judge_queries, heuristic_queries, retrieval, by_behavior

report.py - ReportGenerator:
  - generate_markdown(summary, session_id) -> Path
  - generate_json(summary, session_id) -> Path
  - Imports CheckpointState from evaluation.batch_checkpoint

checkpoint.py / batch_checkpoint.py (identical):
  - CheckpointManager.save_checkpoint(state) -> Path
  - CheckpointManager.load_checkpoint(session_id) -> Optional[CheckpointState]
  - CheckpointState fields: session_id, config, progress, aggregated_metrics, status

runner.py - BatchRunner:
  - Orchestrates: DatasetLoader -> Evaluator -> MetricsCalculator -> ReportGenerator
  - Dependencies: OrchestratorFactory, Evaluator
"""

import json
import pytest
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

# Import from evaluation modules
from evaluation.metrics import (
    EvaluationSummary,
    RetrievalMetrics,
    BehaviorMetrics,
)
from evaluation.evaluator import QuestionResult, SessionReplayResult
from evaluation.batch_checkpoint import CheckpointState, BatchProgress


# ============================================================================
# FAKE DATA FIXTURES
# ============================================================================

FAKE_QUERIES = [
    {
        "query_id": "q1",
        "query_text": "What is 2+2?",
        "relevant_entry_ids": ["e1", "e2"],
        "relevant_content": ["2+2 equals 4", "addition result"],
        "query_type": "single_hop",
        "expected_answer": "4",
    },
    {
        "query_id": "q2",
        "query_text": "Capital of France?",
        "relevant_entry_ids": ["e3"],
        "relevant_content": ["Paris is the capital"],
        "query_type": "single_hop",
        "expected_answer": "Paris",
    },
    {
        "query_id": "q3",
        "query_text": "Color of sky?",
        "relevant_entry_ids": ["e4"],
        "relevant_content": ["The sky is blue"],
        "query_type": "single_hop",
        "expected_answer": "blue",
    },
]

FAKE_ENTRIES = [
    {"content": "[user] 2+2 equals 4", "entry_id": "e1", "tags": ["math"]},
    {"content": "[assistant] Correct, 4", "entry_id": "e2", "tags": ["math"]},
    {"content": "[user] Paris is the capital", "entry_id": "e3", "tags": ["geo"]},
    {"content": "[user] The sky is blue", "entry_id": "e4", "tags": ["science"]},
]


@pytest.fixture
def fake_queries():
    """Standard fake queries for testing."""
    return FAKE_QUERIES.copy()


@pytest.fixture
def fake_entries():
    """Standard fake entries for testing."""
    return FAKE_ENTRIES.copy()


@pytest.fixture
def fake_results():
    """Fake QuestionResult objects with deterministic values."""
    return [
        QuestionResult(
            query_id="q1",
            is_correct=True,
            behavior_type="IE",
            retrieval_trace={"results": [("e1", 1.0)], "latency_ms": 50.0},
            abstained=False,
            latency_ms=100.0,
            judge_result=None,
            validation_method="heuristic",
        ),
        QuestionResult(
            query_id="q2",
            is_correct=True,
            behavior_type="IE",
            retrieval_trace={"results": [("e3", 0.9)], "latency_ms": 60.0},
            abstained=False,
            latency_ms=120.0,
            judge_result=None,
            validation_method="heuristic",
        ),
        QuestionResult(
            query_id="q3",
            is_correct=False,
            behavior_type="IE",
            retrieval_trace={"results": [], "latency_ms": 40.0},
            abstained=True,
            latency_ms=80.0,
            judge_result=None,
            validation_method="heuristic",
        ),
    ]


@pytest.fixture
def fake_session_results():
    """Fake SessionReplayResult objects."""
    return [
        SessionReplayResult(
            session_id="session_1",
            turns_processed=5,
            ltm_entries_added=3,
            stm_tokens_at_end=500,
            learning_scores=[0.8, 0.9],
        ),
        SessionReplayResult(
            session_id="session_2",
            turns_processed=3,
            ltm_entries_added=2,
            stm_tokens_at_end=300,
            learning_scores=[0.7],
        ),
    ]


@pytest.fixture
def fake_summary():
    """Pre-built EvaluationSummary for testing."""
    return EvaluationSummary(
        total_queries=3,
        correct=2,
        accuracy=0.6666666666666666,
        abstained=1,
        avg_latency_ms=100.0,
        llm_judge_queries=0,
        heuristic_queries=3,
        judge_avg_latency_ms=0.0,
        retrieval=RetrievalMetrics(
            mrr_at_1=0.3333,
            mrr_at_5=0.5,
            mrr_at_10=0.5,
            precision_at_1=0.3333,
            precision_at_5=0.2,
            precision_at_10=0.1,
            recall_at_1=0.3333,
            recall_at_5=0.6666,
            recall_at_10=1.0,
            ndcg_at_5=0.5,
            ndcg_at_10=0.6,
            avg_latency_ms=50.0,
        ),
        by_behavior={
            "IE": BehaviorMetrics(behavior="IE", query_count=3, accuracy=0.6666),
        },
        session_replay={},
    )


@pytest.fixture
def fake_checkpoint_state():
    """Pre-built CheckpointState for testing."""
    return CheckpointState(
        session_id="test_session_001",
        config={"batch_size": 10, "mode": "full"},
        progress=BatchProgress(
            total_interactions=30,
            completed_interactions=15,
            completed_batches=2,
            last_batch_id=1,
        ),
        aggregated_metrics={"accuracy": 0.75, "total_queries": 15},
        status="running",
    )


@pytest.fixture
def fake_dataset_json(tmp_path: Path):
    """Create a fake dataset JSON file for testing DatasetLoader."""
    data = [
        {
            "question_id": "q1",
            "question": "What is 2+2?",
            "answer": "4",
            "question_type": "single_hop",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "2+2 equals 4", "has_answer": True},
                ]
            ],
            "haystack_session_ids": ["s1"],
        },
        {
            "question_id": "q2",
            "question": "Capital of France?",
            "answer": "Paris",
            "question_type": "single_hop",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "Paris is the capital", "has_answer": True},
                ]
            ],
            "haystack_session_ids": ["s2"],
        },
        {
            "question_id": "q3",
            "question": "Color of sky?",
            "answer": "blue",
            "question_type": "single_hop",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "The sky is blue", "has_answer": True},
                ]
            ],
            "haystack_session_ids": ["s3"],
        },
    ]
    dataset_path = tmp_path / "test_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return dataset_path


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_orchestrator():
    """Mock Orchestrator for testing without real LLM."""
    mock = MagicMock()
    mock.chat.return_value = "This is a test response containing 4."
    mock.last_trace.return_value = MagicMock(
        ops_applied=[],
        feedback=None,
        latency_ms=50.0,
    )
    mock.ltm_snapshot.return_value = []
    mock.stm_stats.return_value = MagicMock(total_tokens=100)
    return mock


@pytest.fixture
def mock_llm_judge():
    """Mock LLMJudge for testing without real judge server."""
    from evaluation.llm_judge import JudgeResult
    mock = MagicMock()
    mock.evaluate.return_value = JudgeResult(
        is_correct=True,
        raw_response="yes",
        latency_ms=150.0,
        model="test-model",
    )
    return mock