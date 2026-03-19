"""
AgeMem Evaluation Suite
-----------------------

Automated testing suite for evaluating the AgeMem system.

Per TRS-AGEMEM-EVAL-001 Technical Requirements Specification.

Components:
- pipeline.dataset_pipeline: Dataset ingestion and validation
- pipeline.inference_pipeline: AgeMem execution with telemetry capture
- pipeline.metrics_pipeline: KPI metric calculation
- pipeline.report_generator: Evaluation report generation
- orchestrator_test_harness: Orchestrator-based unified evaluation

Usage:
    # Run evaluation with sample dataset
    python -m evaluation.runner --sample --queries 100

    # Run with existing dataset
    python -m evaluation.runner --dataset path/to/dataset.json

    # Generate sample dataset
    python -m evaluation.runner --create-sample --entries 500 --queries 100

    # Use orchestrator test harness for unified evaluation
    from evaluation import EvaluationSession, MultiSessionEvaluation
    session = EvaluationSession(ltm_seed_data=memories)
    session.load_multi_session_history(sessions, behavior_type="IE")
    result = session.send_message("What's my phone number?")
"""

from evaluation.pipeline import (
    DatasetPipeline,
    InferencePipeline,
    MetricsPipeline,
    ReportGenerator,
)
from evaluation.orchestrator_test_harness import (
    EvaluationSession,
    MultiSessionEvaluation,
    TurnResult,
    EvaluationTrace,
    CrossSessionPersistenceTest,
    MockLLMClient,
)
from evaluation.question_evaluator import (
    QuestionEvaluator,
    EvaluationContext,
    QuestionResult,
)

__version__ = "1.0.0"
__all__ = [
    "DatasetPipeline",
    "InferencePipeline",
    "MetricsPipeline",
    "ReportGenerator",
    "EvaluationSession",
    "MultiSessionEvaluation",
    "TurnResult",
    "EvaluationTrace",
    "CrossSessionPersistenceTest",
    "MockLLMClient",
    "QuestionEvaluator",
    "EvaluationContext",
    "QuestionResult",
]