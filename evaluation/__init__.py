"""
AgeMem Evaluation Suite
-----------------------
Simplified automated testing suite for evaluating the AgeMem system.

Components:
- evaluators: Core evaluation logic (session replay + question evaluation)
- metrics: Metrics calculation (MRR@K, Recall@K, etc.)
- factory: Orchestrator factory for evaluation
- mock_llm: Mock LLM for deterministic testing

Usage:
    # Run evaluation with dataset
    python -m evaluation.run --dataset path/to/dataset.json --queries 10

    # Use in code
    from evaluation.evaluators import Evaluator
    from evaluation.metrics import calculate_metrics
"""

from evaluation.evaluators import (
    Evaluator,
    SessionReplayResult,
    QuestionResult,
    EvaluationContext,
)
from evaluation.metrics import (
    calculate_metrics,
    EvaluationSummary,
    RetrievalMetrics,
    BehaviorMetrics,
)
from evaluation.factory import OrchestratorFactory

__version__ = "2.0.0"
__all__ = [
    # Core evaluation
    "Evaluator",
    "SessionReplayResult",
    "QuestionResult",
    "EvaluationContext",
    # Metrics
    "calculate_metrics",
    "EvaluationSummary",
    "RetrievalMetrics",
    "BehaviorMetrics",
    # Factory
    "OrchestratorFactory",
]
