"""
Evaluation Pipeline Module
--------------------------

Implements the AgeMem Automated Testing Suite per TRS-AGEMEM-EVAL-001.

Components:
- dataset_pipeline: Dataset ingestion and validation
- inference_pipeline: AgeMem execution with telemetry capture
- metrics_pipeline: KPI metric calculation
- report_generator: Evaluation report generation
- phase2_pipeline: End-to-end memory lifecycle testing
- end_to_end_runner: Combined Phase 1 + Phase 2 evaluation
- simulation: Deterministic testing without live LLM
"""

from evaluation.pipeline.dataset_pipeline import DatasetPipeline
from evaluation.pipeline.inference_pipeline import InferencePipeline
from evaluation.pipeline.metrics_pipeline import MetricsPipeline
from evaluation.pipeline.report_generator import ReportGenerator
from evaluation.pipeline.phase2_pipeline import (
    Phase2Pipeline,
    Phase2Results,
    MemoryOperationMetrics,
    LearningScoreMetrics,
    ContextAwareRetrievalMetrics,
    QueryExpansionMetrics,
)
from evaluation.pipeline.end_to_end_runner import (
    EndToEndRunner,
    EndToEndResults,
)
from evaluation.pipeline.simulation import (
    MemoryOperationSimulator,
    LearningScoreSimulator,
    LongMemEvalConversationSimulator,
    generate_phase2_test_data,
)

__all__ = [
    "DatasetPipeline",
    "InferencePipeline",
    "MetricsPipeline",
    "ReportGenerator",
    # Phase 2
    "Phase2Pipeline",
    "Phase2Results",
    "MemoryOperationMetrics",
    "LearningScoreMetrics",
    "ContextAwareRetrievalMetrics",
    "QueryExpansionMetrics",
    # End-to-End
    "EndToEndRunner",
    "EndToEndResults",
    # Simulation
    "MemoryOperationSimulator",
    "LearningScoreSimulator",
    "LongMemEvalConversationSimulator",
    "generate_phase2_test_data",
]