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

Usage:
    # Run evaluation with sample dataset
    python -m evaluation.runner --sample --queries 100

    # Run with existing dataset
    python -m evaluation.runner --dataset path/to/dataset.json

    # Generate sample dataset
    python -m evaluation.runner --create-sample --entries 500 --queries 100
"""

from evaluation.pipeline import (
    DatasetPipeline,
    InferencePipeline,
    MetricsPipeline,
    ReportGenerator,
)

__version__ = "1.0.0"
__all__ = [
    "DatasetPipeline",
    "InferencePipeline",
    "MetricsPipeline",
    "ReportGenerator",
]