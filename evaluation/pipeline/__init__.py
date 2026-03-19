"""
Evaluation Pipeline Module
--------------------------

Implements the AgeMem Automated Testing Suite per TRS-AGEMEM-EVAL-001.

Components:
- dataset_pipeline: Dataset ingestion and validation
- inference_pipeline: AgeMem execution with telemetry capture
- metrics_pipeline: KPI metric calculation
- report_generator: Evaluation report generation
"""

from evaluation.pipeline.dataset_pipeline import DatasetPipeline
from evaluation.pipeline.inference_pipeline import InferencePipeline
from evaluation.pipeline.metrics_pipeline import MetricsPipeline
from evaluation.pipeline.report_generator import ReportGenerator

__all__ = [
    "DatasetPipeline",
    "InferencePipeline",
    "MetricsPipeline",
    "ReportGenerator",
]