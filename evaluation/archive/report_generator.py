"""
Evaluation Metrics Report Generator Module
------------------------------------------

Generates evaluation reports per Section 3.4 of TRS-AGEMEM-EVAL-001.

Output formats:
- Markdown: For documentation and version control
- JSON: For programmatic consumption
- HTML: For interactive web-based viewing
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from evaluation.pipeline.metrics_pipeline import (
    EvaluationResults,
    RetrievalMetrics,
    MemoryQualityMetrics,
    ResponseQualityMetrics,
    ComparativeMetrics,
)

logger = logging.getLogger(__name__)


# Competitor benchmark data (placeholder - would be populated from actual runs)
COMPETITOR_BENCHMARKS: dict[str, dict[str, float]] = {
    "MemGPT": {
        "mrr_at_5": 0.72,
        "hallucination_rate": 0.08,
        "tokens_per_query": 2500,
        "latency_ms": 450,
    },
    "Letta": {
        "mrr_at_5": 0.75,
        "hallucination_rate": 0.07,
        "tokens_per_query": 2200,
        "latency_ms": 380,
    },
    "LangChain RAG": {
        "mrr_at_5": 0.68,
        "hallucination_rate": 0.12,
        "tokens_per_query": 3000,
        "latency_ms": 350,
    },
    "LlamaIndex": {
        "mrr_at_5": 0.70,
        "hallucination_rate": 0.10,
        "tokens_per_query": 2800,
        "latency_ms": 320,
    },
    "Base LLM (no memory)": {
        "mrr_at_5": 0.45,
        "hallucination_rate": 0.25,
        "tokens_per_query": 1500,
        "latency_ms": 200,
    },
}


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_competitors: bool = True
    include_recommendations: bool = True
    include_detailed_metrics: bool = True
    version: str = "1.0"


class ReportGenerator:
    """
    Evaluation Metrics Report Generator per Section 3.4 of TRS-AGEMEM-EVAL-001.

    Generates final performance summary against competitor benchmarks.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config: Optional[ReportConfig] = None,
    ) -> None:
        self._output_dir = output_dir or Path("evaluation/results")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or ReportConfig()

    def generate(
        self,
        results: EvaluationResults,
        format: str = "markdown",
    ) -> str:
        """
        Generate evaluation report in specified format.

        Args:
            results: EvaluationResults object
            format: Output format (markdown, json, html)

        Returns:
            Report content as string
        """
        if format == "markdown":
            return self._generate_markdown(results)
        elif format == "json":
            return self._generate_json(results)
        elif format == "html":
            return self._generate_html(results)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_markdown(self, results: EvaluationResults) -> str:
        """Generate Markdown report per Section 7.7 format."""
        lines = []

        # Header
        lines.append("# Evaluation Report: AgeMem")
        lines.append("")
        lines.append(f"**Version:** {self._config.version}")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Session ID:** {results.session_id}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        retrieval_status = results.retrieval.meets_targets()
        memory_status = results.memory_quality.meets_targets()
        response_status = results.response_quality.meets_targets()

        all_passed = (
            all(retrieval_status.values()) and
            all(memory_status.values()) and
            all(response_status.values())
        )

        if all_passed:
            lines.append("**Overall Status:** All targets met.")
        else:
            failed = []
            if not all(retrieval_status.values()):
                failed.append("Retrieval metrics")
            if not all(memory_status.values()):
                failed.append("Memory quality metrics")
            if not all(response_status.values()):
                failed.append("Response quality metrics")
            lines.append(f"**Overall Status:** Some targets not met: {', '.join(failed)}")
        lines.append("")

        # Retrieval Quality
        lines.append("## Retrieval Quality")
        lines.append("")
        lines.append("| Metric | Value | Target | Status |")
        lines.append("|--------|-------|--------|--------|")
        lines.append(self._metric_row("MRR@1", results.retrieval.mrr_at_1, "N/A", None))
        lines.append(self._metric_row("MRR@5", results.retrieval.mrr_at_5, "N/A", None))
        lines.append(self._metric_row("MRR@10", results.retrieval.mrr_at_10, ">= 0.85", retrieval_status.get("mrr_at_10")))
        lines.append(self._metric_row("Precision@5", results.retrieval.precision_at_5, "N/A", None))
        lines.append(self._metric_row("Recall@5", results.retrieval.recall_at_5, ">= 0.90", retrieval_status.get("recall_at_5")))
        lines.append(self._metric_row("NDCG@10", results.retrieval.ndcg_at_10, "N/A", None))
        lines.append(self._metric_row("Avg Latency", results.retrieval.avg_latency_ms, "< 500ms", results.retrieval.avg_latency_ms < 500))
        lines.append("")

        # Memory Persistence
        lines.append("## Memory Persistence")
        lines.append("")
        lines.append("| Metric | Value | Target | Status |")
        lines.append("|--------|-------|--------|--------|")
        lines.append(self._metric_row("Retention Rate", results.memory_quality.retention_rate, ">= 95%", memory_status.get("retention_rate")))
        lines.append(self._metric_row("Dedup Accuracy", results.memory_quality.deduplication_accuracy, ">= 90%", memory_status.get("deduplication_accuracy")))
        lines.append(self._metric_row("Learning Score Corr", results.memory_quality.learning_score_correlation, ">= 0.7", memory_status.get("learning_score_correlation")))
        lines.append(self._metric_row("Context Utilization", results.memory_quality.context_utilization, ">= 60%", memory_status.get("context_utilization")))
        lines.append("")

        # Response Quality
        lines.append("## Response Quality")
        lines.append("")
        lines.append("| Metric | Value | Target | Status |")
        lines.append("|--------|-------|--------|--------|")
        lines.append(self._metric_row("Hallucination Rate", results.response_quality.hallucination_rate, "<= 5%", response_status.get("hallucination_rate")))
        lines.append(self._metric_row("Coherence Score", results.response_quality.coherence_score, ">= 4.0", response_status.get("coherence_score")))
        lines.append(self._metric_row("Memory Grounding", results.response_quality.memory_grounding, ">= 90%", response_status.get("memory_grounding")))
        lines.append(self._metric_row("Preference Accuracy", results.response_quality.preference_accuracy, ">= 95%", response_status.get("preference_accuracy")))
        lines.append("")

        # Comparative Performance
        if self._config.include_competitors:
            lines.append("## Comparative Performance")
            lines.append("")
            lines.append("| System | MRR@5 | Halluc. Rate | Tokens/Query | Latency (ms) |")
            lines.append("|--------|-------|--------------|--------------|--------------|")
            lines.append(self._comparator_row("AgeMem", results.comparative.mrr_at_5, results.response_quality.hallucination_rate, results.comparative.tokens_per_query, results.comparative.latency_ms))

            for name, data in COMPETITOR_BENCHMARKS.items():
                lines.append(self._comparator_row(
                    name,
                    data.get("mrr_at_5", 0.0),
                    data.get("hallucination_rate", 0.0),
                    data.get("tokens_per_query", 0.0),
                    data.get("latency_ms", 0.0),
                ))
            lines.append("")

        # Recommendations
        if self._config.include_recommendations:
            lines.append("## Recommendations")
            lines.append("")
            recommendations = self._generate_recommendations(results)
            for rec in recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Generated by AgeMem Evaluation Suite v{self._config.version}*")

        return "\n".join(lines)

    def _metric_row(
        self,
        name: str,
        value: float,
        target: str,
        passed: Optional[bool],
    ) -> str:
        """Generate a metric table row."""
        if isinstance(value, float):
            if value < 1.0 and "Rate" in name or "Utilization" in name or "Accuracy" in name:
                formatted = f"{value:.2%}"
            elif value < 1.0:
                formatted = f"{value:.4f}"
            else:
                formatted = f"{value:.2f}"
        else:
            formatted = str(value)

        status = "N/A"
        if passed is not None:
            status = "PASS" if passed else "FAIL"

        return f"| {name} | {formatted} | {target} | {status} |"

    def _comparator_row(
        self,
        name: str,
        mrr: float,
        halluc: float,
        tokens: float,
        latency: float,
    ) -> str:
        """Generate a comparator table row."""
        return f"| {name} | {mrr:.2f} | {halluc:.1%} | {tokens:.0f} | {latency:.0f} |"

    def _generate_recommendations(self, results: EvaluationResults) -> list[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        # Check retrieval metrics
        if results.retrieval.mrr_at_10 < 0.85:
            recommendations.append(
                "MRR@10 below target. Consider adjusting LTM_DEDUP_THRESHOLD or "
                "enabling query expansion for better retrieval coverage."
            )

        if results.retrieval.recall_at_5 < 0.90:
            recommendations.append(
                "Recall@5 below target. Consider increasing SEMANTIC_RETRIEVAL_MULTIPLIER "
                "or enabling context-aware retrieval."
            )

        # Check memory quality
        if results.memory_quality.retention_rate < 0.95:
            recommendations.append(
                "Retention rate below target. Review learning score thresholds "
                "(LTM_PROMOTE_THRESHOLD) to ensure important memories are retained."
            )

        if results.memory_quality.deduplication_accuracy < 0.90:
            recommendations.append(
                "Deduplication accuracy below target. Consider adjusting "
                "LTM_DEDUP_THRESHOLD (cosine) or LTM_DEDUP_OVERLAP_THRESHOLD (Jaccard)."
            )

        # Check response quality
        if results.response_quality.hallucination_rate > 0.05:
            recommendations.append(
                "Hallucination rate above target. Enable Tier 3 validation tools "
                "(validate_ltm_relevance) to improve memory grounding."
            )

        if not recommendations:
            recommendations.append("All metrics meet targets. Continue monitoring for regression.")

        return recommendations

    def _generate_json(self, results: EvaluationResults) -> str:
        """Generate JSON report for programmatic consumption."""
        report = {
            "report_type": "agemem_evaluation",
            "version": self._config.version,
            "generated_at": datetime.now().isoformat(),
            "session_id": results.session_id,
            "metrics": results.to_dict(),
            "target_status": {
                "retrieval": results.retrieval.meets_targets(),
                "memory_quality": results.memory_quality.meets_targets(),
                "response_quality": results.response_quality.meets_targets(),
            },
        }

        if self._config.include_competitors:
            report["competitor_benchmarks"] = COMPETITOR_BENCHMARKS

        return json.dumps(report, indent=2)

    def _generate_html(self, results: EvaluationResults) -> str:
        """Generate HTML report for web-based viewing."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgeMem Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .metric-value {{ font-family: monospace; }}
    </style>
</head>
<body>
    <h1>AgeMem Evaluation Report</h1>
    <p><strong>Version:</strong> {self._config.version}</p>
    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Session ID:</strong> {results.session_id}</p>

    <h2>Retrieval Quality</h2>
    {self._html_metrics_table(results.retrieval)}

    <h2>Memory Persistence</h2>
    {self._html_metrics_table(results.memory_quality)}

    <h2>Response Quality</h2>
    {self._html_metrics_table(results.response_quality)}

    <hr>
    <p><em>Generated by AgeMem Evaluation Suite v{self._config.version}</em></p>
</body>
</html>"""
        return html

    def _html_metrics_table(self, metrics: Any) -> str:
        """Generate HTML table for metrics."""
        data = metrics.to_dict()
        html = "<table><tr><th>Metric</th><th>Value</th></tr>"
        for key, value in data.items():
            if isinstance(value, float):
                if value < 1.0 and "rate" in key or "accuracy" in key or "utilization" in key:
                    formatted = f"{value:.2%}"
                else:
                    formatted = f"{value:.4f}"
            else:
                formatted = str(value)
            html += f"<tr><td>{key}</td><td class='metric-value'>{formatted}</td></tr>"
        html += "</table>"
        return html

    # ── File Output ───────────────────────────────────────────────────────────

    def save(
        self,
        results: EvaluationResults,
        base_name: str = "evaluation_report",
        formats: Optional[list[str]] = None,
    ) -> dict[str, Path]:
        """
        Save report in multiple formats.

        Args:
            results: EvaluationResults object
            base_name: Base filename (without extension)
            formats: List of formats to generate (default: all)

        Returns:
            Dictionary mapping format to output path
        """
        formats = formats or ["markdown", "json", "html"]
        paths = {}

        for fmt in formats:
            content = self.generate(results, format=fmt)
            ext = {"markdown": ".md", "json": ".json", "html": ".html"}[fmt]
            path = self._output_dir / f"{base_name}{ext}"

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            paths[fmt] = path
            logger.info(f"Saved {fmt} report to {path}")

        return paths