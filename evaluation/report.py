"""Report generation for evaluation results.

This module provides unified report generation for both full-run and partial/checkpoint
evaluation results. It supports markdown and JSON output formats.
"""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from evaluation.batch_checkpoint import CheckpointState
from evaluation.metrics import EvaluationSummary


class ReportGenerator:
    """Generates evaluation reports in markdown and JSON formats.

    Supports both full-run reports (complete evaluation) and partial reports
    (checkpoint/in-progress state).

    Usage:
        generator = ReportGenerator(output_dir=Path("reports"))
        md_path = generator.generate_markdown(summary, session_id="eval_001")
        json_path = generator.generate_json(summary, session_id="eval_001")

        # For partial/checkpoint reports:
        md_path = generator.generate_partial_markdown(
            summary, session_id="eval_001", checkpoint=checkpoint_state
        )
    """

    def __init__(self, output_dir: Path):
        """Initialize the report generator.

        Args:
            output_dir: Directory where reports will be written.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown(
        self,
        summary: EvaluationSummary,
        session_id: str,
    ) -> Path:
        """Generate a markdown report for a complete evaluation.

        Args:
            summary: The evaluation summary containing all metrics.
            session_id: Unique identifier for this evaluation session.

        Returns:
            Path to the generated markdown file.
        """
        lines = self._build_full_report_lines(summary, session_id)

        md_path = self.output_dir / f"{session_id}_report.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")

        return md_path

    def generate_json(
        self,
        summary: EvaluationSummary,
        session_id: str,
    ) -> Path:
        """Generate a JSON report for a complete evaluation.

        Args:
            summary: The evaluation summary containing all metrics.
            session_id: Unique identifier for this evaluation session.

        Returns:
            Path to the generated JSON file.
        """
        json_path = self.output_dir / f"{session_id}_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": session_id,
                "summary": self._summary_to_dict(summary),
            }, f, indent=2)

        return json_path

    def generate_partial_markdown(
        self,
        summary: EvaluationSummary,
        session_id: str,
        checkpoint: CheckpointState,
    ) -> Path:
        """Generate a markdown report for partial/in-progress evaluation results.

        Args:
            summary: The evaluation summary with partial metrics.
            session_id: Unique identifier for this evaluation session.
            checkpoint: The checkpoint state containing progress information.

        Returns:
            Path to the generated markdown file.
        """
        lines = self._build_partial_report_lines(summary, session_id, checkpoint)

        md_path = self.output_dir / f"{session_id}_partial_report.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")

        return md_path

    def generate_partial_json(
        self,
        summary: EvaluationSummary,
        session_id: str,
        checkpoint: CheckpointState,
    ) -> Path:
        """Generate a JSON report for partial/in-progress evaluation results.

        Args:
            summary: The evaluation summary with partial metrics.
            session_id: Unique identifier for this evaluation session.
            checkpoint: The checkpoint state containing progress information.

        Returns:
            Path to the generated JSON file.
        """
        json_path = self.output_dir / f"{session_id}_partial_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": session_id,
                "status": checkpoint.status,
                "progress": asdict(checkpoint.progress),
                "summary": self._summary_to_dict(summary),
                "checkpoint": checkpoint.to_dict(),
            }, f, indent=2)

        return json_path

    def _build_full_report_lines(
        self,
        summary: EvaluationSummary,
        session_id: str,
    ) -> list[str]:
        """Build the lines for a full markdown report."""
        lines = [
            f"# AgeMem Evaluation Report",
            f"",
            f"**Session ID:** {session_id}",
            f"**Evaluated at:** {datetime.now().isoformat()}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Queries | {summary.total_queries} |",
            f"| Correct | {summary.correct} |",
            f"| Accuracy | {summary.accuracy:.2%} |",
            f"| Abstained | {summary.abstained} |",
            f"| Avg Latency | {summary.avg_latency_ms:.1f}ms |",
            f"",
            f"## LLM-as-Judge Statistics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Queries by LLM Judge | {summary.llm_judge_queries} |",
            f"| Queries by Heuristic | {summary.heuristic_queries} |",
            f"| Judge Avg Latency | {summary.judge_avg_latency_ms:.1f}ms |",
            f"",
            f"## Retrieval Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
        ]

        # Add retrieval metrics
        for key, value in self._get_retrieval_dict(summary).items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            else:
                lines.append(f"| {key} | {value} |")

        # Add behavior breakdown
        lines.extend([
            f"",
            f"## Behavior Breakdown",
            f"",
            f"| Behavior | Count | Accuracy |",
            f"|----------|-------|----------|",
        ])

        for behavior, metrics in self._get_behavior_dict(summary).items():
            query_count = metrics.get("query_count", 0)
            accuracy = metrics.get("accuracy", 0.0)
            lines.append(f"| {behavior} | {query_count} | {accuracy:.2%} |")

        # Add session replay section if present
        session_replay = self._get_session_replay_dict(summary)
        if session_replay:
            lines.extend([
                f"",
                f"## Session Replay",
                f"",
                f"- Total Sessions: {session_replay.get('total_sessions', 0)}",
                f"- Total Turns: {session_replay.get('total_turns', 0)}",
                f"- LTM Entries Added: {session_replay.get('total_ltm_adds', 0)}",
                f"- Avg STM Tokens: {session_replay.get('avg_stm_tokens', 0):.0f}",
            ])

        return lines

    def _build_partial_report_lines(
        self,
        summary: EvaluationSummary,
        session_id: str,
        checkpoint: CheckpointState,
    ) -> list[str]:
        """Build the lines for a partial/checkpoint markdown report."""
        lines = [
            f"# AgeMem Evaluation Report (PARTIAL RESULTS)",
            f"",
            f"**Session ID:** {session_id}",
            f"**Status:** {checkpoint.status}",
            f"**Generated at:** {datetime.now().isoformat()}",
            f"**Progress:** {checkpoint.progress.completed_interactions} / {checkpoint.progress.total_interactions} interactions",
            f"**Batches:** {checkpoint.progress.completed_batches} completed",
            f"**Percent Complete:** {checkpoint.progress.percent_complete:.1f}%",
            f"",
            f"## Partial Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Queries | {summary.total_queries} |",
            f"| Correct | {summary.correct} |",
            f"| Accuracy | {summary.accuracy:.2%} |",
            f"| Abstained | {summary.abstained} |",
            f"| Avg Latency | {summary.avg_latency_ms:.1f}ms |",
            f"",
            f"> **Note:** This report shows partial results. The evaluation is still in progress.",
            f"> Run `runner.resume('{session_id}')` to continue.",
        ]

        return lines

    def _summary_to_dict(self, summary: EvaluationSummary) -> dict:
        """Convert EvaluationSummary to a dictionary, handling nested objects."""
        if hasattr(summary, "to_dict"):
            return summary.to_dict()

        # Fallback: manual conversion
        return {
            "total_queries": summary.total_queries,
            "correct": summary.correct,
            "accuracy": summary.accuracy,
            "abstained": summary.abstained,
            "avg_latency_ms": summary.avg_latency_ms,
            "llm_judge_queries": summary.llm_judge_queries,
            "heuristic_queries": summary.heuristic_queries,
            "judge_avg_latency_ms": summary.judge_avg_latency_ms,
            "retrieval": self._get_retrieval_dict(summary),
            "by_behavior": self._get_behavior_dict(summary),
            "session_replay": self._get_session_replay_dict(summary),
        }

    def _get_retrieval_dict(self, summary: EvaluationSummary) -> dict:
        """Safely get retrieval metrics as dict."""
        retrieval = getattr(summary, "retrieval", None)
        if retrieval is None:
            return {}
        if hasattr(retrieval, "to_dict"):
            return retrieval.to_dict()
        if is_dataclass(retrieval):
            return asdict(retrieval)
        return dict(retrieval) if retrieval else {}

    def _get_behavior_dict(self, summary: EvaluationSummary) -> dict:
        """Safely get behavior metrics as dict."""
        by_behavior = getattr(summary, "by_behavior", None)
        if by_behavior is None:
            return {}

        result = {}
        for k, v in by_behavior.items():
            if hasattr(v, "to_dict"):
                result[k] = v.to_dict()
            elif is_dataclass(v):
                result[k] = asdict(v)
            else:
                result[k] = {
                    "behavior": getattr(v, "behavior", k),
                    "query_count": getattr(v, "query_count", 0),
                    "accuracy": getattr(v, "accuracy", 0.0),
                }
        return result

    def _get_session_replay_dict(self, summary: EvaluationSummary) -> dict:
        """Safely get session replay metrics as dict."""
        session_replay = getattr(summary, "session_replay", None)
        if session_replay is None:
            return {}
        return dict(session_replay) if session_replay else {}


def generate_report(
    summary: EvaluationSummary,
    session_id: str,
    output_dir: Path,
) -> Path:
    """Standalone function for generating a complete evaluation report.

    This is a convenience function that maintains backward compatibility
    with the existing codebase.

    Args:
        summary: The evaluation summary containing all metrics.
        session_id: Unique identifier for this evaluation session.
        output_dir: Directory where reports will be written.

    Returns:
        Path to the generated markdown report.
    """
    generator = ReportGenerator(output_dir)
    generator.generate_markdown(summary, session_id)
    generator.generate_json(summary, session_id)
    return generator.output_dir / f"{session_id}_report.md"