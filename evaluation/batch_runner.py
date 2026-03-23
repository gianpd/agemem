"""
evaluation/batch_runner.py
──────────────────────────
Batch-aware evaluation runner with checkpoint persistence.

DEPRECATED: This module is deprecated. Use evaluation.runner instead.
  - BatchEvaluationRunner -> evaluation.runner.BatchRunner
  - BatchConfig -> evaluation.runner.BatchConfig
  - PartialMetrics -> evaluation.runner.PartialMetrics
  - BatchResult -> (no direct replacement; use dict from BatchRunner._process_single_batch)

Processes evaluations in configurable batch sizes, flushing results
to disk after each batch for crash recovery and partial result visibility.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "evaluation.batch_runner is deprecated. Use evaluation.runner instead.\n"
    "  - BatchEvaluationRunner -> BatchRunner\n"
    "  - BatchConfig, PartialMetrics -> available in evaluation.runner",
    DeprecationWarning,
    stacklevel=2
)

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any

from agents.orchestrator import Orchestrator
from evaluation.batch_checkpoint import (
    CheckpointManager,
    CheckpointState,
    BatchProgress,
)
from evaluation.evaluators import Evaluator, SessionReplayResult, QuestionResult
from evaluation.metrics import calculate_metrics, EvaluationSummary

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch evaluation."""
    batch_size: int = 10
    checkpoint_interval: int = 1  # Save checkpoint every N batches
    output_dir: Path = field(default_factory=lambda: Path("evaluation/results"))
    resume_from_checkpoint: bool = True
    flush_partial_metrics: bool = True  # Write partial metrics after each batch

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class BatchResult:
    """Results from processing a single batch."""
    batch_id: int
    start_idx: int  # Start index in the full dataset
    end_idx: int    # End index in the full dataset
    session_results: list[SessionReplayResult] = field(default_factory=list)
    question_results: list[QuestionResult] = field(default_factory=list)
    latency_ms: float = 0.0
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert batch result to serializable dict."""
        return {
            "batch_id": self.batch_id,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "session_results": [
                {
                    "session_id": r.session_id,
                    "turns_processed": r.turns_processed,
                    "ltm_entries_added": r.ltm_entries_added,
                    "stm_tokens_at_end": r.stm_tokens_at_end,
                    "learning_scores": r.learning_scores,
                }
                for r in self.session_results
            ],
            "question_results": [
                {
                    "query_id": r.query_id,
                    "is_correct": r.is_correct,
                    "behavior_type": r.behavior_type,
                    "abstained": r.abstained,
                    "latency_ms": r.latency_ms,
                    "validation_method": r.validation_method,
                    "judge_result": {
                        "is_correct": r.judge_result.is_correct,
                        "latency_ms": r.judge_result.latency_ms,
                        "model": r.judge_result.model,
                    } if r.judge_result else None,
                }
                for r in self.question_results
            ],
            "latency_ms": self.latency_ms,
            "processed_at": self.processed_at,
        }


@dataclass
class PartialMetrics:
    """Running aggregated metrics for partial results."""
    total_queries: int = 0
    correct: int = 0
    abstained: int = 0
    total_latency_ms: float = 0.0
    judge_latency_ms: float = 0.0
    llm_judge_queries: int = 0
    heuristic_queries: int = 0

    def update(self, result: QuestionResult) -> None:
        """Update metrics with a new question result."""
        self.total_queries += 1
        if result.is_correct:
            self.correct += 1
        if result.abstained:
            self.abstained += 1
        self.total_latency_ms += result.latency_ms

        if result.validation_method == "llm_judge":
            self.llm_judge_queries += 1
            if result.judge_result:
                self.judge_latency_ms += result.judge_result.latency_ms
        else:
            self.heuristic_queries += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "total_queries": self.total_queries,
            "correct": self.correct,
            "accuracy": self.correct / self.total_queries if self.total_queries > 0 else 0.0,
            "abstained": self.abstained,
            "avg_latency_ms": self.total_latency_ms / self.total_queries if self.total_queries > 0 else 0.0,
            "llm_judge_queries": self.llm_judge_queries,
            "heuristic_queries": self.heuristic_queries,
            "judge_avg_latency_ms": (
                self.judge_latency_ms / self.llm_judge_queries if self.llm_judge_queries > 0 else 0.0
            ),
        }


class BatchEvaluationRunner:
    """
    Batch-aware evaluation runner with checkpoint persistence.

    Processes evaluations in configurable batch sizes, writing results
to disk after each batch. Supports resuming from checkpoints after
    crashes or interruptions.

    Usage:
        runner = BatchEvaluationRunner(
            config=BatchConfig(batch_size=10),
            factory=orchestrator_factory,
        )

        # Run evaluation with automatic checkpointing
        summary = runner.run(
            dataset_path=Path("evaluation/data/dataset.json"),
            mode="full",
            max_interactions=100,
        )

        # Or resume an interrupted evaluation
        summary = runner.resume("eval_20260322_120000")

    File Outputs:
        {session_id}_checkpoint.json       - Recovery state
        {session_id}_batch_{N}.jsonl       - Results for batch N (one per line)
        {session_id}_partial_metrics.json  - Running aggregates
        {session_id}_report.md             - Final report (on completion)
        {session_id}_metrics.json          - Final metrics (on completion)
    """

    def __init__(
        self,
        config: BatchConfig,
        factory: Any,  # OrchestratorFactory
        evaluator_factory: Optional[Callable[[Orchestrator], Evaluator]] = None,
    ) -> None:
        """
        Initialize batch evaluation runner.

        Args:
            config: Batch evaluation configuration.
            factory: OrchestratorFactory for creating orchestrator instances.
            evaluator_factory: Optional factory for creating Evaluator instances.
                             Defaults to lambda orch: Evaluator(orch).
        """
        self.config = config
        self.factory = factory
        self.evaluator_factory = evaluator_factory or (lambda orch: Evaluator(orch))
        self.checkpoint_manager = CheckpointManager(config.output_dir)

    def run(
        self,
        dataset_path: Path,
        mode: str = "full",
        max_interactions: int = 0,
        max_batches: int = 0,
        session_id: Optional[str] = None,
    ) -> EvaluationSummary:
        """
        Run batch evaluation, resuming from checkpoint if available.

        Args:
            dataset_path: Path to the evaluation dataset.
            mode: Evaluation mode - "full", "lifecycle", or "retrieval".
            max_interactions: Maximum total interactions to process (0 = all).
            max_batches: Maximum batches to process (0 = unlimited).
            session_id: Optional session ID. If None, generates new ID.
                        If checkpoint exists for this ID, resumes from it.

        Returns:
            EvaluationSummary with final results.
        """
        # Generate or use provided session ID
        if session_id is None:
            session_id = datetime.now().strftime("eval_%Y%m%d_%H%M%S")

        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(session_id)

        if checkpoint and self.config.resume_from_checkpoint:
            logger.info(f"Found checkpoint for {session_id}, resuming...")
            return self._resume_from_checkpoint(checkpoint, dataset_path, mode, max_batches)

        # Start fresh evaluation
        logger.info(f"Starting new batch evaluation: {session_id}")
        return self._run_fresh(session_id, dataset_path, mode, max_interactions, max_batches)

    def _run_fresh(
        self,
        session_id: str,
        dataset_path: Path,
        mode: str,
        max_interactions: int,
        max_batches: int,
    ) -> EvaluationSummary:
        """Run a fresh evaluation from the beginning."""
        # Load dataset
        entries, queries, raw_data = self._load_dataset(dataset_path, max_interactions)

        total_interactions = len(entries)
        logger.info(f"Loaded {total_interactions} interactions, batch_size={self.config.batch_size}")

        # Initialize checkpoint state
        progress = BatchProgress(
            total_interactions=total_interactions,
            completed_interactions=0,
            completed_batches=0,
            last_batch_id=-1,
        )
        partial_metrics = PartialMetrics()

        state = CheckpointState(
            session_id=session_id,
            config={
                "batch_size": self.config.batch_size,
                "mode": mode,
                "dataset": str(dataset_path),
                "max_interactions": max_interactions,
            },
            progress=progress,
            aggregated_metrics=partial_metrics.to_dict(),
            status="running",
        )

        # Save initial checkpoint
        self.checkpoint_manager.save_checkpoint(state)

        try:
            # Process batches
            return self._process_batches(
                session_id=session_id,
                entries=entries,
                queries=queries,
                raw_data=raw_data,
                mode=mode,
                max_batches=max_batches,
                state=state,
                partial_metrics=partial_metrics,
            )
        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            self.checkpoint_manager.mark_failed(session_id, str(e))
            raise

    def _resume_from_checkpoint(
        self,
        checkpoint: CheckpointState,
        dataset_path: Path,
        mode: str,
        max_batches: int,
    ) -> EvaluationSummary:
        """Resume evaluation from a checkpoint."""
        session_id = checkpoint.session_id
        completed_batches = checkpoint.progress.completed_batches

        logger.info(f"Resuming {session_id} from batch {completed_batches}")

        # Load dataset
        max_interactions = checkpoint.config.get("max_interactions", 0)
        entries, queries, raw_data = self._load_dataset(dataset_path, max_interactions)

        # Skip already processed entries
        processed_count = checkpoint.progress.completed_interactions
        remaining_entries = entries[processed_count:]
        remaining_queries = queries[processed_count:] if len(queries) > processed_count else []

        logger.info(f"Skipping {processed_count} already processed interactions")

        # Reconstruct partial metrics from checkpoint
        partial_metrics = self._reconstruct_partial_metrics(checkpoint.aggregated_metrics)

        # Update checkpoint status
        checkpoint.status = "running"
        self.checkpoint_manager.save_checkpoint(checkpoint)

        try:
            # Process remaining batches
            return self._process_batches(
                session_id=session_id,
                entries=remaining_entries,
                queries=remaining_queries,
                raw_data=raw_data,
                mode=mode,
                max_batches=max_batches,
                state=checkpoint,
                partial_metrics=partial_metrics,
                starting_batch_id=completed_batches,
                starting_interaction_idx=processed_count,
            )
        except Exception as e:
            logger.exception(f"Evaluation failed during resume: {e}")
            self.checkpoint_manager.mark_failed(session_id, str(e))
            raise

    def _process_batches(
        self,
        session_id: str,
        entries: list[dict],
        queries: list[dict],
        raw_data: list[dict],
        mode: str,
        max_batches: int,
        state: CheckpointState,
        partial_metrics: PartialMetrics,
        starting_batch_id: int = 0,
        starting_interaction_idx: int = 0,
    ) -> EvaluationSummary:
        """Process all batches and return final summary."""
        batch_size = self.config.batch_size
        total_batches = (len(entries) + batch_size - 1) // batch_size

        if max_batches > 0:
            total_batches = min(total_batches, max_batches)

        logger.info(f"Processing {total_batches} batches (batch_size={batch_size})")

        all_session_results: list[SessionReplayResult] = []
        all_question_results: list[QuestionResult] = []

        for batch_idx in range(total_batches):
            batch_id = starting_batch_id + batch_idx
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(entries))

            batch_entries = entries[start_idx:end_idx]
            batch_queries = queries[start_idx:end_idx] if start_idx < len(queries) else []

            logger.info(f"Processing batch {batch_id} (interactions {start_idx}-{end_idx})")

            # Create FRESH orchestrator for this batch (prevents STM accumulation)
            import tempfile
            batch_persist_dir = Path(tempfile.mkdtemp(prefix=f"agemem_eval_batch_{batch_id}_"))
            orchestrator = self.factory.build_for_evaluation(
                persist_dir=batch_persist_dir,
                config_overrides={"STM_TOKEN_LIMIT": 8000}
            )
            evaluator = self.evaluator_factory(orchestrator)

            # Process the batch
            t0 = time.time()
            batch_result = self._process_single_batch(
                batch_id=batch_id,
                start_idx=starting_interaction_idx + start_idx,
                end_idx=starting_interaction_idx + end_idx,
                entries=batch_entries,
                queries=batch_queries,
                raw_data=raw_data,
                mode=mode,
                orchestrator=orchestrator,
                evaluator=evaluator,
            )
            batch_latency_ms = (time.time() - t0) * 1000
            batch_result.latency_ms = batch_latency_ms

            # Clean up batch orchestrator
            import shutil
            shutil.rmtree(batch_persist_dir, ignore_errors=True)

            # Update partial metrics
            for qr in batch_result.question_results:
                partial_metrics.update(qr)

            # Flush batch results to disk
            self._flush_batch_results(session_id, batch_result)

            # Accumulate for final summary
            all_session_results.extend(batch_result.session_results)
            all_question_results.extend(batch_result.question_results)

            # Update checkpoint periodically
            if (batch_idx + 1) % self.config.checkpoint_interval == 0:
                state.progress.completed_batches = batch_id + 1
                state.progress.completed_interactions = len(all_question_results)
                state.progress.last_batch_id = batch_id
                state.aggregated_metrics = partial_metrics.to_dict()
                self.checkpoint_manager.save_checkpoint(state)
                logger.info(
                    f"Checkpoint saved: batch {batch_id + 1}, "
                    f"{len(all_question_results)} interactions processed"
                )

            # Flush partial metrics if enabled
            if self.config.flush_partial_metrics:
                self._flush_partial_metrics(session_id, partial_metrics)

        # Mark as completed
        state.status = "completed"
        state.progress.completed_batches = starting_batch_id + total_batches
        state.progress.completed_interactions = len(all_question_results)
        state.aggregated_metrics = partial_metrics.to_dict()
        self.checkpoint_manager.save_checkpoint(state)

        # Generate final report
        summary = calculate_metrics(queries, all_question_results, all_session_results)
        self._generate_final_report(session_id, summary)

        logger.info(f"Evaluation complete: {session_id}")
        return summary

    def _process_single_batch(
        self,
        batch_id: int,
        start_idx: int,
        end_idx: int,
        entries: list[dict],
        queries: list[dict],
        raw_data: list[dict],
        mode: str,
        orchestrator: Orchestrator,
        evaluator: Evaluator,
    ) -> BatchResult:
        """Process a single batch of interactions."""
        batch_result = BatchResult(
            batch_id=batch_id,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        # Session replay mode - feed interactions into orchestrator
        if mode in ["lifecycle", "full"]:
            for entry in entries:
                content = entry.get("content", "")
                if content:
                    # Extract actual content from "[role] content" format
                    if content.startswith("[") and "]" in content:
                        content = content.split("]", 1)[1].strip()
                    orchestrator.chat(content)

            # Capture session metrics (simplified)
            batch_result.session_results.append(
                SessionReplayResult(
                    session_id=f"batch_{batch_id}",
                    turns_processed=len(entries),
                    ltm_entries_added=len(orchestrator.ltm_snapshot()),
                    stm_tokens_at_end=orchestrator.stm_stats().total_tokens,
                    learning_scores=[],
                )
            )

        # Question evaluation mode - evaluate queries
        if mode in ["retrieval", "full"] and queries:
            for query in queries:
                # Find corresponding raw data for this query
                query_result = self._evaluate_query(query, raw_data, evaluator)
                batch_result.question_results.append(query_result)

        return batch_result

    def _evaluate_query(
        self,
        query: dict,
        raw_data: list[dict],
        evaluator: Evaluator,
    ) -> QuestionResult:
        """Evaluate a single query."""
        query_text = query.get("query_text", "")
        query_id = query.get("query_id", "")

        # Find expected answer from raw data
        expected_answer = ""
        question = ""
        for instance in raw_data:
            if instance.get("question_id") == query_id:
                expected_answer = instance.get("answer", "")
                question = instance.get("question", "")
                break

        # Use evaluator to evaluate
        from evaluation.evaluators import EvaluationContext

        t0 = time.time()
        response = evaluator._orchestrator.chat(query_text)
        latency_ms = (time.time() - t0) * 1000

        # Build trace
        last_trace = evaluator._orchestrator.last_trace()
        retrieval_trace = evaluator._build_trace(last_trace)
        abstained = evaluator._detect_abstention(response)

        # Simple heuristic validation (or could use LLM judge)
        is_correct = evaluator._match_answer(response, expected_answer) if expected_answer else False

        return QuestionResult(
            query_id=query_id,
            is_correct=is_correct,
            behavior_type="IE",  # Simplified - could map from question type
            retrieval_trace=retrieval_trace,
            abstained=abstained,
            latency_ms=latency_ms,
            judge_result=None,
            validation_method="heuristic",
        )

    def _flush_batch_results(self, session_id: str, result: BatchResult) -> Path:
        """Flush batch results to disk as JSONL."""
        batch_path = self.checkpoint_manager.get_batch_path(session_id, result.batch_id)

        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f)
            f.write("\n")

        return batch_path

    def _flush_partial_metrics(self, session_id: str, metrics: PartialMetrics) -> Path:
        """Flush partial metrics to disk."""
        metrics_path = self.config.output_dir / f"{session_id}_partial_metrics.json"

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": session_id,
                "updated_at": datetime.now().isoformat(),
                "metrics": metrics.to_dict(),
            }, f, indent=2)

        return metrics_path

    def _load_dataset(
        self,
        dataset_path: Path,
        max_interactions: int = 0,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Load evaluation dataset."""
        logger.info(f"Loading dataset from {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Build entries and queries from dataset
        entries = []
        queries = []

        for instance in raw_data:
            question_id = instance.get("question_id", "")
            question_type = instance.get("question_type", "retrieval")
            question_text = instance.get("question", "")
            answer = instance.get("answer", "")

            # Process sessions into entries
            sessions = instance.get("haystack_sessions", [])
            for session_idx, session in enumerate(sessions):
                for turn_idx, turn in enumerate(session):
                    role = turn.get("role", "user")
                    content = turn.get("content", "")

                    entry_id = f"{question_id}_{session_idx}_{turn_idx}"
                    entries.append({
                        "content": f"[{role}] {content}",
                        "entry_id": entry_id,
                        "tags": [question_type, f"session_{session_idx}"],
                    })

            # Add query
            queries.append({
                "query_id": question_id,
                "query_text": question_text,
                "relevant_entry_ids": [],
                "relevant_content": [],
                "query_type": question_type,
                "expected_answer": answer,
            })

        # Limit if requested
        if max_interactions > 0 and len(entries) > max_interactions:
            entries = entries[:max_interactions]
            queries = queries[:max_interactions]

        logger.info(f"Loaded {len(entries)} entries and {len(queries)} queries")
        return entries, queries, raw_data

    def _reconstruct_partial_metrics(self, aggregated: dict) -> PartialMetrics:
        """Reconstruct PartialMetrics from checkpoint data."""
        return PartialMetrics(
            total_queries=aggregated.get("total_queries", 0),
            correct=aggregated.get("correct", 0),
            abstained=aggregated.get("abstained", 0),
            total_latency_ms=aggregated.get("total_latency_ms", 0.0),
            judge_latency_ms=aggregated.get("judge_latency_ms", 0.0),
            llm_judge_queries=aggregated.get("llm_judge_queries", 0),
            heuristic_queries=aggregated.get("heuristic_queries", 0),
        )

    def _generate_final_report(self, session_id: str, summary: EvaluationSummary) -> Path:
        """Generate final evaluation report."""
        from evaluation.run import generate_report
        return generate_report(summary, session_id, self.config.output_dir)

    def generate_partial_report(self, session_id: str) -> Optional[Path]:
        """
        Generate report from partial results at any point during evaluation.

        Args:
            session_id: The evaluation session ID.

        Returns:
            Path to generated report, or None if no checkpoint found.
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(session_id)
        if not checkpoint:
            logger.warning(f"No checkpoint found for {session_id}")
            return None

        # Load all batch results
        all_question_results = []
        all_session_results = []

        for batch_id in self.checkpoint_manager.list_completed_batches(session_id):
            batch_path = self.checkpoint_manager.get_batch_path(session_id, batch_id)
            with open(batch_path, "r", encoding="utf-8") as f:
                batch_data = json.load(f)

            # Reconstruct results (simplified)
            for qr_data in batch_data.get("question_results", []):
                all_question_results.append(QuestionResult(
                    query_id=qr_data["query_id"],
                    is_correct=qr_data["is_correct"],
                    behavior_type=qr_data["behavior_type"],
                    retrieval_trace={},
                    abstained=qr_data["abstained"],
                    latency_ms=qr_data["latency_ms"],
                    validation_method=qr_data["validation_method"],
                ))

        # Generate summary from partial data
        summary = calculate_metrics([], all_question_results, all_session_results)

        # Generate report with partial flag
        return self._generate_partial_report_markdown(session_id, summary, checkpoint)

    def _generate_partial_report_markdown(
        self,
        session_id: str,
        summary: EvaluationSummary,
        checkpoint: CheckpointState,
    ) -> Path:
        """Generate markdown report for partial results."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

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

        # Write report
        md_path = output_dir / f"{session_id}_partial_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Generated partial report: {md_path}")
        return md_path


# Re-exports from new modules for backward compatibility
# Note: BatchEvaluationRunner is retained here; use evaluation.runner.BatchRunner in new code
from evaluation.runner import BatchRunner, BatchConfig as _BatchConfig, PartialMetrics as _PartialMetrics

# Alias for backward compatibility - new code should use BatchRunner
# BatchEvaluationRunner = BatchRunner  # Disabled: different interface, keep old class
__all__ = [
    "BatchConfig",
    "BatchResult",
    "PartialMetrics",
    "BatchEvaluationRunner",
]
