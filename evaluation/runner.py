"""
evaluation/runner.py - Slimmed batch runner that delegates to specialized modules.
"""
from __future__ import annotations
import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from agents.orchestrator import Orchestrator
from evaluation.checkpoint import BatchProgress, CheckpointManager, CheckpointState
from evaluation.evaluator import Evaluator, QuestionResult, SessionReplayResult
from evaluation.loader import DatasetLoader
from evaluation.metrics import EvaluationSummary, calculate_metrics
from evaluation.report import ReportGenerator

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    batch_size: int = 10
    checkpoint_interval: int = 1
    output_dir: Path = field(default_factory=lambda: Path("evaluation/results"))
    resume_from_checkpoint: bool = True
    flush_partial_metrics: bool = True
    use_mock_llm: bool = False
    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

@dataclass
class PartialMetrics:
    total_queries: int = 0
    correct: int = 0
    abstained: int = 0
    total_latency_ms: float = 0.0
    judge_latency_ms: float = 0.0
    llm_judge_queries: int = 0
    heuristic_queries: int = 0
    def update(self, r: QuestionResult) -> None:
        self.total_queries += 1
        if r.is_correct: self.correct += 1
        if r.abstained: self.abstained += 1
        self.total_latency_ms += r.latency_ms
        if r.validation_method == "llm_judge":
            self.llm_judge_queries += 1
            if r.judge_result: self.judge_latency_ms += r.judge_result.latency_ms
        else: self.heuristic_queries += 1
    def to_dict(self) -> dict:
        n = self.total_queries
        return {"total_queries": n, "correct": self.correct, "accuracy": self.correct/n if n else 0.0,
                "abstained": self.abstained, "avg_latency_ms": self.total_latency_ms/n if n else 0.0,
                "llm_judge_queries": self.llm_judge_queries, "heuristic_queries": self.heuristic_queries,
                "judge_avg_latency_ms": self.judge_latency_ms/self.llm_judge_queries if self.llm_judge_queries else 0.0}

class BatchRunner:
    """Batch-aware evaluation runner with checkpoint persistence."""
    def __init__(self, config: BatchConfig, factory: Any,
                 evaluator_factory: Optional[Callable[[Orchestrator], Evaluator]] = None) -> None:
        self.config = config
        self.factory = factory
        self.evaluator_factory = evaluator_factory or (lambda orch: Evaluator(orch))
        self.checkpoint_manager = CheckpointManager(config.output_dir)
        self.loader = DatasetLoader()
        self.reporter = ReportGenerator(config.output_dir)

    def run(self, dataset_path: Path, mode: str = "full", max_interactions: int = 0,
            max_batches: int = 0, session_id: Optional[str] = None) -> EvaluationSummary:
        """Run batch evaluation, resuming from checkpoint if available."""
        if session_id is None: session_id = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        checkpoint = self.checkpoint_manager.load_checkpoint(session_id)
        if checkpoint and self.config.resume_from_checkpoint:
            logger.info(f"Found checkpoint for {session_id}, resuming...")
            return self._resume_from_checkpoint(checkpoint, dataset_path, mode, max_batches)
        logger.info(f"Starting new batch evaluation: {session_id}")
        return self._run_fresh(session_id, dataset_path, mode, max_interactions, max_batches)

    def _run_fresh(self, session_id: str, dataset_path: Path, mode: str,
                   max_interactions: int, max_batches: int) -> EvaluationSummary:
        """Run a fresh evaluation from the beginning."""
        entries, queries, raw_data = self.loader.load(dataset_path, limit=max_interactions)
        total_interactions = len(entries)
        logger.info(f"Loaded {total_interactions} interactions, batch_size={self.config.batch_size}")
        state = CheckpointState(
            session_id=session_id,
            config={"batch_size": self.config.batch_size, "mode": mode,
                    "dataset": str(dataset_path), "max_interactions": max_interactions},
            progress=BatchProgress(total_interactions=total_interactions),
            aggregated_metrics={}, status="running")
        self.checkpoint_manager.save_checkpoint(state)
        try:
            return self._process_batches(session_id, entries, queries, raw_data, mode,
                                         max_batches, state, PartialMetrics())
        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            self.checkpoint_manager.mark_failed(session_id, str(e))
            raise

    def _resume_from_checkpoint(self, checkpoint: CheckpointState, dataset_path: Path,
                                 mode: str, max_batches: int) -> EvaluationSummary:
        """Resume evaluation from a checkpoint."""
        session_id, completed_batches = checkpoint.session_id, checkpoint.progress.completed_batches
        logger.info(f"Resuming {session_id} from batch {completed_batches}")
        max_interactions = checkpoint.config.get("max_interactions", 0)
        entries, queries, raw_data = self.loader.load(dataset_path, limit=max_interactions)
        processed_count = checkpoint.progress.completed_interactions
        remaining_entries, remaining_queries = entries[processed_count:], \
            queries[processed_count:] if len(queries) > processed_count else []
        logger.info(f"Skipping {processed_count} already processed interactions")
        partial_metrics = self._reconstruct_partial_metrics(checkpoint.aggregated_metrics)
        checkpoint.status = "running"
        self.checkpoint_manager.save_checkpoint(checkpoint)
        try:
            return self._process_batches(session_id, remaining_entries, remaining_queries, raw_data,
                                         mode, max_batches, checkpoint, partial_metrics,
                                         starting_batch_id=completed_batches, starting_interaction_idx=processed_count)
        except Exception as e:
            logger.exception(f"Evaluation failed during resume: {e}")
            self.checkpoint_manager.mark_failed(session_id, str(e))
            raise

    def _process_batches(self, session_id: str, entries: list[dict], queries: list[dict],
                         raw_data: list[dict], mode: str, max_batches: int,
                         state: CheckpointState, partial_metrics: PartialMetrics,
                         starting_batch_id: int = 0, starting_interaction_idx: int = 0) -> EvaluationSummary:
        """Process all batches and return final summary."""
        batch_size = self.config.batch_size
        total_batches = (len(entries) + batch_size - 1) // batch_size
        if max_batches > 0: total_batches = min(total_batches, max_batches)
        logger.info(f"Processing {total_batches} batches (batch_size={batch_size})")
        all_session_results, all_question_results, all_evaluated_queries = [], [], []
        for batch_idx in range(total_batches):
            batch_id = starting_batch_id + batch_idx
            start_idx, end_idx = batch_idx * batch_size, min((batch_idx+1)*batch_size, len(entries))
            batch_entries = entries[start_idx:end_idx]
            batch_queries = queries[start_idx:end_idx] if start_idx < len(queries) else []
            logger.info(f"Processing batch {batch_id} (interactions {start_idx}-{end_idx})")
            batch_persist_dir = Path(tempfile.mkdtemp(prefix=f"agemem_eval_batch_{batch_id}_"))
            try:
                build_kwargs = {"persist_dir": batch_persist_dir, "config_overrides": {"STM_TOKEN_LIMIT": 8000}}
                if self.config.use_mock_llm:
                    from evaluation.mock_llm import StatefulMockLLM
                    build_kwargs["llm_client"] = StatefulMockLLM(strategy="template")
                    build_kwargs["use_real_llm"] = False
                orchestrator = self.factory.build_for_evaluation(**build_kwargs)
                evaluator = self.evaluator_factory(orchestrator)
                t0 = time.time()
                batch_result = self._process_single_batch(
                    batch_id, starting_interaction_idx+start_idx, starting_interaction_idx+end_idx,
                    batch_entries, batch_queries, raw_data, mode, orchestrator, evaluator)
                batch_result["latency_ms"] = (time.time() - t0) * 1000
            finally:
                shutil.rmtree(batch_persist_dir, ignore_errors=True)
            for qr in batch_result["question_results"]: partial_metrics.update(qr)
            self._flush_batch_results(session_id, batch_result)
            all_session_results.extend(batch_result["session_results"])
            all_question_results.extend(batch_result["question_results"])
            all_evaluated_queries.extend(batch_queries)
            if (batch_idx + 1) % self.config.checkpoint_interval == 0:
                state.progress.completed_batches, state.progress.completed_interactions = batch_id + 1, len(all_question_results)
                state.progress.last_batch_id, state.aggregated_metrics = batch_id, partial_metrics.to_dict()
                self.checkpoint_manager.save_checkpoint(state)
                logger.info(f"Checkpoint saved: batch {batch_id+1}, {len(all_question_results)} interactions")
            if self.config.flush_partial_metrics: self._flush_partial_metrics(session_id, partial_metrics)
        state.status, state.progress.completed_batches = "completed", starting_batch_id + total_batches
        state.progress.completed_interactions, state.aggregated_metrics = len(all_question_results), partial_metrics.to_dict()
        self.checkpoint_manager.save_checkpoint(state)
        summary = calculate_metrics(all_evaluated_queries, all_question_results, all_session_results)
        self.reporter.generate_markdown(summary, session_id)
        self.reporter.generate_json(summary, session_id)
        logger.info(f"Evaluation complete: {session_id}")
        return summary

    def _process_single_batch(self, batch_id: int, start_idx: int, end_idx: int,
                               entries: list[dict], queries: list[dict], raw_data: list[dict],
                               mode: str, orchestrator: Orchestrator, evaluator: Evaluator) -> dict:
        """Process a single batch of interactions."""
        session_results, question_results = [], []
        if mode in ["lifecycle", "full"]:
            for entry in entries:
                content = entry.get("content", "")
                if content:
                    if content.startswith("[") and "]" in content: content = content.split("]", 1)[1].strip()
                    orchestrator.chat(content)
            session_results.append(SessionReplayResult(
                session_id=f"batch_{batch_id}", turns_processed=len(entries),
                ltm_entries_added=len(orchestrator.ltm_snapshot()),
                stm_tokens_at_end=orchestrator.stm_stats().total_tokens, learning_scores=[]))
        if mode in ["retrieval", "full"] and queries:
            for query in queries:
                instance = next((i for i in raw_data if i.get("question_id") == query.get("query_id")), {})
                question_results.append(evaluator.evaluate_query(
                    {"query_text": query.get("query_text", ""), "query_id": query.get("query_id", "")}, instance))
        return {"batch_id": batch_id, "start_idx": start_idx, "end_idx": end_idx,
                "session_results": session_results, "question_results": question_results}

    def _flush_batch_results(self, session_id: str, result: dict) -> Path:
        batch_path = self.checkpoint_manager.get_batch_path(session_id, result["batch_id"])
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(result, f, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o))
            f.write("\n")
        return batch_path

    def _flush_partial_metrics(self, session_id: str, metrics: PartialMetrics) -> Path:
        metrics_path = self.config.output_dir / f"{session_id}_partial_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"session_id": session_id, "updated_at": datetime.now().isoformat(),
                       "metrics": metrics.to_dict()}, f, indent=2)
        return metrics_path

    def _reconstruct_partial_metrics(self, a: dict) -> PartialMetrics:
        return PartialMetrics(total_queries=a.get("total_queries", 0), correct=a.get("correct", 0),
                              abstained=a.get("abstained", 0), total_latency_ms=a.get("total_latency_ms", 0.0),
                              judge_latency_ms=a.get("judge_latency_ms", 0.0),
                              llm_judge_queries=a.get("llm_judge_queries", 0), heuristic_queries=a.get("heuristic_queries", 0))

    def generate_partial_report(self, session_id: str) -> Optional[Path]:
        """Generate report from partial results during evaluation."""
        checkpoint = self.checkpoint_manager.load_checkpoint(session_id)
        if not checkpoint:
            logger.warning(f"No checkpoint found for {session_id}")
            return None
        all_question_results = []
        for batch_id in self.checkpoint_manager.list_completed_batches(session_id):
            with open(self.checkpoint_manager.get_batch_path(session_id, batch_id), "r", encoding="utf-8") as f:
                for qr in json.load(f).get("question_results", []):
                    all_question_results.append(QuestionResult(
                        query_id=qr["query_id"], is_correct=qr["is_correct"], behavior_type=qr["behavior_type"],
                        retrieval_trace={}, abstained=qr["abstained"], latency_ms=qr["latency_ms"],
                        validation_method=qr["validation_method"]))
        summary = calculate_metrics([], all_question_results, [])
        return self.reporter.generate_partial_markdown(summary, session_id, checkpoint)