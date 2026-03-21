"""
evaluation/metrics.py
---------------------
Simplified metrics calculation for AgeMem evaluation.

Calculates only the essential retrieval metrics:
- MRR@K (Mean Reciprocal Rank)
- Recall@K
- Precision@K
- NDCG@K
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional


@dataclass
class RetrievalMetrics:
    """Essential retrieval metrics."""
    mrr_at_1: float = 0.0
    mrr_at_5: float = 0.0
    mrr_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mrr_at_1": self.mrr_at_1,
            "mrr_at_5": self.mrr_at_5,
            "mrr_at_10": self.mrr_at_10,
            "precision_at_1": self.precision_at_1,
            "precision_at_5": self.precision_at_5,
            "precision_at_10": self.precision_at_10,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "ndcg_at_5": self.ndcg_at_5,
            "ndcg_at_10": self.ndcg_at_10,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class BehaviorMetrics:
    """Metrics for a specific behavior type."""
    behavior: str
    query_count: int = 0
    accuracy: float = 0.0


@dataclass
class EvaluationSummary:
    """Complete evaluation summary with LLM judge stats."""
    total_queries: int = 0
    correct: int = 0
    accuracy: float = 0.0
    abstained: int = 0
    avg_latency_ms: float = 0.0
    # New judge metrics
    llm_judge_queries: int = 0
    heuristic_queries: int = 0
    judge_avg_latency_ms: float = 0.0
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    by_behavior: dict[str, BehaviorMetrics] = field(default_factory=dict)
    session_replay: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "abstained": self.abstained,
            "avg_latency_ms": self.avg_latency_ms,
            "llm_judge_queries": self.llm_judge_queries,
            "heuristic_queries": self.heuristic_queries,
            "judge_avg_latency_ms": self.judge_avg_latency_ms,
            "retrieval": self.retrieval.to_dict(),
            "by_behavior": {k: {"behavior": v.behavior, "query_count": v.query_count, "accuracy": v.accuracy}
                          for k, v in self.by_behavior.items()},
            "session_replay": self.session_replay,
        }


def calculate_metrics(
    queries: list[dict],
    question_results: list,
    session_results: list,
) -> EvaluationSummary:
    """
    Calculate all evaluation metrics.

    Args:
        queries: List of query dicts with 'relevant_entry_ids' and 'relevant_content'
        question_results: List of QuestionResult objects
        session_results: List of SessionReplayResult objects

    Returns:
        EvaluationSummary with all metrics
    """
    summary = EvaluationSummary()

    # Basic question metrics
    summary.total_queries = len(question_results)
    summary.correct = sum(1 for r in question_results if r.is_correct)
    summary.accuracy = summary.correct / summary.total_queries if summary.total_queries else 0.0
    summary.abstained = sum(1 for r in question_results if r.abstained)
    summary.avg_latency_ms = mean([r.latency_ms for r in question_results]) if question_results else 0.0

    # LLM-as-Judge metrics
    summary.llm_judge_queries = sum(1 for r in question_results if r.validation_method == "llm_judge")
    summary.heuristic_queries = sum(1 for r in question_results if r.validation_method == "heuristic")
    judge_latencies = [r.judge_result.latency_ms for r in question_results if r.judge_result]
    summary.judge_avg_latency_ms = mean(judge_latencies) if judge_latencies else 0.0

    # Retrieval metrics
    summary.retrieval = _calculate_retrieval_metrics(queries, question_results)

    # Behavior breakdown
    summary.by_behavior = _calculate_behavior_breakdown(question_results)

    # Session replay metrics
    if session_results:
        summary.session_replay = {
            "total_sessions": len(session_results),
            "total_turns": sum(r.turns_processed for r in session_results),
            "total_ltm_adds": sum(r.ltm_entries_added for r in session_results),
            "avg_stm_tokens": mean([r.stm_tokens_at_end for r in session_results]),
            "avg_learning_score": mean([mean(r.learning_scores) if r.learning_scores else 0.0
                                        for r in session_results]),
        }

    return summary


def _calculate_retrieval_metrics(
    queries: list[dict],
    question_results: list,
) -> RetrievalMetrics:
    """Calculate retrieval metrics from question results."""
    if not queries or not question_results:
        return RetrievalMetrics()

    # Build traces from question results
    traces = []
    for qr in question_results:
        trace_results = qr.retrieval_trace.get("results", []) if qr.retrieval_trace else []
        traces.append({
            "results": trace_results,
            "latency_ms": qr.latency_ms,
        })

    return RetrievalMetrics(
        mrr_at_1=_mrr_at_k(queries, traces, k=1),
        mrr_at_5=_mrr_at_k(queries, traces, k=5),
        mrr_at_10=_mrr_at_k(queries, traces, k=10),
        precision_at_1=_precision_at_k(queries, traces, k=1),
        precision_at_5=_precision_at_k(queries, traces, k=5),
        precision_at_10=_precision_at_k(queries, traces, k=10),
        recall_at_1=_recall_at_k(queries, traces, k=1),
        recall_at_5=_recall_at_k(queries, traces, k=5),
        recall_at_10=_recall_at_k(queries, traces, k=10),
        ndcg_at_5=_ndcg_at_k(queries, traces, k=5),
        ndcg_at_10=_ndcg_at_k(queries, traces, k=10),
        avg_latency_ms=mean([t["latency_ms"] for t in traces]) if traces else 0.0,
    )


def _mrr_at_k(queries: list[dict], traces: list[dict], k: int) -> float:
    """Calculate Mean Reciprocal Rank at K."""
    if not queries or not traces:
        return 0.0

    reciprocal_ranks = []

    for query, trace in zip(queries, traces):
        relevant_ids = set(query.get("relevant_entry_ids", []))
        found = False

        for rank, (entry_id, score) in enumerate(trace["results"][:k], start=1):
            if entry_id in relevant_ids:
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break

        if not found:
            reciprocal_ranks.append(0.0)

    return mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def _precision_at_k(queries: list[dict], traces: list[dict], k: int) -> float:
    """Calculate Precision at K."""
    if not queries or not traces:
        return 0.0

    precisions = []

    for query, trace in zip(queries, traces):
        relevant_ids = set(query.get("relevant_entry_ids", []))
        top_k = trace["results"][:k]

        relevant_count = sum(1 for entry_id, _ in top_k if entry_id in relevant_ids)
        precisions.append(relevant_count / k)

    return mean(precisions) if precisions else 0.0


def _recall_at_k(queries: list[dict], traces: list[dict], k: int) -> float:
    """Calculate Recall at K."""
    if not queries or not traces:
        return 0.0

    recalls = []

    for query, trace in zip(queries, traces):
        relevant_ids = set(query.get("relevant_entry_ids", []))
        total_relevant = len(relevant_ids)

        if total_relevant == 0:
            continue

        top_k = trace["results"][:k]
        relevant_count = sum(1 for entry_id, _ in top_k if entry_id in relevant_ids)
        recalls.append(relevant_count / total_relevant)

    return mean(recalls) if recalls else 0.0


def _ndcg_at_k(queries: list[dict], traces: list[dict], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K."""
    if not queries or not traces:
        return 0.0

    ndcgs = []

    for query, trace in zip(queries, traces):
        relevant_ids = set(query.get("relevant_entry_ids", []))

        # Calculate DCG
        dcg = 0.0
        for rank, (entry_id, _) in enumerate(trace["results"][:k], start=1):
            rel = 1.0 if entry_id in relevant_ids else 0.0
            dcg += (2 ** rel - 1) / math.log2(rank + 1)

        # Calculate IDCG
        num_relevant = len(relevant_ids)
        ideal_relevances = [1.0] * min(num_relevant, k)
        idcg = sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(ideal_relevances)
        )

        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)

    return mean(ndcgs) if ndcgs else 0.0


def _calculate_behavior_breakdown(question_results: list) -> dict[str, BehaviorMetrics]:
    """Calculate accuracy breakdown by behavior type."""
    behavior_stats: dict[str, dict] = {}

    for r in question_results:
        behavior = r.behavior_type
        if behavior not in behavior_stats:
            behavior_stats[behavior] = {"total": 0, "correct": 0}
        behavior_stats[behavior]["total"] += 1
        if r.is_correct:
            behavior_stats[behavior]["correct"] += 1

    return {
        behavior: BehaviorMetrics(
            behavior=behavior,
            query_count=stats["total"],
            accuracy=stats["correct"] / stats["total"] if stats["total"] else 0.0,
        )
        for behavior, stats in behavior_stats.items()
    }
