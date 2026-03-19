"""
KPI Metrics Collection Pipeline Module
--------------------------------------

Calculates evaluation metrics per Section 3.3 of TRS-AGEMEM-EVAL-001.

Metrics implemented:
- Retrieval: MRR@K, Precision@K, Recall@K, NDCG@K
- Memory Quality: Retention Rate, Deduplication Accuracy, Learning Score Correlation
- Response Quality: Hallucination Rate, Coherence Score, Memory Grounding, Preference Accuracy
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from statistics import mean, stdev

from evaluation.pipeline.dataset_pipeline import BenchmarkQuery
from evaluation.pipeline.inference_pipeline import SearchTrace

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """
    Retrieval quality metrics per Section 7.3.1 of the technical specification.

    Targets per Section 3.3.3:
    - MRR@10 >= 0.85
    - Recall@5 >= 0.90
    """
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
        return asdict(self)

    def meets_targets(self) -> dict[str, bool]:
        """Check if metrics meet specified targets."""
        return {
            "mrr_at_10": self.mrr_at_10 >= 0.85,
            "recall_at_5": self.recall_at_5 >= 0.90,
        }


@dataclass
class MemoryQualityMetrics:
    """
    Memory quality metrics per Section 7.3.2 of the technical specification.

    Targets:
    - Retention Rate >= 95%
    - Deduplication Accuracy >= 90%
    - Learning Score Correlation >= 0.7
    - Context Utilization >= 60%
    """
    retention_rate: float = 0.0
    deduplication_accuracy: float = 0.0
    learning_score_correlation: float = 0.0
    context_utilization: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def meets_targets(self) -> dict[str, bool]:
        """Check if metrics meet specified targets."""
        return {
            "retention_rate": self.retention_rate >= 0.95,
            "deduplication_accuracy": self.deduplication_accuracy >= 0.90,
            "learning_score_correlation": self.learning_score_correlation >= 0.7,
            "context_utilization": self.context_utilization >= 0.60,
        }


@dataclass
class ResponseQualityMetrics:
    """
    Response quality metrics per Section 7.3.3 of the technical specification.

    Targets:
    - Hallucination Rate <= 5%
    - Coherence Score >= 4.0
    - Memory Grounding >= 90%
    - Preference Accuracy >= 95%
    """
    hallucination_rate: float = 0.0
    coherence_score: float = 0.0
    memory_grounding: float = 0.0
    preference_accuracy: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def meets_targets(self) -> dict[str, bool]:
        """Check if metrics meet specified targets."""
        return {
            "hallucination_rate": self.hallucination_rate <= 0.05,
            "coherence_score": self.coherence_score >= 4.0,
            "memory_grounding": self.memory_grounding >= 0.90,
            "preference_accuracy": self.preference_accuracy >= 0.95,
        }


@dataclass
class ComparativeMetrics:
    """
    Comparative performance metrics per Section 7.4 of the technical specification.

    Benchmarks AgeMem against:
    - MemGPT
    - Letta
    - LangChain RAG
    - LlamaIndex
    - Base LLM (no memory)
    """
    system_name: str = "AgeMem"
    mrr_at_5: float = 0.0
    hallucination_rate: float = 0.0
    tokens_per_query: float = 0.0
    latency_ms: float = 0.0
    memory_footprint_mb: float = 0.0

    # Competitor benchmarks (populated separately)
    competitors: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BehaviorMetrics:
    """
    Per-behavior metrics per LongMemEval benchmark specification.

    Maps to the five core memory behaviors:
    - Information Extraction (IE): Single-Session-User/Assistant questions
    - Multi-Session Reasoning (MR): Aggregation/Comparison questions
    - Knowledge Updates (KU): Most-Recent-Value questions
    - Temporal Reasoning (TR): Time-Reference/Date-Filtering questions
    - Abstention (ABS): Unknown-Information questions
    """
    behavior_name: str
    query_count: int = 0
    mrr_at_10: float = 0.0
    recall_at_5: float = 0.0
    precision_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResults:
    """Complete evaluation results for a session."""
    session_id: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    memory_quality: MemoryQualityMetrics = field(default_factory=MemoryQualityMetrics)
    response_quality: ResponseQualityMetrics = field(default_factory=ResponseQualityMetrics)
    comparative: ComparativeMetrics = field(default_factory=ComparativeMetrics)
    behavior_metrics: dict[str, BehaviorMetrics] = field(default_factory=dict)
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "retrieval": self.retrieval.to_dict(),
            "memory_quality": self.memory_quality.to_dict(),
            "response_quality": self.response_quality.to_dict(),
            "comparative": self.comparative.to_dict(),
            "behavior_metrics": {k: v.to_dict() for k, v in self.behavior_metrics.items()},
            "evaluated_at": self.evaluated_at,
        }


class MetricsPipeline:
    """
    KPI Metrics Collection Pipeline per Section 3.3 of TRS-AGEMEM-EVAL-001.

    Calculates MRR@K and other accuracy benchmarks.
    """

    def __init__(self, session_id: str = "") -> None:
        self._session_id = session_id
        self._results: Optional[EvaluationResults] = None

    # ── Retrieval Metrics ─────────────────────────────────────────────────────

    def calculate_mrr_at_k(
        self,
        queries: list[BenchmarkQuery],
        traces: list[SearchTrace],
        k: int = 10,
    ) -> float:
        """
        Calculate Mean Reciprocal Rank at K.

        Formula: MRR@K = (1/N) * sum(1/rank_of_first_relevant)

        Per Section 7.3.1 of the technical specification.
        """
        if not queries or not traces:
            return 0.0

        reciprocal_ranks = []

        for query, trace in zip(queries, traces):
            relevant_ids = set(query.relevant_entry_ids)
            found = False

            for rank, (entry_id, score) in enumerate(trace.results[:k], start=1):
                if entry_id in relevant_ids:
                    reciprocal_ranks.append(1.0 / rank)
                    found = True
                    break

            if not found:
                reciprocal_ranks.append(0.0)

        return mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def calculate_precision_at_k(
        self,
        queries: list[BenchmarkQuery],
        traces: list[SearchTrace],
        k: int = 5,
    ) -> float:
        """
        Calculate Precision at K.

        Formula: Precision@K = relevant_in_topK / K

        Per Section 7.3.1 of the technical specification.
        """
        if not queries or not traces:
            return 0.0

        precisions = []

        for query, trace in zip(queries, traces):
            relevant_ids = set(query.relevant_entry_ids)
            top_k = trace.results[:k]

            relevant_in_topk = sum(
                1 for entry_id, _ in top_k if entry_id in relevant_ids
            )

            precisions.append(relevant_in_topk / k)

        return mean(precisions) if precisions else 0.0

    def calculate_recall_at_k(
        self,
        queries: list[BenchmarkQuery],
        traces: list[SearchTrace],
        k: int = 10,
    ) -> float:
        """
        Calculate Recall at K.

        Formula: Recall@K = relevant_in_topK / total_relevant

        Per Section 7.3.1 of the technical specification.
        """
        if not queries or not traces:
            return 0.0

        recalls = []

        for query, trace in zip(queries, traces):
            relevant_ids = set(query.relevant_entry_ids)
            if not relevant_ids:
                continue

            top_k = trace.results[:k]
            relevant_in_topk = sum(
                1 for entry_id, _ in top_k if entry_id in relevant_ids
            )

            recalls.append(relevant_in_topk / len(relevant_ids))

        return mean(recalls) if recalls else 0.0

    def calculate_ndcg_at_k(
        self,
        queries: list[BenchmarkQuery],
        traces: list[SearchTrace],
        k: int = 10,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        Formula: NDCG@K = DCG@K / IDCG@K
        where DCG = sum((2^relevance - 1) / log2(rank + 1))

        Per Section 7.3.1 of the technical specification.
        """
        if not queries or not traces:
            return 0.0

        ndcgs = []

        for query, trace in zip(queries, traces):
            relevance_scores = query.relevance_scores
            if not relevance_scores:
                # Binary relevance: 1 if relevant, 0 otherwise
                relevance_scores = {
                    entry_id: 1.0 for entry_id in query.relevant_entry_ids
                }

            # Calculate DCG
            dcg = 0.0
            for rank, (entry_id, _) in enumerate(trace.results[:k], start=1):
                rel = relevance_scores.get(entry_id, 0.0)
                dcg += (2 ** rel - 1) / math.log2(rank + 1)

            # Calculate IDCG (ideal DCG)
            ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
            idcg = sum(
                (2 ** rel - 1) / math.log2(i + 2)
                for i, rel in enumerate(ideal_scores)
            )

            # Calculate NDCG
            if idcg > 0:
                ndcgs.append(dcg / idcg)
            else:
                ndcgs.append(0.0)

        return mean(ndcgs) if ndcgs else 0.0

    # ── Full Retrieval Metrics Calculation ────────────────────────────────────

    def calculate_retrieval_metrics(
        self,
        queries: list[BenchmarkQuery],
        traces: list[SearchTrace],
    ) -> RetrievalMetrics:
        """Calculate all retrieval metrics."""
        avg_latency = mean(t.latency_ms for t in traces) if traces else 0.0

        return RetrievalMetrics(
            mrr_at_1=self.calculate_mrr_at_k(queries, traces, k=1),
            mrr_at_5=self.calculate_mrr_at_k(queries, traces, k=5),
            mrr_at_10=self.calculate_mrr_at_k(queries, traces, k=10),
            precision_at_1=self.calculate_precision_at_k(queries, traces, k=1),
            precision_at_5=self.calculate_precision_at_k(queries, traces, k=5),
            precision_at_10=self.calculate_precision_at_k(queries, traces, k=10),
            recall_at_1=self.calculate_recall_at_k(queries, traces, k=1),
            recall_at_5=self.calculate_recall_at_k(queries, traces, k=5),
            recall_at_10=self.calculate_recall_at_k(queries, traces, k=10),
            ndcg_at_5=self.calculate_ndcg_at_k(queries, traces, k=5),
            ndcg_at_10=self.calculate_ndcg_at_k(queries, traces, k=10),
            avg_latency_ms=avg_latency,
        )

    # ── Behavior-Segmented Metrics ─────────────────────────────────────────────

    # Mapping from LongMemEval question types to behavior categories
    BEHAVIOR_MAPPING = {
        # Information Extraction
        "single-session-user": "information_extraction",
        "single-session-assistant": "information_extraction",
        "preference": "information_extraction",
        # Multi-Session Reasoning
        "multi-session": "multi_session_reasoning",
        "aggregation": "multi_session_reasoning",
        "comparison": "multi_session_reasoning",
        # Knowledge Updates
        "knowledge-update": "knowledge_updates",
        "knowledge": "knowledge_updates",
        # Temporal Reasoning
        "temporal-reasoning": "temporal_reasoning",
        "temporal": "temporal_reasoning",
        "time-reference": "temporal_reasoning",
        "date-filtering": "temporal_reasoning",
        # Abstention
        "unknown": "abstention",
        "abstention": "abstention",
    }

    def calculate_behavior_metrics(
        self,
        queries: list[BenchmarkQuery],
        traces: list[SearchTrace],
    ) -> dict[str, BehaviorMetrics]:
        """
        Calculate metrics segmented by question_type (behavior category).

        Per LongMemEval benchmark specification, results should be segmented
        by the five core memory behaviors:
        - Information Extraction (IE)
        - Multi-Session Reasoning (MR)
        - Knowledge Updates (KU)
        - Temporal Reasoning (TR)
        - Abstention (ABS)

        Returns:
            Dictionary mapping behavior names to BehaviorMetrics objects
        """
        if not queries or not traces:
            return {}

        # Group queries and traces by behavior category
        behavior_groups: dict[str, tuple[list[BenchmarkQuery], list[SearchTrace]]] = {}

        for query, trace in zip(queries, traces):
            # Map query_type to behavior category
            query_type_lower = query.query_type.lower().replace("_", "-")
            behavior = self.BEHAVIOR_MAPPING.get(
                query_type_lower,
                query.query_type.lower().replace(" ", "_")
            )

            if behavior not in behavior_groups:
                behavior_groups[behavior] = ([], [])
            behavior_groups[behavior][0].append(query)
            behavior_groups[behavior][1].append(trace)

        # Calculate metrics for each behavior
        results = {}
        for behavior, (behavior_queries, behavior_traces) in behavior_groups.items():
            avg_latency = mean(t.latency_ms for t in behavior_traces) if behavior_traces else 0.0

            metrics = BehaviorMetrics(
                behavior_name=behavior,
                query_count=len(behavior_queries),
                mrr_at_10=self.calculate_mrr_at_k(behavior_queries, behavior_traces, k=10),
                recall_at_5=self.calculate_recall_at_k(behavior_queries, behavior_traces, k=5),
                precision_at_5=self.calculate_precision_at_k(behavior_queries, behavior_traces, k=5),
                ndcg_at_10=self.calculate_ndcg_at_k(behavior_queries, behavior_traces, k=10),
                avg_latency_ms=avg_latency,
            )
            results[behavior] = metrics

        return results

    # ── Memory Quality Metrics ────────────────────────────────────────────────

    def calculate_memory_quality_metrics(
        self,
        promoted_entries: list[str],
        retained_after_n_turns: list[str],
        dedup_true_positives: int,
        dedup_total: int,
        learning_scores: list[float],
        utility_scores: list[float],
        context_utilizations: list[float],
    ) -> MemoryQualityMetrics:
        """
        Calculate memory quality metrics.

        Args:
            promoted_entries: IDs of entries promoted to LTM
            retained_after_n_turns: IDs of entries still in LTM after N turns
            dedup_true_positives: Correctly identified duplicates
            dedup_total: Total duplicate pairs
            learning_scores: Learning scores assigned to entries
            utility_scores: Actual utility scores (human-rated or task-based)
            context_utilizations: Context utilization ratios per turn
        """
        # Retention rate
        if promoted_entries:
            retained_set = set(retained_after_n_turns)
            retention_rate = len([e for e in promoted_entries if e in retained_set]) / len(promoted_entries)
        else:
            retention_rate = 1.0

        # Deduplication accuracy
        dedup_accuracy = dedup_true_positives / dedup_total if dedup_total > 0 else 1.0

        # Learning score correlation (Pearson)
        if len(learning_scores) > 1 and len(utility_scores) > 1:
            correlation = self._pearson_correlation(learning_scores, utility_scores)
        else:
            correlation = 0.0

        # Context utilization
        avg_utilization = mean(context_utilizations) if context_utilizations else 0.0

        return MemoryQualityMetrics(
            retention_rate=retention_rate,
            deduplication_accuracy=dedup_accuracy,
            learning_score_correlation=correlation,
            context_utilization=avg_utilization,
        )

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        mean_x = mean(x)
        mean_y = mean(y)

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

        denominator = math.sqrt(sum_sq_x * sum_sq_y)

        return numerator / denominator if denominator > 0 else 0.0

    # ── Response Quality Metrics ───────────────────────────────────────────────

    def calculate_response_quality_metrics(
        self,
        total_responses: int,
        hallucinated_responses: int,
        coherence_ratings: list[float],
        grounded_claims: int,
        total_memory_claims: int,
        correct_preferences: int,
        total_preferences: int,
    ) -> ResponseQualityMetrics:
        """
        Calculate response quality metrics.

        Args:
            total_responses: Total number of responses evaluated
            hallucinated_responses: Responses with unsupported claims
            coherence_ratings: Human-rated coherence scores (1-5)
            grounded_claims: Memory-dependent claims with valid citations
            total_memory_claims: Total memory-dependent claims
            correct_preferences: Correctly recalled user preferences
            total_preferences: Total preference queries
        """
        hallucination_rate = hallucinated_responses / total_responses if total_responses > 0 else 0.0
        coherence_score = mean(coherence_ratings) if coherence_ratings else 0.0
        memory_grounding = grounded_claims / total_memory_claims if total_memory_claims > 0 else 1.0
        preference_accuracy = correct_preferences / total_preferences if total_preferences > 0 else 1.0

        return ResponseQualityMetrics(
            hallucination_rate=hallucination_rate,
            coherence_score=coherence_score,
            memory_grounding=memory_grounding,
            preference_accuracy=preference_accuracy,
        )

    # ── Full Evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        queries: list[BenchmarkQuery],
        traces: list[SearchTrace],
        memory_quality_data: Optional[dict] = None,
        response_quality_data: Optional[dict] = None,
    ) -> EvaluationResults:
        """
        Run full evaluation and return results.

        Args:
            queries: Benchmark queries
            traces: Search traces from inference pipeline
            memory_quality_data: Optional dict with memory quality metrics data
            response_quality_data: Optional dict with response quality metrics data

        Returns:
            EvaluationResults object with all metrics including behavior-segmented metrics
        """
        # Calculate retrieval metrics
        retrieval = self.calculate_retrieval_metrics(queries, traces)

        # Calculate behavior-segmented metrics per LongMemEval specification
        behavior_metrics = self.calculate_behavior_metrics(queries, traces)

        # Calculate memory quality metrics if data provided
        if memory_quality_data:
            memory_quality = self.calculate_memory_quality_metrics(**memory_quality_data)
        else:
            memory_quality = MemoryQualityMetrics()

        # Calculate response quality metrics if data provided
        if response_quality_data:
            response_quality = self.calculate_response_quality_metrics(**response_quality_data)
        else:
            response_quality = ResponseQualityMetrics()

        # Build comparative metrics
        comparative = ComparativeMetrics(
            system_name="AgeMem",
            mrr_at_5=retrieval.mrr_at_5,
            hallucination_rate=response_quality.hallucination_rate,
            tokens_per_query=retrieval.avg_latency_ms,  # Placeholder
            latency_ms=retrieval.avg_latency_ms,
        )

        self._results = EvaluationResults(
            session_id=self._session_id,
            retrieval=retrieval,
            memory_quality=memory_quality,
            response_quality=response_quality,
            comparative=comparative,
            behavior_metrics=behavior_metrics,
        )

        return self._results

    # ── Output ─────────────────────────────────────────────────────────────────

    def export_json(self, output_path: Path) -> None:
        """Export results to JSON file."""
        if not self._results:
            raise ValueError("No results to export. Run evaluate() first.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._results.to_dict(), f, indent=2)

        logger.info(f"Exported results to {output_path}")

    def get_results(self) -> Optional[EvaluationResults]:
        return self._results