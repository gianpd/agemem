"""
evaluation/pipeline/end_to_end_runner.py
────────────────────────────────────────
Combined Phase 1 + Phase 2 Evaluation Runner

Executes a comprehensive evaluation that tests both:
- Phase 1: Retrieval quality (MRR, Recall)
- Phase 2: End-to-end memory lifecycle (operations, learning scores, context, expansion)

This provides a complete picture of memory system effectiveness aligned with
the coherence analysis recommendations.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict

from core.config import AgememConfig, DEFAULT_CONFIG
from evaluation.pipeline.phase2_pipeline import (
    Phase2Pipeline,
    Phase2Results,
    MemoryOperationMetrics,
    LearningScoreMetrics,
    ContextAwareRetrievalMetrics,
    QueryExpansionMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class EndToEndResults:
    """Complete evaluation results combining Phase 1 and Phase 2."""
    # Phase 1: Retrieval Quality
    phase1_overall_mrr: float = 0.0
    phase1_overall_recall: float = 0.0
    phase1_behavior_metrics: dict = field(default_factory=dict)

    # Phase 2: Memory Lifecycle
    memory_operations: MemoryOperationMetrics = field(default_factory=MemoryOperationMetrics)
    learning_scores: LearningScoreMetrics = field(default_factory=LearningScoreMetrics)
    context_aware_retrieval: ContextAwareRetrievalMetrics = field(default_factory=ContextAwareRetrievalMetrics)
    query_expansion: QueryExpansionMetrics = field(default_factory=QueryExpansionMetrics)

    # Combined metrics
    coverage_score: float = 0.0  # What % of production features are tested
    representativeness_score: float = 0.0  # How well results predict production behavior

    # Metadata
    session_id: str = ""
    dataset_name: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            'phase1': {
                'overall_mrr': self.phase1_overall_mrr,
                'overall_recall': self.phase1_overall_recall,
                'behavior_metrics': self.phase1_behavior_metrics,
            },
            'phase2': {
                'memory_operations': asdict(self.memory_operations),
                'learning_scores': asdict(self.learning_scores),
                'context_aware_retrieval': asdict(self.context_aware_retrieval),
                'query_expansion': asdict(self.query_expansion),
            },
            'combined': {
                'coverage_score': self.coverage_score,
                'representativeness_score': self.representativeness_score,
            },
            'metadata': {
                'session_id': self.session_id,
                'dataset_name': self.dataset_name,
                'started_at': self.started_at,
                'completed_at': self.completed_at,
                'duration_seconds': self.duration_seconds,
            },
        }


class EndToEndRunner:
    """
    Runs comprehensive Phase 1 + Phase 2 evaluation.

    This addresses the coherence analysis findings by testing:
    1. Memory operation triggers (ADD/UPDATE/DELETE)
    2. Learning score evolution
    3. Context-aware retrieval effectiveness
    4. Query expansion contribution to recall

    The results provide a complete view of memory system behavior.
    """

    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        config: Optional[AgememConfig] = None,
        enable_query_expansion: bool = True,
        enable_context_aware: bool = True,
    ) -> None:
        self._dataset_path = dataset_path
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or DEFAULT_CONFIG
        self._enable_query_expansion = enable_query_expansion
        self._enable_context_aware = enable_context_aware

    def run(
        self,
        num_queries: int = 0,
        seed: int = 42,
    ) -> EndToEndResults:
        """
        Execute complete end-to-end evaluation.

        Args:
            num_queries: Number of queries to test (0 = all)
            seed: Random seed for reproducibility

        Returns:
            EndToEndResults with combined Phase 1 + Phase 2 metrics
        """
        import random
        random.seed(seed)

        session_id = datetime.now().strftime("e2e_%Y%m%d_%H%M%S")
        started_at = datetime.now().isoformat()
        start_time = time.time()

        results = EndToEndResults(
            session_id=session_id,
            dataset_name=self._dataset_path.stem,
            started_at=started_at,
        )

        # ── Phase 1: Retrieval Quality ───────────────────────────────────────
        logger.info("Running Phase 1: Retrieval Quality...")
        phase1_results = self._run_phase1(num_queries, seed)

        results.phase1_overall_mrr = phase1_results.retrieval.mrr_at_10
        results.phase1_overall_recall = phase1_results.retrieval.recall_at_5
        results.phase1_behavior_metrics = {
            behavior: {
                'mrr': m.mrr_at_10,
                'recall': m.recall_at_5,
                'count': m.query_count,
            }
            for behavior, m in (phase1_results.behavior_metrics or {}).items()
        }

        # ── Phase 2: Memory Lifecycle ────────────────────────────────────────
        logger.info("Running Phase 2: Memory Lifecycle...")
        phase2_results = self._run_phase2(num_queries)

        results.memory_operations = phase2_results.memory_operations
        results.learning_scores = phase2_results.learning_scores
        results.context_aware_retrieval = phase2_results.context_aware_retrieval
        results.query_expansion = phase2_results.query_expansion

        # ── Combined Scores ───────────────────────────────────────────────────
        results.coverage_score = self._calculate_coverage_score(results)
        results.representativeness_score = self._calculate_representativeness_score(results)

        # Finalize
        results.completed_at = datetime.now().isoformat()
        results.duration_seconds = time.time() - start_time

        # Save results
        self._save_results(results)

        # Generate report
        self._generate_report(results)

        return results

    def _run_phase1(self, num_queries: int, seed: int):
        """Run Phase 1 retrieval quality evaluation."""
        from evaluation.reproducible_runner import run_reproducible_evaluation

        phase1_output = self._output_dir / "phase1"
        phase1_output.mkdir(parents=True, exist_ok=True)

        results, _ = run_reproducible_evaluation(
            dataset_path=self._dataset_path,
            output_dir=phase1_output,
            num_queries=num_queries,
            top_k=10,
            mode="semantic",
            seed=seed,
        )

        return results

    def _run_phase2(self, num_queries: int) -> Phase2Results:
        """Run Phase 2 memory lifecycle evaluation."""
        from evaluation.pipeline.phase2_pipeline import Phase2Runner

        phase2_output = self._output_dir / "phase2"
        phase2_output.mkdir(parents=True, exist_ok=True)

        runner = Phase2Runner(
            dataset_path=self._dataset_path,
            output_dir=phase2_output,
            config=self._config,
        )

        return runner.run(num_queries=num_queries)

    def _calculate_coverage_score(self, results: EndToEndResults) -> float:
        """
        Calculate what percentage of production features are tested.

        Based on the coherence analysis module coverage matrix:
        - LTMStore.search() = tested
        - Query expansion = tested if improvement > 0
        - Context-aware retrieval = tested if comparison done
        - Memory operations = tested if any operations recorded
        - Learning scores = tested if scores measured
        - STM overflow = not tested (evaluation intentionally focuses on LTM)
        - Skill injection = not tested
        - Orchestrator = not tested directly
        """
        score = 0.0

        # Core retrieval (20%)
        if results.phase1_overall_mrr > 0:
            score += 0.20

        # Query expansion (20%)
        if self._enable_query_expansion and results.query_expansion.baseline_mrr > 0:
            score += 0.20

        # Context-aware retrieval (20%)
        if self._enable_context_aware and results.context_aware_retrieval.baseline_mrr > 0:
            score += 0.20

        # Memory operations (20%)
        if results.memory_operations.total_operations > 0:
            score += 0.20

        # Learning scores (10%)
        if results.learning_scores.scores_measured > 0:
            score += 0.10

        # Behavior segmentation (10%)
        if len(results.phase1_behavior_metrics) >= 5:
            score += 0.10

        return score

    def _calculate_representativeness_score(self, results: EndToEndResults) -> float:
        """
        Calculate how well results predict production behavior.

        Higher scores mean the evaluation better represents what users experience.
        """
        score = 0.0

        # Retrieval quality must be reasonable
        if results.phase1_overall_mrr >= 0.5:
            score += 0.30

        # Context-aware retrieval should improve results (or fallback gracefully)
        if results.context_aware_retrieval.mrr_improvement >= 0:
            score += 0.20

        # Query expansion should help recall
        if results.query_expansion.recall_improvement >= 0:
            score += 0.20

        # Memory operations should be accurate
        if results.memory_operations.correct_rate >= 0.8:
            score += 0.15

        # Learning score promotion should work
        if results.learning_scores.promotion_recall >= 0.8:
            score += 0.15

        return score

    def _save_results(self, results: EndToEndResults) -> None:
        """Save results to JSON."""
        results_path = self._output_dir / "end_to_end_results.json"
        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Saved results to {results_path}")

    def _generate_report(self, results: EndToEndResults) -> None:
        """Generate comprehensive end-to-end report."""
        report_path = self._output_dir / "end_to_end_report.md"

        report = f"""# End-to-End Evaluation Report

**Session ID:** {results.session_id}
**Dataset:** {results.dataset_name}
**Duration:** {results.duration_seconds:.2f}s

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Coverage Score** | {results.coverage_score * 100:.1f}% |
| **Representativeness Score** | {results.representativeness_score * 100:.1f}% |
| Phase 1 MRR@10 | {results.phase1_overall_mrr:.4f} |
| Phase 1 Recall@5 | {results.phase1_overall_recall:.4f} |

Coverage measures what percentage of production features are tested.
Representativeness measures how well results predict production behavior.

---

## Phase 1: Retrieval Quality

### Overall Metrics

| Metric | Value |
|--------|-------|
| MRR@10 | {results.phase1_overall_mrr:.4f} |
| Recall@5 | {results.phase1_overall_recall:.4f} |

### Behavior-Segmented Results

| Behavior | MRR@10 | Recall@5 |
|----------|--------|----------|
"""
        for behavior, metrics in results.phase1_behavior_metrics.items():
            report += f"| {behavior} | {metrics['mrr']:.4f} | {metrics['recall']:.4f} |\n"

        report += f"""
---

## Phase 2: Memory Lifecycle

### 2.1 Memory Operation Triggers

Tests ADD/UPDATE/DELETE operations triggered by conversation context.

| Operation | Total | Correct | Precision | Recall |
|-----------|-------|---------|-----------|--------|
| ADD | {results.memory_operations.add_operations_total} | {results.memory_operations.add_operations_correct} | {results.memory_operations.add_precision:.4f} | {results.memory_operations.add_recall:.4f} |
| UPDATE | {results.memory_operations.update_operations_total} | {results.memory_operations.update_operations_correct} | {results.memory_operations.update_precision:.4f} | {results.memory_operations.update_recall:.4f} |
| DELETE | {results.memory_operations.delete_operations_total} | {results.memory_operations.delete_operations_correct} | {results.memory_operations.delete_precision:.4f} | {results.memory_operations.delete_recall:.4f} |

**Overall Correct Rate:** {results.memory_operations.correct_rate:.4f}

### 2.2 Learning Score Evolution

Tests how learning scores evolve and drive LTM promotion.

| Metric | Value |
|--------|-------|
| Scores Measured | {results.learning_scores.scores_measured} |
| Average Score | {results.learning_scores.avg_score:.4f} |
| Score Range | [{results.learning_scores.min_score:.4f}, {results.learning_scores.max_score:.4f}] |
| Promotion Recall | {results.learning_scores.promotion_recall:.4f} |

### 2.3 Context-Aware Retrieval Effectiveness

Compares context-aware retrieval vs baseline query-only search.

| Metric | Baseline | Context-Aware | Improvement |
|--------|----------|---------------|-------------|
| MRR@10 | {results.context_aware_retrieval.baseline_mrr:.4f} | {results.context_aware_retrieval.context_aware_mrr:.4f} | {results.context_aware_retrieval.mrr_improvement * 100:.2f}% |
| Recall@10 | {results.context_aware_retrieval.baseline_recall:.4f} | {results.context_aware_retrieval.context_aware_recall:.4f} | {results.context_aware_retrieval.recall_improvement * 100:.2f}% |

**Fallback Rate:** {results.context_aware_retrieval.fallback_rate:.4f}

#### Per-Behavior Context-Aware Improvement

| Behavior | Baseline MRR | Context MRR | Improvement |
|----------|--------------|-------------|-------------|
"""
        for behavior, metrics in results.context_aware_retrieval.behavior_improvements.items():
            report += f"| {behavior} | {metrics['baseline_mrr']:.4f} | {metrics['context_aware_mrr']:.4f} | {metrics['mrr_improvement'] * 100:.2f}% |\n"

        report += f"""
### 2.4 Query Expansion Contribution

Measures how query expansion variants improve recall.

| Metric | Baseline | Expanded | Improvement |
|--------|----------|----------|-------------|
| MRR@10 | {results.query_expansion.baseline_mrr:.4f} | {results.query_expansion.expanded_mrr:.4f} | {results.query_expansion.mrr_improvement * 100:.2f}% |
| Recall@10 | {results.query_expansion.baseline_recall:.4f} | {results.query_expansion.expanded_recall:.4f} | {results.query_expansion.recall_improvement * 100:.2f}% |

**Variant Statistics:**
- Avg Variants per Query: {results.query_expansion.avg_variants_per_query:.2f}
- Variant Hit Rate: {results.query_expansion.variant_hit_rate:.4f}

---

## Coherence Analysis Summary

This evaluation addresses the coherence analysis findings by testing:

| Gap | Resolution | Status |
|-----|------------|--------|
| Query Expansion Bypass | Tested via Phase 2.4 | {'✅' if results.query_expansion.baseline_mrr > 0 else '⚠️'} |
| Context-Aware Retrieval Bypass | Tested via Phase 2.3 | {'✅' if results.context_aware_retrieval.baseline_mrr > 0 else '⚠️'} |
| Memory Operations Not Tested | Tested via Phase 2.1 | {'✅' if results.memory_operations.total_operations > 0 else '⚠️'} |
| Learning Score Dynamics | Tested via Phase 2.2 | {'✅' if results.learning_scores.scores_measured > 0 else '⚠️'} |

---

## Recommendations

Based on these results:

"""
        if results.coverage_score < 0.6:
            report += "1. **Low Coverage**: Consider enabling additional Phase 2 tests to improve evaluation comprehensiveness.\n"
        if results.context_aware_retrieval.mrr_improvement < 0:
            report += "2. **Context-Aware Regression**: Context-aware retrieval is underperforming. Review context window weights.\n"
        if results.query_expansion.recall_improvement < 0:
            report += "3. **Query Expansion Regression**: Expansion is not improving recall. Review variant generation.\n"
        if results.memory_operations.correct_rate < 0.8:
            report += "4. **Memory Operation Accuracy**: Operation triggers need tuning.\n"

        if results.coverage_score >= 0.8 and results.representativeness_score >= 0.7:
            report += "**Good Coverage and Representativeness**: Results should predict production behavior well.\n"

        report += """
---

*Report generated by EndToEndRunner*
"""

        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Generated report at {report_path}")


def main():
    """CLI entry point for end-to-end evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run end-to-end Phase 1 + Phase 2 evaluation",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/data/longmemeval_s_cleaned.json"),
        help="Path to dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Output directory",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=0,
        help="Number of queries (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--disable-query-expansion",
        action="store_true",
        help="Disable query expansion testing",
    )
    parser.add_argument(
        "--disable-context-aware",
        action="store_true",
        help="Disable context-aware retrieval testing",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.dataset.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        return 1

    runner = EndToEndRunner(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        enable_query_expansion=not args.disable_query_expansion,
        enable_context_aware=not args.disable_context_aware,
    )

    results = runner.run(
        num_queries=args.queries,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("END-TO-END EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Session: {results.session_id}")
    print(f"Dataset: {results.dataset_name}")
    print(f"Duration: {results.duration_seconds:.2f}s")
    print("-" * 70)
    print(f"Coverage Score: {results.coverage_score * 100:.1f}%")
    print(f"Representativeness Score: {results.representativeness_score * 100:.1f}%")
    print("-" * 70)
    print("Phase 1 (Retrieval Quality):")
    print(f"  MRR@10: {results.phase1_overall_mrr:.4f}")
    print(f"  Recall@5: {results.phase1_overall_recall:.4f}")
    print("-" * 70)
    print("Phase 2 (Memory Lifecycle):")
    print(f"  Memory Ops Correct Rate: {results.memory_operations.correct_rate:.4f}")
    print(f"  Context-Aware MRR Improvement: {results.context_aware_retrieval.mrr_improvement * 100:.2f}%")
    print(f"  Query Expansion Recall Improvement: {results.query_expansion.recall_improvement * 100:.2f}%")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())