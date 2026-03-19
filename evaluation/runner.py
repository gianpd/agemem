#!/usr/bin/env python
"""
AgeMem Evaluation Runner
------------------------

Main entry point for running the AgeMem evaluation pipeline.

Usage:
    python -m evaluation.runner --dataset longmemeval --queries 100

Per Phase 1-4 protocols from Section 7.5 of the technical specification.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.pipeline.dataset_pipeline import DatasetPipeline, BenchmarkEntry, BenchmarkQuery
from evaluation.pipeline.inference_pipeline import InferencePipeline
from evaluation.pipeline.metrics_pipeline import MetricsPipeline, EvaluationResults
from evaluation.pipeline.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_sample_dataset(
    output_path: Path,
    num_entries: int = 500,
    num_queries: int = 100,
) -> None:
    """
    Create a sample dataset for testing the evaluation pipeline.

    This generates synthetic data that mimics the structure of LongMemEval.
    The queries are designed to have clear matches with entries.
    """
    import hashlib
    import random

    entries = []
    queries = []

    # Generate entries with distinctive content that can be queried
    topics = [
        "python programming",
        "machine learning",
        "database design",
        "cloud infrastructure",
        "api development",
        "security practices",
        "testing strategies",
        "performance optimization",
        "user preferences",
        "project requirements",
    ]

    entry_ids = []

    for i in range(num_entries):
        topic = topics[i % len(topics)]
        detail_id = random.randint(1000, 9999)
        content = f"Memory about {topic}. Key detail: item_{detail_id}. " \
                  f"This is important information about {topic} that should be remembered. " \
                  f"Additional context: Entry {i} from the evaluation dataset."

        entry_id = hashlib.sha1(content.encode()).hexdigest()[:12]
        entry_ids.append(entry_id)

        entry = {
            "content": content,
            "entry_id": entry_id,
            "created_at": time.time() - random.randint(0, 86400 * 30),  # Last 30 days
            "learning_score": random.uniform(0.5, 1.0),
            "tags": [topic.replace(" ", "_")],
            "entities": {"topic": [topic], "detail": [f"item_{detail_id}"]},
        }
        entries.append(entry)

    # Generate queries that match entries by content keywords
    for i in range(num_queries):
        # Pick a random entry as the target
        target_idx = random.randint(0, len(entries) - 1)
        target_entry = entries[target_idx]
        topic = target_entry["tags"][0].replace("_", " ")

        # Create a query using the topic and some content from the entry
        query_text = f"What do I know about {topic}?"

        queries.append({
            "query_id": f"q_{i}",
            "query_text": query_text,
            "relevant_entry_ids": [target_entry["entry_id"]],
            "query_type": "retrieval",
        })

    # Save dataset
    data = {
        "memories": entries,
        "questions": queries,
        "metadata": {
            "name": "sample_dataset",
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "entry_count": len(entries),
            "query_count": len(queries),
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Created sample dataset at {output_path}")


def run_phase1_evaluation(
    dataset_path: Path,
    output_dir: Path,
    num_queries: int = 1000,
    top_k: int = 10,
    mode: str = "semantic",
) -> EvaluationResults:
    """
    Run Phase 1: Retrieval Quality evaluation.

    Per Section 7.5 Phase 1 protocol:
    1. Populate LTM with 500 curated memories
    2. Execute 1000 queries across benchmark test sets
    3. Measure MRR@K, Precision@K, Recall@K at K=1,5,10
    4. Compare semantic vs overlap retrieval modes
    5. Log variant hit-rate for query expansion
    """
    logger.info("Starting Phase 1: Retrieval Quality Evaluation")

    # Initialize pipelines
    dataset_pipeline = DatasetPipeline(output_dir=output_dir)
    inference_pipeline = InferencePipeline(
        db_path=output_dir / "traces.db",
        session_id=f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    metrics_pipeline = MetricsPipeline(session_id=inference_pipeline._session_id)

    # Load and validate dataset
    logger.info(f"Loading dataset from {dataset_path}")
    entries, queries = dataset_pipeline.ingest_dataset(dataset_path, "longmemeval")

    # Validate
    validation_report = dataset_pipeline.validate()
    logger.info(f"Validation: {validation_report.is_valid}, {validation_report.total_entries} entries, {validation_report.total_queries} queries")

    if not validation_report.is_valid:
        logger.warning(f"Validation errors: {validation_report.errors}")

    # Limit queries if specified
    if num_queries and num_queries < len(queries):
        queries = queries[:num_queries]
        logger.info(f"Limiting to {num_queries} queries")

    # Create mock LTM store for testing
    # In production, this would use the actual LTMStore
    from core.config import AgememConfig
    from memory.ltm_store import LTMStore

    config = AgememConfig()

    # Create temporary LTM store
    ltm_path = output_dir / "eval_ltm.json"
    semantic_db_path = output_dir / "eval_semantic.db"

    ltm_store = LTMStore(
        config=config,
        persist_path=ltm_path,
        semantic_db_path=semantic_db_path,
        enable_semantic_search=(mode == "semantic"),
    )

    # Populate LTM with entries
    logger.info(f"Populating LTM with {len(entries)} entries...")
    inference_pipeline.start_session()
    count, id_mapping = inference_pipeline.populate_ltm(entries[:500], ltm_store)
    logger.info(f"Added {count} entries to LTM")

    # Update queries with actual LTM entry IDs
    # The benchmark queries have entry_ids from the dataset, but LTM may have
    # assigned different entry_ids due to deduplication
    for query in queries:
        # Translate benchmark entry_ids to actual LTM entry_ids
        query.relevant_entry_ids = [
            id_mapping.get(bid, bid) for bid in query.relevant_entry_ids
            if bid in id_mapping
        ]

    # Execute queries
    logger.info(f"Executing {len(queries)} queries...")
    traces = inference_pipeline.execute_queries(queries, ltm_store, top_k=top_k, mode=mode)

    # End session
    session_stats = inference_pipeline.end_session()
    logger.info(f"Session completed: {session_stats.total_queries} queries, avg latency {session_stats.avg_latency_ms:.2f}ms")

    # Calculate metrics
    logger.info("Calculating metrics...")
    results = metrics_pipeline.evaluate(queries, traces)

    # Export results
    metrics_pipeline.export_json(output_dir / "metrics.json")

    # Cleanup
    ltm_store.close()
    inference_pipeline.close()

    logger.info("Phase 1 evaluation complete")
    return results


def run_full_evaluation(
    dataset_path: Path,
    output_dir: Path,
    phases: list[str] = None,
) -> dict[str, EvaluationResults]:
    """
    Run full evaluation across all phases.

    Phases:
    - phase1: Retrieval Quality (automated)
    - phase2: Memory Persistence (multi-session)
    - phase3: Response Quality (human evaluation)
    - phase4: Comparative Benchmarking
    """
    phases = phases or ["phase1"]
    results = {}

    if "phase1" in phases:
        results["phase1"] = run_phase1_evaluation(
            dataset_path=dataset_path,
            output_dir=output_dir,
        )

    # Phase 2-4 would require additional implementation
    # Phase 2: Multi-session conversations with LTM persistence
    # Phase 3: Human evaluation of responses
    # Phase 4: Run against competitor systems

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AgeMem Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run Phase 1 evaluation with sample dataset
    python -m evaluation.runner --sample --queries 100

    # Run with existing dataset
    python -m evaluation.runner --dataset path/to/dataset.json --queries 1000

    # Generate only sample dataset
    python -m evaluation.runner --create-sample --entries 500 --queries 100
        """
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to dataset file (JSON, CSV, or Parquet)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample dataset for testing",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample dataset and exit",
    )
    parser.add_argument(
        "--entries",
        type=int,
        default=500,
        help="Number of entries for sample dataset (default: 500)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of queries to execute (default: 100)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--mode",
        choices=["semantic", "overlap"],
        default="semantic",
        help="Retrieval mode (default: semantic)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Output directory for results (default: evaluation/results)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["phase1", "phase2", "phase3", "phase4"],
        default=["phase1"],
        help="Evaluation phases to run (default: phase1)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Create sample dataset if requested
    if args.create_sample:
        sample_path = args.output_dir / "sample_dataset.json"
        create_sample_dataset(sample_path, args.entries, args.queries)
        print(f"Sample dataset created at: {sample_path}")
        return 0

    # Determine dataset path
    if args.sample:
        sample_path = args.output_dir / "sample_dataset.json"
        if not sample_path.exists():
            logger.info("Creating sample dataset...")
            create_sample_dataset(sample_path, args.entries, args.queries)
        dataset_path = sample_path
    elif args.dataset:
        dataset_path = args.dataset
    else:
        # Default to sample dataset
        sample_path = args.output_dir / "sample_dataset.json"
        if not sample_path.exists():
            logger.info("Creating sample dataset...")
            create_sample_dataset(sample_path, args.entries, args.queries)
        dataset_path = sample_path

    # Run evaluation
    logger.info(f"Starting evaluation with dataset: {dataset_path}")
    start_time = time.time()

    results = run_full_evaluation(
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        phases=args.phases,
    )

    # Generate reports
    report_generator = ReportGenerator(output_dir=args.output_dir)

    for phase_name, phase_results in results.items():
        paths = report_generator.save(
            phase_results,
            base_name=f"{phase_name}_report",
        )
        logger.info(f"Reports generated for {phase_name}:")
        for fmt, path in paths.items():
            logger.info(f"  {fmt}: {path}")

    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.2f} seconds")

    # Print summary
    if "phase1" in results:
        r = results["phase1"].retrieval
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  MRR@10:    {r.mrr_at_10:.4f} (target: >= 0.85)")
        print(f"  Recall@5:  {r.recall_at_5:.4f} (target: >= 0.90)")
        print(f"  Latency:   {r.avg_latency_ms:.2f}ms")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())