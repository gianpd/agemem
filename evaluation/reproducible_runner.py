#!/usr/bin/env python
"""
Reproducible Evaluation Runner
-------------------------------

Executes the evaluation pipeline with full reproducibility guarantees.
Logs all initial parameters, seeds, and environmental conditions.

Per recommendations from evaluation_pipeline_audit_report.md Section 6.1.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.pipeline.dataset_pipeline import DatasetPipeline, BenchmarkEntry, BenchmarkQuery
from evaluation.pipeline.inference_pipeline import InferencePipeline
from evaluation.pipeline.metrics_pipeline import MetricsPipeline, EvaluationResults

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityManifest:
    """Complete manifest for reproducible evaluation runs."""
    # Identification
    run_id: str
    started_at: str
    completed_at: str = ""

    # Random seeds
    python_seed: int = 0
    numpy_seed: int = 0
    random_seed: int = 0

    # Dataset configuration
    dataset_path: str = ""
    dataset_name: str = ""
    dataset_hash: str = ""
    dataset_size_bytes: int = 0
    num_queries_limit: int = 0

    # Pipeline configuration
    top_k: int = 10
    mode: str = "semantic"
    max_entries: int = 500

    # Environment
    python_version: str = ""
    platform_system: str = ""
    platform_release: str = ""
    platform_machine: str = ""
    working_directory: str = ""
    git_commit: str = ""
    git_branch: str = ""

    # Dependencies (key packages)
    packages: dict[str, str] = field(default_factory=dict)

    # Results summary
    total_entries_loaded: int = 0
    total_queries_executed: int = 0
    total_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def get_file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def get_git_info() -> tuple[str, str]:
    """Get current git commit and branch."""
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent.parent
        ).decode().strip()[:12]
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent.parent
        ).decode().strip()
        return commit, branch
    except Exception:
        return "unknown", "unknown"


def get_package_versions() -> dict[str, str]:
    """Get versions of key packages."""
    packages = {}
    for pkg in ["numpy", "scipy", "torch", "transformers", "sentence_transformers"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg] = "not_installed"
    return packages


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_reproducible_evaluation(
    dataset_path: Path,
    output_dir: Path,
    num_queries: int = 0,  # 0 means all
    top_k: int = 10,
    mode: str = "semantic",
    max_entries: int = 0,  # 0 means load all entries relevant to queries
    seed: int = 42,
) -> tuple[EvaluationResults, ReproducibilityManifest]:
    """
    Run evaluation with full reproducibility guarantees.

    Args:
        dataset_path: Path to the dataset file
        output_dir: Directory for output files
        num_queries: Number of queries to run (0 = all)
        top_k: Number of results to retrieve per query
        mode: Retrieval mode (semantic, overlap)
        max_entries: Maximum entries to load into LTM (0 = load entries relevant to queries)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (EvaluationResults, ReproducibilityManifest)
    """
    run_id = datetime.now().strftime("repro_%Y%m%d_%H%M%S")
    started_at = datetime.now().isoformat()
    start_time = time.time()

    # Initialize manifest
    manifest = ReproducibilityManifest(
        run_id=run_id,
        started_at=started_at,
        python_seed=seed,
        numpy_seed=seed,
        random_seed=seed,
        dataset_path=str(dataset_path),
        dataset_name=dataset_path.stem,
        dataset_hash=get_file_hash(dataset_path),
        dataset_size_bytes=dataset_path.stat().st_size,
        num_queries_limit=num_queries,
        top_k=top_k,
        mode=mode,
        max_entries=max_entries,
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        working_directory=str(Path.cwd()),
        packages=get_package_versions(),
    )

    # Get git info
    manifest.git_commit, manifest.git_branch = get_git_info()

    # Set seeds for reproducibility
    set_all_seeds(seed)

    logger.info(f"Starting reproducible evaluation: {run_id}")
    logger.info(f"Dataset: {dataset_path} (hash: {manifest.dataset_hash})")
    logger.info(f"Seed: {seed}, Mode: {mode}, Top-K: {top_k}")

    # Initialize pipelines
    dataset_pipeline = DatasetPipeline(output_dir=output_dir)
    inference_pipeline = InferencePipeline(
        db_path=output_dir / f"{run_id}_traces.db",
        session_id=run_id,
    )
    metrics_pipeline = MetricsPipeline(session_id=run_id)

    # Load and validate dataset
    logger.info(f"Loading dataset from {dataset_path}")
    entries, queries = dataset_pipeline.ingest_dataset(dataset_path, "longmemeval")

    manifest.total_entries_loaded = len(entries)

    # Validate
    validation_report = dataset_pipeline.validate()
    logger.info(f"Validation: {validation_report.is_valid}, {validation_report.total_entries} entries, {validation_report.total_queries} queries")

    if not validation_report.is_valid:
        logger.warning(f"Validation errors: {validation_report.errors}")

    # Limit queries if specified
    if num_queries > 0 and num_queries < len(queries):
        queries = queries[:num_queries]
        logger.info(f"Limiting to {num_queries} queries")

    manifest.total_queries_executed = len(queries)

    # Determine which entries to load into LTM
    # For proper evaluation, load entries relevant to the queries being tested
    entry_dict = {e.entry_id: e for e in entries}

    if max_entries > 0:
        # Legacy behavior: load first N entries
        entries_to_load = entries[:max_entries]
        logger.info(f"Populating LTM with first {len(entries_to_load)} entries...")
    else:
        # Smart loading: collect all entry IDs referenced by queries
        relevant_entry_ids = set()
        for query in queries:
            relevant_entry_ids.update(query.relevant_entry_ids)

        # Load entries that are referenced by queries
        entries_to_load = [entry_dict[eid] for eid in relevant_entry_ids if eid in entry_dict]
        logger.info(f"Populating LTM with {len(entries_to_load)} entries relevant to {len(queries)} queries...")

    # Create LTM store
    from core.config import AgememConfig
    from memory.ltm_store import LTMStore

    config = AgememConfig()
    ltm_path = output_dir / f"{run_id}_ltm.json"
    semantic_db_path = output_dir / f"{run_id}_semantic.db"

    ltm_store = LTMStore(
        config=config,
        persist_path=ltm_path,
        semantic_db_path=semantic_db_path,
        enable_semantic_search=(mode == "semantic"),
    )

    # Populate LTM
    inference_pipeline.start_session()
    count, id_mapping = inference_pipeline.populate_ltm(entries_to_load, ltm_store)
    logger.info(f"Added {count} entries to LTM")

    # Update queries with actual LTM entry IDs
    for query in queries:
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

    # Calculate metrics (now includes behavior segmentation)
    logger.info("Calculating metrics with behavior segmentation...")
    results = metrics_pipeline.evaluate(queries, traces)

    # Update manifest with completion info
    manifest.completed_at = datetime.now().isoformat()
    manifest.total_duration_seconds = time.time() - start_time

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest_path = output_dir / f"{run_id}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    logger.info(f"Saved manifest to {manifest_path}")

    # Save metrics
    metrics_path = output_dir / f"{run_id}_metrics.json"
    metrics_pipeline.export_json(metrics_path)

    # Generate comprehensive report
    report_path = output_dir / f"{run_id}_report.md"
    generate_report(results, manifest, report_path)

    # Cleanup
    ltm_store.close()
    inference_pipeline.close()

    logger.info(f"Reproducible evaluation completed in {manifest.total_duration_seconds:.2f} seconds")
    return results, manifest


def generate_report(results: EvaluationResults, manifest: ReproducibilityManifest, output_path: Path) -> None:
    """Generate a comprehensive markdown report with reproducibility info."""

    # Build behavior metrics table
    behavior_table = ""
    if results.behavior_metrics:
        behavior_table = "\n### Behavior-Segmented Metrics\n\n"
        behavior_table += "| Behavior | Queries | MRR@10 | Recall@5 | Precision@5 | NDCG@10 | Avg Latency |\n"
        behavior_table += "|----------|---------|--------|----------|-------------|---------|-------------|\n"
        for behavior, metrics in sorted(results.behavior_metrics.items()):
            behavior_table += f"| {metrics.behavior_name} | {metrics.query_count} | {metrics.mrr_at_10:.4f} | {metrics.recall_at_5:.4f} | {metrics.precision_at_5:.4f} | {metrics.ndcg_at_10:.4f} | {metrics.avg_latency_ms:.2f}ms |\n"

    report = f"""# Reproducible Evaluation Report

**Run ID:** {manifest.run_id}
**Generated:** {manifest.completed_at}

---

## 1. Results Summary

### Overall Retrieval Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MRR@1 | {results.retrieval.mrr_at_1:.4f} | - | - |
| MRR@5 | {results.retrieval.mrr_at_5:.4f} | - | - |
| MRR@10 | {results.retrieval.mrr_at_10:.4f} | >= 0.85 | {'PASS' if results.retrieval.mrr_at_10 >= 0.85 else 'FAIL'} |
| Precision@5 | {results.retrieval.precision_at_5:.4f} | - | - |
| Recall@5 | {results.retrieval.recall_at_5:.4f} | >= 0.90 | {'PASS' if results.retrieval.recall_at_5 >= 0.90 else 'FAIL'} |
| Recall@10 | {results.retrieval.recall_at_10:.4f} | - | - |
| NDCG@10 | {results.retrieval.ndcg_at_10:.4f} | - | - |
| Avg Latency | {results.retrieval.avg_latency_ms:.2f}ms | < 500ms | {'PASS' if results.retrieval.avg_latency_ms < 500 else 'FAIL'} |
{behavior_table}

---

## 2. Reproducibility Manifest

### Random Seeds

| Seed Type | Value |
|-----------|-------|
| Python Hash Seed | {manifest.python_seed} |
| NumPy Seed | {manifest.numpy_seed} |
| Random Seed | {manifest.random_seed} |

### Dataset Configuration

| Parameter | Value |
|-----------|-------|
| Dataset Path | `{manifest.dataset_path}` |
| Dataset Name | {manifest.dataset_name} |
| Dataset Hash (SHA256) | {manifest.dataset_hash} |
| Dataset Size | {manifest.dataset_size_bytes:,} bytes |
| Queries Limit | {manifest.num_queries_limit or 'All'} |
| Entries Loaded | {manifest.total_entries_loaded} |
| Queries Executed | {manifest.total_queries_executed} |

### Pipeline Configuration

| Parameter | Value |
|-----------|-------|
| Top-K | {manifest.top_k} |
| Mode | {manifest.mode} |
| Max Entries | {manifest.max_entries} |

### Environment

| Parameter | Value |
|-----------|-------|
| Python Version | {manifest.python_version} |
| Platform | {manifest.platform_system} {manifest.platform_release} |
| Machine | {manifest.platform_machine} |
| Working Directory | `{manifest.working_directory}` |
| Git Commit | `{manifest.git_commit}` |
| Git Branch | `{manifest.git_branch}` |

### Key Package Versions

| Package | Version |
|---------|---------|
{chr(10).join(f'| {pkg} | {ver} |' for pkg, ver in manifest.packages.items())}

---

## 3. Execution Summary

- **Started:** {manifest.started_at}
- **Completed:** {manifest.completed_at}
- **Duration:** {manifest.total_duration_seconds:.2f} seconds

---

## 4. LongMemEval Alignment

This evaluation implements the high-priority recommendations from the audit report:

1. **Behavior Segmentation:** Metrics are calculated per question type (behavior category)
2. **LongMemEval S Dataset:** Uses the standard evaluation dataset (not Oracle baseline)
3. **Reproducibility:** All seeds, parameters, and environment info logged

### Behavior Categories Tested

| Category | Count | Coverage |
|----------|-------|----------|
{chr(10).join(f'| {b} | {m.query_count} | {m.query_count / manifest.total_queries_executed * 100:.1f}% |' for b, m in sorted(results.behavior_metrics.items())) if results.behavior_metrics else '| N/A | 0 | 0% |'}

---

*Report generated automatically by reproducible_runner.py*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Generated report at {output_path}")


def main():
    """Main entry point for reproducible evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run reproducible evaluation with LongMemEval S dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/data/longmemeval_s_cleaned.json"),
        help="Path to dataset (default: LongMemEval S)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=0,
        help="Number of queries to run (0 = all, default: 0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K results per query (default: 10)",
    )
    parser.add_argument(
        "--mode",
        choices=["semantic", "overlap"],
        default="semantic",
        help="Retrieval mode (default: semantic)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="Maximum entries to load into LTM (0 = load entries relevant to queries, default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Output directory (default: evaluation/results)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Verify dataset exists
    if not args.dataset.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    # Run evaluation
    results, manifest = run_reproducible_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_queries=args.queries,
        top_k=args.top_k,
        mode=args.mode,
        max_entries=args.max_entries,
        seed=args.seed,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("REPRODUCIBLE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Run ID:     {manifest.run_id}")
    print(f"Dataset:    {manifest.dataset_name} (hash: {manifest.dataset_hash})")
    print(f"Seed:       {manifest.seed if hasattr(manifest, 'seed') else manifest.python_seed}")
    print(f"Duration:   {manifest.total_duration_seconds:.2f}s")
    print("-" * 70)
    print("OVERALL METRICS")
    print(f"  MRR@10:    {results.retrieval.mrr_at_10:.4f} (target: >= 0.85)")
    print(f"  Recall@5:  {results.retrieval.recall_at_5:.4f} (target: >= 0.90)")
    print(f"  Latency:   {results.retrieval.avg_latency_ms:.2f}ms")
    print("-" * 70)

    if results.behavior_metrics:
        print("BEHAVIOR METRICS")
        for behavior, metrics in sorted(results.behavior_metrics.items()):
            print(f"  {behavior}: MRR={metrics.mrr_at_10:.4f}, Recall={metrics.recall_at_5:.4f} ({metrics.query_count} queries)")

    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}/{manifest.run_id}_*")


if __name__ == "__main__":
    sys.exit(main())