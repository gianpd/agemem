#!/usr/bin/env python3
"""
Resume HotpotQA evaluation from the last successful checkpoint.

Parses evaluation logs to find which samples were successfully evaluated,
then continues from where the run left off.

Usage:
    python evaluation/resume_hotpotqa.py --log evaluation/logs/hotpotqa_20260326_120905.log

The script will:
1. Parse the log to extract successfully evaluated sample IDs
2. Load the HotpotQA dataset
3. Filter out already-evaluated samples
4. Run evaluation on remaining samples
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.run_hotpotqa import (
    HotpotEvaluator,
    HotpotSample,
    RunConfig,
    EvalResult,
    EvalMode,
    ParseStatus,
    load_hotpotqa_hf,
    load_hotpotqa_json,
    _print_summary,
    _save_results,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

@dataclass
class EvaluatedSample:
    """Tracks a sample that was evaluated in a previous run."""
    sample_id: str
    j_score: float
    status: str  # "ok", "parse_failure", "exception"
    latency_ms: float
    index: int  # Position in the run (e.g., 69 from [69/7405])


@dataclass
class LogParseResult:
    """Result of parsing an evaluation log."""
    evaluated: list[EvaluatedSample]
    successful_ids: set[str]
    failed_ids: set[str]
    last_index: int
    total_samples: int


def parse_eval_log(log_path: Path) -> LogParseResult:
    """
    Parse an evaluation log to extract evaluated samples.

    The log format is:
        INFO [69/7405] id=5ac2acff55429921a00ab02b type=bridge level=hard
        ... evaluation happens ...
        INFO   j_score=1.00 status=ok latency=53387ms error=None

    We match sample ID lines with their subsequent j_score lines.
    """
    evaluated: list[EvaluatedSample] = []
    successful_ids: set[str] = set()
    failed_ids: set[str] = set()
    last_index = 0
    total_samples = 0

    # Patterns
    sample_pattern = re.compile(
        r"INFO\s+\[(\d+)/(\d+)\]\s+id=([a-f0-9]+)\s+type=\w+\s+level=\w+"
    )
    score_pattern = re.compile(
        r"j_score=([0-9.]+)\s+status=(\w+)\s+latency=(\d+)ms"
    )

    current_sample: Optional[dict] = None

    with open(log_path, "r") as f:
        for line in f:
            # Check for sample start
            sample_match = sample_pattern.search(line)
            if sample_match:
                idx = int(sample_match.group(1))
                total = int(sample_match.group(2))
                sid = sample_match.group(3)

                last_index = idx
                total_samples = total
                current_sample = {
                    "sample_id": sid,
                    "index": idx,
                }
                continue

            # Check for j_score (completes the current sample)
            score_match = score_pattern.search(line)
            if score_match and current_sample:
                j_score = float(score_match.group(1))
                status = score_match.group(2)
                latency = int(score_match.group(3))

                eval_sample = EvaluatedSample(
                    sample_id=current_sample["sample_id"],
                    j_score=j_score,
                    status=status,
                    latency_ms=float(latency),
                    index=current_sample["index"],
                )
                evaluated.append(eval_sample)

                if status == "ok":
                    successful_ids.add(current_sample["sample_id"])
                else:
                    failed_ids.add(current_sample["sample_id"])

                current_sample = None

    return LogParseResult(
        evaluated=evaluated,
        successful_ids=successful_ids,
        failed_ids=failed_ids,
        last_index=last_index,
        total_samples=total_samples,
    )


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

@dataclass
class ResumeConfig:
    """Configuration for resuming evaluation."""
    log_path: Path
    output_path: Path
    split: str = "validation"
    setting: str = "distractor"
    mode: str = "ltm"
    corpus_dir: Optional[Path] = None
    persist_dir: Optional[Path] = None
    judge_model: Optional[str] = None
    hard_fail: bool = False
    skip_ingest: bool = False
    retry_failed: bool = False  # If True, also retry parse_failure samples
    limit: int = 0


def build_resume_config(args: argparse.Namespace) -> ResumeConfig:
    """Build ResumeConfig from CLI args."""
    import tempfile

    persist_dir = args.persist_dir or Path(tempfile.mkdtemp(prefix="hotpot_eval_resume_"))
    corpus_dir = args.corpus_dir if args.mode == "corpus" else None

    return ResumeConfig(
        log_path=args.log,
        output_path=args.output,
        split=args.split,
        setting=args.setting,
        mode=args.mode,
        corpus_dir=corpus_dir,
        persist_dir=persist_dir,
        judge_model=args.judge_model,
        hard_fail=args.hard_fail,
        skip_ingest=args.skip_ingest,
        retry_failed=args.retry_failed,
        limit=args.limit,
    )


def filter_samples(
    samples: list[HotpotSample],
    parse_result: LogParseResult,
    retry_failed: bool = False,
) -> list[HotpotSample]:
    """
    Filter out already-evaluated samples.

    If retry_failed is True, include samples that had parse_failure status
    (they will be re-evaluated).
    """
    if retry_failed:
        # Only skip successful ones
        skip_ids = parse_result.successful_ids
        logger.info(
            "Skipping %d successful samples, retrying %d failed samples",
            len(parse_result.successful_ids),
            len(parse_result.failed_ids),
        )
    else:
        # Skip all previously attempted samples
        skip_ids = parse_result.successful_ids | parse_result.failed_ids
        logger.info(
            "Skipping %d previously evaluated samples",
            len(skip_ids),
        )

    remaining = [s for s in samples if s.id not in skip_ids]
    logger.info("Remaining samples to evaluate: %d", len(remaining))
    return remaining


def merge_results(
    previous: list[EvaluatedSample],
    new_results: list[EvalResult],
) -> list[dict]:
    """Merge previous log results with new evaluation results."""
    merged = []

    # Convert previous to dict format
    for p in previous:
        if p.status == "ok":
            merged.append({
                "sample_id": p.sample_id,
                "j_score": p.j_score,
                "parse_status": "ok",
                "latency_ms": p.latency_ms,
                "error": None,
                "source": "previous_run",
            })

    # Add new results
    for r in new_results:
        merged.append({
            "sample_id": r.sample_id,
            "j_score": r.j_score,
            "parse_status": r.judge.status.value,
            "latency_ms": r.latency_ms,
            "error": r.error,
            "source": "current_run",
        })

    return merged


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Resume HotpotQA evaluation from checkpoint"
    )
    parser.add_argument(
        "--log", type=Path, required=True,
        help="Path to evaluation log file to parse for checkpoint",
    )
    parser.add_argument(
        "--split", default="validation",
        help="Dataset split",
    )
    parser.add_argument(
        "--setting", default="distractor",
        choices=["distractor", "fullwiki"],
    )
    parser.add_argument(
        "--mode", default="corpus",
        choices=["ltm", "corpus"],
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=None,
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Local JSON dataset (uses HF if not specified)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation/results/hotpot_resume.json"),
    )
    parser.add_argument(
        "--persist-dir", type=Path, default=None,
    )
    parser.add_argument(
        "--judge-model", type=str, default=None,
    )
    parser.add_argument(
        "--hard-fail", action="store_true",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry samples that had parse_failure status",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of samples (for testing). Use -1 for dry-run (parse only, no evaluation).",
    )
    args = parser.parse_args()

    # Parse the log
    logger.info("Parsing log file: %s", args.log)
    parse_result = parse_eval_log(args.log)

    print(f"\n{'='*60}")
    print("CHECKPOINT SUMMARY")
    print(f"{'='*60}")
    print(f"Log file              : {args.log}")
    print(f"Samples evaluated     : {len(parse_result.evaluated)}")
    print(f"  - Successful (ok)   : {len(parse_result.successful_ids)}")
    print(f"  - Failed            : {len(parse_result.failed_ids)}")
    print(f"Last sample index     : {parse_result.last_index}/{parse_result.total_samples}")
    print(f"{'='*60}\n")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint info
    checkpoint_info = {
        "log_file": str(args.log),
        "total_evaluated": len(parse_result.evaluated),
        "successful": len(parse_result.successful_ids),
        "failed": len(parse_result.failed_ids),
        "last_index": parse_result.last_index,
        "total_samples": parse_result.total_samples,
        "evaluated_samples": [
            {
                "sample_id": e.sample_id,
                "j_score": e.j_score,
                "status": e.status,
                "latency_ms": e.latency_ms,
                "index": e.index,
            }
            for e in parse_result.evaluated
        ],
    }
    checkpoint_path = args.output.with_suffix(".checkpoint.json")
    checkpoint_path.write_text(json.dumps(checkpoint_info, indent=2))
    logger.info("Checkpoint saved to: %s", checkpoint_path)

    # Load the dataset
    resume_config = build_resume_config(args)

    if args.data:
        all_samples = load_hotpotqa_json(args.data, 0)
    else:
        all_samples = load_hotpotqa_hf(resume_config.split, resume_config.setting, 0)

    # Filter to get remaining samples
    remaining = filter_samples(all_samples, parse_result, args.retry_failed)

    if resume_config.limit > 0:
        remaining = remaining[:resume_config.limit]
        logger.info("Limited to %d samples", len(remaining))

    if not remaining:
        logger.info("No samples remaining to evaluate. Done!")
        return

    # Check for --dry-run mode
    if resume_config.limit < 0:
        logger.info("Dry-run mode (--limit -1): not starting evaluation")
        return

    # Build run config
    import os
    import tempfile

    judge_model = resume_config.judge_model or os.getenv(
        "JUDGE_BASE_MODEL", "Qwen3.5-9B-UD-Q4_K_XL.gguf"
    )

    run_config = RunConfig(
        mode=EvalMode(resume_config.mode),
        split=resume_config.split,
        setting=resume_config.setting,
        judge_model=judge_model,
        persist_dir=resume_config.persist_dir,
        corpus_dir=resume_config.corpus_dir,
        limit=0,  # Already filtered
        hard_fail=resume_config.hard_fail,
        skip_ingest=resume_config.skip_ingest,
    )

    # Run evaluation
    print(f"\n{'='*60}")
    print("RESUMING EVALUATION")
    print(f"{'='*60}")
    print(f"Samples to evaluate: {len(remaining)}")
    print(f"Mode: {run_config.mode.value}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    evaluator = HotpotEvaluator(run_config)
    new_results = evaluator.run(remaining, args.output)

    # Merge and save final results
    merged = merge_results(parse_result.evaluated, new_results)
    final_path = args.output.with_suffix(".merged.json")
    final_path.write_text(json.dumps(merged, indent=2))
    logger.info("Merged results saved to: %s", final_path)


if __name__ == "__main__":
    main()