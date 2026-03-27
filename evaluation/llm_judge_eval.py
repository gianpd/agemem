#!/usr/bin/env python3
"""
Evaluate LLM-as-Judge correctness.

Parses evaluation logs to extract LLM-as-Judge scores, then cross-references
with the HotpotQA dataset to produce a comprehensive report.

Output schema:
- sample_id: Unique sample identifier
- index: Position in evaluation run
- query: The question from HotpotQA
- gold: Expected answer
- gold_titles: Supporting fact titles (for multi-hop verification)
- llm_judge_score: Score from LLM-as-Judge (0.0-1.0)
- judge_status: "ok" or "parse_failure"
- latency_ms: Evaluation latency in milliseconds

Usage:
    # Parse log file and generate report
    python evaluation/llm_judge_eval.py --log evaluation/logs/hotpotqa_20260326_120905.log

    # Output to specific files
    python evaluation/llm_judge_eval.py --log evaluation/logs/hotpotqa_20260326_120905.log --output reports/judge_eval.json --csv reports/judge_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log parsing (from resume_hotpotqa.py)
# ---------------------------------------------------------------------------

@dataclass
class ParsedSample:
    """Sample parsed from evaluation log."""
    sample_id: str
    index: int
    j_score: float
    status: str
    latency_ms: float
    question_type: str = ""
    level: str = ""


@dataclass
class LogParseResult:
    """Result of parsing an evaluation log."""
    evaluated: list[ParsedSample] = field(default_factory=list)
    successful_ids: set[str] = field(default_factory=set)
    failed_ids: set[str] = field(default_factory=set)
    last_index: int = 0
    total_samples: int = 0


def parse_eval_log(log_path: Path) -> LogParseResult:
    """
    Parse an evaluation log to extract evaluated samples.

    The log format is:
        INFO [69/7405] id=5ac2acff55429921a00ab02b type=bridge level=hard
        ... evaluation happens ...
        INFO   j_score=1.00 status=ok latency=53387ms error=None

    We match sample ID lines with their subsequent j_score lines.
    """
    evaluated: list[ParsedSample] = []
    successful_ids: set[str] = set()
    failed_ids: set[str] = set()
    last_index = 0
    total_samples = 0

    # Patterns
    sample_pattern = re.compile(
        r"INFO\s+\[(\d+)/(\d+)\]\s+id=([a-f0-9]+)\s+type=(\w+)\s+level=(\w+)"
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
                qtype = sample_match.group(4)
                qlevel = sample_match.group(5)

                last_index = idx
                total_samples = total
                current_sample = {
                    "sample_id": sid,
                    "index": idx,
                    "question_type": qtype,
                    "level": qlevel,
                }
                continue

            # Check for j_score (completes the current sample)
            score_match = score_pattern.search(line)
            if score_match and current_sample:
                j_score = float(score_match.group(1))
                status = score_match.group(2)
                latency = int(score_match.group(3))

                parsed = ParsedSample(
                    sample_id=current_sample["sample_id"],
                    index=current_sample["index"],
                    j_score=j_score,
                    status=status,
                    latency_ms=float(latency),
                    question_type=current_sample["question_type"],
                    level=current_sample["level"],
                )
                evaluated.append(parsed)

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
# HotpotQA dataset loading
# ---------------------------------------------------------------------------

@dataclass
class HotpotSample:
    """A single HotpotQA sample."""
    id: str
    question: str
    answer: str
    supporting_facts: list[tuple[str, int]]
    context: list[tuple[str, list[str]]]
    question_type: str
    level: str

    @property
    def gold_titles(self) -> set[str]:
        return {title for title, _ in self.supporting_facts}


def load_hotpotqa_hf(split: str = "validation", setting: str = "distractor", limit: int = 0) -> list[HotpotSample]:
    """Load HotpotQA from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading HotpotQA split=%s setting=%s limit=%s", split, setting, limit or "all")
    dataset = load_dataset("hotpot_qa", setting, split=split)

    samples = []
    for item in (dataset if not limit else list(dataset)[:limit]):
        sample = HotpotSample(
            id=item.get("_id") or item.get("id"),
            question=item["question"],
            answer=item["answer"],
            supporting_facts=list(zip(
                item["supporting_facts"]["title"],
                item["supporting_facts"]["sent_id"],
            )),
            context=list(zip(
                item["context"]["title"],
                item["context"]["sentences"],
            )),
            question_type=item.get("type", "unknown"),
            level=item.get("level", "unknown"),
        )
        samples.append(sample)

    logger.info("Loaded %d samples", len(samples))
    return samples


def load_hotpotqa_json(path: Path, limit: int = 0) -> list[HotpotSample]:
    """Load HotpotQA from a local JSON file."""
    logger.info("Loading HotpotQA from %s (limit=%s)", path, limit or "all")
    raw = json.loads(path.read_text())
    if limit:
        raw = raw[:limit]

    samples = []
    for item in raw:
        sample = HotpotSample(
            id=item.get("_id") or item.get("id"),
            question=item["question"],
            answer=item["answer"],
            supporting_facts=list(zip(
                item["supporting_facts"]["title"],
                item["supporting_facts"]["sent_id"],
            )),
            context=list(zip(
                item["context"]["title"],
                item["context"]["sentences"],
            )),
            question_type=item.get("type", "unknown"),
            level=item.get("level", "unknown"),
        )
        samples.append(sample)

    logger.info("Loaded %d samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------

@dataclass
class JudgeEvalReport:
    """Single row in the LLM-as-Judge evaluation report."""
    sample_id: str
    index: int
    query: str
    gold: str
    gold_titles: list[str]
    prediction: str
    llm_judge_score: float
    judge_status: str
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_row(self) -> list:
        """Convert to CSV row format."""
        return [
            self.sample_id,
            self.index,
            self.query,
            self.gold,
            "|".join(self.gold_titles),  # Pipe-separated for CSV
            self.prediction[:200] + "..." if len(self.prediction) > 200 else self.prediction,
            self.llm_judge_score,
            self.judge_status,
            self.latency_ms,
        ]


@dataclass
class JudgeEvalSummary:
    """Summary statistics for LLM-as-Judge evaluation."""
    total_samples: int
    successful_judges: int
    failed_judges: int
    coverage_rate: float  # Percentage of successful judge calls
    mean_score_valid_only: float  # Mean of successful scores only (excludes failures)
    std_score_valid_only: float  # Std dev of successful scores only
    mean_score_all: float  # Mean with failures imputed as 0.0 (for reference)
    predictions_available: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results_from_json(path: Path) -> dict:
    """Load evaluation results from JSON file."""
    data = json.loads(path.read_text())

    # Handle different formats
    if "results" in data:
        # Format from run_hotpotqa.py
        return {
            r["sample_id"]: {
                "sample_id": r["sample_id"],
                "j_score": r["j_score"],
                "status": r["parse_status"],
                "prediction": r.get("predicted_answer", ""),
                "latency_ms": r.get("latency_ms", 0),
            }
            for r in data["results"]
        }
    elif isinstance(data, list):
        # Format from checkpoint or merged
        if data and "sample_id" in data[0]:
            return {
                r["sample_id"]: {
                    "sample_id": r["sample_id"],
                    "j_score": r.get("j_score", r.get("value", 0)),
                    "status": r.get("status", r.get("parse_status", "unknown")),
                    "prediction": r.get("prediction", ""),
                    "latency_ms": r.get("latency_ms", 0),
                }
                for r in data
            }

    raise ValueError(f"Unknown results format in {path}")


def load_results_from_checkpoint(path: Path) -> dict:
    """Load results from checkpoint JSON."""
    data = json.loads(path.read_text())

    if "evaluated_samples" not in data:
        raise ValueError(f"Not a checkpoint file: {path}")

    return {
        r["sample_id"]: {
            "sample_id": r["sample_id"],
            "j_score": r["j_score"],
            "status": r["status"],
            "prediction": "",  # Not available in checkpoint
            "latency_ms": r["latency_ms"],
            "index": r["index"],
        }
        for r in data["evaluated_samples"]
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(
    dataset: list[HotpotSample],
    results: dict,
    predictions_available: bool = False,
) -> tuple[list[JudgeEvalReport], JudgeEvalSummary]:
    """
    Build the evaluation report by matching results to dataset.

    Args:
        dataset: List of HotpotQA samples
        results: Dict of sample_id -> evaluation result
        predictions_available: Whether predictions are in the results

    Returns:
        Tuple of (report rows, summary)
    """
    # Create dataset lookup
    dataset_by_id = {s.id: s for s in dataset}

    reports: list[JudgeEvalReport] = []
    scores: list[float] = []
    successful = 0
    failed = 0

    for sample_id, result in results.items():
        if sample_id not in dataset_by_id:
            logger.warning("Sample %s not found in dataset", sample_id)
            continue

        sample = dataset_by_id[sample_id]

        # Track stats - only count successful scores for mean/std calculation
        if result["status"] == "ok":
            successful += 1
            scores.append(result["j_score"])
        else:
            failed += 1
            # Do NOT add 0.0 for failed judges - they are excluded from score stats

        report = JudgeEvalReport(
            sample_id=sample_id,
            index=result.get("index", 0),
            query=sample.question,
            gold=sample.answer,
            gold_titles=list(sample.gold_titles),
            prediction=result.get("prediction", ""),
            llm_judge_score=result["j_score"],
            judge_status=result["status"],
            latency_ms=result["latency_ms"],
        )
        reports.append(report)

    # Sort by index
    reports.sort(key=lambda r: r.index)

    # Calculate summary stats
    # Coverage: percentage of successful judge calls
    total = len(reports)
    coverage_rate = (successful / total * 100) if total > 0 else 0.0

    # Mean/std of VALID scores only (exclude parse failures)
    mean_score_valid = sum(scores) / len(scores) if scores else 0.0
    std_score_valid = (sum((s - mean_score_valid) ** 2 for s in scores) / len(scores)) ** 0.5 if len(scores) > 1 else 0.0

    # Mean with failures imputed as 0.0 (for reference/comparison)
    all_scores = scores + [0.0] * failed  # Add zeros for failed judges
    mean_score_all = sum(all_scores) / len(all_scores) if all_scores else 0.0

    summary = JudgeEvalSummary(
        total_samples=len(reports),
        successful_judges=successful,
        failed_judges=failed,
        coverage_rate=coverage_rate,
        mean_score_valid_only=mean_score_valid,
        std_score_valid_only=std_score_valid,
        mean_score_all=mean_score_all,
        predictions_available=predictions_available,
    )

    return reports, summary


def save_report_json(reports: list[JudgeEvalReport], summary: JudgeEvalSummary, path: Path) -> None:
    """Save report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "summary": summary.to_dict(),
        "reports": [r.to_dict() for r in reports],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("JSON report saved to %s", path)


def save_report_csv(reports: list[JudgeEvalReport], path: Path) -> None:
    """Save report as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "sample_id", "index", "query", "gold", "gold_titles",
        "prediction", "llm_judge_score", "judge_status", "latency_ms"
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for report in reports:
            writer.writerow(report.to_row())

    logger.info("CSV report saved to %s", path)


def save_report_markdown(reports: list[JudgeEvalReport], summary: JudgeEvalSummary, path: Path) -> None:
    """Save report as Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# LLM-as-Judge Evaluation Report",
        "",
        "## Summary",
        "",
        f"- **Total samples:** {summary.total_samples}",
        f"- **Successful judges:** {summary.successful_judges}",
        f"- **Failed judges:** {summary.failed_judges}",
        f"- **Coverage rate:** {summary.coverage_rate:.1f}%",
        f"- **Mean J-score (valid only):** {summary.mean_score_valid_only:.4f}",
        f"- **Std J-score (valid only):** {summary.std_score_valid_only:.4f}",
        f"- **Mean J-score (all):** {summary.mean_score_all:.4f} (with failures as 0.0)",
        f"- **Predictions available:** {summary.predictions_available}",
        "",
        "## Detailed Results",
        "",
        "| # | Sample ID | J-score | Status | Query | Gold |",
        "|---|-----------|---------|--------|-------|------|",
    ]

    for r in reports[:50]:  # Limit to first 50 for readability
        query_short = r.query[:50] + "..." if len(r.query) > 50 else r.query
        gold_short = r.gold[:30] + "..." if len(r.gold) > 30 else r.gold
        lines.append(f"| {r.index} | `{r.sample_id[:16]}...` | {r.llm_judge_score:.2f} | {r.judge_status} | {query_short} | {gold_short} |")

    if len(reports) > 50:
        lines.append(f"| ... | ({len(reports) - 50} more rows) | | | | |")

    path.write_text("\n".join(lines))
    logger.info("Markdown report saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate LLM-as-Judge evaluation report"
    )
    parser.add_argument(
        "--log", type=Path,
        help="Evaluation log file to parse for judge scores",
    )
    parser.add_argument(
        "--results", type=Path,
        help="Results JSON file (contains predictions if available)",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Checkpoint JSON file from resume script",
    )
    parser.add_argument(
        "--dataset", type=Path,
        help="Local JSON dataset (uses HF if not specified)",
    )
    parser.add_argument(
        "--split", default="validation",
        help="Dataset split",
    )
    parser.add_argument(
        "--setting", default="distractor",
        help="Dataset setting",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation/reports/judge_eval.json"),
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--csv", type=Path,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--md", type=Path,
        help="Optional Markdown output path",
    )
    args = parser.parse_args()

    # Determine source of results
    results: dict = {}
    predictions_available = False

    if args.results:
        logger.info("Loading results from: %s", args.results)
        results = load_results_from_json(args.results)
        predictions_available = True
    elif args.checkpoint:
        logger.info("Loading results from checkpoint: %s", args.checkpoint)
        results = load_results_from_checkpoint(args.checkpoint)
    elif args.log:
        logger.info("Parsing log file: %s", args.log)
        parse_result = parse_eval_log(args.log)
        results = {
            s.sample_id: {
                "sample_id": s.sample_id,
                "j_score": s.j_score,
                "status": s.status,
                "prediction": "",  # Not available from log
                "latency_ms": s.latency_ms,
                "index": s.index,
            }
            for s in parse_result.evaluated
        }
    else:
        parser.error("Must specify --log, --results, or --checkpoint")

    if not results:
        logger.error("No results found")
        sys.exit(1)

    # Load dataset
    if args.dataset:
        dataset = load_hotpotqa_json(args.dataset, 0)
    else:
        dataset = load_hotpotqa_hf(args.split, args.setting, 0)

    logger.info("Loaded %d samples from dataset", len(dataset))
    logger.info("Loaded %d results", len(results))

    # Build report
    reports, summary = build_report(dataset, results, predictions_available)

    # Print summary
    print("\n" + "=" * 60)
    print("LLM-AS-JUDGE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples       : {summary.total_samples}")
    print(f"Successful judges   : {summary.successful_judges}")
    print(f"Failed judges       : {summary.failed_judges}")
    print(f"Coverage rate       : {summary.coverage_rate:.1f}%")
    print("-" * 60)
    print(f"Mean J-score (valid): {summary.mean_score_valid_only:.4f}")
    print(f"Std J-score (valid) : {summary.std_score_valid_only:.4f}")
    print(f"Mean J-score (all)  : {summary.mean_score_all:.4f} (with failures as 0.0)")
    print("-" * 60)
    print(f"Predictions         : {'Available' if predictions_available else 'NOT AVAILABLE'}")
    print("=" * 60 + "\n")

    # Show sample of results
    print("Sample of results:")
    print("-" * 60)
    for r in reports[:5]:
        status_marker = "✓" if r.judge_status == "ok" else "✗"
        print(f"[{r.index}] {status_marker} J={r.llm_judge_score:.2f} | {r.judge_status}")
        print(f"    Q: {r.query[:60]}...")
        print(f"    Gold: {r.gold}")
        if r.prediction:
            print(f"    Pred: {r.prediction[:60]}...")
        elif r.judge_status != "ok":
            print(f"    Pred: [parse failure - prediction not evaluated]")
        print()

    # Save outputs
    save_report_json(reports, summary, args.output)

    if args.csv:
        save_report_csv(reports, args.csv)

    if args.md:
        save_report_markdown(reports, summary, args.md)
    else:
        # Default markdown output
        default_md = args.output.with_suffix(".md")
        save_report_markdown(reports, summary, default_md)


if __name__ == "__main__":
    main()