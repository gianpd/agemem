"""
evaluation/run.py
-----------------
CLI entry point for evaluation pipeline.

Usage:
    python evaluation/run.py --dataset evaluation/data/longmemeval_s_cleaned.json --mode full
    python evaluation/run.py --dataset evaluation/data/longmemeval_s_cleaned.json --mode lifecycle --queries 10
    python evaluation/run.py --dataset evaluation/data/longmemeval_s_cleaned.json --mode retrieval --output-dir evaluation/results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Evaluation components
from evaluation.factory import OrchestratorFactory
from evaluation.mock_llm import StatefulMockLLM
from evaluation.session_replay import SessionReplayEngine, SessionReplayResult
from evaluation.question_evaluator import QuestionEvaluator, EvaluationContext, QuestionResult
from evaluation.trace_capture import TraceCapture, TurnTraceSummary

# Pipeline components
from evaluation.pipeline.dataset_pipeline import DatasetPipeline, BenchmarkQuery
from evaluation.pipeline.metrics_pipeline import MetricsPipeline
from evaluation.pipeline.report_generator import ReportGenerator

# Core types
from agents.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for evaluation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AgeMem Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  retrieval   - Test retrieval quality (MRR, Recall, NDCG)
  lifecycle   - Test memory lifecycle (STM overflow, triggers, LTM promotion)
  full        - Complete evaluation with all metrics
        """,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to benchmark dataset (JSON format)",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "lifecycle", "full"],
        default="full",
        help="Evaluation mode (default: full)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=0,
        help="Number of queries to evaluate (0 = all, default: 0). Also limits sessions to those relevant to these queries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Output directory for results (default: evaluation/results)",
    )
    parser.add_argument(
        "--persist-session",
        action="store_true",
        help="Persist session data after evaluation (for debugging)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_dataset(
    dataset_path: Path,
    query_limit: int = 0,
) -> tuple[list, list, list]:
    """
    Load benchmark dataset with optional query limit.

    When query_limit is set, only loads the relevant subset of data
    (the N instances and their associated sessions).

    Args:
        dataset_path: Path to the dataset JSON file
        query_limit: Max number of queries to load (0 = all)

    Returns:
        Tuple of (entries, queries, raw_data)
    """
    logger.info(f"Loading dataset from {dataset_path}")

    # Load raw JSON first
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Apply query limit at instance level
    if query_limit > 0:
        raw_data = raw_data[:query_limit]
        logger.info(f"Limited to {len(raw_data)} query instances")

    # Create minimal entries/queries for the limited data
    # This avoids parsing all 246k entries when we only need a few
    entries = []
    queries = []
    entry_id_counter = 0

    for instance in raw_data:
        question_id = instance.get("question_id", f"q_{entry_id_counter}")
        question_text = instance.get("question", "")
        question_type = instance.get("question_type", "retrieval")

        # Extract entries from sessions
        relevant_entry_ids = []
        haystack_sessions = instance.get("haystack_sessions", [])

        for session_idx, session in enumerate(haystack_sessions):
            session_id = instance.get("haystack_session_ids", [])[session_idx] if session_idx < len(instance.get("haystack_session_ids", [])) else f"s_{session_idx}"

            for turn_idx, turn in enumerate(session):
                role = turn.get("role", "user")
                content = turn.get("content", "")
                has_answer = turn.get("has_answer", False)

                entry_id = f"{question_id}_{session_id}_{turn_idx}"
                entries.append({
                    "content": f"[{role}] {content}",
                    "entry_id": entry_id,
                    "tags": [question_type, f"session_{session_id}"],
                })

                if has_answer:
                    relevant_entry_ids.append(entry_id)

                entry_id_counter += 1

        queries.append({
            "query_id": question_id,
            "query_text": question_text,
            "relevant_entry_ids": relevant_entry_ids,
            "query_type": question_type,
            "metadata": {
                "answer": instance.get("answer", ""),
                "question_date": instance.get("question_date"),
                "answer_session_ids": instance.get("answer_session_ids", []),
            },
        })

    logger.info(f"Loaded {len(entries)} entries and {len(queries)} queries")

    return entries, queries, raw_data


def run_session_replay(
    orchestrator: Orchestrator,
    raw_data: list,
    behavior_type: str = "IE",
) -> list[SessionReplayResult]:
    """
    Replay sessions through orchestrator.chat().

    This tests the production codepath users experience.
    """
    logger.info("Starting session replay...")

    engine = SessionReplayEngine(orchestrator=orchestrator)

    # Extract sessions from LongMemEval format
    # raw_data is already limited by query_limit/session_limit in load_dataset
    all_sessions = []
    for instance in raw_data:
        sessions = instance.get("haystack_sessions", [])
        for session in sessions:
            all_sessions.append(session)

    logger.info(f"Replaying {len(all_sessions)} sessions")

    # Load and replay sessions
    engine.load_sessions(all_sessions, behavior_type)
    results = engine.replay_all()

    # Log summary
    total_turns = sum(r.turns_processed for r in results)
    total_ltm_adds = sum(r.ltm_entries_added for r in results)
    logger.info(f"Replayed {total_turns} turns, added {total_ltm_adds} LTM entries")

    return results


def run_question_evaluation(
    orchestrator: Orchestrator,
    queries: list,
    raw_data: list,
    limit: int = 0,
) -> list[QuestionResult]:
    """
    Evaluate questions through orchestrator.chat().
    """
    logger.info("Starting question evaluation...")

    evaluator = QuestionEvaluator(orchestrator=orchestrator)
    results = []

    # Limit queries if specified
    eval_queries = queries[:limit] if limit > 0 else queries

    # Build question_id to instance mapping
    instance_map = {}
    for instance in raw_data:
        qid = instance.get("question_id", "")
        if qid:
            instance_map[qid] = instance

    for query in eval_queries:
        # Find corresponding instance for behavior type and answer
        # queries is a list of dicts from load_dataset
        query_id = query.get("query_id", "") if isinstance(query, dict) else query.query_id
        instance = instance_map.get(query_id, {})
        expected_answer = instance.get("answer", "")
        question_type = instance.get("question_type", "retrieval")
        query_text = query.get("query_text", "") if isinstance(query, dict) else query.query_text

        # Map question type to behavior
        behavior_map = {
            "single-session-user": "IE",
            "single-session-assistant": "IE",
            "preference": "IE",
            "multi-session": "MR",
            "aggregation": "MR",
            "comparison": "MR",
            "knowledge-update": "KU",
            "knowledge": "KU",
            "temporal-reasoning": "TR",
            "temporal": "TR",
            "time-reference": "TR",
            "date-filtering": "TR",
            "unknown": "ABS",
            "abstention": "ABS",
        }
        behavior_type = behavior_map.get(question_type.lower().replace("_", "-"), "IE")

        # Get evidence session IDs
        evidence_session_ids = instance.get("answer_session_ids", [])
        if not evidence_session_ids:
            # Fallback: extract from has_answer markers
            evidence_session_ids = []
            for idx, session in enumerate(instance.get("haystack_sessions", [])):
                for turn in session:
                    if turn.get("has_answer", False):
                        evidence_session_ids.append(idx)
                        break

        context = EvaluationContext(
            behavior_type=behavior_type,
            expected_answer=expected_answer,
            evidence_session_ids=evidence_session_ids,
        )

        result = evaluator.evaluate_question(query_text, context)
        results.append(result)

        logger.debug(
            f"Query {query_id}: correct={result.is_correct}, "
            f"behavior={result.behavior_type}, abstained={result.abstained}"
        )

    # Log summary
    correct = sum(1 for r in results if r.is_correct)
    logger.info(f"Evaluated {len(results)} queries: {correct}/{len(results)} correct")

    return results


def compute_and_export_metrics(
    queries: list,
    question_results: list[QuestionResult],
    session_results: list[SessionReplayResult],
    output_dir: Path,
    session_id: str,
) -> dict:
    """
    Compute metrics and export results.
    """
    logger.info("Computing metrics...")

    # Build trace data for metrics pipeline
    # We need SearchTrace format for MetricsPipeline
    from evaluation.pipeline.inference_pipeline import SearchTrace
    from evaluation.pipeline.dataset_pipeline import BenchmarkQuery

    traces = []
    for qr in question_results:
        if qr.retrieval_trace:
            trace = SearchTrace(
                query=qr.query_id,  # Use query_id as query identifier
                results=qr.retrieval_trace.get("results", []),
                latency_ms=qr.latency_ms,
            )
        else:
            trace = SearchTrace(
                query=qr.query_id,
                results=[],
                latency_ms=qr.latency_ms,
            )
        traces.append(trace)

    # Convert query dicts to BenchmarkQuery objects if needed
    benchmark_queries = []
    for q in queries:
        if isinstance(q, dict):
            benchmark_queries.append(BenchmarkQuery(
                query_id=q.get("query_id", ""),
                query_text=q.get("query_text", ""),
                relevant_entry_ids=q.get("relevant_entry_ids", []),
                query_type=q.get("query_type", "retrieval"),
            ))
        else:
            benchmark_queries.append(q)

    # Calculate metrics
    metrics_pipeline = MetricsPipeline(session_id=session_id)
    retrieval_metrics = metrics_pipeline.calculate_retrieval_metrics(benchmark_queries, traces)
    behavior_metrics = metrics_pipeline.calculate_behavior_metrics(benchmark_queries, traces)

    # Build session replay metrics
    replay_metrics = {
        "total_sessions": len(session_results),
        "total_turns": sum(r.turns_processed for r in session_results),
        "total_ltm_adds": sum(r.ltm_entries_added for r in session_results),
        "avg_stm_tokens": sum(r.stm_tokens_at_end for r in session_results) / len(session_results) if session_results else 0,
        "avg_learning_score": sum(
            sum(r.learning_scores) / len(r.learning_scores) if r.learning_scores else 0
            for r in session_results
        ) / len(session_results) if session_results else 0,
    }

    # Build question evaluation metrics
    question_metrics = {
        "total_queries": len(question_results),
        "correct": sum(1 for r in question_results if r.is_correct),
        "accuracy": sum(1 for r in question_results if r.is_correct) / len(question_results) if question_results else 0,
        "abstained": sum(1 for r in question_results if r.abstained),
        "avg_latency_ms": sum(r.latency_ms for r in question_results) / len(question_results) if question_results else 0,
    }

    # Behavior breakdown
    behavior_breakdown = {}
    for r in question_results:
        if r.behavior_type not in behavior_breakdown:
            behavior_breakdown[r.behavior_type] = {"total": 0, "correct": 0}
        behavior_breakdown[r.behavior_type]["total"] += 1
        if r.is_correct:
            behavior_breakdown[r.behavior_type]["correct"] += 1

    results = {
        "session_id": session_id,
        "retrieval": retrieval_metrics.to_dict(),
        "behavior_metrics": {k: v.to_dict() for k, v in behavior_metrics.items()},
        "replay_metrics": replay_metrics,
        "question_metrics": question_metrics,
        "behavior_breakdown": behavior_breakdown,
        "evaluated_at": datetime.now().isoformat(),
    }

    # Export to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{session_id}_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Exported metrics to {output_path}")

    return results


def generate_report(
    results: dict,
    question_results: list[QuestionResult],
    session_results: list[SessionReplayResult],
    output_dir: Path,
    session_id: str,
) -> Path:
    """
    Generate HTML and markdown reports.
    """
    logger.info("Generating reports...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate markdown report
    report_lines = [
        f"# AgeMem Evaluation Report",
        f"",
        f"**Session ID:** {session_id}",
        f"**Evaluated at:** {results['evaluated_at']}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Queries | {results['question_metrics']['total_queries']} |",
        f"| Correct | {results['question_metrics']['correct']} |",
        f"| Accuracy | {results['question_metrics']['accuracy']:.2%} |",
        f"| Abstained | {results['question_metrics']['abstained']} |",
        f"| Avg Latency | {results['question_metrics']['avg_latency_ms']:.1f}ms |",
        f"",
        f"## Retrieval Metrics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]

    for key, value in results["retrieval"].items():
        if isinstance(value, float):
            report_lines.append(f"| {key} | {value:.4f} |")
        else:
            report_lines.append(f"| {key} | {value} |")

    report_lines.extend([
        f"",
        f"## Behavior Breakdown",
        f"",
        f"| Behavior | Total | Correct | Accuracy |",
        f"|----------|-------|---------|----------|",
    ])

    for behavior, stats in results["behavior_breakdown"].items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        report_lines.append(f"| {behavior} | {stats['total']} | {stats['correct']} | {acc:.2%} |")

    report_lines.extend([
        f"",
        f"## Session Replay Metrics",
        f"",
        f"- Total Sessions: {results['replay_metrics']['total_sessions']}",
        f"- Total Turns: {results['replay_metrics']['total_turns']}",
        f"- LTM Entries Added: {results['replay_metrics']['total_ltm_adds']}",
        f"- Avg STM Tokens: {results['replay_metrics']['avg_stm_tokens']:.0f}",
        f"- Avg Learning Score: {results['replay_metrics']['avg_learning_score']:.3f}",
        f"",
    ])

    # Write markdown report
    md_path = output_dir / f"{session_id}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Generated report: {md_path}")

    return md_path


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    session_id = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    logger.info(f"Starting evaluation session: {session_id}")

    # Create temp directory for session data
    if args.persist_session:
        persist_dir = args.output_dir / "session" / session_id
        persist_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="agemem_eval_")
        persist_dir = Path(temp_dir)

    try:
        # Load dataset
        entries, queries, raw_data = load_dataset(
            args.dataset,
            query_limit=args.queries,
        )

        if not queries:
            logger.error("No queries found in dataset")
            return 1

        # Build orchestrator with mock LLM
        logger.info("Building orchestrator with mock LLM...")
        mock_llm = StatefulMockLLM(strategy="template")

        # Add some default response templates
        mock_llm.add_response_template("phone", "Based on our conversation, your phone number is mentioned in the context.")
        mock_llm.add_response_template("email", "I can see your email address in our conversation history.")
        mock_llm.add_response_template("address", "Your address was mentioned earlier in our conversation.")
        mock_llm.add_response_template("preference", "I remember your preference from our earlier discussion.")
        mock_llm.add_response_template("name", "Your name was mentioned in our conversation.")

        orchestrator = OrchestratorFactory().build_for_evaluation(
            llm_client=mock_llm,
            persist_dir=persist_dir,
            config_overrides={
                "STM_TOKEN_LIMIT": 8000,
                "LTM_PROMOTE_THRESHOLD": 0.5,
            },
        )

        # Run evaluation based on mode
        session_results = []
        question_results = []

        if args.mode in ["lifecycle", "full"]:
            # Session replay tests the full memory lifecycle
            session_results = run_session_replay(
                orchestrator=orchestrator,
                raw_data=raw_data,
            )

        if args.mode in ["retrieval", "full"]:
            # Question evaluation tests retrieval quality
            question_results = run_question_evaluation(
                orchestrator=orchestrator,
                queries=queries,
                raw_data=raw_data,
                limit=args.queries,
            )

        # Compute metrics and generate report
        results = compute_and_export_metrics(
            queries=queries,
            question_results=question_results,
            session_results=session_results,
            output_dir=args.output_dir,
            session_id=session_id,
        )

        report_path = generate_report(
            results=results,
            question_results=question_results,
            session_results=session_results,
            output_dir=args.output_dir,
            session_id=session_id,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Session ID: {session_id}")
        print(f"Mode: {args.mode}")
        print(f"Queries evaluated: {results['question_metrics']['total_queries']}")
        print(f"Accuracy: {results['question_metrics']['accuracy']:.2%}")
        print(f"Report: {report_path}")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1

    finally:
        # Cleanup temp directory unless persisting
        if not args.persist_session and persist_dir.exists():
            import shutil
            shutil.rmtree(persist_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())