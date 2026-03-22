"""
evaluation/run.py
-----------------
Simplified CLI entry point for AgeMem evaluation.

DEPRECATED: This module is deprecated.
  - load_dataset -> evaluation.loader.load_dataset
  - generate_report -> evaluation.report.generate_report
  - For batch evaluation, use evaluation.runner.BatchRunner

Usage:
    python evaluation/run.py --dataset evaluation/data/longmemeval_s_cleaned.json --queries 5
    python evaluation/run.py --dataset evaluation/data/longmemeval_s_cleaned.json --mode lifecycle --queries 10
"""

from __future__ import annotations

import warnings

warnings.warn(
    "evaluation.run is deprecated. Use the new modular components instead:\n"
    "  - load_dataset -> evaluation.loader.load_dataset\n"
    "  - generate_report -> evaluation.report.generate_report\n"
    "  - For batch evaluation, use evaluation.runner.BatchRunner",
    DeprecationWarning,
    stacklevel=2
)

import argparse
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import Orchestrator
from evaluation.factory import OrchestratorFactory
from evaluation.mock_llm import StatefulMockLLM
from evaluation.evaluators import Evaluator
from evaluation.metrics import calculate_metrics, EvaluationSummary
from evaluation.llm_judge import LLMJudge  # Import judge

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgeMem Evaluation Pipeline with LLM-as-Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full       - Complete evaluation (session replay + questions)
  lifecycle  - Session replay only (test memory lifecycle)
  retrieval  - Question evaluation only (test retrieval quality)

LLM-as-Judge:
  Enable with --use-llm-judge. Requires llama.cpp server running.
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
        choices=["full", "lifecycle", "retrieval"],
        default="full",
        help="Evaluation mode (default: full)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=0,
        help="Number of queries to evaluate (0 = all)",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=0,
        help="Max sessions to replay in lifecycle mode (0 = all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--persist-session",
        action="store_true",
        help="Persist session data after evaluation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM instead of real LLM",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM-as-Judge for answer evaluation (requires judge server)",
    )
    parser.add_argument(
        "--judge-api-base",
        type=str,
        default="http://localhost:8080/v1",
        help="Judge server API endpoint (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="llama-3.1-70b-instruct",
        help="Judge model name (default: llama-3.1-70b-instruct)",
    )
    return parser.parse_args()


def load_dataset(
    dataset_path: Path,
    query_limit: int = 0,
    load_sessions: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load LongMemEval dataset.

    Args:
        dataset_path: Path to the dataset JSON file
        query_limit: Max number of queries to load (0 = all)
        load_sessions: If True, include session data (expensive for large datasets)

    Returns:
        Tuple of (entries, queries, raw_data)
    """
    logger.info(f"Loading dataset from {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if query_limit > 0:
        raw_data = raw_data[:query_limit]
        logger.info(f"Limited to {len(raw_data)} query instances")

    # Build entries and queries from LongMemEval format
    entries = []
    queries = []
    entry_id_counter = 0

    for instance in raw_data:
        question_id = instance.get("question_id", f"q_{entry_id_counter}")
        question_text = instance.get("question", "")
        question_type = instance.get("question_type", "retrieval")

        relevant_entry_ids = []
        relevant_content = []
        session_ids = instance.get("haystack_session_ids", [])

        # Only process session data if requested
        if load_sessions:
            haystack_sessions = instance.get("haystack_sessions", [])
            for session_idx, session in enumerate(haystack_sessions):
                session_id = session_ids[session_idx] if session_idx < len(session_ids) else f"s_{session_idx}"

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
                        relevant_content.append(content)

                    entry_id_counter += 1

        queries.append({
            "query_id": question_id,
            "query_text": question_text,
            "relevant_entry_ids": relevant_entry_ids,
            "relevant_content": relevant_content,
            "query_type": question_type,
            "expected_answer": instance.get("answer", ""),
        })

    logger.info(f"Loaded {len(entries)} entries and {len(queries)} queries")
    return entries, queries, raw_data


def generate_report(
    summary: EvaluationSummary,
    session_id: str,
    output_dir: Path,
) -> Path:
    """Generate markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# AgeMem Evaluation Report",
        f"",
        f"**Session ID:** {session_id}",
        f"**Evaluated at:** {datetime.now().isoformat()}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Queries | {summary.total_queries} |",
        f"| Correct | {summary.correct} |",
        f"| Accuracy | {summary.accuracy:.2%} |",
        f"| Abstained | {summary.abstained} |",
        f"| Avg Latency | {summary.avg_latency_ms:.1f}ms |",
        f"",
        f"## LLM-as-Judge Statistics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Queries by LLM Judge | {summary.llm_judge_queries} |",
        f"| Queries by Heuristic | {summary.heuristic_queries} |",
        f"| Judge Avg Latency | {summary.judge_avg_latency_ms:.1f}ms |",
        f"",
        f"## Retrieval Metrics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]

    for key, value in summary.retrieval.to_dict().items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")

    lines.extend([
        f"",
        f"## Behavior Breakdown",
        f"",
        f"| Behavior | Count | Accuracy |",
        f"|----------|-------|----------|",
    ])

    for behavior, metrics in summary.by_behavior.items():
        lines.append(f"| {behavior} | {metrics.query_count} | {metrics.accuracy:.2%} |")

    if summary.session_replay:
        lines.extend([
            f"",
            f"## Session Replay",
            f"",
            f"- Total Sessions: {summary.session_replay.get('total_sessions', 0)}",
            f"- Total Turns: {summary.session_replay.get('total_turns', 0)}",
            f"- LTM Entries Added: {summary.session_replay.get('total_ltm_adds', 0)}",
            f"- Avg STM Tokens: {summary.session_replay.get('avg_stm_tokens', 0):.0f}",
        ])

    # Write report
    md_path = output_dir / f"{session_id}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Write JSON results
    json_path = output_dir / f"{session_id}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "session_id": session_id,
            "summary": summary.to_dict(),
        }, f, indent=2)

    logger.info(f"Generated report: {md_path}")
    return md_path


def main() -> int:
    """Main entry point with LLM-as-Judge support."""
    args = parse_args()
    setup_logging(args.verbose)

    session_id = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    logger.info(f"Starting evaluation session: {session_id}")

    # Initialize LLM-as-Judge if requested
    llm_judge = None
    if args.use_llm_judge:
        logger.info(f"Initializing LLM-as-Judge at {args.judge_api_base}")
        llm_judge = LLMJudge(
            api_base=args.judge_api_base,
            model=args.judge_model,
        )
        if not llm_judge.health_check():
            logger.error("LLM-as-Judge server is not accessible!")
            logger.error(f"Please start llama.cpp server at {args.judge_api_base}")
            logger.error("Example: llama-server --model llama-3.1-70b-instruct.Q4_K_M.gguf --port 8080")
            return 1
        logger.info("LLM-as-Judge initialized successfully")

    # Create temp directory for session data
    if args.persist_session:
        persist_dir = args.output_dir / "session" / session_id
        persist_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="agemem_eval_")
        persist_dir = Path(temp_dir)

    try:
        # Load dataset - only load sessions if doing lifecycle evaluation
        load_sessions = args.mode in ["lifecycle", "full"]
        entries, queries, raw_data = load_dataset(
            args.dataset,
            query_limit=args.queries,
            load_sessions=load_sessions,
        )

        if not queries:
            logger.error("No queries found in dataset")
            return 1

        # Build orchestrator
        if args.mock:
            logger.info("Building orchestrator with MOCK LLM...")
            mock_llm = StatefulMockLLM(strategy="template")
            mock_llm.add_response_template("phone", "Based on our conversation, your phone number is mentioned.")
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
        else:
            logger.info("Building orchestrator with REAL LLM...")
            orchestrator = OrchestratorFactory().build_for_evaluation(
                persist_dir=persist_dir,
                config_overrides={
                    "STM_TOKEN_LIMIT": 8000,
                    "LTM_PROMOTE_THRESHOLD": 0.5,
                },
            )

        # Create evaluator with LLM-as-Judge support
        evaluator = Evaluator(
            orchestrator,
            llm_judge=llm_judge,
            use_llm_judge=args.use_llm_judge,
        )

        # Run evaluation based on mode
        session_results = []
        question_results = []

        if args.mode in ["lifecycle", "full"]:
            logger.info("Running session replay...")
            # Extract sessions from raw data
            all_sessions = []
            for instance in raw_data:
                sessions = instance.get("haystack_sessions", [])
                all_sessions.extend(sessions)

            # Limit sessions if specified
            if args.sessions > 0:
                all_sessions = all_sessions[:args.sessions]
                logger.info(f"Limited to {len(all_sessions)} sessions for replay")

            session_results = evaluator.replay_sessions(all_sessions, behavior_type="IE")
            logger.info(f"Replayed {len(session_results)} sessions")

        if args.mode in ["retrieval", "full"]:
            logger.info("Running question evaluation...")
            question_results = evaluator.evaluate_questions(queries, raw_data)
            correct = sum(1 for r in question_results if r.is_correct)
            logger.info(f"Evaluated {len(question_results)} queries: {correct}/{len(question_results)} correct")

        # Calculate metrics
        summary = calculate_metrics(queries, question_results, session_results)

        # Generate report
        report_path = generate_report(summary, session_id, args.output_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Session ID: {session_id}")
        print(f"Mode: {args.mode}")
        print(f"Queries evaluated: {summary.total_queries}")
        print(f"Accuracy: {summary.accuracy:.2%}")
        if summary.llm_judge_queries > 0:
            print(f"LLM-as-Judge: {summary.llm_judge_queries} queries (avg {summary.judge_avg_latency_ms:.0f}ms)")
            print(f"Heuristic: {summary.heuristic_queries} queries")
        if summary.session_replay:
            print(f"Sessions replayed: {summary.session_replay.get('total_sessions', 0)}")
        print(f"Report: {report_path}")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1

    finally:
        # Cleanup temp directory unless persisting
        if not args.persist_session and persist_dir.exists():
            shutil.rmtree(persist_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())


# Re-exports from new modules for backward compatibility
# These functions are defined locally above; new code should import from:
#   - evaluation.loader.load_dataset
#   - evaluation.report.generate_report
__all__ = [
    "setup_logging",
    "parse_args",
    "load_dataset",
    "generate_report",
    "main",
]
