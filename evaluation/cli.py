"""
evaluation/cli.py - Unified CLI entry point for AgeMem evaluation.

Usage:
    python -m evaluation.cli --dataset data.json --batch-size 10
    python -m evaluation.cli --resume eval_20260322_120000
    python -m evaluation.cli --report eval_20260322_120000
    python -m evaluation.cli --list-checkpoints
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.runner import BatchConfig, BatchRunner
from evaluation.checkpoint import CheckpointManager
from evaluation.factory import OrchestratorFactory
from evaluation.llm_judge import LLMJudge
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgeMem Unified Evaluation CLI",
        epilog="Modes: full (default), lifecycle, retrieval. Use --batch-size 0 for no checkpointing.")
    # Actions
    p.add_argument("--dataset", type=Path, help="Path to evaluation dataset (JSON)")
    p.add_argument("--resume", metavar="SESSION_ID", help="Resume from checkpoint")
    p.add_argument("--report", metavar="SESSION_ID", help="Generate report from checkpoint")
    p.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints")
    p.add_argument("--cleanup", metavar="SESSION_ID", help="Remove checkpoint files")
    # Config
    p.add_argument("--mode", choices=["full", "lifecycle", "retrieval"], default="full")
    p.add_argument("--batch-size", type=int, default=0, help="Interactions per batch (0=no checkpointing)")
    p.add_argument("--max-interactions", type=int, default=0, help="Max interactions (0=all)")
    p.add_argument("--max-batches", type=int, default=0, help="Max batches (0=unlimited)")
    p.add_argument("--session-id", help="Session ID (auto-generated if not set)")
    p.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    p.add_argument("--checkpoint-interval", type=int, default=1, help="Save every N batches")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoint")
    p.add_argument("--mock", action="store_true", help="Use mock LLM for deterministic testing")
    p.add_argument("--use-llm-judge", action="store_true", help="Use LLM-as-Judge for answer evaluation")
    p.add_argument("--judge-api-base", type=str, default="http://localhost:8080/v1",
                   help="Judge server API endpoint (default: http://localhost:8080/v1)")
    p.add_argument("--judge-model", type=str, default="Qwen3.5-9B-UD-Q4_K_XL.gguf",
                   help="Judge model name (default: Qwen3.5-9B-UD-Q4_K_XL.gguf)")
    p.add_argument("--judge-timeout", type=float, default=120.0,
                   help="Judge request timeout in seconds (default: 120.0)")
    return p.parse_args()


def list_checkpoints(output_dir: Path) -> int:
    manager = CheckpointManager(output_dir)
    checkpoints = manager.list_checkpoints()
    if not checkpoints:
        print(f"No checkpoints found in {output_dir}")
        return 0
    print(f"\nCheckpoints in {output_dir}:")
    print("-" * 65)
    for sid in checkpoints:
        s = manager.load_checkpoint(sid)
        if s:
            print(f"{sid:<30} {s.status:<10} {s.progress.completed_interactions}/{s.progress.total_interactions}")
    return 0


def generate_report(session_id: str, output_dir: Path) -> int:
    runner = BatchRunner(BatchConfig(output_dir=output_dir), OrchestratorFactory())
    path = runner.generate_partial_report(session_id)
    if path:
        print(f"Report generated: {path}")
        return 0
    print(f"Error: No checkpoint found for '{session_id}'")
    return 1


def cleanup_session(session_id: str, output_dir: Path) -> int:
    manager = CheckpointManager(output_dir)
    state = manager.load_checkpoint(session_id)
    if not state:
        print(f"No checkpoint found for '{session_id}'")
        return 1
    print(f"Cleaning up: {session_id} ({state.progress.completed_batches} batches)")
    manager.cleanup(session_id, keep_checkpoint=False)
    return 0


def run_evaluation(args: argparse.Namespace) -> int:
    if not args.dataset:
        print("Error: --dataset is required")
        return 1
    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        return 1

    # Initialize LLM-as-Judge if requested
    llm_judge = None
    if args.use_llm_judge:
        print(f"Initializing LLM-as-Judge at {args.judge_api_base}")
        llm_judge = LLMJudge(
            api_base=args.judge_api_base,
            model=args.judge_model,
            timeout=args.judge_timeout,
        )
        if not llm_judge.health_check():
            print(f"Error: LLM-as-Judge server not accessible at {args.judge_api_base}")
            return 1
        print("LLM-as-Judge initialized successfully")

    config = BatchConfig(
        batch_size=args.batch_size if args.batch_size > 0 else 10,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        resume_from_checkpoint=not args.no_resume,
        use_mock_llm=args.mock,
    )

    # Create evaluator factory with LLM judge support
    def evaluator_factory(orch):
        return Evaluator(orch, llm_judge=llm_judge, use_llm_judge=args.use_llm_judge)

    runner = BatchRunner(config, OrchestratorFactory(), evaluator_factory=evaluator_factory)

    print(f"\nEvaluation: {args.dataset} | mode={args.mode} | batch_size={args.batch_size or 'none'}")
    try:
        summary = runner.run(args.dataset, args.mode, args.max_interactions, args.max_batches, args.session_id)
        print("\n" + "=" * 50 + "\nCOMPLETE")
        print(f"Queries: {summary.total_queries} | Correct: {summary.correct} | Accuracy: {summary.accuracy:.2%}")
        print("=" * 50)
        return 0
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume: python -m evaluation.cli --resume {args.session_id or '<id>'}")
        return 130
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return 1


def resume_evaluation(session_id: str, args: argparse.Namespace) -> int:
    state = CheckpointManager(args.output_dir).load_checkpoint(session_id)
    if not state:
        print(f"Error: No checkpoint found for '{session_id}'")
        return 1

    print(f"\nResuming: {session_id} ({state.progress.completed_interactions}/{state.progress.total_interactions})")

    # Initialize LLM-as-Judge if requested
    llm_judge = None
    if args.use_llm_judge:
        print(f"Initializing LLM-as-Judge at {args.judge_api_base}")
        llm_judge = LLMJudge(
            api_base=args.judge_api_base,
            model=args.judge_model,
            timeout=args.judge_timeout,
        )
        if not llm_judge.health_check():
            print(f"Error: LLM-as-Judge server not accessible at {args.judge_api_base}")
            return 1

    config = BatchConfig(batch_size=state.config.get("batch_size", 10),
        checkpoint_interval=args.checkpoint_interval, output_dir=args.output_dir, resume_from_checkpoint=True)

    def evaluator_factory(orch):
        return Evaluator(orch, llm_judge=llm_judge, use_llm_judge=args.use_llm_judge)

    runner = BatchRunner(config, OrchestratorFactory(), evaluator_factory=evaluator_factory)
    dataset_path = Path(state.config.get("dataset", ""))
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return 1

    try:
        summary = runner.run(dataset_path, state.config.get("mode", "full"),
            state.config.get("max_interactions", 0), args.max_batches, session_id)
        print("\n" + "=" * 50 + f"\nCOMPLETE | Accuracy: {summary.accuracy:.2%}")
        return 0
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume: python -m evaluation.cli --resume {session_id}")
        return 130
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return 1


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if args.list_checkpoints:
        return list_checkpoints(args.output_dir)
    if args.cleanup:
        return cleanup_session(args.cleanup, args.output_dir)
    if args.report:
        return generate_report(args.report, args.output_dir)
    if args.resume:
        return resume_evaluation(args.resume, args)
    return run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())