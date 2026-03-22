"""
evaluation/run_batch.py
───────────────────────
CLI entry point for batch evaluation with checkpoint persistence.

DEPRECATED: This module is deprecated. Use evaluation.runner.BatchRunner instead.
  - BatchEvaluationRunner -> evaluation.runner.BatchRunner
  - BatchConfig -> evaluation.runner.BatchConfig

Usage:
    # Run evaluation in batches of 10
    python evaluation/run_batch.py --dataset data.json --batch-size 10

    # Resume interrupted evaluation
    python evaluation/run_batch.py --resume eval_20260322_120000

    # Run with specific session ID (enables resume later)
    python evaluation/run_batch.py --dataset data.json --session-id my_eval_001

    # Generate report from partial results
    python evaluation/run_batch.py --report eval_20260322_120000
"""

from __future__ import annotations

import warnings

warnings.warn(
    "evaluation.run_batch is deprecated. Use evaluation.runner.BatchRunner instead.\n"
    "  - BatchEvaluationRunner -> BatchRunner\n"
    "  - BatchConfig -> evaluation.runner.BatchConfig",
    DeprecationWarning,
    stacklevel=2
)

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.batch_runner import BatchEvaluationRunner, BatchConfig
from evaluation.batch_checkpoint import CheckpointManager
from evaluation.factory import OrchestratorFactory

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
        description="AgeMem Batch Evaluation with Checkpoint Persistence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New evaluation with batching
  python evaluation/run_batch.py --dataset evaluation/data/longmemeval_s_cleaned.json

  # Specify batch size and max interactions
  python evaluation/run_batch.py --dataset data.json --batch-size 10 --max-interactions 100

  # Resume from checkpoint
  python evaluation/run_batch.py --resume eval_20260322_120000

  # Generate partial report
  python evaluation/run_batch.py --report eval_20260322_120000

  # List available checkpoints
  python evaluation/run_batch.py --list-checkpoints
""",
    )

    # Main modes
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to evaluation dataset (JSON format)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="SESSION_ID",
        help="Resume evaluation from checkpoint with given session ID",
    )
    parser.add_argument(
        "--report",
        type=str,
        metavar="SESSION_ID",
        help="Generate report from partial results for given session ID",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List all available checkpoints",
    )

    # Evaluation configuration
    parser.add_argument(
        "--session-id",
        type=str,
        help="Session ID for this evaluation (allows resuming later)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of interactions per batch (default: 10)",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=0,
        help="Maximum interactions to process (0 = all, default: 0)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Maximum batches to process (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "lifecycle", "retrieval"],
        default="full",
        help="Evaluation mode (default: full)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Output directory for results (default: evaluation/results)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save checkpoint every N batches (default: 1)",
    )

    # Utility flags
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing checkpoint (start fresh)",
    )
    parser.add_argument(
        "--cleanup",
        type=str,
        metavar="SESSION_ID",
        help="Clean up checkpoint and batch files for given session ID",
    )

    return parser.parse_args()


def list_checkpoints(output_dir: Path) -> int:
    """List all available checkpoints."""
    manager = CheckpointManager(output_dir)
    checkpoints = manager.list_checkpoints()

    if not checkpoints:
        print(f"No checkpoints found in {output_dir}")
        return 0

    print(f"\nAvailable checkpoints in {output_dir}:")
    print("-" * 80)
    print(f"{'Session ID':<40} {'Status':<12} {'Progress':<15}")
    print("-" * 80)

    for session_id in checkpoints:
        state = manager.load_checkpoint(session_id)
        if state:
            progress = state.progress
            status = state.status
            progress_str = f"{progress.completed_interactions}/{progress.total_interactions}"
            print(f"{session_id:<40} {status:<12} {progress_str:<15}")

    print("-" * 80)
    print(f"\nTotal checkpoints: {len(checkpoints)}")
    print(f"\nTo resume: python evaluation/run_batch.py --resume <session_id>")
    print(f"To report: python evaluation/run_batch.py --report <session_id>")

    return 0


def generate_report(session_id: str, output_dir: Path) -> int:
    """Generate report from partial results."""
    manager = CheckpointManager(output_dir)
    state = manager.load_checkpoint(session_id)

    if not state:
        print(f"Error: No checkpoint found for session '{session_id}'")
        print(f"       Checked: {output_dir / (session_id + '_checkpoint.json')}")
        return 1

    print(f"\nGenerating report for session: {session_id}")
    print(f"Status: {state.status}")
    print(f"Progress: {state.progress.completed_interactions}/{state.progress.total_interactions}")
    print(f"Batches: {state.progress.completed_batches}")

    # Create runner and generate report
    config = BatchConfig(output_dir=output_dir)
    runner = BatchEvaluationRunner(config, OrchestratorFactory())

    report_path = runner.generate_partial_report(session_id)

    if report_path:
        print(f"\nReport generated: {report_path}")
        return 0
    else:
        print("Error: Failed to generate report")
        return 1


def cleanup_session(session_id: str, output_dir: Path) -> int:
    """Clean up checkpoint and batch files."""
    manager = CheckpointManager(output_dir)

    # Check if checkpoint exists
    state = manager.load_checkpoint(session_id)
    if not state:
        print(f"No checkpoint found for session '{session_id}'")
        return 1

    print(f"Cleaning up session: {session_id}")
    print(f"Status: {state.status}")
    print(f"Batches to remove: {state.progress.completed_batches}")

    # Clean up
    manager.cleanup(session_id, keep_checkpoint=False)

    print(f"Cleanup complete. Removed checkpoint and {state.progress.completed_batches} batch files.")
    return 0


def run_evaluation(args: argparse.Namespace) -> int:
    """Run batch evaluation."""
    if not args.dataset:
        print("Error: --dataset is required for new evaluations")
        return 1

    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        return 1

    # Create configuration
    config = BatchConfig(
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        resume_from_checkpoint=not args.no_resume,
    )

    # Create runner
    factory = OrchestratorFactory()
    runner = BatchEvaluationRunner(config, factory)

    # Run evaluation
    print(f"\nStarting batch evaluation:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Mode: {args.mode}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max interactions: {args.max_interactions if args.max_interactions > 0 else 'all'}")
    print(f"  Output: {args.output_dir}")
    if args.session_id:
        print(f"  Session ID: {args.session_id}")
    print()

    try:
        summary = runner.run(
            dataset_path=args.dataset,
            mode=args.mode,
            max_interactions=args.max_interactions,
            max_batches=args.max_batches,
            session_id=args.session_id,
        )

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Total queries: {summary.total_queries}")
        print(f"Correct: {summary.correct}")
        print(f"Accuracy: {summary.accuracy:.2%}")
        print(f"Average latency: {summary.avg_latency_ms:.1f}ms")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        print(f"Progress has been saved. To resume, run:")
        print(f"  python evaluation/run_batch.py --resume {args.session_id or '<session_id>'}")
        return 130

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        print(f"\nError: {e}")
        print(f"Progress has been saved. To resume, run:")
        print(f"  python evaluation/run_batch.py --resume {args.session_id or '<session_id>'}")
        return 1


def resume_evaluation(session_id: str, args: argparse.Namespace) -> int:
    """Resume evaluation from checkpoint."""
    output_dir = args.output_dir
    manager = CheckpointManager(output_dir)

    # Load checkpoint
    state = manager.load_checkpoint(session_id)
    if not state:
        print(f"Error: No checkpoint found for session '{session_id}'")
        print(f"       Checked: {output_dir / (session_id + '_checkpoint.json')}")
        print(f"\nTo list available checkpoints:")
        print(f"  python evaluation/run_batch.py --list-checkpoints")
        return 1

    print(f"\nResuming evaluation:")
    print(f"  Session ID: {session_id}")
    print(f"  Status: {state.status}")
    print(f"  Progress: {state.progress.completed_interactions}/{state.progress.total_interactions}")
    print(f"  Batches completed: {state.progress.completed_batches}")
    print()

    # Create configuration from checkpoint
    config = BatchConfig(
        batch_size=state.config.get("batch_size", 10),
        checkpoint_interval=args.checkpoint_interval,
        output_dir=output_dir,
        resume_from_checkpoint=True,
    )

    # Create runner and resume
    factory = OrchestratorFactory()
    runner = BatchEvaluationRunner(config, factory)

    # Get dataset path from checkpoint
    dataset_path = Path(state.config.get("dataset", "evaluation/data/longmemeval_s_cleaned.json"))
    if not dataset_path.exists():
        print(f"Error: Original dataset not found: {dataset_path}")
        print(f"       Please specify with --dataset")
        return 1

    try:
        summary = runner.run(
            dataset_path=dataset_path,
            mode=state.config.get("mode", "full"),
            max_interactions=state.config.get("max_interactions", 0),
            max_batches=args.max_batches,
            session_id=session_id,
        )

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE (Resumed)")
        print("=" * 60)
        print(f"Total queries: {summary.total_queries}")
        print(f"Correct: {summary.correct}")
        print(f"Accuracy: {summary.accuracy:.2%}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        print(f"Progress has been saved. To resume, run:")
        print(f"  python evaluation/run_batch.py --resume {session_id}")
        return 130

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        print(f"\nError: {e}")
        print(f"Progress has been saved. To resume, run:")
        print(f"  python evaluation/run_batch.py --resume {session_id}")
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Handle utility commands first
    if args.list_checkpoints:
        return list_checkpoints(args.output_dir)

    if args.cleanup:
        return cleanup_session(args.cleanup, args.output_dir)

    if args.report:
        return generate_report(args.report, args.output_dir)

    if args.resume:
        return resume_evaluation(args.resume, args)

    # Run new evaluation
    return run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())


# Re-exports from new modules for backward compatibility
# CLI functions are defined locally above; new code should use:
#   - evaluation.runner.BatchRunner for programmatic access
#   - evaluation.checkpoint.CheckpointManager for checkpoint operations
__all__ = [
    "setup_logging",
    "parse_args",
    "list_checkpoints",
    "generate_report",
    "cleanup_session",
    "run_evaluation",
    "resume_evaluation",
    "main",
]
