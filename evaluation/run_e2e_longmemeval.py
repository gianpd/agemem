"""
evaluation/run_e2e_longmemeval.py
─────────────────────────────────
End-to-End LongMemEval runner with production-identical configuration.

This script runs the AgeMem system against the LongMemEval benchmark
using the EXACT same configuration as main.py (no batch isolation).

Features:
- Uses production Orchestrator configuration (same as main.py)
- Incremental JSONL logging after each interaction
- Crash-resistant: preserves all completed interactions
- Outputs session history for LLM-as-judge analysis

Usage:
    python -m evaluation.run_e2e_longmemeval
    python -m evaluation.run_e2e_longmemeval --limit 1
    python -m evaluation.run_e2e_longmemeval --resume session_20260323.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from core.tracing import init_tracing, get_tracer, shutdown_tracing
from core.llm_factory import LLMClientFactory
from core.config import AgememConfig
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator

# Import tools from main.py configuration
from tools.corpus import tool_definitions as CORPUS_TOOL_DEFINITIONS
from tools.web_tools import tool_definitions as WEB_TOOL_DEFINITIONS
from memory import introspection_tool_definitions

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (Mirrors main.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings

def get_env_with_fallback(new_name: str, old_name: str, default: str) -> str:
    """Get env var with fallback to old name for backward compatibility."""
    value = os.getenv(new_name) or os.getenv(old_name, default)
    if os.getenv(old_name) and not os.getenv(new_name):
        warnings.warn(
            f"Environment variable '{old_name}' is deprecated. Use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
    return value

# Environment configuration (same as main.py)
BASE_URL = get_env_with_fallback("BASE_URL", "LLAMA_HOST", "http://localhost:8080")
BASE_MODEL = get_env_with_fallback("BASE_MODEL", "LLAMA_MODEL", "Qwen3.5-9B-UD-Q4_K_XL.gguf")
BASE_MAX_TOKENS = int(get_env_with_fallback("BASE_MAX_TOKENS", "LLAMA_MAX_TOKENS", "2048"))
BASE_TEMPERATURE = float(get_env_with_fallback("BASE_TEMPERATURE", "LLAMA_TEMPERATURE", "0.2"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "4000"))
PERSIST_DIR = os.getenv("PERSIST_DIR", "agent_memory")
STM_TOKEN_LIMIT = int(os.getenv("STM_TOKEN_LIMIT", "6000"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
TRACE_LOG_DIR = os.getenv("TRACE_LOG_DIR", "logs")
TRACE_RETENTION_DAYS = int(os.getenv("TRACE_RETENTION_DAYS", "30"))


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InteractionRecord:
    """Single interaction record for JSONL output."""
    interaction_id: int
    timestamp: str
    question_id: str
    question_type: str
    user_input: str
    expected_answer: str
    agent_response: str
    latency_ms: float
    stm_tokens: int
    stm_utilization: float
    ltm_entries: int
    tool_calls: list[dict] = field(default_factory=list)
    memory_ops: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionMetadata:
    """Session-level metadata."""
    session_id: str
    started_at: str
    completed_at: Optional[str]
    total_interactions: int
    completed_interactions: int
    failed_interactions: int
    config: dict
    status: str  # "running", "completed", "crashed", "interrupted"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class E2EConfig:
    """Configuration for E2E evaluation."""
    dataset_path: Path
    output_dir: Path
    limit: int  # Number of instances (0 = all)
    target_messages: int  # Target message count (0 = all)
    persist_dir: Path  # Where to store LTM/STM
    resume_from: Optional[Path] = None


# ─────────────────────────────────────────────────────────────────────────────
# Incremental Logger (Crash-Safe)
# ─────────────────────────────────────────────────────────────────────────────

class IncrementalSessionLogger:
    """
    Crash-safe incremental logger that writes after each interaction.

    Uses JSONL format (one JSON object per line) for:
    - Easy appending
    - Crash recovery (partial file is still valid)
    - Streaming processing for LLM-as-judge
    """

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.metadata_path = output_path.with_suffix(".metadata.json")
        self.interactions: list[InteractionRecord] = []
        self.metadata: Optional[SessionMetadata] = None

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize_session(self, config: E2EConfig, total_interactions: int) -> str:
        """Initialize a new session and write initial metadata."""
        session_id = datetime.now().strftime("e2e_%Y%m%d_%H%M%S")

        self.metadata = SessionMetadata(
            session_id=session_id,
            started_at=datetime.now().isoformat(),
            completed_at=None,
            total_interactions=total_interactions,
            completed_interactions=0,
            failed_interactions=0,
            config={
                "dataset_path": str(config.dataset_path),
                "limit": config.limit,
                "target_messages": config.target_messages,
                "persist_dir": str(config.persist_dir),
                "base_model": BASE_MODEL,
                "base_url": BASE_URL,
                "stm_token_limit": STM_TOKEN_LIMIT,
            },
            status="running",
        )

        self._write_metadata()
        return session_id

    def log_interaction(self, record: InteractionRecord) -> None:
        """Write interaction immediately to JSONL file."""
        self.interactions.append(record)

        # Append to JSONL file immediately (crash-safe)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            f.flush()  # Force write to disk

        # Update metadata
        if self.metadata:
            self.metadata.completed_interactions = len(self.interactions)
            if record.error:
                self.metadata.failed_interactions += 1
            self._write_metadata()

    def finalize_session(self, status: str = "completed") -> None:
        """Mark session as complete."""
        if self.metadata:
            self.metadata.completed_at = datetime.now().isoformat()
            self.metadata.status = status
            self._write_metadata()

    def _write_metadata(self) -> None:
        """Write metadata file (overwrites each time)."""
        if self.metadata:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata.to_dict(), f, indent=2, ensure_ascii=False)

    def load_existing(self) -> int:
        """Load existing interactions from JSONL file. Returns count."""
        if not self.output_path.exists():
            return 0

        count = 0
        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self.interactions.append(InteractionRecord(**data))
                        count += 1
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line: {line[:100]}")

        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
                self.metadata = SessionMetadata(**meta_data)
                self.metadata.status = "resuming"

        return count


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Builder (Exact copy of main.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_orchestrator(persist_dir: Optional[Path] = None) -> Orchestrator:
    """
    Build Orchestrator with EXACT configuration from main.py.

    This ensures the evaluation runs with the same memory stack,
    learning scorer, and tool configuration as production.

    Args:
        persist_dir: Optional override for persistence directory.
                    If None, uses PERSIST_DIR from environment.
    """
    # Main LLM - from environment (can be local or external)
    factory = LLMClientFactory()
    llm = factory.create()

    # Learning Scorer LLM - always OpenRouter (if enabled and API key available)
    learning_scorer_llm: Optional[LLMClient] = None
    try:
        learning_factory = LLMClientFactory.for_learning_scorer()
        learning_scorer_llm = learning_factory.create()
        print(f"[INFO] Learning scorer using OpenRouter with model: {learning_factory.config.model}")
    except ValueError as e:
        print(f"[WARNING] Learning scorer disabled: {e}")
        print("[WARNING] Set OPENROUTER_API_KEY environment variable to enable learning feedback.")

    # Build config (same as main.py)
    config = AgememConfig(
        DEFAULT_MODEL=factory.config.model,
        MEMORY_AGENT_MODEL=factory.config.model,
        STM_TOKEN_LIMIT=STM_TOKEN_LIMIT,
        STM_WARNING_THRESHOLD=0.75,
        STM_CRITICAL_THRESHOLD=0.90,
        LTM_PROMOTE_THRESHOLD=0.65,
        LEARNING_SCORE_PROMPT_EVERY_N=5,
        TRIGGER_EVERY_N_TURNS=10,
        DEFAULT_MAX_TOKENS=factory.config.max_tokens,
        DEFAULT_TEMPERATURE=factory.config.temperature,
        PERSIST_DIR=str(persist_dir) if persist_dir else PERSIST_DIR,
    )

    # Build orchestrator (same as main.py)
    orch = Orchestrator(
        llm=llm,
        config=config,
        learning_scorer_llm=learning_scorer_llm,
    )

    # Set up tools - combine all tool definitions (same as main.py)
    all_tools = (
        # WEB_TOOL_DEFINITIONS +
        CORPUS_TOOL_DEFINITIONS +
        introspection_tool_definitions
    )
    orch.set_tools(all_tools)

    return orch


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_longmemeval_dataset(
    dataset_path: Path,
    target_messages: int = 0,
    limit: int = 0,
) -> list[dict]:
    """
    Load LongMemEval dataset.

    Args:
        dataset_path: Path to the JSON dataset
        target_messages: Target total messages (0 = all).
                        Stops when cumulative turn count reaches this.
        limit: Max instances to load (0 = all)

    Returns:
        List of instance dicts with metadata
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"[INFO] Loaded {len(raw_data)} instances from {dataset_path}")

    # Apply instance limit
    if limit > 0:
        raw_data = raw_data[:limit]
        print(f"[INFO] Limited to {len(raw_data)} instances")

    # Calculate total turns and apply message limit
    instances = []
    total_turns = 0

    for instance in raw_data:
        sessions = instance.get("haystack_sessions", [])
        instance_turns = sum(len(s) for s in sessions)

        instances.append({
            "question_id": instance.get("question_id", "unknown"),
            "question_type": instance.get("question_type", "unknown"),
            "question": instance.get("question", ""),
            "answer": instance.get("answer", ""),
            "sessions": sessions,
            "session_ids": instance.get("haystack_session_ids", []),
            "turn_count": instance_turns,
        })

        total_turns += instance_turns

        if target_messages > 0 and total_turns >= target_messages:
            print(f"[INFO] Reached target of ~{target_messages} messages after {len(instances)} instances")
            break

    print(f"[INFO] Total instances: {len(instances)}, Total turns: {total_turns}")
    return instances


# ─────────────────────────────────────────────────────────────────────────────
# E2E Runner
# ─────────────────────────────────────────────────────────────────────────────

class E2ERunner:
    """
    End-to-End evaluation runner with crash recovery.

    Key features:
    - Single orchestrator instance (like real user sessions)
    - Incremental logging after each interaction
    - Graceful shutdown on SIGINT/SIGTERM
    - Resume from last completed interaction
    """

    def __init__(self, config: E2EConfig):
        self.config = config
        self.orchestrator: Optional[Orchestrator] = None
        self.logger = IncrementalSessionLogger(
            config.output_dir / "session.jsonl"
        )
        self.shutdown_requested = False
        self.interaction_count = 0

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        print(f"\n[INFO] Shutdown requested (signal {signum})")
        self.shutdown_requested = True

    def run(self) -> SessionMetadata:
        """Run the E2E evaluation."""
        # Load dataset
        instances = load_longmemeval_dataset(
            self.config.dataset_path,
            target_messages=self.config.target_messages,
            limit=self.config.limit,
        )

        # Check for resume
        start_idx = 0
        if self.config.resume_from:
            existing_count = self.logger.load_existing()
            if existing_count > 0:
                start_idx = existing_count
                print(f"[INFO] Resuming from interaction {start_idx}")
            else:
                print(f"[WARNING] Resume file not found or empty: {self.config.resume_from}")

        # Initialize session
        session_id = self.logger.initialize_session(
            self.config,
            total_interactions=len(instances),
        )
        print(f"[INFO] Session ID: {session_id}")

        # Build orchestrator (production config)
        print(f"[INFO] Building orchestrator with persist_dir={self.config.persist_dir}")
        self.orchestrator = build_orchestrator(self.config.persist_dir)

        print(f"[INFO] Initial LTM entries: {len(self.orchestrator.ltm_snapshot())}")
        print(f"[INFO] Initial STM tokens: {self.orchestrator.stm_stats().total_tokens}")

        try:
            # Process each instance
            for idx, instance in enumerate(instances):
                if self.shutdown_requested:
                    print(f"\n[INFO] Shutdown requested, stopping at instance {idx}")
                    break

                if idx < start_idx:
                    continue  # Skip already processed

                self._process_instance(idx, instance)

            # Finalize
            status = "interrupted" if self.shutdown_requested else "completed"
            self.logger.finalize_session(status)

        except Exception as e:
            print(f"\n[ERROR] Fatal error: {e}")
            traceback.print_exc()
            self.logger.finalize_session("crashed")
            raise

        return self.logger.metadata

    def _process_instance(self, idx: int, instance: dict) -> None:
        """
        Process a single instance: replay sessions and evaluate question.

        This mirrors how a real user would interact:
        1. User has conversations (sessions) with the assistant
        2. Later, user asks a question that requires remembering
        """
        question_id = instance["question_id"]
        question_type = instance["question_type"]
        question = instance["question"]
        expected_answer = instance["answer"]
        sessions = instance["sessions"]

        print(f"\n[INFO] Instance {idx + 1}: {question_id} ({question_type})")
        print(f"       Question: {question[:80]}{'...' if len(question) > 80 else ''}")
        print(f"       Sessions: {len(sessions)}, Turns: {instance['turn_count']}")

        # Track turn-level interactions
        turn_records = []
        instance_start_time = time.time()

        try:
            # Phase 1: Replay sessions (build up memory)
            for session_idx, session in enumerate(sessions):
                if self.shutdown_requested:
                    break

                for turn_idx, turn in enumerate(session):
                    if self.shutdown_requested:
                        break

                    role = turn.get("role", "user")
                    content = turn.get("content", "")

                    if role != "user" or not content:
                        continue  # Skip assistant turns and empty content

                    # Process turn through orchestrator
                    turn_start = time.time()
                    time.sleep(1)
                    try:
                        response = self.orchestrator.chat(content)
                        turn_latency = (time.time() - turn_start) * 1000

                        # Get memory state after turn
                        stm_stats = self.orchestrator.stm_stats()
                        trace = self.orchestrator.last_trace()

                        turn_record = InteractionRecord(
                            interaction_id=self.interaction_count,
                            timestamp=datetime.now().isoformat(),
                            question_id=question_id,
                            question_type=f"session_turn_{session_idx}_{turn_idx}",
                            user_input=content,
                            expected_answer="",  # No expected answer for session turns
                            agent_response=response,
                            latency_ms=turn_latency,
                            stm_tokens=stm_stats.total_tokens,
                            stm_utilization=stm_stats.utilisation_ratio,
                            ltm_entries=len(self.orchestrator.ltm_snapshot()),
                            tool_calls=[{"name": tc.name, "arguments": tc.arguments}
                                       for tc in trace.tool_calls] if trace else [],
                            memory_ops=[{"op": op.op.value, "detail": op.detail}
                                       for op in trace.ops_applied] if trace else [],
                        )

                        # Log immediately (crash-safe)
                        self.logger.log_interaction(turn_record)
                        self.interaction_count += 1

                        # Progress indicator
                        print(f"         Turn {turn_idx + 1}: {turn_latency:.0f}ms, "
                              f"STM={stm_stats.total_tokens}tok, "
                              f"LTM={len(self.orchestrator.ltm_snapshot())}")

                    except Exception as e:
                        print(f"         [ERROR] Turn {turn_idx + 1} failed: {e}")
                        # Log error but continue
                        error_record = InteractionRecord(
                            interaction_id=self.interaction_count,
                            timestamp=datetime.now().isoformat(),
                            question_id=question_id,
                            question_type=f"session_turn_{session_idx}_{turn_idx}",
                            user_input=content,
                            expected_answer="",
                            agent_response="",
                            latency_ms=0,
                            stm_tokens=0,
                            stm_utilization=0,
                            ltm_entries=0,
                            error=str(e),
                        )
                        self.logger.log_interaction(error_record)
                        self.interaction_count += 1

            # Phase 2: Evaluate the question
            if not self.shutdown_requested and question:
                print(f"       Evaluating question...")
                question_start = time.time()

                try:
                    response = self.orchestrator.chat(question)
                    question_latency = (time.time() - question_start) * 1000

                    stm_stats = self.orchestrator.stm_stats()
                    trace = self.orchestrator.last_trace()

                    question_record = InteractionRecord(
                        interaction_id=self.interaction_count,
                        timestamp=datetime.now().isoformat(),
                        question_id=question_id,
                        question_type=question_type,
                        user_input=question,
                        expected_answer=expected_answer,
                        agent_response=response,
                        latency_ms=question_latency,
                        stm_tokens=stm_stats.total_tokens,
                        stm_utilization=stm_stats.utilisation_ratio,
                        ltm_entries=len(self.orchestrator.ltm_snapshot()),
                        tool_calls=[{"name": tc.name, "arguments": tc.arguments}
                                   for tc in trace.tool_calls] if trace else [],
                        memory_ops=[{"op": op.op.value, "detail": op.detail}
                                   for op in trace.ops_applied] if trace else [],
                    )

                    self.logger.log_interaction(question_record)
                    self.interaction_count += 1

                    # Print summary
                    print(f"       Response: {response[:100]}{'...' if len(response) > 100 else ''}")
                    print(f"       Expected: {expected_answer[:100]}{'...' if len(expected_answer) > 100 else ''}")

                except Exception as e:
                    print(f"       [ERROR] Question evaluation failed: {e}")
                    error_record = InteractionRecord(
                        interaction_id=self.interaction_count,
                        timestamp=datetime.now().isoformat(),
                        question_id=question_id,
                        question_type=question_type,
                        user_input=question,
                        expected_answer=expected_answer,
                        agent_response="",
                        latency_ms=0,
                        stm_tokens=0,
                        stm_utilization=0,
                        ltm_entries=0,
                        error=str(e),
                    )
                    self.logger.log_interaction(error_record)
                    self.interaction_count += 1

            # Instance summary
            instance_latency = (time.time() - instance_start_time) * 1000
            print(f"       Instance complete: {instance_latency / 1000:.1f}s")

            # Reset memory for next instance (benchmark isolation)
            # Each LongMemEval instance represents an independent user/session
            self.orchestrator.reset_stm()
            self.orchestrator.clear_ltm()
            print(f"       Memory reset (STM cleared, LTM wiped)")

        except Exception as e:
            print(f"       [ERROR] Instance failed: {e}")
            traceback.print_exc()

            # Reset memory even on failure to ensure clean state for next instance
            self.orchestrator.reset_stm()
            self.orchestrator.clear_ltm()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-End LongMemEval runner with production configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run single instance (~550 messages)
    python -m evaluation.run_e2e_longmemeval --limit 1

    # Run with custom message target
    python -m evaluation.run_e2e_longmemeval --target-messages 1000

    # Resume interrupted session
    python -m evaluation.run_e2e_longmemeval --resume evaluation/results/session.jsonl
        """,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/data/longmemeval_s_cleaned.json"),
        help="Path to LongMemEval dataset (default: evaluation/data/longmemeval_s_cleaned.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of instances to process (default: 1, ~550 messages)",
    )
    parser.add_argument(
        "--target-messages",
        type=int,
        default=0,
        help="Target message count (0 = all in limited instances)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Output directory for session logs (default: evaluation/results)",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        help="Override persistence directory (default: temp dir)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from existing session JSONL file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Initialize tracing (same as main.py)
    init_tracing(
        log_dir=TRACE_LOG_DIR,
        debug=DEBUG_MODE,
        retention_days=TRACE_RETENTION_DAYS,
    )

    # Create temp persist dir if not specified
    import tempfile
    if args.persist_dir:
        persist_dir = args.persist_dir
    else:
        persist_dir = Path(tempfile.mkdtemp(prefix="agemem_e2e_"))
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    config = E2EConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        limit=args.limit,
        target_messages=args.target_messages,
        persist_dir=persist_dir,
        resume_from=args.resume,
    )

    print("=" * 60)
    print("AgeMem E2E LongMemEval Runner")
    print("=" * 60)
    print(f"Dataset:       {config.dataset_path}")
    print(f"Limit:         {config.limit} instances")
    print(f"Target msgs:   {config.target_messages or 'all'}")
    print(f"Output dir:    {config.output_dir}")
    print(f"Persist dir:   {config.persist_dir}")
    print(f"Model:         {BASE_MODEL} @ {BASE_URL}")
    print(f"STM limit:     {STM_TOKEN_LIMIT} tokens")
    print("=" * 60)

    # Run evaluation
    try:
        runner = E2ERunner(config)
        metadata = runner.run()

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Session ID:     {metadata.session_id}")
        print(f"Status:         {metadata.status}")
        print(f"Interactions:   {metadata.completed_interactions}")
        print(f"Failed:         {metadata.failed_interactions}")
        print(f"Output:         {config.output_dir / 'session.jsonl'}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        return 130

    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        traceback.print_exc()
        return 1

    finally:
        shutdown_tracing()


if __name__ == "__main__":
    sys.exit(main())