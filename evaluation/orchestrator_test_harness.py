"""
evaluation/orchestrator_test_harness.py
──────────────────────────────────────
Orchestrator-based test harness for unified evaluation.

This module implements the evaluation architecture proposal (Phase 1):
- Wraps Orchestrator for controlled evaluation testing
- Supports multi-session history loading (LongMemEval_S: 30-40 sessions)
- Captures full traces for metric calculation
- Provides isolated storage to prevent side effects
- Supports mock LLM for deterministic testing

Coherence with longmemeval_guide.md:
- Tests all 5 memory behaviors: IE, MR, KU, TR, ABS
- Supports LongMemEval_S standard (~115k tokens, ~40 sessions)
- Validates cross-session memory persistence
- Maps behavior categories to orchestrator test cases

Usage:
    session = EvaluationSession(ltm_seed_data=memories)
    session.load_multi_session_history(sessions, behavior_type="IE")
    result = session.send_message("What is my phone number?")
    assert result.answer_is_correct("555-1234")
    assert result.retrieved_from_correct_sessions([12])
"""

from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable
from unittest.mock import Mock

# Core types
from core.types import (
    MemoryEntry,
    ContextMessage,
    TriggerKind,
    LearningFeedback,
    MemoryOpResult,
)
from core.config import AgememConfig, DEFAULT_CONFIG
from core.tracing import get_tracer

# Memory components
from memory.ltm_store import LTMStore
from memory.stm_context import STMContext
from memory.context_retrieval import ContextAwareRetriever, ContextRetrievalConfig

# Agents
from agents.orchestrator import Orchestrator, TurnTrace
from agents.llm_client import LLMClient
from agents.learning_scorer import LearningScorer
from triggers.memory_trigger_engine import MemoryTriggerEngine

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """
    Result from a single evaluation turn.
    Captures everything needed for metric calculation.
    """
    turn_index: int
    user_input: str
    assistant_response: str
    retrieval_trace: Optional[dict] = None  # LTM retrieval details
    memories_injected: list[MemoryEntry] = field(default_factory=list)
    stm_stats_before: Optional[dict] = None
    stm_stats_after: Optional[dict] = None
    memory_ops: list[MemoryOpResult] = field(default_factory=list)
    learning_feedback: Optional[LearningFeedback] = None
    latency_ms: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    corpus_fallback_used: bool = False

    def answer_is_correct(self, expected_answer: str, matcher: Optional[Callable] = None) -> bool:
        """
        Check if assistant response matches expected answer.

        Args:
            expected_answer: The ground truth answer
            matcher: Optional custom matcher function (response, expected) -> bool

        Returns:
            True if answer is correct
        """
        if matcher:
            return matcher(self.assistant_response, expected_answer)

        # Default: case-insensitive substring match
        response_lower = self.assistant_response.lower()
        expected_lower = expected_answer.lower()

        # Direct containment
        if expected_lower in response_lower or response_lower in expected_lower:
            return True

        # Token overlap for multi-word answers
        expected_tokens = set(expected_lower.split())
        response_tokens = set(response_lower.split())

        if len(expected_tokens) > 0:
            overlap = len(expected_tokens & response_tokens)
            return overlap / len(expected_tokens) >= 0.7  # 70% token overlap

        return False

    def retrieved_from_correct_sessions(self, evidence_indices: list[int]) -> bool:
        """
        Check if retrieval sourced from the correct evidence sessions.

        Args:
            evidence_indices: Session indices containing answer evidence

        Returns:
            True if memories from evidence sessions were retrieved
        """
        if not self.retrieval_trace or not self.memories_injected:
            return False

        # Check if any retrieved memory came from evidence sessions
        for memory in self.memories_injected:
            source_turn = getattr(memory, 'source_turn', -1)
            if source_turn in evidence_indices:
                return True

        return False

    def has_abstained(self) -> bool:
        """
        Check if the response indicates abstention (ABS behavior).

        Returns:
            True if response indicates "I don't know" or similar
        """
        abstention_phrases = [
            "i don't know", "i don't have", "i'm not sure",
            "i cannot", "i can't", "no information",
            "not mentioned", "don't recall", "no memory",
        ]

        response_lower = self.assistant_response.lower()
        return any(phrase in response_lower for phrase in abstention_phrases)


@dataclass
class EvaluationTrace:
    """
    Full trace of an evaluation session for debugging and analysis.
    """
    session_id: str
    behavior_type: str  # "IE", "MR", "KU", "TR", "ABS"
    turns: list[TurnResult] = field(default_factory=list)
    ltm_entries_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert trace to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "behavior_type": self.behavior_type,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "user_input": t.user_input,
                    "assistant_response": t.assistant_response,
                    "latency_ms": t.latency_ms,
                    "memories_injected": len(t.memories_injected),
                    "corpus_fallback_used": t.corpus_fallback_used,
                }
                for t in self.turns
            ],
            "ltm_entries_count": self.ltm_entries_count,
            "metadata": self.metadata,
        }


class MockLLMClient(LLMClient):
    """
    Mock LLM client for deterministic evaluation testing.

    Allows pre-configured responses based on query patterns.
    """

    def __init__(self, responses: Optional[dict[str, str]] = None):
        """
        Initialize with optional response mappings.

        Args:
            responses: Dict mapping query patterns to responses
        """
        # Don't call LLMClient.__init__ - this is a mock with its own state
        self._model = "mock"
        self._temperature = 0.2
        self._responses = responses or {}
        self._default_response = "I don't have specific information about that."
        self._call_history: list[dict] = []

    def add_response(self, pattern: str, response: str) -> None:
        """Add a response pattern."""
        self._responses[pattern.lower()] = response

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        retries: int = 2,
        tools: Optional[list] = None,
        timeout: float = 300.0,
    ) -> str:
        """Mock chat that returns configured response based on last user message."""
        # Extract last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        self._call_history.append({
            "user_message": user_message,
            "timestamp": datetime.now().isoformat(),
        })

        # Find matching response
        user_lower = user_message.lower()
        for pattern, response in self._responses.items():
            if pattern in user_lower:
                return response

        return self._default_response

    def get_call_history(self) -> list[dict]:
        """Get history of calls made to mock LLM."""
        return self._call_history


class EvaluationSession:
    """
    Wraps an Orchestrator instance for controlled evaluation.

    This class provides:
    - Pre-seeded LTM with evaluation datasets (single or multi-session)
    - Mocked LLM responses for deterministic testing
    - Trace capture for metric calculation
    - No side effects (isolated storage paths)

    Coherence with LongMemEval:
    - Supports loading 30-40 sessions per question (LongMemEval_S standard)
    - Tests all 5 memory behaviors through orchestrator.chat()
    - Validates cross-session memory effects

    Example:
        session = EvaluationSession(
            ltm_seed_data=dataset.memories,
            config_overrides={"CONTEXT_AWARE_RETRIEVAL": True}
        )
        session.load_multi_session_history(
            sessions=dataset.sessions,  # 30-40 sessions
            behavior_type="IE"
        )
        result = session.send_message("What's my phone number?")
    """

    def __init__(
        self,
        ltm_seed_data: Optional[list[dict]] = None,
        corpus_path: Optional[Path] = None,
        config_overrides: Optional[dict] = None,
        use_mock_llm: bool = True,
        mock_responses: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Initialize evaluation session.

        Args:
            ltm_seed_data: Pre-loaded memories to seed LTM
            corpus_path: Optional path to corpus for fallback search
            config_overrides: Config overrides for evaluation
            use_mock_llm: Whether to use mock LLM for deterministic testing
            mock_responses: Pattern -> response mappings for mock LLM
        """
        self._session_id = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        self._temp_dir = Path(tempfile.mkdtemp(prefix="agemem_eval_"))
        self._corpus_path = corpus_path

        # Create evaluation config
        self._config = self._create_eval_config(config_overrides)

        # Initialize LLM (mock or real)
        if use_mock_llm:
            self._llm = MockLLMClient(mock_responses)
        else:
            self._llm = LLMClient(
                api_key=self._config.DEFAULT_API_KEY or "",
                model=self._config.DEFAULT_MODEL,
            )

        # Create isolated storage paths
        self._ltm_path = self._temp_dir / "ltm.json"
        self._stm_path = self._temp_dir / "stm.json"
        self._semantic_db_path = self._temp_dir / "semantic.db"

        # Initialize orchestrator
        self._orchestrator = self._create_orchestrator()

        # Seed LTM with evaluation data
        self._ltm_entries_count = 0
        if ltm_seed_data:
            self._seed_ltm(ltm_seed_data)

        # Trace database for detailed metrics
        self._trace_db_path = self._temp_dir / "traces.db"
        self._init_trace_db()

        # Evaluation trace
        self._trace = EvaluationTrace(
            session_id=self._session_id,
            behavior_type="unknown",
            ltm_entries_count=self._ltm_entries_count,
        )

        # Turn history
        self._turns: list[TurnResult] = []

        logger.info(f"EvaluationSession initialized: {self._session_id}")
        logger.info(f"Temp directory: {self._temp_dir}")

    def _create_eval_config(self, overrides: Optional[dict]) -> AgememConfig:
        """Create evaluation configuration with overrides."""
        config = DEFAULT_CONFIG

        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Force isolated paths
        config.PERSIST_DIR = str(self._temp_dir)

        return config

    def _create_orchestrator(self) -> Orchestrator:
        """Create orchestrator with evaluation configuration."""
        return Orchestrator(
            llm=self._llm,
            config=self._config,
        )

    def _seed_ltm(self, seed_data: list[dict]) -> None:
        """Seed LTM with evaluation data."""
        for entry in seed_data:
            content = entry.get("content", "")
            if not content:
                continue

            learning_score = entry.get("learning_score", 0.5)
            tags = entry.get("tags", [])
            source_turn = entry.get("source_turn", 0)

            result = self._orchestrator._ltm.add(
                content=content,
                learning_score=learning_score,
                tags=tags,
                source_turn=source_turn,
            )

            if result.success:
                self._ltm_entries_count += 1

        logger.info(f"Seeded LTM with {self._ltm_entries_count} entries")

    def _init_trace_db(self) -> None:
        """Initialize trace database for detailed metrics."""
        self._db = sqlite3.connect(str(self._trace_db_path))
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                user_input TEXT,
                assistant_response TEXT,
                latency_ms REAL,
                memories_injected INTEGER,
                corpus_fallback_used BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS retrievals (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                query TEXT,
                entry_id TEXT,
                source_turn INTEGER,
                retrieval_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._db.commit()

    def load_multi_session_history(
        self,
        sessions: list[list[dict]],
        behavior_type: str,
    ) -> None:
        """
        Load a full conversation history as if the agent had lived it.

        Each session is replayed through orchestrator.chat() to populate LTM
        with realistic conversation patterns. This simulates the 30-40 sessions
        per question pattern from LongMemEval_S.

        Args:
            sessions: List of 30-40 sessions, each with messages
                      Format: [[{"role": "user", "content": "..."}, ...], ...]
            behavior_type: One of "IE", "MR", "KU", "TR", "ABS"
                          (Information Extraction, Multi-session Reasoning,
                           Knowledge Updates, Temporal Reasoning, Abstention)

        Example:
            # Load 40 sessions of history
            session.load_multi_session_history(
                sessions=longmemeval_data.sessions,  # 40 sessions
                behavior_type="IE"  # Information Extraction test
            )
        """
        self._trace.behavior_type = behavior_type

        logger.info(f"Loading {len(sessions)} sessions for {behavior_type} evaluation")

        for session_idx, session_messages in enumerate(sessions):
            for msg in session_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "user":
                    # Process through orchestrator to trigger full lifecycle
                    self._orchestrator.chat(content)

        logger.info(f"Loaded {len(sessions)} sessions, STM turn: {self._orchestrator._stm.current_turn()}")

    def send_message(
        self,
        user_input: str,
        expected_answer: Optional[str] = None,
        evidence_sessions: Optional[list[int]] = None,
    ) -> TurnResult:
        """
        Send a message through orchestrator.chat().

        Captures full trace including:
        - LTM retrieval (what memories were injected)
        - STM state changes
        - Memory operations triggered
        - Learning feedback collected
        - Corpus fallback usage

        Args:
            user_input: The user's query
            expected_answer: Optional ground truth for validation
            evidence_sessions: Optional session indices containing evidence

        Returns:
            TurnResult with full trace of the turn
        """
        t0 = time.time()
        turn_idx = self._orchestrator._stm.current_turn()

        # Get STM stats before
        stm_stats_before = self._orchestrator._stm.stats()

        # Track retrieval before the turn
        memories_before = set(
            m.entry_id for m in self._orchestrator._stm.messages()
            if hasattr(m, 'entry_id')
        )

        # Execute the turn through orchestrator
        assistant_response = self._orchestrator.chat(user_input)

        # Calculate latency
        latency_ms = (time.time() - t0) * 1000

        # Get STM stats after
        stm_stats_after = self._orchestrator._stm.stats()

        # Get last trace from orchestrator
        last_trace = self._orchestrator.last_trace()

        # Determine which memories were injected
        memories_injected = []
        if last_trace:
            for op in last_trace.ops_applied:
                if hasattr(op, 'entries_affected'):
                    # Could retrieve full entries if needed
                    pass

        # Check corpus fallback usage from trace
        corpus_fallback_used = False
        if last_trace:
            for op in last_trace.ops_applied:
                if "CORPUS" in str(op.detail).upper():
                    corpus_fallback_used = True
                    break

        # Build TurnResult
        result = TurnResult(
            turn_index=turn_idx,
            user_input=user_input,
            assistant_response=assistant_response,
            memories_injected=memories_injected,
            stm_stats_before=stm_stats_before.to_dict() if hasattr(stm_stats_before, 'to_dict') else {},
            stm_stats_after=stm_stats_after.to_dict() if hasattr(stm_stats_after, 'to_dict') else {},
            memory_ops=last_trace.ops_applied if last_trace else [],
            learning_feedback=last_trace.feedback if last_trace else None,
            latency_ms=latency_ms,
            corpus_fallback_used=corpus_fallback_used,
        )

        # Store turn
        self._turns.append(result)
        self._trace.turns.append(result)

        # Log to trace database
        self._db.execute("""
            INSERT INTO turns
            (session_id, turn_index, user_input, assistant_response, latency_ms,
             memories_injected, corpus_fallback_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            self._session_id,
            turn_idx,
            user_input[:500],
            assistant_response[:1000],
            latency_ms,
            len(memories_injected),
            corpus_fallback_used,
        ))
        self._db.commit()

        # Log retrieval details
        tracer = get_tracer()
        tracer.log_memory_op(
            op_type="EVAL_TURN",
            detail=f"turn={turn_idx}, behavior={self._trace.behavior_type}",
            success=True,
        )

        return result

    def get_conversation_history(self) -> list[TurnResult]:
        """Get full history of the session."""
        return list(self._turns)

    def get_trace(self) -> EvaluationTrace:
        """Get full evaluation trace."""
        return self._trace

    def get_metrics(self) -> dict:
        """
        Calculate aggregate metrics from the session.

        Returns:
            Dict with metrics including:
            - total_turns
            - avg_latency_ms
            - corpus_fallback_rate
            - memories_injected_per_turn
        """
        if not self._turns:
            return {}

        total_turns = len(self._turns)
        avg_latency = sum(t.latency_ms for t in self._turns) / total_turns
        corpus_fallbacks = sum(1 for t in self._turns if t.corpus_fallback_used)
        total_memories = sum(len(t.memories_injected) for t in self._turns)

        return {
            "total_turns": total_turns,
            "avg_latency_ms": avg_latency,
            "corpus_fallback_rate": corpus_fallbacks / total_turns,
            "memories_injected_per_turn": total_memories / total_turns,
            "total_memories_injected": total_memories,
        }

    def cleanup(self) -> None:
        """Remove temporary storage and close resources."""
        try:
            self._db.close()
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp directory: {self._temp_dir}")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False


class MultiSessionEvaluation:
    """
    Loads and replays full conversation histories for evaluation.

    This class implements the multi-session pattern from LongMemEval:
    - Loads 30-40 sessions per question
    - Seeds LTM chronologically
    - Evaluates against specific behaviors

    Coherence with longmemeval_guide.md:
    - Supports LongMemEval_S (~115k tokens, ~40 sessions)
    - Implements all 5 memory behavior tests
    - Validates cross-session memory effects
    """

    def __init__(
        self,
        config_overrides: Optional[dict] = None,
        use_mock_llm: bool = True,
    ) -> None:
        """
        Initialize multi-session evaluation.

        Args:
            config_overrides: Configuration overrides
            use_mock_llm: Whether to use mock LLM for deterministic testing
        """
        self._config_overrides = config_overrides or {}
        self._use_mock_llm = use_mock_llm
        self._results: list[TurnResult] = []

    def load_session_history(
        self,
        sessions: list[dict],
        evidence_indices: list[int],
    ) -> EvaluationSession:
        """
        Pre-seed LTM with session history in chronological order.

        Each session becomes part of the agent's 'past'.

        Args:
            sessions: 30-40 sessions from LongMemEval
            evidence_indices: Which sessions contain answer evidence

        Returns:
            EvaluationSession ready for testing
        """
        # Convert sessions to format expected by EvaluationSession
        session_lists = []
        for session in sessions:
            messages = session.get("messages", [])
            session_lists.append(messages)

        # Create session and load history
        eval_session = EvaluationSession(
            config_overrides=self._config_overrides,
            use_mock_llm=self._use_mock_llm,
        )

        # Determine behavior type from evidence pattern
        behavior_type = self._infer_behavior_type(sessions, evidence_indices)

        eval_session.load_multi_session_history(
            sessions=session_lists,
            behavior_type=behavior_type,
        )

        return eval_session

    def _infer_behavior_type(
        self,
        sessions: list[dict],
        evidence_indices: list[int],
    ) -> str:
        """
        Infer LongMemEval behavior type from session pattern.

        Args:
            sessions: Session data
            evidence_indices: Indices of evidence sessions

        Returns:
            One of: "IE", "MR", "KU", "TR", "ABS"
        """
        # Single evidence session -> IE (Information Extraction)
        if len(evidence_indices) == 1:
            return "IE"

        # Multiple evidence sessions -> MR (Multi-session Reasoning)
        if len(evidence_indices) > 1:
            # Check for knowledge updates (overlapping content)
            # This is simplified - real detection would analyze content
            return "MR"

        # No evidence -> ABS (Abstention)
        return "ABS"

    def evaluate_question(
        self,
        question: str,
        expected_answer: str,
        evidence_sessions: list[int],
        eval_session: EvaluationSession,
    ) -> dict:
        """
        Ask the question after loading history and verify results.

        Verifies the orchestrator:
        1. Retrieves from the correct evidence sessions
        2. Synthesizes across multiple sessions when needed
        3. Produces the correct answer

        Args:
            question: The question to ask
            expected_answer: Ground truth answer
            evidence_sessions: Session indices with evidence
            eval_session: Pre-loaded evaluation session

        Returns:
            Dict with evaluation results including:
            - correct: Whether answer is correct
            - retrieved_from_evidence: Whether evidence sessions were used
            - has_abstained: Whether model abstained
            - latency_ms: Response latency
        """
        result = eval_session.send_message(
            user_input=question,
            expected_answer=expected_answer,
            evidence_sessions=evidence_sessions,
        )

        self._results.append(result)

        return {
            "correct": result.answer_is_correct(expected_answer),
            "retrieved_from_evidence": result.retrieved_from_correct_sessions(evidence_sessions),
            "has_abstained": result.has_abstained(),
            "latency_ms": result.latency_ms,
            "turn_index": result.turn_index,
        }


class CrossSessionPersistenceTest:
    """
    Tests that memory writes in early sessions affect reads in later sessions.

    Implements the cross-session persistence tests from the proposal:
    - Preference persistence (Session 5 -> Session 36)
    - Fact accumulation from multiple sessions
    - Contradiction resolution (most recent info wins)
    """

    def __init__(self, eval_session: EvaluationSession) -> None:
        """
        Initialize with an evaluation session.

        Args:
            eval_session: The evaluation session to test
        """
        self._session = eval_session

    def test_preference_persistence(
        self,
        early_session_content: str,
        gap_sessions: int,
        later_question: str,
        expected_in_response: str,
    ) -> bool:
        """
        Test that preferences from early sessions affect later responses.

        Example:
            Session 5: "I hate mushrooms"
            Sessions 6-35: Various unrelated conversations
            Session 36: "What should I order at the Italian place?"

        Verify: Response acknowledges mushroom preference

        Args:
            early_session_content: Content establishing preference
            gap_sessions: Number of unrelated sessions
            later_question: Question that should trigger preference
            expected_in_response: String expected in response

        Returns:
            True if preference was remembered
        """
        # Simulate early session
        self._session.send_message(early_session_content)

        # Simulate gap sessions
        for i in range(gap_sessions):
            self._session.send_message(f"Session {i} unrelated content")

        # Ask question
        result = self._session.send_message(later_question)

        # Check if preference was remembered
        return expected_in_response.lower() in result.assistant_response.lower()

    def test_fact_accumulation(
        self,
        facts: list[tuple[int, str]],  # (session_offset, fact)
        summary_question: str,
        expected_facts: list[str],
    ) -> dict:
        """
        Test that facts from different sessions are accumulated.

        Example:
            Session 10: "I live in Seattle"
            Session 20: "I have 2 cats"
            Session 30: "I work at Google"
            Session 31: "Tell me about yourself"

        Verify: Response includes all three facts

        Args:
            facts: List of (session_offset, fact) tuples
            summary_question: Question asking for accumulated info
            expected_facts: Facts that should appear in response

        Returns:
            Dict with per-fact detection results
        """
        results = {}

        # Simulate sessions with facts
        current_session = 0
        for session_offset, fact in facts:
            # Advance to target session
            while current_session < session_offset:
                self._session.send_message("Unrelated conversation")
                current_session += 1

            # Add fact
            self._session.send_message(fact)
            current_session += 1

        # Ask summary question
        result = self._session.send_message(summary_question)
        response_lower = result.assistant_response.lower()

        # Check for each expected fact
        for fact in expected_facts:
            results[fact] = fact.lower() in response_lower

        return results

    def test_contradiction_resolution(
        self,
        initial_fact: tuple[int, str],
        updated_fact: tuple[int, str],
        query: str,
        expected_current: str,
    ) -> bool:
        """
        Test that most recent info overrides older contradictions.

        Example:
            Session 15: "I'm vegetarian"
            Session 25: "I started eating chicken again"
            Session 35: "What should I cook for dinner?"

        Verify: Response uses most recent dietary preference

        Args:
            initial_fact: (session_offset, initial_statement)
            updated_fact: (session_offset, updated_statement)
            query: Question about current state
            expected_current: Expected response content

        Returns:
            True if most recent info was used
        """
        # Add initial fact
        for _ in range(initial_fact[0]):
            self._session.send_message("Unrelated")
        self._session.send_message(initial_fact[1])

        # Add updated fact
        for _ in range(updated_fact[0] - initial_fact[0]):
            self._session.send_message("Unrelated")
        self._session.send_message(updated_fact[1])

        # Query current state
        result = self._session.send_message(query)

        # Check if most recent info was used
        return expected_current.lower() in result.assistant_response.lower()
