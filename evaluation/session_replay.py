"""
evaluation/session_replay.py
----------------------------
Replay LongMemEval sessions through the orchestrator to test the production codepath.

This module provides the SessionReplayEngine which processes 30-40 LongMemEval sessions
through orchestrator.chat() to validate end-to-end memory lifecycle behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.orchestrator import Orchestrator, TurnTrace
from core.types import ContextStats, MemoryOp


@dataclass
class SessionReplayResult:
    """Result of replaying a single session through the orchestrator."""
    session_id: str
    turns_processed: int
    ltm_entries_added: int
    stm_tokens_at_end: int
    learning_scores: list[float] = field(default_factory=list)
    retrieval_traces: list[dict] = field(default_factory=list)


class SessionReplayEngine:
    """
    Engine for replaying LongMemEval sessions through the orchestrator.

    This tests the production codepath by processing sessions through
    orchestrator.chat() and capturing memory operations.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        """
        Initialize the replay engine.

        Args:
            orchestrator: The Orchestrator instance to replay sessions through.
        """
        self._orchestrator = orchestrator
        self._sessions: list[tuple[list[dict], str]] = []  # (session, behavior_type)

    def load_sessions(self, sessions: list[list[dict]], behavior_type: str) -> None:
        """
        Load sessions for replay.

        Args:
            sessions: List of sessions, each session is a list of turn dicts.
            behavior_type: The behavior type category for these sessions.
        """
        self._sessions = [
            (session, behavior_type)
            for session in sessions
        ]

    def replay_all(self) -> list[SessionReplayResult]:
        """
        Replay all loaded sessions through the orchestrator.

        Returns:
            List of SessionReplayResult for each session.
        """
        results = []
        for i, (session, behavior_type) in enumerate(self._sessions):
            session_id = f"{behavior_type}_{i}"
            result = self.replay_session(session, session_id)
            results.append(result)
        return results

    def replay_session(self, session: list[dict], session_id: str) -> SessionReplayResult:
        """
        Replay a single session through the orchestrator.

        Args:
            session: List of turn dicts with 'role' and 'content' keys.
            session_id: Unique identifier for this session.

        Returns:
            SessionReplayResult with captured metrics.
        """
        turns_processed = 0
        ltm_entries_added = 0
        learning_scores: list[float] = []
        retrieval_traces: list[dict] = []

        # Track LTM size at start
        ltm_before = len(self._orchestrator.ltm_snapshot())

        for turn in session:
            if turn.get('role') != 'user':
                continue

            user_content = turn.get('content', '')
            if not user_content:
                continue

            # Process through orchestrator
            self._orchestrator.chat(user_content)
            turns_processed += 1

            # Capture trace data
            trace = self._orchestrator.last_trace()
            if trace:
                # Track learning scores from feedback
                if trace.feedback:
                    learning_scores.append(trace.feedback.score)

                # Track LTM adds from ops_applied
                for op_result in trace.ops_applied:
                    if op_result.op == MemoryOp.ADD and op_result.success:
                        ltm_entries_added += len(op_result.entries_affected)

                    # Track retrieval operations
                    if op_result.op == MemoryOp.RETRIEVE:
                        retrieval_traces.append({
                            'turn_index': trace.turn_index,
                            'trigger': op_result.trigger.value,
                            'entries_count': len(op_result.entries_affected),
                            'detail': op_result.detail,
                        })

        # Get final STM token count
        stm_stats = self._orchestrator.stm_stats()
        stm_tokens_at_end = stm_stats.total_tokens

        return SessionReplayResult(
            session_id=session_id,
            turns_processed=turns_processed,
            ltm_entries_added=ltm_entries_added,
            stm_tokens_at_end=stm_tokens_at_end,
            learning_scores=learning_scores,
            retrieval_traces=retrieval_traces,
        )