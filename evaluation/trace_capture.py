"""
evaluation/trace_capture.py
--------------------------

Trace capture utilities for evaluation.

Wraps orchestrator inspection methods for structured trace capture
during evaluation runs.
"""

from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from agents.orchestrator import Orchestrator, TurnTrace
from core.types import ContextStats, MemoryOp


@dataclass
class TurnTraceSummary:
    """Summary metrics computed from a list of TurnTrace objects."""
    total_turns: int
    ltm_adds: int
    ltm_updates: int
    ltm_deletes: int
    avg_learning_score: float
    stm_overflow_count: int
    trigger_count: int


class TraceCapture:
    """
    Captures and persists traces from orchestrator runs.

    Wraps orchestrator inspection methods for structured trace capture
    during evaluation. Optionally persists to SQLite for offline analysis.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize trace capture.

        Args:
            db_path: Optional path to SQLite database for trace persistence.
                     If None, traces are kept in memory only.
        """
        self._traces: list[TurnTrace] = []
        self._db_path = db_path

        if db_path is not None:
            self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite tables for trace persistence."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_index INTEGER NOT NULL,
                    user_input TEXT,
                    assistant_response TEXT,
                    stm_stats_before TEXT,
                    stm_stats_after TEXT,
                    ops_applied TEXT,
                    feedback_score REAL,
                    feedback_rationale TEXT,
                    memory_agent_rationale TEXT,
                    latency_ms REAL,
                    prompt_versions TEXT,
                    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def capture_turn(self, orchestrator: Orchestrator) -> Optional[TurnTrace]:
        """
        Capture the most recent turn trace from the orchestrator.

        Args:
            orchestrator: The orchestrator instance to capture from.

        Returns:
            The captured TurnTrace, or None if no traces exist.
        """
        trace = orchestrator.last_trace()
        if trace is not None:
            self._traces.append(trace)
            if self._db_path is not None:
                self._persist_trace(trace)
        return trace

    def capture_ltm_snapshot(self, orchestrator: Orchestrator) -> list[dict]:
        """
        Capture a snapshot of the LTM state.

        Args:
            orchestrator: The orchestrator instance to capture from.

        Returns:
            List of LTM entry dicts.
        """
        return orchestrator.ltm_snapshot()

    def capture_stm_stats(self, orchestrator: Orchestrator) -> ContextStats:
        """
        Capture current STM statistics.

        Args:
            orchestrator: The orchestrator instance to capture from.

        Returns:
            Current ContextStats for STM.
        """
        return orchestrator.stm_stats()

    def _persist_trace(self, trace: TurnTrace) -> None:
        """Persist a single trace to the SQLite database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO traces (
                    turn_index, user_input, assistant_response,
                    stm_stats_before, stm_stats_after, ops_applied,
                    feedback_score, feedback_rationale,
                    memory_agent_rationale, latency_ms, prompt_versions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.turn_index,
                trace.user_input,
                trace.assistant_response,
                json.dumps(asdict(trace.stm_stats_before)) if trace.stm_stats_before else None,
                json.dumps(asdict(trace.stm_stats_after)) if trace.stm_stats_after else None,
                json.dumps([asdict(op) for op in trace.ops_applied]),
                trace.feedback.score if trace.feedback else None,
                trace.feedback.rationale if trace.feedback else None,
                trace.memory_agent_rationale,
                trace.latency_ms,
                json.dumps(trace.prompt_versions),
            ))
            conn.commit()

    def compute_metrics(self, traces: Optional[list[TurnTrace]] = None) -> TurnTraceSummary:
        """
        Compute summary metrics from traces.

        Args:
            traces: Optional list of traces to compute from. If None,
                    uses all captured traces.

        Returns:
            TurnTraceSummary with computed metrics.
        """
        if traces is None:
            traces = self._traces

        if not traces:
            return TurnTraceSummary(
                total_turns=0,
                ltm_adds=0,
                ltm_updates=0,
                ltm_deletes=0,
                avg_learning_score=0.0,
                stm_overflow_count=0,
                trigger_count=0,
            )

        ltm_adds = 0
        ltm_updates = 0
        ltm_deletes = 0
        learning_scores = []
        overflow_count = 0
        trigger_count = 0

        for trace in traces:
            # Count LTM operations
            for op in trace.ops_applied:
                if op.op == MemoryOp.ADD:
                    ltm_adds += 1
                elif op.op == MemoryOp.UPDATE:
                    ltm_updates += 1
                elif op.op == MemoryOp.DELETE:
                    ltm_deletes += 1
                trigger_count += 1

            # Collect learning scores
            if trace.feedback and trace.feedback.score is not None:
                learning_scores.append(trace.feedback.score)

            # Count STM overflow events
            if trace.stm_stats_after and trace.stm_stats_after.overflow_risk:
                overflow_count += 1

        avg_score = sum(learning_scores) / len(learning_scores) if learning_scores else 0.0

        return TurnTraceSummary(
            total_turns=len(traces),
            ltm_adds=ltm_adds,
            ltm_updates=ltm_updates,
            ltm_deletes=ltm_deletes,
            avg_learning_score=round(avg_score, 4),
            stm_overflow_count=overflow_count,
            trigger_count=trigger_count,
        )

    def all_traces(self) -> list[TurnTrace]:
        """Return all captured traces."""
        return list(self._traces)

    def clear(self) -> None:
        """Clear all captured traces from memory."""
        self._traces.clear()