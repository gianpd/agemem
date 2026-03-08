"""
triggers/system_rules.py
─────────────────────────
Deterministic, heuristic-based trigger layer.

These rules fire *without* any LLM call.  They are the
"system decides" part of the hybrid approach.

Rules implemented
─────────────────
R1  Context overflow warning  → force SUMMARY
R2  Context critical overflow → force FILTER + SUMMARY
R3  Every N turns             → invoke MemoryAgent review cycle
R4  Learning score spike      → immediate LTM ADD candidate
R5  STM message relevance     → tag messages for FILTER

Separation of concerns
───────────────────────
SystemRules only *detects* conditions and returns a list of
TriggerDecision objects.  The Orchestrator executes the actual
memory operations.  This makes rules unit-testable without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.types import ContextStats, LearningFeedback, MemoryOp, TriggerKind
from core.config import AgememConfig, DEFAULT_CONFIG


class RuleID(str, Enum):
    OVERFLOW_WARN     = "R1_overflow_warn"
    OVERFLOW_CRITICAL = "R2_overflow_critical"
    PERIODIC_REVIEW   = "R3_periodic_review"
    LEARNING_SPIKE    = "R4_learning_spike"
    RELEVANCE_DECAY   = "R5_relevance_decay"


@dataclass
class TriggerDecision:
    rule_id: RuleID
    recommended_op: MemoryOp
    priority: int                   # higher = more urgent
    reason: str
    metadata: dict = field(default_factory=dict)


class SystemRules:

    def __init__(self, config: AgememConfig = DEFAULT_CONFIG) -> None:
        self._config = config
        self._last_review_turn: int = -1

    def evaluate(
        self,
        stats: ContextStats,
        turn_index: int,
        feedback: Optional[LearningFeedback] = None,
    ) -> list[TriggerDecision]:
        """
        Evaluate all rules and return a (possibly empty) list of decisions,
        sorted by priority descending.
        """
        decisions: list[TriggerDecision] = []

        decisions.extend(self._check_overflow(stats))
        decisions.extend(self._check_periodic(turn_index))
        if feedback:
            decisions.extend(self._check_learning_spike(feedback))

        decisions.sort(key=lambda d: d.priority, reverse=True)
        return decisions

    # ── Individual rules ──────────────────────────────────────────────────────

    def _check_overflow(self, stats: ContextStats) -> list[TriggerDecision]:
        out: list[TriggerDecision] = []

        if stats.utilisation_ratio >= self._config.STM_CRITICAL_THRESHOLD:
            out.append(TriggerDecision(
                rule_id=RuleID.OVERFLOW_CRITICAL,
                recommended_op=MemoryOp.FILTER,
                priority=100,
                reason=(
                    f"Context at {stats.utilisation_ratio:.0%} of limit "
                    f"({stats.total_tokens}/{self._config.STM_TOKEN_LIMIT} tokens). "
                    "Critical: forcing FILTER then SUMMARY."
                ),
                metadata={"utilisation": stats.utilisation_ratio},
            ))
        elif stats.utilisation_ratio >= self._config.STM_WARNING_THRESHOLD:
            out.append(TriggerDecision(
                rule_id=RuleID.OVERFLOW_WARN,
                recommended_op=MemoryOp.SUMMARY,
                priority=70,
                reason=(
                    f"Context at {stats.utilisation_ratio:.0%} — "
                    "approaching limit, recommending SUMMARY."
                ),
                metadata={"utilisation": stats.utilisation_ratio},
            ))

        return out

    def _check_periodic(self, turn_index: int) -> list[TriggerDecision]:
        n = self._config.TRIGGER_EVERY_N_TURNS
        if n <= 0:
            return []
        if (
            turn_index > 0
            and turn_index % n == 0
            and turn_index != self._last_review_turn
        ):
            self._last_review_turn = turn_index
            return [TriggerDecision(
                rule_id=RuleID.PERIODIC_REVIEW,
                recommended_op=MemoryOp.ADD,  # memory-agent decides exact op
                priority=40,
                reason=f"Periodic memory review at turn {turn_index}.",
                metadata={"turn": turn_index},
            )]
        return []

    def _check_learning_spike(
        self, feedback: LearningFeedback
    ) -> list[TriggerDecision]:
        if feedback.score >= self._config.LEARNING_SCORE_THRESHOLD_IMMEDIATE:
            return [TriggerDecision(
                rule_id=RuleID.LEARNING_SPIKE,
                recommended_op=MemoryOp.ADD,
                priority=90,
                reason=(
                    f"Agent self-reported learning_score={feedback.score:.2f} "
                    f"(>= {self._config.LEARNING_SCORE_THRESHOLD_IMMEDIATE}). "
                    "Immediate LTM candidacy."
                ),
                metadata={
                    "score": feedback.score,
                    "content": feedback.affected_content,
                },
            )]
        return []
