"""
triggers/memory_trigger_engine.py
─────────────────────────────────
Unified entry point for the Memory Trigger System.

Hides the complexity of:
- SystemRules evaluation (threshold checking)
- MemoryAgent LLM calls (prompt building, JSON parsing)
- Operation execution ordering (FILTER before SUMMARY, etc.)
- Error handling (LLM failures, malformed responses)
- Graceful degradation (fallback when agent unavailable)

This is a "deep module" per John Ousterhout's philosophy:
small interface, large implementation. Callers only need
process_turn() and receive a complete report of what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.types import (
    ContextStats,
    LearningFeedback,
    MemoryOp,
    MemoryOpResult,
    TriggerKind,
)
from core.config import AgememConfig
from triggers.system_rules import SystemRules, RuleID
from agents.memory_agent import MemoryAgent, LTMOperation
from agents.llm_client import LLMClient
from memory.stm_context import STMContext
from memory.ltm_store import LTMStore
from core.tracing import get_tracer


@dataclass(frozen=True)
class MemoryCycleReport:
    """
    Complete summary of what the Memory Trigger System did this turn.

    The Orchestrator receives this and logs/traces it. No further action needed.
    """
    # What rules fired (for observability)
    rules_triggered: list[RuleID] = field(default_factory=list)

    # All memory operations that were executed (in order)
    operations: list[MemoryOpResult] = field(default_factory=list)

    # Human-readable summary of what happened
    summary: str = ""

    # Agent's reasoning if MemoryAgent was invoked
    agent_rationale: Optional[str] = None

    # Whether any LTM mutations occurred (for downstream sync decisions)
    ltm_modified: bool = False

    # Current STM stats after all operations
    stm_stats: Optional[ContextStats] = None


class MemoryTriggerEngine:
    """
    Unified entry point for the Memory Trigger System.

    Dependencies (all injected):
    - config: AgememConfig (thresholds, cadences)
    - llm: LLMClient (for MemoryAgent calls)
    - stm: STMContext (operations execute here)
    - ltm: LTMStore (operations execute here)
    """

    def __init__(
        self,
        config: AgememConfig,
        llm: LLMClient,
        stm: STMContext,
        ltm: LTMStore,
    ) -> None:
        self._config = config
        self._rules = SystemRules(config)
        self._agent = MemoryAgent(llm, config)
        self._stm = stm
        self._ltm = ltm

    def process_turn(
        self,
        turn_index: int,
        feedback: Optional[LearningFeedback] = None,
        assistant_response: Optional[str] = None,
    ) -> MemoryCycleReport:
        """
        Execute the complete memory trigger cycle for one turn.

        This is the ONLY method callers need. It:
        1. Evaluates system rules against current STM stats
        2. Executes FILTER/SUMMARY operations immediately
        3. Invokes MemoryAgent if periodic review or learning spike detected
        4. Applies MemoryAgent decisions (ADD/UPDATE/DELETE) to LTM
        5. Applies context relevance scores to STM messages
        6. Handles learning spike immediate promotion

        All error handling is internal. Returns a complete report of what happened.

        Args:
            turn_index: Current conversation turn number
            feedback: Optional learning feedback from the main agent
            assistant_response: Optional assistant response for content fallback

        Returns:
            MemoryCycleReport summarizing all actions taken
        """
        tracer = get_tracer()
        operations: list[MemoryOpResult] = []
        rules_triggered: list[RuleID] = []
        agent_rationale: Optional[str] = None
        ltm_modified = False

        # 1. Get current STM stats and evaluate rules
        stats = self._stm.stats()
        decisions = self._rules.evaluate(stats, turn_index, feedback)

        # Collect rule IDs for observability
        for decision in decisions:
            rules_triggered.append(decision.rule_id)

        # 2. Execute immediate operations (overflow handling)
        should_run_memory_agent = False

        for decision in decisions:
            if decision.rule_id == RuleID.OVERFLOW_CRITICAL:
                # FILTER then SUMMARY
                filter_op = self._stm.filter(trigger=TriggerKind.SYSTEM_RULE)
                operations.append(filter_op)
                tracer.log_memory_op(
                    op_type="STM_FILTER",
                    detail="Critical overflow",
                    success=True,
                    trigger="SYSTEM_RULE",
                )

                summary_op = self._stm.summary(trigger=TriggerKind.SYSTEM_RULE)
                operations.append(summary_op)
                tracer.log_memory_op(
                    op_type="STM_SUMMARY",
                    detail="Critical overflow summary",
                    success=True,
                    trigger="SYSTEM_RULE",
                )

            elif decision.rule_id == RuleID.OVERFLOW_WARN:
                summary_op = self._stm.summary(trigger=TriggerKind.SYSTEM_RULE)
                operations.append(summary_op)
                tracer.log_memory_op(
                    op_type="STM_SUMMARY",
                    detail="Warning overflow summary",
                    success=True,
                    trigger="SYSTEM_RULE",
                )

            elif decision.rule_id in (RuleID.PERIODIC_REVIEW, RuleID.LEARNING_SPIKE):
                should_run_memory_agent = True

        # 3. Immediate LTM promotion on learning spike (threshold-based)
        if feedback and feedback.score >= self._config.LTM_PROMOTE_THRESHOLD:
            content_to_store = feedback.affected_content
            if not content_to_store:
                # Build content from recent assistant messages
                budget = self._config.LTM_ENTRY_MAX_CHARS
                segments: list[str] = []
                for msg in reversed(self._stm.messages()):
                    if msg.role == "assistant" and msg.content:
                        segments.append(msg.content[:budget])
                        budget -= len(msg.content)
                        if budget <= 0:
                            break
                content_to_store = "\n\n".join(reversed(segments))
            else:
                content_to_store = content_to_store[:self._config.LTM_ENTRY_MAX_CHARS]

            if content_to_store:
                add_op = self._ltm.add(
                    content=content_to_store,
                    learning_score=feedback.score,
                    source_turn=turn_index,
                    trigger=TriggerKind.LEARNING_SCORE,
                )
                operations.append(add_op)
                ltm_modified = True
                tracer.log_memory_op(
                    op_type="LTM_ADD",
                    detail=f"Learning spike score={feedback.score:.2f}",
                    success=add_op.success,
                    trigger="LEARNING_SCORE",
                )

        # 4. Run MemoryAgent if triggered
        if should_run_memory_agent:
            agent_ops, rationale = self._run_memory_agent(turn_index, feedback)
            operations.extend(agent_ops)
            if rationale:
                agent_rationale = rationale
            if any(op.op in (MemoryOp.ADD, MemoryOp.UPDATE, MemoryOp.DELETE) for op in agent_ops):
                ltm_modified = True

        # 5. Build summary
        stats_after = self._stm.stats()
        summary = self._build_summary(rules_triggered, operations, agent_rationale)

        return MemoryCycleReport(
            rules_triggered=rules_triggered,
            operations=operations,
            summary=summary,
            agent_rationale=agent_rationale,
            ltm_modified=ltm_modified,
            stm_stats=stats_after,
        )

    def force_summary(self) -> MemoryOpResult:
        """
        Emergency summary when context is about to overflow.
        Bypasses normal rule evaluation for immediate action.
        """
        return self._stm.summary(trigger=TriggerKind.SYSTEM_RULE)

    def check_health(self) -> dict:
        """
        Diagnostic info: rule hit rates, agent call frequency, etc.
        """
        stats = self._stm.stats()
        return {
            "stm_utilisation": stats.utilisation_ratio,
            "stm_message_count": stats.message_count,
            "stm_overflow_risk": stats.overflow_risk,
            "ltm_entry_count": len(self._ltm.all_entries()),
            "config": {
                "trigger_every_n_turns": self._config.TRIGGER_EVERY_N_TURNS,
                "ltm_promote_threshold": self._config.LTM_PROMOTE_THRESHOLD,
                "stm_warning_threshold": self._config.STM_WARNING_THRESHOLD,
                "stm_critical_threshold": self._config.STM_CRITICAL_THRESHOLD,
            },
        }

    # ── Internal methods ──────────────────────────────────────────────────────

    def _run_memory_agent(
        self,
        turn_index: int,
        feedback: Optional[LearningFeedback],
    ) -> tuple[list[MemoryOpResult], Optional[str]]:
        """
        Run MemoryAgent and apply its decisions.

        Returns:
            Tuple of (operations executed, agent rationale)
        """
        tracer = get_tracer()
        operations: list[MemoryOpResult] = []
        rationale: Optional[str] = None

        try:
            decision_obj = self._agent.review(
                recent_messages=self._stm.messages(),
                ltm_entries=self._ltm.all_entries(),
                feedback=feedback,
            )
            rationale = decision_obj.rationale

            # Apply LTM operations
            for ltm_op in decision_obj.ltm_operations:
                if ltm_op.confidence < 0.6:
                    continue  # Skip low-confidence ops

                score = feedback.score if feedback else ltm_op.confidence

                if ltm_op.op == MemoryOp.ADD:
                    result = self._ltm.add(
                        content=ltm_op.content,
                        learning_score=score,
                        tags=ltm_op.tags,
                        source_turn=turn_index,
                        trigger=TriggerKind.MEMORY_AGENT,
                    )
                    operations.append(result)
                    tracer.log_memory_op(
                        op_type="LTM_ADD",
                        detail=f"MemoryAgent ADD: {ltm_op.content[:100]}...",
                        success=result.success,
                        trigger="MEMORY_AGENT",
                    )

                elif ltm_op.op == MemoryOp.UPDATE and ltm_op.entry_id:
                    result = self._ltm.update(
                        entry_id=ltm_op.entry_id,
                        content=ltm_op.content,
                        learning_score=score,
                        trigger=TriggerKind.MEMORY_AGENT,
                    )
                    operations.append(result)
                    tracer.log_memory_op(
                        op_type="LTM_UPDATE",
                        detail=f"MemoryAgent UPDATE: entry_id={ltm_op.entry_id}",
                        success=result.success,
                        trigger="MEMORY_AGENT",
                    )

                elif ltm_op.op == MemoryOp.DELETE and ltm_op.entry_id:
                    result = self._ltm.delete(
                        entry_id=ltm_op.entry_id,
                        trigger=TriggerKind.MEMORY_AGENT,
                    )
                    operations.append(result)
                    tracer.log_memory_op(
                        op_type="LTM_DELETE",
                        detail=f"MemoryAgent DELETE: entry_id={ltm_op.entry_id}",
                        success=result.success,
                        trigger="MEMORY_AGENT",
                    )

            # Apply context relevance scores to STM messages
            if decision_obj.context_relevance:
                for msg in self._stm.messages():
                    if msg.turn_index in decision_obj.context_relevance:
                        msg.relevance_score = decision_obj.context_relevance[msg.turn_index]

            # SUMMARY if MemoryAgent requested it
            if decision_obj.summary_needed:
                summary_op = self._stm.summary(trigger=TriggerKind.MEMORY_AGENT)
                operations.append(summary_op)
                tracer.log_memory_op(
                    op_type="STM_SUMMARY",
                    detail="MemoryAgent requested summary",
                    success=True,
                    trigger="MEMORY_AGENT",
                )

        except Exception as exc:
            # Graceful degradation: log and continue
            print(f"[DEBUG] MemoryTriggerEngine agent error: {exc}", flush=True)

        return operations, rationale

    def _build_summary(
        self,
        rules_triggered: list[RuleID],
        operations: list[MemoryOpResult],
        agent_rationale: Optional[str],
    ) -> str:
        """Build a human-readable summary of the cycle."""
        parts: list[str] = []

        if rules_triggered:
            rule_names = [r.value for r in rules_triggered]
            parts.append(f"Rules: {', '.join(rule_names)}")

        if operations:
            op_counts: dict[MemoryOp, int] = {}
            for op in operations:
                op_counts[op.op] = op_counts.get(op.op, 0) + 1
            op_summary = ", ".join(f"{k.value}:{v}" for k, v in op_counts.items())
            parts.append(f"Ops: {op_summary}")

        if agent_rationale:
            parts.append(f"Agent: {agent_rationale[:100]}")

        return " | ".join(parts) if parts else "No actions"