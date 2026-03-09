"""
agents/memory_agent.py
───────────────────────
The dedicated MemoryAgent sub-agent.

Responsibility
──────────────
Given a snapshot of recent context and the current LTM store, the
MemoryAgent decides:
  • Which ADD/UPDATE/DELETE operations should be applied to LTM
  • Which context messages are low-relevance (for FILTER scoring)
  • A SUMMARY of the context window if requested

This is the "agent-based" component of the hybrid approach.  It runs
inside the same inference budget as the main agent — no separate model
required (though MEMORY_AGENT_MODEL can differ).

Key design constraints (inference-only):
─────────────────────────────────────────
* The MemoryAgent does NOT learn from rewards.  It relies on a
  well-crafted system prompt that encodes the AgeMem heuristics as
  instructions.
* Its decisions are deterministic given the same context and a low
  temperature (0.1).
* It returns structured JSON so decisions can be validated before
  being applied to the stores.

Output schema (JSON)
─────────────────────
{
  "ltm_operations": [
    {"op": "add"|"update"|"delete",
     "entry_id": "<id or null>",
     "content": "<text>",
     "tags": ["..."],
     "confidence": 0.0-1.0}
  ],
  "context_relevance": [
    {"turn_index": <int>, "relevance_score": 0.0-1.0}
  ],
  "summary_needed": true|false,
  "rationale": "<brief explanation>"
}
"""

from __future__ import annotations

import json
from typing import Optional

from core.types import (
    ContextMessage,
    LearningFeedback,
    MemoryEntry,
    MemoryOp,
    MemoryOpResult,
    TriggerKind,
)
from core.config import AgememConfig, DEFAULT_CONFIG
from agents.llm_client import LLMClient


_SYSTEM_PROMPT = """\
You are a Memory Management Agent for an LLM assistant system.
Your sole task is to analyse a conversation window and the current long-term memory store,
then decide what memory operations to perform.

You MUST return valid JSON only, following this exact schema:
{
  "ltm_operations": [
    {
      "op": "add" | "update" | "delete",
      "entry_id": "<existing_id or null for add>",
      "content": "<text to store or updated text>",
      "tags": ["<tag1>", "<tag2>"],
      "confidence": <float 0.0-1.0>
    }
  ],
  "context_relevance": [
    {"turn_index": <int>, "relevance_score": <float 0.0-1.0>}
  ],
  "summary_needed": <true|false>,
  "rationale": "<one sentence explanation>"
}

Rules for ltm_operations:
- ADD:    Only add information that is novel, factual, and likely to be reused.
          Do NOT add trivial pleasantries or one-off queries.
- UPDATE: If an existing LTM entry becomes outdated or can be enriched, update it.
- DELETE: Remove entries that are superseded or were stored incorrectly.
- Keep confidence >= 0.7 for ADD operations.  Lower-confidence items should be omitted.

Rules for context_relevance:
- Score 1.0 = critical to ongoing task (never filter)
- Score 0.5 = moderately relevant
- Score 0.2 = likely noise (filter candidate)
- Assign scores only to non-system messages.

Rules for summary_needed:
- Return true only if there are 6+ consecutive exchanges about the same topic
  that could be safely compressed without losing key facts.

Return ONLY the JSON object, no preamble, no explanation outside the JSON.
"""


class MemoryAgent:

    def __init__(
        self,
        llm: LLMClient,
        config: AgememConfig = DEFAULT_CONFIG,
    ) -> None:
        self._llm = llm
        self._config = config

    def review(
        self,
        recent_messages: list[ContextMessage],
        ltm_entries: list[MemoryEntry],
        feedback: Optional[LearningFeedback] = None,
    ) -> "MemoryAgentDecision":
        """
        Run a full review cycle.

        Returns a MemoryAgentDecision with all recommended operations.
        This never raises: on parse failure it returns an empty decision.
        """
        prompt = self._build_prompt(recent_messages, ltm_entries, feedback)
        try:
            raw = self._llm.chat_json(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                model=self._config.MEMORY_AGENT_MODEL,
                max_tokens=self._config.MEMORY_AGENT_MAX_TOKENS,
            )
            return MemoryAgentDecision.from_dict(raw)
        except Exception as exc:
            return MemoryAgentDecision(
                ltm_operations=[],
                context_relevance={},
                summary_needed=False,
                rationale=f"MemoryAgent parse error: {exc}",
            )

    def summarise_context(self, messages: list[ContextMessage]) -> str:
        """
        Produce a concise summary of a list of messages.
        Used as the STMContext.summary_fn callback.
        """
        if not messages:
            return ""
        formatted = "\n".join(
            f"[{m.role}] (turn {m.turn_index}): {m.content}"
            for m in messages
        )
        prompt = (
            "Summarise the following conversation segment in 3-4 sentences. "
            "Preserve all key facts, decisions, and user preferences. "
            "Be concise.\n\n" + formatted
        )
        try:
            return self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self._config.MEMORY_AGENT_MODEL,
                max_tokens=512,
                temperature=0.1,
            )
        except Exception as exc:
            # Graceful fallback: return truncated original
            return f"[Summary unavailable: {exc}] " + formatted[:300]

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_prompt(
        self,
        messages: list[ContextMessage],
        ltm_entries: list[MemoryEntry],
        feedback: Optional[LearningFeedback],
    ) -> str:
        parts: list[str] = []

        parts.append("=== RECENT CONVERSATION (last messages) ===")
        for m in messages[-12:]:  # cap at 12 messages to control prompt size
            parts.append(f"[turn={m.turn_index}, role={m.role}]: {m.content[:400]}")

        parts.append("\n=== CURRENT LTM STORE (sample) ===")
        if ltm_entries:
            for e in ltm_entries[:10]:
                parts.append(
                    f"[id={e.entry_id}, score={e.learning_score:.2f}]: {e.content[:200]}"
                )
        else:
            parts.append("(empty)")

        if feedback:
            parts.append(
                f"\n=== AGENT LEARNING FEEDBACK ===\n"
                f"score={feedback.score:.2f}, "
                f"content='{feedback.affected_content[:200]}', "
                f"rationale='{feedback.rationale}'"
            )

        return "\n".join(parts)


# ── Decision object ───────────────────────────────────────────────────────────

class LTMOperation:
    __slots__ = ("op", "entry_id", "content", "tags", "confidence")

    def __init__(
        self,
        op: str,
        content: str,
        entry_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        confidence: float = 0.8,
    ) -> None:
        self.op = MemoryOp(op)
        self.entry_id = entry_id
        self.content = content
        self.tags = tags or []
        self.confidence = confidence


class MemoryAgentDecision:

    def __init__(
        self,
        ltm_operations: list[LTMOperation],
        context_relevance: dict[int, float],  # turn_index -> score
        summary_needed: bool,
        rationale: str,
    ) -> None:
        self.ltm_operations = ltm_operations
        self.context_relevance = context_relevance
        self.summary_needed = summary_needed
        self.rationale = rationale

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryAgentDecision":
        ops: list[LTMOperation] = []
        for item in data.get("ltm_operations", []):
            try:
                ops.append(LTMOperation(
                    op=item["op"],
                    content=item.get("content", ""),
                    entry_id=item.get("entry_id"),
                    tags=item.get("tags", []),
                    confidence=float(item.get("confidence", 0.8)),
                ))
            except (KeyError, ValueError):
                continue  # skip malformed op

        relevance: dict[int, float] = {}
        for item in data.get("context_relevance", []):
            try:
                relevance[int(item["turn_index"])] = float(item["relevance_score"])
            except (KeyError, ValueError):
                continue

        return cls(
            ltm_operations=ops,
            context_relevance=relevance,
            summary_needed=bool(data.get("summary_needed", False)),
            rationale=data.get("rationale", ""),
        )

    def has_work(self) -> bool:
        return bool(self.ltm_operations) or self.summary_needed
