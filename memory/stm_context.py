"""
memory/stm_context.py
──────────────────────
Short-Term Memory: the active context window.

Responsibilities
────────────────
* Maintain an ordered list of ContextMessages
* RETRIEVE  – inject LTM entries into context
* SUMMARY   – compress a sliding window of messages into one summarised message
* FILTER    – drop low-relevance messages
* overflow_check – emit ContextStats; callers decide what to do

Critical invariant
──────────────────
  total_tokens < STM_TOKEN_LIMIT at the start of every LLM call.

The manager enforces this internally; callers do not need to gate on it,
but can inspect ContextStats for monitoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from core.types import (
    ContextMessage,
    ContextStats,
    MemoryEntry,
    MemoryOp,
    MemoryOpResult,
    TriggerKind,
    TokenCounter,
)
from core.config import AgememConfig, DEFAULT_CONFIG


# Type alias for a summarise callback: takes list of messages, returns str
SummaryFn = Callable[[list[ContextMessage]], str]


class STMContext:

    def __init__(
        self,
        config: AgememConfig = DEFAULT_CONFIG,
        token_counter: Optional[TokenCounter] = None,
        summary_fn: Optional[SummaryFn] = None,
    ) -> None:
        self._config = config
        self._tc = token_counter or TokenCounter()
        self._summary_fn = summary_fn  # injected by Orchestrator (calls LLM)
        self._messages: list[ContextMessage] = []
        self._turn_index: int = 0

    # ── Public accessors ──────────────────────────────────────────────────────

    def messages(self) -> list[ContextMessage]:
        """Return all messages in insertion order."""
        return list(self._messages)

    def openai_messages(self) -> list[dict]:
        """Return messages in OpenAI-compatible format.

        Consolidates all system messages into a single message at the beginning
        to comply with llama.cpp Jinja template requirements.
        Converts unsupported 'tool' role messages to 'user' messages for compatibility.
        """
        # Separate system messages from conversation messages
        system_parts = []
        conversation = []

        for m in self._messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "tool":
                # Keep tool messages as proper tool role with tool_call_id
                # This is required for proper tool calling with models that support it
                tool_msg = m.to_openai_dict()
                conversation.append(tool_msg)
            else:
                conversation.append(m.to_openai_dict())

        result = []
        if system_parts:
            # Merge all system messages into one
            result.append({"role": "system", "content": "\n\n".join(system_parts)})

        result.extend(conversation)
        return result

    def stats(self) -> ContextStats:
        total = self._tc.count_messages(self._messages)
        limit = self._config.STM_TOKEN_LIMIT
        ratio = total / limit if limit > 0 else 0.0
        return ContextStats(
            total_tokens=total,
            message_count=len(self._messages),
            pinned_count=sum(1 for m in self._messages if m.is_pinned),
            utilisation_ratio=ratio,
            overflow_risk=ratio >= self._config.STM_WARNING_THRESHOLD,
        )

    def current_turn(self) -> int:
        return self._turn_index

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add_message(
        self,
        role: str,
        content: Optional[str] = None,
        is_pinned: bool = False,
        relevance_score: float = 1.0,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[list[dict]] = None,
    ) -> None:
        """Append a new message.  Pinned = never evicted by FILTER."""
        if not content:
            # convert no content to empty string
            content = ""
        msg = ContextMessage(
            role=role,
            content=content,
            turn_index=self._turn_index,
            token_estimate=self._tc.count(content or ""),
            relevance_score=relevance_score,
            is_pinned=is_pinned,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        )
        self._messages.append(msg)

    def update_pinned_system_message(self, new_content: str) -> bool:
        """
        Update the content of the pinned system message (main prompt).

        Args:
            new_content: The new system prompt content

        Returns:
            True if a pinned system message was found and updated, False otherwise
        """
        for i, msg in enumerate(self._messages):
            if msg.role == "system" and msg.is_pinned:
                # Create updated message with new content
                from dataclasses import replace
                updated_msg = replace(
                    msg,
                    content=new_content,
                    token_estimate=self._tc.count(new_content),
                )
                self._messages[i] = updated_msg
                return True
        return False

    def increment_turn(self) -> None:
        self._turn_index += 1

    # ── Memory operations ─────────────────────────────────────────────────────

    def retrieve(
        self,
        entries: list[MemoryEntry],
        trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
    ) -> MemoryOpResult:
        """
        Inject LTM entries into STM as system messages.
        Only injects entries that are not already in context.

        Pinning and relevance are driven by semantic similarity:
        - relevance_score = entry.similarity_score (not learning_score)
        - is_pinned only when similarity_score >= STM_MEMORY_PIN_THRESHOLD
        - Entries below the pin threshold are still injected but evictable.

        Handles both MemoryEntry objects and dict entries for robustness.
        """
        existing_ids = {
            m.content.split("]")[0].lstrip("[MEMORY:")
            for m in self._messages
            if m.role == "system" and "[MEMORY:" in m.content
        }
        pin_threshold = self._config.STM_MEMORY_PIN_THRESHOLD
        injected: list[str] = []

        def _get_entry_id(entry) -> str:
            """Get entry_id from MemoryEntry object or dict."""
            if isinstance(entry, dict):
                # Dicts from trigger_retrieval use 'source' or generate from content hash
                return entry.get("source") or entry.get("entry_id") or str(hash(entry.get("content", "")))
            return entry.entry_id

        def _get_content(entry) -> str:
            """Get content from MemoryEntry object or dict."""
            if isinstance(entry, dict):
                return entry.get("content", "")
            return entry.content

        def _get_similarity_score(entry) -> float:
            """Get similarity_score from MemoryEntry object or dict."""
            if isinstance(entry, dict):
                # Dicts use 'score' key for similarity
                return entry.get("score", entry.get("similarity_score", 0.0))
            return entry.similarity_score

        for entry in entries:
            entry_id = _get_entry_id(entry)
            content = _get_content(entry)
            similarity_score = _get_similarity_score(entry)

            tag = f"[MEMORY:{entry_id}]"
            if entry_id in existing_ids:
                continue
            should_pin = similarity_score >= pin_threshold
            self.add_message(
                role="system",
                content=f"{tag} {content}",
                is_pinned=should_pin,
                relevance_score=similarity_score,
            )
            injected.append(entry_id)

        return MemoryOpResult(
            op=MemoryOp.RETRIEVE,
            success=True,
            trigger=trigger,
            detail=f"Injected {len(injected)} LTM entries",
            entries_affected=injected,
        )

    def summary(
        self,
        trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
    ) -> MemoryOpResult:
        """
        Summarise the oldest STM_SUMMARY_WINDOW non-pinned messages.

        Requires self._summary_fn to be set (injected by Orchestrator).
        Falls back to a simple concatenation stub if not set, so the system
        never fails — but callers should always inject a real summary_fn.
        """
        candidates = [m for m in self._messages if not m.is_pinned]
        window = candidates[: self._config.STM_SUMMARY_WINDOW]
        if len(window) < 2:
            return MemoryOpResult(
                op=MemoryOp.SUMMARY,
                success=False,
                trigger=trigger,
                detail="Not enough non-pinned messages to summarise",
            )

        summary_text = (
            self._summary_fn(window)
            if self._summary_fn
            else self._fallback_summary(window)
        )

        # Replace the window with a single summarised message
        window_set = set(id(m) for m in window)
        self._messages = [m for m in self._messages if id(m) not in window_set]
        # Insert the summary at the position of the first removed message
        self._messages.insert(
            0,
            ContextMessage(
                role="system",
                content=f"[SUMMARY] {summary_text}",
                turn_index=window[0].turn_index,
                token_estimate=self._tc.count(summary_text),
                relevance_score=1.0,
                is_pinned=False,
            ),
        )

        saved = sum(m.token_estimate for m in window)
        return MemoryOpResult(
            op=MemoryOp.SUMMARY,
            success=True,
            trigger=trigger,
            detail=f"Compressed {len(window)} messages (~{saved} tokens saved)",
        )

    def filter(
        self,
        trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
        threshold: Optional[float] = None,
    ) -> MemoryOpResult:
        """
        Drop non-pinned messages whose relevance_score <= threshold.
        Never drops below STM_MIN_MESSAGES total.
        """
        threshold = threshold if threshold is not None else self._config.STM_EVICT_THRESHOLD
        min_keep = self._config.STM_MIN_MESSAGES

        # Separate pinned from eviction candidates
        pinned = [m for m in self._messages if m.is_pinned]
        evictable = [m for m in self._messages if not m.is_pinned]

        # Sort so highest-relevance are kept
        evictable.sort(key=lambda m: m.relevance_score, reverse=True)

        # Compute how many we can drop while honouring the minimum
        can_drop = max(0, len(self._messages) - min_keep)
        to_drop = [
            m for m in evictable if m.relevance_score <= threshold
        ][:can_drop]

        drop_ids = set(id(m) for m in to_drop)
        before = len(self._messages)
        self._messages = [m for m in self._messages if id(m) not in drop_ids]

        dropped = before - len(self._messages)
        return MemoryOpResult(
            op=MemoryOp.FILTER,
            success=True,
            trigger=trigger,
            detail=f"Dropped {dropped} low-relevance messages",
        )

    def force_fit(self) -> list[MemoryOpResult]:
        """
        Emergency overflow handler: ensure total tokens < STM_TOKEN_LIMIT.
        Called automatically before each LLM call.

        Strategy (in order):
          1. FILTER all sub-threshold messages
          2. SUMMARY if still over warning threshold
          3. Hard-drop oldest non-pinned messages until under limit
        Returns list of ops applied.
        """
        ops: list[MemoryOpResult] = []
        stats = self.stats()

        if stats.utilisation_ratio < self._config.STM_WARNING_THRESHOLD:
            return ops

        # Step 1: filter
        ops.append(self.filter(trigger=TriggerKind.SYSTEM_RULE))

        # Step 2: summary if still warm
        if self.stats().utilisation_ratio >= self._config.STM_WARNING_THRESHOLD:
            ops.append(self.summary(trigger=TriggerKind.SYSTEM_RULE))

        # Step 3: hard drop if still over critical
        while (
            self.stats().utilisation_ratio >= self._config.STM_CRITICAL_THRESHOLD
            and len(self._messages) > self._config.STM_MIN_MESSAGES
        ):
            non_pinned = [m for m in self._messages if not m.is_pinned]
            if not non_pinned:
                break
            # Drop the oldest non-pinned message
            oldest = min(non_pinned, key=lambda m: m.turn_index)
            self._messages.remove(oldest)
            ops.append(MemoryOpResult(
                op=MemoryOp.FILTER,
                success=True,
                trigger=TriggerKind.SYSTEM_RULE,
                detail="Emergency hard-drop oldest non-pinned message",
            ))

        return ops

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist current STM messages to disk."""
        data = [
            {
                "role": m.role,
                "content": m.content,
                "turn_index": m.turn_index,
                "token_estimate": m.token_estimate,
                "relevance_score": m.relevance_score,
                "is_pinned": m.is_pinned,
                "tool_call_id": m.tool_call_id,
                "tool_calls": m.tool_calls,
            }
            for m in self._messages
        ]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def load(self, path: Path) -> None:
        """Restore STM messages from disk."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._messages = [
                ContextMessage(
                    role=m["role"],
                    content=m["content"],
                    turn_index=m["turn_index"],
                    token_estimate=m["token_estimate"],
                    relevance_score=m["relevance_score"],
                    is_pinned=m["is_pinned"],
                    tool_call_id=m.get("tool_call_id"),
                    tool_calls=m.get("tool_calls"),
                )
                for m in data
            ]
            if self._messages:
                self._turn_index = max(m.turn_index for m in self._messages) + 1
        except Exception:
            # Corrupt state — start fresh rather than crash
            self._messages = []
            self._turn_index = 0

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_summary(messages: list[ContextMessage]) -> str:
        """
        Used when no summary_fn is injected.  Produces a lossless concatenation
        (no compression, but prevents crashes during testing).

        NOTE: This should never be used in production. It defeats the purpose
        of SUMMARY. Always inject a real summary_fn via Orchestrator.
        """
        parts = [f"[{m.role}]: {m.content or '(tool call)' if m.tool_calls else m.content or ''}" for m in messages]
        return "Previous context: " + " | ".join(parts)
