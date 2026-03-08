"""
core/types.py
─────────────
Canonical data contracts for AgeMem-hybrid.

Design notes
────────────
* Everything is a plain dataclass so it is trivially JSON-serialisable and
  works without any third-party library.
* Token counts are *estimated* via a simple word-split heuristic when the
  real tiktoken library is absent (network-restricted environment).  Callers
  that have tiktoken should override `TokenCounter` at construction time.
* LearningScore is the acceptance-criterion feedback signal: the agent rates
  0-1 how much it "learned" from an interaction.  This drives LTM promotion
  and STM eviction priority.
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class MemoryTier(str, Enum):
    STM = "stm"   # active context window
    LTM = "ltm"   # persistent store


class MemoryOp(str, Enum):
    """All operations the system or agent can invoke."""
    ADD      = "add"       # LTM: store new entry
    UPDATE   = "update"    # LTM: overwrite existing entry
    DELETE   = "delete"    # LTM: discard entry
    RETRIEVE = "retrieve"  # STM: pull LTM entry into context
    SUMMARY  = "summary"   # STM: compress a context segment
    FILTER   = "filter"    # STM: drop irrelevant context segment


class TriggerKind(str, Enum):
    """Who/what fired the memory operation."""
    SYSTEM_RULE    = "system_rule"    # deterministic threshold trigger
    MEMORY_AGENT   = "memory_agent"   # dedicated sub-agent decision
    MAIN_AGENT     = "main_agent"     # main LLM explicitly called the tool
    LEARNING_SCORE = "learning_score" # post-turn score drove promotion/eviction


# ──────────────────────────────────────────────────────────────────────────────
# Core data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single unit of long-term memory."""
    content: str
    entry_id: str = field(default_factory=lambda: "")
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    learning_score: float = 0.0     # aggregated signal from LearningFeedback
    tags: list[str] = field(default_factory=list)
    source_turn: int = 0            # which conversation turn created this

    def __post_init__(self) -> None:
        if not self.entry_id:
            self.entry_id = hashlib.sha1(
                f"{self.content}{self.created_at}".encode()
            ).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "learning_score": self.learning_score,
            "tags": self.tags,
            "source_turn": self.source_turn,
        }


@dataclass
class ContextMessage:
    """A single message in the active STM context window."""
    role: str                        # "system" | "user" | "assistant" | "tool"
    content: str
    turn_index: int = 0
    token_estimate: int = 0
    relevance_score: float = 1.0    # used by FILTER to rank eviction priority
    is_pinned: bool = False          # pinned messages are never evicted
    tool_call_id: Optional[str] = None  # for tool role: links to the tool call

    def to_openai_dict(self) -> dict:
        result = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class LearningFeedback:
    """
    Post-turn feedback from the main agent.

    The main agent answers: 'On a 0-1 scale, how much new information did
    you just encounter that you want to retain?'

    This score drives:
      • score >= LTM_PROMOTE_THRESHOLD  → candidate for LTM ADD
      • score <= STM_EVICT_THRESHOLD    → low-value message, FILTER candidate
    """
    score: float            # 0.0 (nothing learned) – 1.0 (highly novel)
    rationale: str = ""     # optional explanation from agent
    turn_index: int = 0
    affected_content: str = "" # excerpt the agent considers learnable


@dataclass
class MemoryOpResult:
    """Returned by every memory operation to caller."""
    op: MemoryOp
    success: bool
    trigger: TriggerKind
    detail: str = ""
    entries_affected: list[str] = field(default_factory=list)  # entry_ids


@dataclass
class ContextStats:
    """Snapshot of STM health, emitted after every turn."""
    total_tokens: int
    message_count: int
    pinned_count: int
    utilisation_ratio: float   # total_tokens / STM_TOKEN_LIMIT
    overflow_risk: bool        # utilisation_ratio >= WARNING_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# Token estimation (stdlib fallback)
# ──────────────────────────────────────────────────────────────────────────────

class TokenCounter:
    """
    Approximate token counter.

    Replace with a real tiktoken-based implementation when the environment
    allows network access:

        from tiktoken import encoding_for_model
        counter = TokenCounter(encoding_for_model("gpt-4o"))
    """

    def __init__(self, encoder=None):
        self._enc = encoder  # optional tiktoken encoder

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        # Heuristic: ~0.75 tokens per word is a reasonable approximation for
        # English text. We add 4 per message for OpenAI's framing overhead.
        words = len(text.split())
        return max(1, int(words * 0.75)) + 4

    def count_messages(self, messages: list[ContextMessage]) -> int:
        return sum(self.count(m.content) for m in messages)
