"""
memory/ltm_store.py
────────────────────
Long-Term Memory store.

Responsibilities
────────────────
* ADD     – store a new MemoryEntry
* UPDATE  – overwrite an existing entry (by entry_id or content similarity)
* DELETE  – remove an entry
* SEARCH  – naive keyword retrieval (no embeddings; inference-only constraint)
* PRUNE   – enforce LTM_MAX_ENTRIES by dropping lowest-scored entries

Design decisions
────────────────
* No vector DB.  Retrieval is done by token-overlap scoring, which is
  feasible without any network or heavy library.  The expected entry count
  (≤500) makes O(n) scan acceptable.
* Thread safety is not a concern for a single-agent inference loop but a
  reentrant lock is included for completeness.
* Persistence: entries are serialised to JSON.  Callers can pass a path to
  enable persistence across sessions.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Optional

from core.types import MemoryEntry, MemoryOp, MemoryOpResult, TriggerKind
from core.config import AgememConfig, DEFAULT_CONFIG


class LTMStore:

    def __init__(
        self,
        config: AgememConfig = DEFAULT_CONFIG,
        persist_path: Optional[Path] = None,
    ) -> None:
        self._config = config
        self._entries: dict[str, MemoryEntry] = {}
        self._lock = threading.RLock()
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load(persist_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        learning_score: float = 0.0,
        tags: list[str] | None = None,
        source_turn: int = 0,
        trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
    ) -> MemoryOpResult:
        """
        Add a new entry.  If a near-duplicate exists and the learning_score is
        above UPDATE_THRESHOLD, updates it instead.
        """
        with self._lock:
            existing = self._find_similar(content)
            if existing and learning_score >= self._config.LTM_UPDATE_THRESHOLD:
                return self.update(
                    existing.entry_id,
                    content=content,
                    learning_score=learning_score,
                    trigger=trigger,
                )

            entry = MemoryEntry(
                content=content,
                learning_score=learning_score,
                tags=tags or [],
                source_turn=source_turn,
            )
            self._entries[entry.entry_id] = entry
            self._maybe_prune()
            self._maybe_persist()

            return MemoryOpResult(
                op=MemoryOp.ADD,
                success=True,
                trigger=trigger,
                detail=f"Stored entry {entry.entry_id}",
                entries_affected=[entry.entry_id],
            )

    def update(
        self,
        entry_id: str,
        content: str,
        learning_score: float = 0.0,
        trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
    ) -> MemoryOpResult:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return MemoryOpResult(
                    op=MemoryOp.UPDATE,
                    success=False,
                    trigger=trigger,
                    detail=f"Entry {entry_id} not found",
                )
            entry.content = content
            entry.updated_at = time.time()
            # Exponential moving average of learning scores
            entry.learning_score = 0.6 * entry.learning_score + 0.4 * learning_score
            self._maybe_persist()
            return MemoryOpResult(
                op=MemoryOp.UPDATE,
                success=True,
                trigger=trigger,
                detail=f"Updated entry {entry_id}",
                entries_affected=[entry_id],
            )

    def delete(
        self,
        entry_id: str,
        trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
    ) -> MemoryOpResult:
        with self._lock:
            if entry_id not in self._entries:
                return MemoryOpResult(
                    op=MemoryOp.DELETE,
                    success=False,
                    trigger=trigger,
                    detail=f"Entry {entry_id} not found",
                )
            del self._entries[entry_id]
            self._maybe_persist()
            return MemoryOpResult(
                op=MemoryOp.DELETE,
                success=True,
                trigger=trigger,
                detail=f"Deleted entry {entry_id}",
                entries_affected=[entry_id],
            )

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        Retrieve the top_k most relevant entries for a query.

        Scoring: TF-IDF-inspired token overlap + recency decay + learning_score.
        All O(n) — adequate for ≤500 entries without any embedding model.
        """
        with self._lock:
            if not self._entries:
                return []
            query_tokens = self._tokenise(query)
            scored: list[tuple[float, MemoryEntry]] = []
            now = time.time()
            for entry in self._entries.values():
                overlap = self._overlap_score(query_tokens, entry.content)
                # Recency: exponential decay over 7 days
                age_days = (now - entry.updated_at) / 86_400
                recency = math.exp(-age_days / 7)
                score = 0.5 * overlap + 0.3 * entry.learning_score + 0.2 * recency
                scored.append((score, entry))

            scored.sort(key=lambda t: t[0], reverse=True)
            results = [e for _, e in scored[:top_k]]
            # Increment access count for retrieved entries
            for e in results:
                e.access_count += 1
            return results

    def all_entries(self) -> list[MemoryEntry]:
        with self._lock:
            return list(self._entries.values())

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> set[str]:
        """Lowercase word tokens, stopwords excluded."""
        STOPWORDS = {
            "the", "a", "an", "is", "in", "on", "at", "to", "of",
            "and", "or", "but", "for", "with", "was", "be", "are",
        }
        return {
            w.lower().strip(".,!?;:\"'()")
            for w in text.split()
            if w.lower() not in STOPWORDS and len(w) > 2
        }

    def _overlap_score(self, query_tokens: set[str], content: str) -> float:
        """Jaccard-like overlap between query and content token sets."""
        content_tokens = self._tokenise(content)
        if not query_tokens or not content_tokens:
            return 0.0
        intersection = query_tokens & content_tokens
        union = query_tokens | content_tokens
        return len(intersection) / len(union)

    def _find_similar(self, content: str) -> Optional[MemoryEntry]:
        """
        Naive duplicate detection: compare first N words.
        Returns an existing entry if it looks like the same knowledge unit.
        """
        n = self._config.LTM_SIMILARITY_WORDS
        lead = " ".join(content.split()[:n]).lower()
        for entry in self._entries.values():
            entry_lead = " ".join(entry.content.split()[:n]).lower()
            if entry_lead == lead:
                return entry
        return None

    def _maybe_prune(self) -> None:
        """Drop entries beyond LTM_MAX_ENTRIES, lowest learning_score first."""
        if len(self._entries) <= self._config.LTM_MAX_ENTRIES:
            return
        sorted_ids = sorted(
            self._entries,
            key=lambda eid: self._entries[eid].learning_score,
        )
        excess = len(self._entries) - self._config.LTM_MAX_ENTRIES
        for eid in sorted_ids[:excess]:
            del self._entries[eid]

    def _maybe_persist(self) -> None:
        if self._persist_path is None:
            return
        data = [e.to_dict() for e in self._entries.values()]
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
            for item in data:
                entry = MemoryEntry(**item)
                self._entries[entry.entry_id] = entry
        except Exception:
            pass  # corrupt file → start fresh
