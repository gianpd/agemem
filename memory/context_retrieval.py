"""
memory/context_retrieval.py
---------------------------
Context-aware LTM retrieval using conversation window embeddings.

Responsibilities
────────────────
* Compute weighted context embeddings from recent conversation turns
* Retrieve LTM entries using context-aware vector search
* Cache recent message embeddings for efficiency
* Provide fallback to query-only search when needed

Design decisions
────────────────
* Uses weighted average of embeddings across a sliding context window
* Weights decay for older turns (current > previous > older)
* Embeddings are cached by turn_index to avoid re-computation
* Falls back to query-only search if context-aware retrieval returns no results
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from core.config import AgememConfig
from core.types import ContextMessage, MemoryEntry
from memory.embedding import embed_text, embed_batch, cosine_similarity

if TYPE_CHECKING:
    from memory.ltm_store import LTMStore

logger = logging.getLogger(__name__)


@dataclass
class ContextRetrievalConfig:
    """Configuration for context-aware retrieval."""

    window_size: int = 3
    """Number of recent turns to include in context window."""

    current_weight: float = 0.50
    """Weight for current query embedding."""

    previous_weight: float = 0.30
    """Weight for previous turn embedding."""

    turn_before_weight: float = 0.15
    """Weight for turn-before-previous embedding."""

    oldest_weight: float = 0.05
    """Weight for oldest turn in context window."""

    min_similarity_threshold: float = 0.65
    """Minimum similarity score to include an LTM entry."""

    fallback_to_query_only: bool = True
    """If True, falls back to query-only search when context-aware returns no results."""

    enable_caching: bool = True
    """Cache embeddings for recent messages to avoid re-computation."""

    cache_size: int = 20
    """Maximum number of embeddings to cache."""

    @classmethod
    def from_agemem_config(cls, config: AgememConfig) -> "ContextRetrievalConfig":
        """Create config from AgememConfig settings."""
        return cls(
            window_size=config.CONTEXT_WINDOW_SIZE,
            current_weight=config.CONTEXT_CURRENT_QUERY_WEIGHT,
            previous_weight=config.CONTEXT_PREVIOUS_TURN_WEIGHT,
            turn_before_weight=config.CONTEXT_TURN_BEFORE_WEIGHT,
            oldest_weight=config.CONTEXT_OLDEST_TURN_WEIGHT,
            min_similarity_threshold=config.CONTEXT_MIN_SIMILARITY_THRESHOLD,
            fallback_to_query_only=config.CONTEXT_FALLBACK_TO_QUERY_ONLY,
            enable_caching=True,
            cache_size=config.CONTEXT_WINDOW_SIZE * 2 + 4,
        )


class ContextAwareRetriever:
    """
    Retrieves LTM entries based on conversation context window.

    This retriever considers not just the current query, but the recent
    conversation flow when searching for relevant LTM entries. This improves
    retrieval relevance when the conversation has established context that
    should influence memory retrieval.

    Usage:
        retriever = ContextAwareRetriever(ltm_store, config)
        entries = retriever.retrieve(
            current_query="What about Python?",
            recent_messages=stm.messages(),
            current_turn=stm.current_turn(),
            top_k=5,
        )
    """

    def __init__(
        self,
        ltm_store: LTMStore,
        config: Optional[ContextRetrievalConfig] = None,
    ) -> None:
        self._ltm = ltm_store
        self._config = config or ContextRetrievalConfig()
        self._embedding_cache: dict[int, np.ndarray] = {}
        self._fallback_count: int = 0
        self._total_calls: int = 0

    def retrieve(
        self,
        current_query: str,
        recent_messages: list[ContextMessage],
        current_turn: int,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """
        Retrieve LTM entries using context-aware embedding.

        Args:
            current_query: The current user query string.
            recent_messages: List of recent messages from STM.
            current_turn: The current turn index.
            top_k: Number of results to return.

        Returns:
            List of MemoryEntry objects sorted by contextual relevance.
        """
        self._total_calls += 1

        # Extract recent user messages for context
        context_messages = self._extract_user_context(recent_messages, current_turn)

        # Compute weighted context embedding
        context_emb = self._compute_context_embedding(current_query, context_messages)

        if context_emb is None:
            logger.warning("Failed to compute context embedding, falling back to query-only")
            self._fallback_count += 1
            return self._ltm.search(current_query, top_k=top_k)

        # Search LTM using context-aware vector
        results = self._ltm.search_by_vector(
            query_vector=context_emb,
            top_k=top_k,
            min_similarity=self._config.min_similarity_threshold,
        )

        # Fallback if needed
        if not results and self._config.fallback_to_query_only:
            logger.debug("Context-aware retrieval returned no results, falling back to query-only")
            self._fallback_count += 1
            results = self._ltm.search(current_query, top_k=top_k)

        return results

    def get_stats(self) -> dict:
        """Return retrieval statistics for monitoring."""
        return {
            "total_calls": self._total_calls,
            "fallback_count": self._fallback_count,
            "fallback_rate": self._fallback_count / max(1, self._total_calls),
            "cache_size": len(self._embedding_cache),
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    def _extract_user_context(
        self,
        messages: list[ContextMessage],
        current_turn: int,
    ) -> list[tuple[str, int]]:
        """
        Extract recent user messages for context window.

        Returns:
            List of (content, turn_index) tuples for recent user messages,
            ordered from most recent to oldest (excluding current turn).
        """
        # Filter user messages from recent turns
        user_msgs = [
            (m.content, m.turn_index)
            for m in messages
            if m.role == "user"
            and m.content
            and m.turn_index < current_turn  # Exclude current turn
        ]

        # Sort by turn_index descending (most recent first)
        user_msgs.sort(key=lambda x: x[1], reverse=True)

        # Take up to window_size messages
        return user_msgs[: self._config.window_size]

    def _compute_context_embedding(
        self,
        current_query: str,
        context_messages: list[tuple[str, int]],
    ) -> Optional[np.ndarray]:
        """
        Compute weighted average embedding across context window.

        Uses batched embedding for efficiency and caches results.

        Args:
            current_query: The current user query.
            context_messages: List of (content, turn_index) tuples.

        Returns:
            Normalized context embedding vector, or None on failure.
        """
        try:
            # Gather texts and their weights
            texts = [current_query]
            weights = [self._config.current_weight]
            turn_indices = [-1]  # -1 indicates current query (not cached)

            # Add context messages with their weights
            weight_map = [
                self._config.previous_weight,
                self._config.turn_before_weight,
                self._config.oldest_weight,
            ]

            for i, (content, turn_idx) in enumerate(context_messages):
                if i < len(weight_map):
                    texts.append(content)
                    weights.append(weight_map[i])
                    turn_indices.append(turn_idx)

            # Compute or retrieve embeddings
            embeddings = []
            for text, turn_idx in zip(texts, turn_indices):
                emb = self._get_or_compute_embedding(text, turn_idx)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    # If any embedding fails, skip this text but continue
                    weights = weights[: len(embeddings)]

            if not embeddings:
                return None

            # Normalize weights to sum to 1
            effective_weights = weights[: len(embeddings)]
            total_weight = sum(effective_weights)
            if total_weight == 0:
                return None

            normalized_weights = [w / total_weight for w in effective_weights]

            # Compute weighted average
            result = np.zeros_like(embeddings[0])
            for emb, weight in zip(embeddings, normalized_weights):
                result += emb * weight

            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm

            return result

        except Exception as e:
            logger.error(f"Failed to compute context embedding: {e}")
            return None

    def _get_or_compute_embedding(
        self,
        text: str,
        turn_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Get embedding from cache or compute it.

        Args:
            text: Text to embed.
            turn_idx: Turn index for caching (-1 for current query, not cached).

        Returns:
            Embedding vector or None on failure.
        """
        # Check cache for message embeddings (not for current query)
        if (
            self._config.enable_caching
            and turn_idx >= 0
            and turn_idx in self._embedding_cache
        ):
            return self._embedding_cache[turn_idx]

        # Compute embedding
        try:
            emb = embed_text(text)

            # Cache if applicable
            if self._config.enable_caching and turn_idx >= 0:
                self._embedding_cache[turn_idx] = emb
                self._prune_cache()

            return emb

        except Exception as e:
            logger.warning(f"Failed to embed text: {e}")
            return None

    def _prune_cache(self) -> None:
        """Prune cache to keep size under limit."""
        if len(self._embedding_cache) <= self._config.cache_size:
            return

        # Remove oldest entries (lowest turn indices)
        sorted_turns = sorted(self._embedding_cache.keys())
        to_remove = len(self._embedding_cache) - self._config.cache_size
        for turn in sorted_turns[:to_remove]:
            del self._embedding_cache[turn]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function for direct usage
# ──────────────────────────────────────────────────────────────────────────────


def retrieve_with_context(
    ltm_store: LTMStore,
    current_query: str,
    recent_messages: list[ContextMessage],
    current_turn: int,
    config: Optional[AgememConfig] = None,
    top_k: int = 5,
) -> list[MemoryEntry]:
    """
    Convenience function for one-off context-aware retrieval.

    Args:
        ltm_store: The LTM store to search.
        current_query: Current user query.
        recent_messages: Recent messages from STM.
        current_turn: Current turn index.
        config: Optional AgememConfig (uses defaults if not provided).
        top_k: Number of results to return.

    Returns:
        List of MemoryEntry objects.
    """
    if config and config.CONTEXT_AWARE_RETRIEVAL:
        ctx_config = ContextRetrievalConfig.from_agemem_config(config)
        retriever = ContextAwareRetriever(ltm_store, ctx_config)
        return retriever.retrieve(current_query, recent_messages, current_turn, top_k)
    else:
        # Fallback to standard search
        return ltm_store.search(current_query, top_k=top_k)
