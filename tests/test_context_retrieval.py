"""
Test Context-Aware LTM Retrieval
================================

Tests for the context-aware retrieval functionality.

Run with: python -m pytest tests/test_context_retrieval.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from memory.context_retrieval import (
    ContextAwareRetriever,
    ContextRetrievalConfig,
    retrieve_with_context,
)
from core.types import ContextMessage, MemoryEntry
from core.config import AgememConfig


@dataclass
class MockLTMStore:
    """Mock LTM store for testing."""

    def __init__(self, entries=None):
        self._entries = entries or []
        self.search_calls = []
        self.vector_search_calls = []

    def search(self, query, top_k=5):
        self.search_calls.append((query, top_k))
        return self._entries[:top_k]

    def search_by_vector(self, query_vector, top_k=5, min_similarity=None):
        self.vector_search_calls.append((query_vector, top_k, min_similarity))
        return self._entries[:top_k]


class TestContextRetrievalConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        config = ContextRetrievalConfig()
        assert config.window_size == 3
        assert config.current_weight == 0.50
        assert config.previous_weight == 0.30
        assert config.turn_before_weight == 0.15
        assert config.oldest_weight == 0.05
        assert config.min_similarity_threshold == 0.65
        assert config.fallback_to_query_only is True

    def test_from_agemem_config(self):
        agemem_config = AgememConfig(
            CONTEXT_AWARE_RETRIEVAL=True,
            CONTEXT_WINDOW_SIZE=5,
            CONTEXT_CURRENT_QUERY_WEIGHT=0.60,
            CONTEXT_PREVIOUS_TURN_WEIGHT=0.25,
            CONTEXT_MIN_SIMILARITY_THRESHOLD=0.70,
        )

        ctx_config = ContextRetrievalConfig.from_agemem_config(agemem_config)

        assert ctx_config.window_size == 5
        assert ctx_config.current_weight == 0.60
        assert ctx_config.previous_weight == 0.25
        assert ctx_config.min_similarity_threshold == 0.70


class TestContextAwareRetriever:
    """Test the context-aware retriever."""

    def test_initialization(self):
        ltm = MockLTMStore()
        config = ContextRetrievalConfig()
        retriever = ContextAwareRetriever(ltm, config)

        assert retriever._ltm == ltm
        assert retriever._config == config
        assert retriever._fallback_count == 0
        assert retriever._total_calls == 0

    def test_extract_user_context(self):
        ltm = MockLTMStore()
        retriever = ContextAwareRetriever(ltm)

        # Create test messages
        messages = [
            ContextMessage(role="system", content="System prompt", turn_index=0),
            ContextMessage(role="user", content="First question", turn_index=0),
            ContextMessage(role="assistant", content="First answer", turn_index=0),
            ContextMessage(role="user", content="Second question", turn_index=1),
            ContextMessage(role="assistant", content="Second answer", turn_index=1),
            ContextMessage(role="user", content="Current question", turn_index=2),
        ]

        context = retriever._extract_user_context(messages, current_turn=2)

        # Should get previous user messages (not current turn)
        assert len(context) == 2
        assert context[0] == ("Second question", 1)
        assert context[1] == ("First question", 0)

    def test_extract_user_context_with_pinned(self):
        ltm = MockLTMStore()
        retriever = ContextAwareRetriever(ltm)

        messages = [
            ContextMessage(role="user", content="Pinned memory", turn_index=0, is_pinned=True),
            ContextMessage(role="user", content="Regular question", turn_index=0),
        ]

        # Pinned messages should still be included (they're still user messages)
        context = retriever._extract_user_context(messages, current_turn=1)
        assert len(context) == 2

    def test_empty_messages(self):
        ltm = MockLTMStore()
        retriever = ContextAwareRetriever(ltm)

        context = retriever._extract_user_context([], current_turn=0)
        assert context == []

    def test_stats_tracking(self):
        ltm = MockLTMStore()
        retriever = ContextAwareRetriever(ltm)

        # Simulate some calls
        retriever._total_calls = 10
        retriever._fallback_count = 3

        stats = retriever.get_stats()
        assert stats["total_calls"] == 10
        assert stats["fallback_count"] == 3
        assert stats["fallback_rate"] == 0.3

    def test_clear_cache(self):
        ltm = MockLTMStore()
        retriever = ContextAwareRetriever(ltm)

        # Add some fake cache entries
        retriever._embedding_cache = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}

        retriever.clear_cache()
        assert len(retriever._embedding_cache) == 0


class TestContextEmbeddingComputation:
    """Test the context embedding computation."""

    @patch("memory.context_retrieval.embed_text")
    def test_compute_context_embedding(self, mock_embed):
        ltm = MockLTMStore()
        retriever = ContextAwareRetriever(ltm)

        # Mock embeddings (unit vectors for simplicity)
        def mock_embedding(text):
            if text == "current":
                return np.array([1.0, 0.0])  # x-axis
            elif text == "previous":
                return np.array([0.0, 1.0])  # y-axis
            else:
                return np.array([0.707, 0.707])  # 45 degrees

        mock_embed.side_effect = mock_embedding

        result = retriever._compute_context_embedding(
            current_query="current",
            context_messages=[("previous", 0)],
        )

        # Result should be a normalized vector
        assert result is not None
        assert len(result) == 2
        assert np.abs(np.linalg.norm(result) - 1.0) < 1e-6  # Should be unit vector

    @patch("memory.context_retrieval.embed_text")
    def test_compute_with_multiple_context(self, mock_embed):
        ltm = MockLTMStore()
        config = ContextRetrievalConfig(
            current_weight=0.5,
            previous_weight=0.3,
            turn_before_weight=0.2,
        )
        retriever = ContextAwareRetriever(ltm, config)

        # Create orthogonal embeddings to test weighting
        mock_embed.side_effect = lambda text: {
            "current": np.array([1.0, 0.0]),
            "prev": np.array([0.0, 1.0]),
            "before": np.array([-1.0, 0.0]),
        }.get(text, np.array([1.0, 0.0]))

        result = retriever._compute_context_embedding(
            current_query="current",
            context_messages=[("prev", 1), ("before", 0)],
        )

        assert result is not None
        # With these weights and orthogonal vectors, result should be
        # dominated by the current query (x-axis)
        assert result[0] >= result[1]  # x component should be larger or equal

    def test_compute_embedding_failure(self):
        ltm = MockLTMStore()
        retriever = ContextAwareRetriever(ltm)

        # Test with no embeddings available
        with patch("memory.context_retrieval.embed_text", side_effect=Exception("Embedding failed")):
            result = retriever._compute_context_embedding("query", [])
            assert result is None


class TestRetrieveWithContext:
    """Test the full retrieval flow."""

    @patch("memory.context_retrieval.embed_text")
    def test_retrieve_with_fallback(self, mock_embed):
        """Test that fallback works when context-aware returns no results."""
        ltm = MockLTMStore(entries=[])
        config = ContextRetrievalConfig(fallback_to_query_only=True)
        retriever = ContextAwareRetriever(ltm, config)

        mock_embed.return_value = np.array([1.0, 0.0, 0.0, 0.0])  # Unit vector

        messages = [
            ContextMessage(role="user", content="Previous", turn_index=0),
        ]

        results = retriever.retrieve(
            current_query="Current query",
            recent_messages=messages,
            current_turn=1,
            top_k=5,
        )

        # Should have fallen back to regular search
        assert len(ltm.search_calls) == 1
        assert ltm.search_calls[0][0] == "Current query"

    @patch("memory.context_retrieval.embed_text")
    def test_retrieve_without_fallback(self, mock_embed):
        """Test that no fallback occurs when disabled."""
        ltm = MockLTMStore(entries=[])
        config = ContextRetrievalConfig(fallback_to_query_only=False)
        retriever = ContextAwareRetriever(ltm, config)

        mock_embed.return_value = np.array([1.0, 0.0, 0.0, 0.0])

        messages = []

        results = retriever.retrieve(
            current_query="Query",
            recent_messages=messages,
            current_turn=0,
            top_k=5,
        )

        # Should not have called regular search
        assert len(ltm.search_calls) == 0
        # But should have called vector search
        assert len(ltm.vector_search_calls) == 1

    @patch("memory.context_retrieval.embed_text")
    def test_convenience_function(self, mock_embed):
        """Test the convenience function retrieve_with_context."""
        ltm = MockLTMStore()
        config = AgememConfig(CONTEXT_AWARE_RETRIEVAL=True)

        mock_embed.return_value = np.array([1.0, 0.0, 0.0, 0.0])

        results = retrieve_with_context(
            ltm_store=ltm,
            current_query="test",
            recent_messages=[],
            current_turn=0,
            config=config,
            top_k=3,
        )

        # Should use vector search
        assert len(ltm.vector_search_calls) == 1
        assert ltm.vector_search_calls[0][1] == 3  # top_k

    def test_convenience_function_disabled(self):
        """Test that convenience function falls back when disabled."""
        ltm = MockLTMStore()
        config = AgememConfig(CONTEXT_AWARE_RETRIEVAL=False)

        results = retrieve_with_context(
            ltm_store=ltm,
            current_query="test",
            recent_messages=[],
            current_turn=0,
            config=config,
        )

        # Should use regular search
        assert len(ltm.search_calls) == 1
        assert len(ltm.vector_search_calls) == 0


class TestCacheManagement:
    """Test embedding cache behavior."""

    @patch("memory.context_retrieval.embed_text")
    def test_caching_behavior(self, mock_embed):
        ltm = MockLTMStore()
        config = ContextRetrievalConfig(enable_caching=True, cache_size=3)
        retriever = ContextAwareRetriever(ltm, config)

        mock_embed.return_value = np.array([1.0, 0.0])

        # First call should compute and cache
        retriever._get_or_compute_embedding("text", turn_idx=0)
        assert mock_embed.call_count == 1
        assert 0 in retriever._embedding_cache

        # Second call with same turn_idx should use cache
        retriever._get_or_compute_embedding("different text", turn_idx=0)
        assert mock_embed.call_count == 1  # No new embedding computed

        # Call with new turn_idx should compute new embedding
        retriever._get_or_compute_embedding("text", turn_idx=1)
        assert mock_embed.call_count == 2

    @patch("memory.context_retrieval.embed_text")
    def test_cache_pruning(self, mock_embed):
        ltm = MockLTMStore()
        config = ContextRetrievalConfig(enable_caching=True, cache_size=2)
        retriever = ContextAwareRetriever(ltm, config)

        mock_embed.return_value = np.array([1.0, 0.0])

        # Add more entries than cache size
        for i in range(5):
            retriever._get_or_compute_embedding(f"text {i}", turn_idx=i)

        # Cache should be pruned to size 2
        assert len(retriever._embedding_cache) == 2
        # Oldest entries should be removed
        assert 0 not in retriever._embedding_cache
        assert 1 not in retriever._embedding_cache
        assert 2 not in retriever._embedding_cache
        # Newest should remain
        assert 3 in retriever._embedding_cache
        assert 4 in retriever._embedding_cache

    @patch("memory.context_retrieval.embed_text")
    def test_no_caching_for_current_query(self, mock_embed):
        ltm = MockLTMStore()
        retriever = ContextAwareRetriever(ltm)

        mock_embed.return_value = np.array([1.0, 0.0])

        # Current query (turn_idx=-1) should not be cached
        retriever._get_or_compute_embedding("query", turn_idx=-1)
        assert len(retriever._embedding_cache) == 0
        assert mock_embed.call_count == 1

        # Second call should recompute
        retriever._get_or_compute_embedding("query", turn_idx=-1)
        assert mock_embed.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
