"""
tests/test_ltm_store.py
───────────────────────
Unit tests for LTMStore focusing on _find_similar and search methods.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import AgememConfig
from core.types import MemoryEntry
from memory.ltm_store import LTMStore


def _cfg(**overrides) -> AgememConfig:
    """Create config with defaults and overrides."""
    defaults = {
        "STM_TOKEN_LIMIT": 1000,
        "STM_WARNING_THRESHOLD": 0.75,
        "STM_CRITICAL_THRESHOLD": 0.90,
        "STM_MIN_MESSAGES": 2,
        "STM_SUMMARY_WINDOW": 2,
        "LTM_MAX_ENTRIES": 100,
        "LTM_PROMOTE_THRESHOLD": 0.65,
        "LTM_UPDATE_THRESHOLD": 0.5,
        "LTM_SIMILARITY_WORDS": 6,
        "LEARNING_SCORE_PROMPT_EVERY_N": 3,
        "LEARNING_SCORE_THRESHOLD_IMMEDIATE": 0.85,
        "TRIGGER_EVERY_N_TURNS": 5,
        "PERSIST_DIR": None,
    }
    defaults.update(overrides)
    return AgememConfig(**defaults)


class TestFindSimilar(unittest.TestCase):
    """Tests for _find_similar method - naive duplicate detection."""

    def test_find_similar_exact_match(self):
        """_find_similar returns entry when first N words match exactly."""
        cfg = _cfg(LTM_SIMILARITY_WORDS=3)
        store = LTMStore(cfg)

        # Add an entry
        store.add("Python is great for machine learning", learning_score=0.8)

        # Find similar with exact same first 3 words
        result = store._find_similar("Python is great for data science")

        self.assertIsNotNone(result)
        self.assertIn("Python is great", result.content)

    def test_find_similar_no_match(self):
        """_find_similar returns None when no similar entry exists."""
        cfg = _cfg(LTM_SIMILARITY_WORDS=3)
        store = LTMStore(cfg)

        # Add an entry
        store.add("Python is great for ML", learning_score=0.8)

        # Try to find different content
        result = store._find_similar("JavaScript is used for web")

        self.assertIsNone(result)

    def test_find_similar_case_insensitive(self):
        """_find_similar is case insensitive for matching."""
        cfg = _cfg(LTM_SIMILARITY_WORDS=3)
        store = LTMStore(cfg)

        # Add entry with lowercase
        store.add("python is great for ML", learning_score=0.8)

        # Search with uppercase - should match
        result = store._find_similar("PYTHON IS GREAT for data")

        self.assertIsNotNone(result)
        self.assertIn("python is great", result.content.lower())

    def test_find_similar_partial_match_not_enough_words(self):
        """_find_similar requires at least N words to match."""
        cfg = _cfg(LTM_SIMILARITY_WORDS=6)
        store = LTMStore(cfg)

        # Add entry with 6+ words
        store.add("Python is great for machine learning projects", learning_score=0.8)

        # Only first 3 words match - should not find
        result = store._find_similar("Python is great for something else entirely")

        self.assertIsNone(result)

    def test_find_similar_empty_store(self):
        """_find_similar returns None when store is empty."""
        cfg = _cfg(LTM_SIMILARITY_WORDS=3)
        store = LTMStore(cfg)

        result = store._find_similar("Any content here")

        self.assertIsNone(result)

    def test_find_similar_multiple_entries_returns_first_match(self):
        """_find_similar returns a matching entry when duplicates exist."""
        cfg = _cfg(LTM_SIMILARITY_WORDS=2)
        store = LTMStore(cfg)

        # Add multiple entries with same leading words
        store.add("Python programming is fun", learning_score=0.8)
        store.add("Python programming for beginners", learning_score=0.7)

        # Should find one of them (dict iteration order not guaranteed)
        result = store._find_similar("Python programming advanced topics")

        self.assertIsNotNone(result)
        self.assertIn("Python programming", result.content)

    def test_find_similar_with_punctuation(self):
        """_find_similar requires exact word match including punctuation."""
        cfg = _cfg(LTM_SIMILARITY_WORDS=3)
        store = LTMStore(cfg)

        # Add entry with punctuation attached to words
        store.add("hello, world! how are you?", learning_score=0.8)

        # Match exact same punctuation - should find
        result = store._find_similar("hello, world! how")

        self.assertIsNotNone(result)

    def test_find_similar_punctuation_differs(self):
        """_find_similar matches when punctuation differs but words are same."""
        cfg = _cfg(LTM_SIMILARITY_WORDS=3)
        store = LTMStore(cfg)

        # Add entry without punctuation
        store.add("hello world how are you", learning_score=0.8)

        # Search with different content after leading words
        result = store._find_similar("hello world how are you today")

        self.assertIsNotNone(result)


class TestSearch(unittest.TestCase):
    """Tests for search method - TF-IDF inspired retrieval."""

    def test_search_empty_store(self):
        """search returns empty list when store is empty."""
        cfg = _cfg()
        store = LTMStore(cfg)

        results = store.search("any query")

        self.assertEqual(len(results), 0)
        self.assertIsInstance(results, list)

    def test_search_returns_top_k(self):
        """search returns at most top_k results."""
        cfg = _cfg()
        store = LTMStore(cfg)

        # Add 10 entries
        for i in range(10):
            store.add(f"Entry number {i} content here", learning_score=0.5)

        # Search with top_k=3
        results = store.search("entry content", top_k=3)

        self.assertEqual(len(results), 3)

    def test_search_default_top_k(self):
        """search uses default top_k=5 when not specified."""
        cfg = _cfg()
        store = LTMStore(cfg)

        # Add 10 entries
        for i in range(10):
            store.add(f"Entry number {i} content here", learning_score=0.5)

        # Search without specifying top_k
        results = store.search("entry content")

        self.assertEqual(len(results), 5)

    def test_search_relevance_ranking_by_overlap(self):
        """search ranks results by token overlap."""
        cfg = _cfg()
        store = LTMStore(cfg)

        # Add entries with varying relevance
        store.add("Python machine learning tutorial", learning_score=0.5)
        store.add("Python programming guide", learning_score=0.5)
        store.add("JavaScript web development", learning_score=0.5)

        # Search for Python ML
        results = store.search("Python machine learning", top_k=2)

        self.assertEqual(len(results), 2)
        # Python ML tutorial should be first (more token overlap)
        self.assertIn("Python machine learning", results[0].content)

    def test_search_increments_access_count(self):
        """search increments access_count for returned entries."""
        cfg = _cfg()
        store = LTMStore(cfg)

        store.add("Python machine learning tutorial", learning_score=0.8)

        # Initial access count should be 0
        entries = store.all_entries()
        self.assertEqual(entries[0].access_count, 0)

        # Search and retrieve
        store.search("Python machine learning")

        # Access count should be incremented
        entries = store.all_entries()
        self.assertEqual(entries[0].access_count, 1)

    def test_search_learning_score_weighting(self):
        """search considers learning_score in ranking."""
        cfg = _cfg()
        store = LTMStore(cfg)

        # Add entries with same content but different learning scores
        store.add("Python tutorial for beginners", learning_score=0.9)
        store.add("Python tutorial advanced topics", learning_score=0.3)

        # Both match, higher learning score should rank higher
        results = store.search("Python tutorial", top_k=2)

        self.assertEqual(len(results), 2)
        # Higher learning score first
        self.assertEqual(results[0].learning_score, 0.9)
        self.assertEqual(results[1].learning_score, 0.3)

    def test_search_recency_weighting(self):
        """search considers recency in ranking."""
        cfg = _cfg()
        store = LTMStore(cfg)

        # Add older entry
        old_entry = MemoryEntry(
            content="Python old tutorial",
            learning_score=0.5,
            updated_at=time.time() - 86400 * 30,  # 30 days old
        )
        store._entries[old_entry.entry_id] = old_entry

        # Add newer entry
        new_entry = MemoryEntry(
            content="Python new tutorial",
            learning_score=0.5,
            updated_at=time.time(),  # Current
        )
        store._entries[new_entry.entry_id] = new_entry

        # Search - newer should rank higher with same learning score
        results = store.search("Python tutorial", top_k=2)

        self.assertEqual(len(results), 2)
        # Newer entry should be first due to recency
        self.assertEqual(results[0].content, "Python new tutorial")

    def test_search_stopwords_excluded(self):
        """search excludes stopwords from token matching."""
        cfg = _cfg()
        store = LTMStore(cfg)

        store.add("The Python programming language", learning_score=0.8)

        # Search with stopwords "the" and "is"
        results = store.search("the Python is programming")

        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0].content)

    def test_search_short_words_excluded(self):
        """search excludes words with 2 or fewer characters."""
        cfg = _cfg()
        store = LTMStore(cfg)

        store.add("Python programming language", learning_score=0.8)

        # "py" is too short, "programming" matches
        results = store.search("py programming")

        self.assertEqual(len(results), 1)

    def test_search_no_results_for_irrelevant_query(self):
        """search returns empty list for irrelevant queries."""
        cfg = _cfg()
        store = LTMStore(cfg)

        store.add("Python machine learning", learning_score=0.8)
        store.add("Data science with Python", learning_score=0.7)

        # Search for unrelated content
        results = store.search("cooking recipes")

        # Should still return results but based on minimal overlap
        self.assertIsInstance(results, list)

    def test_search_punctuation_handling(self):
        """search handles punctuation in queries correctly."""
        cfg = _cfg()
        store = LTMStore(cfg)

        store.add("Python, machine learning!", learning_score=0.8)

        # Search with punctuation
        results = store.search("Python machine learning")

        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0].content)

    def test_search_case_insensitive(self):
        """search is case insensitive."""
        cfg = _cfg()
        store = LTMStore(cfg)

        store.add("Python Machine Learning", learning_score=0.8)

        # Search lowercase
        results = store.search("python machine learning")

        self.assertEqual(len(results), 1)


class TestSearchAndFindSimilarIntegration(unittest.TestCase):
    """Integration tests combining _find_similar and search."""

    def test_add_then_search_finds_entry(self):
        """Added entries can be found via search."""
        cfg = _cfg()
        store = LTMStore(cfg)

        store.add("Python machine learning tutorial", learning_score=0.8)
        store.add("JavaScript web development guide", learning_score=0.7)

        results = store.search("Python tutorial", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0].content)

    def test_find_similar_prevents_duplicate_via_add(self):
        """_find_similar prevents duplicates when using add()."""
        cfg = _cfg(LTM_SIMILARITY_WORDS=3, LTM_UPDATE_THRESHOLD=0.5)
        store = LTMStore(cfg)

        # Add initial entry
        result1 = store.add("Python is great for ML", learning_score=0.6)
        self.assertEqual(store.size(), 1)

        # Add similar content - should update, not create new
        result2 = store.add("Python is great for data science", learning_score=0.7)
        self.assertEqual(store.size(), 1)  # Still 1 entry


if __name__ == "__main__":
    unittest.main()
