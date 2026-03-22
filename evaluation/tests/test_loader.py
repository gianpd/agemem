"""
Tests for evaluation/loader.py - DatasetLoader

Tests that the loader returns non-empty results with correct structure,
and that limit application works correctly.
"""

import json
import pytest
from pathlib import Path

from evaluation.loader import DatasetLoader


class TestDatasetLoader:
    """Test DatasetLoader functionality."""

    def test_load_returns_non_empty_tuple(self, fake_dataset_json: Path):
        """DatasetLoader.load() returns a non-empty (entries, queries, raw_data) tuple."""
        loader = DatasetLoader()
        entries, queries, raw_data = loader.load(fake_dataset_json)

        assert isinstance(entries, list), "entries should be a list"
        assert isinstance(queries, list), "queries should be a list"
        assert isinstance(raw_data, list), "raw_data should be a list"

        # Critical assertion: must not be empty
        assert len(queries) > 0, "queries list must not be empty - silent zero bug!"
        assert len(raw_data) > 0, "raw_data list must not be empty"

    def test_queries_have_expected_keys(self, fake_dataset_json: Path):
        """Each query dict has all expected keys."""
        loader = DatasetLoader()
        entries, queries, raw_data = loader.load(fake_dataset_json)

        expected_keys = {
            "query_id", "query_text", "relevant_entry_ids",
            "relevant_content", "query_type", "expected_answer"
        }

        for i, query in enumerate(queries):
            missing = expected_keys - set(query.keys())
            assert not missing, f"Query {i} missing keys: {missing}"

    def test_entries_have_expected_keys(self, fake_dataset_json: Path):
        """Each entry dict has expected keys."""
        loader = DatasetLoader()
        entries, queries, raw_data = loader.load(fake_dataset_json)

        expected_keys = {"content", "entry_id", "tags"}

        for i, entry in enumerate(entries):
            missing = expected_keys - set(entry.keys())
            assert not missing, f"Entry {i} missing keys: {missing}"

    def test_limit_returns_exact_count(self, fake_dataset_json: Path):
        """Limit parameter returns exactly N records."""
        loader = DatasetLoader()

        # Test with limit=1
        entries, queries, raw_data = loader.load(fake_dataset_json, limit=1)
        assert len(raw_data) == 1, "limit=1 should return exactly 1 raw_data record"
        assert len(queries) == 1, "limit=1 should return exactly 1 query"

        # Test with limit=2
        entries, queries, raw_data = loader.load(fake_dataset_json, limit=2)
        assert len(raw_data) == 2, "limit=2 should return exactly 2 raw_data records"
        assert len(queries) == 2, "limit=2 should return exactly 2 queries"

    def test_limit_zero_returns_all(self, fake_dataset_json: Path):
        """Limit=0 returns all records."""
        loader = DatasetLoader()
        entries, queries, raw_data = loader.load(fake_dataset_json, limit=0)

        # With 3 records in fake dataset, should get all 3
        assert len(raw_data) == 3, "limit=0 should return all records"

    def test_load_full_convenience_method(self, fake_dataset_json: Path):
        """load_full() returns same as load() with default subset."""
        loader = DatasetLoader()

        entries1, queries1, raw1 = loader.load_full(fake_dataset_json)
        entries2, queries2, raw2 = loader.load(fake_dataset_json)

        assert len(entries1) == len(entries2)
        assert len(queries1) == len(queries2)
        assert len(raw1) == len(raw2)

    def test_load_queries_only_returns_empty_entries(self, fake_dataset_json: Path):
        """load_queries_only() returns empty entries but populated queries."""
        loader = DatasetLoader()
        entries, queries, raw_data = loader.load_queries_only(fake_dataset_json)

        assert len(entries) == 0, "load_queries_only should return empty entries"
        assert len(queries) > 0, "queries should still be populated"

    def test_query_id_never_empty(self, fake_dataset_json: Path):
        """Every query has a non-empty query_id."""
        loader = DatasetLoader()
        entries, queries, raw_data = loader.load(fake_dataset_json)

        for i, query in enumerate(queries):
            assert query.get("query_id"), f"Query {i} has empty query_id"


class TestDatasetLoaderEdgeCases:
    """Test edge cases and error handling."""

    def test_load_nonexistent_file_raises_error(self, tmp_path: Path):
        """Loading a non-existent file raises FileNotFoundError."""
        loader = DatasetLoader()
        nonexistent = tmp_path / "does_not_exist.json"

        with pytest.raises(FileNotFoundError):
            loader.load(nonexistent)

    def test_load_empty_json_returns_empty_lists(self, tmp_path: Path):
        """Loading empty JSON array returns empty lists (not crash)."""
        loader = DatasetLoader()
        empty_path = tmp_path / "empty.json"
        empty_path.write_text("[]", encoding="utf-8")

        entries, queries, raw_data = loader.load(empty_path)

        assert entries == []
        assert queries == []
        assert raw_data == []

    def test_load_with_missing_optional_fields(self, tmp_path: Path):
        """Loader handles records with missing optional fields gracefully."""
        loader = DatasetLoader()

        # Minimal record with missing optional fields
        data = [{"question": "Test?", "answer": "42"}]
        dataset_path = tmp_path / "minimal.json"
        dataset_path.write_text(json.dumps(data), encoding="utf-8")

        entries, queries, raw_data = loader.load(dataset_path)

        assert len(queries) == 1
        # Should generate a query_id
        assert queries[0]["query_id"], "Should generate query_id when missing"
        # Should have defaults for other fields
        assert queries[0]["query_type"] == "retrieval"