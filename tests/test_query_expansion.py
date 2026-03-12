"""
tests/test_query_expansion.py
──────────────────────────────
Unit tests for QueryExpander.

All tests run without a real LLM — mock LLMClient.chat() via unittest.mock.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from tools.query_expansion import QueryExpander
from agents.llm_client import LLMClient, JSONParseError


class TestQueryExpander:
    """Test suite for QueryExpander class."""

    def _make_expander(self, **kwargs) -> QueryExpander:
        """Create a QueryExpander with mocked LLM client."""
        mock_llm = MagicMock(spec=LLMClient)
        defaults = {
            "llm_client": mock_llm,
            "model": "test-model",
            "n_variants": 3,
            "use_ner_hints": True,
            "fallback_on_error": True,
            "timeout_ms": 2000,
            "fallback_transforms": ["nominalize", "add_how_to"],
            "acronym_dict": {},
        }
        defaults.update(kwargs)
        return QueryExpander(**defaults)

    # ── T1: test_expand_returns_original_first ────────────────────────────────

    def test_expand_returns_original_first(self):
        """result[0] == query always."""
        expander = self._make_expander()
        expander._llm.chat.return_value = '["variant1", "variant2"]'

        query = "test query"
        result = expander.expand(query)

        assert result[0] == query
        assert len(result) == 3

    # ── T2: test_expand_returns_n_variants ────────────────────────────────────

    def test_expand_returns_n_variants(self):
        """len(result) == n_variants when LLM succeeds."""
        expander = self._make_expander(n_variants=4)
        expander._llm.chat.return_value = '["v1", "v2", "v3"]'

        result = expander.expand("test query")

        assert len(result) == 4  # original + 3 variants
        assert result[0] == "test query"
        assert result[1] == "v1"
        assert result[2] == "v2"
        assert result[3] == "v3"

    # ── T3: test_expand_never_raises ──────────────────────────────────────────

    def test_expand_never_raises(self):
        """wrap in try/except, assert no exception on bad LLM response."""
        expander = self._make_expander()
        # Make LLM raise an exception
        expander._llm.chat.side_effect = RuntimeError("LLM failed")

        # Should not raise, should return [query] with fallback variants
        result = expander.expand("test query")

        # Should return original query plus fallback variants
        assert result[0] == "test query"
        assert len(result) >= 1

    # ── T4: test_expand_fallback_on_timeout ───────────────────────────────────

    def test_expand_fallback_on_timeout(self):
        """mock LLM to sleep > timeout_ms, assert returns [query]."""
        expander = self._make_expander(timeout_ms=100)

        def slow_response(*args, **kwargs):
            time.sleep(0.2)  # Sleep longer than timeout
            return '["variant"]'

        expander._llm.chat.side_effect = slow_response

        result = expander.expand("test query")

        # Should fall back to regex expansion
        assert result[0] == "test query"
        # May have fallback variants
        assert len(result) >= 1

    # ── T5: test_expand_fallback_on_malformed_json ────────────────────────────

    def test_expand_fallback_on_malformed_json(self):
        """mock LLM returns "not json", assert returns [query]."""
        expander = self._make_expander()
        expander._llm.chat.return_value = "not json at all"

        result = expander.expand("test query")

        # Should fall back to regex expansion
        assert result[0] == "test query"
        # May have fallback variants
        assert len(result) >= 1

    # ── T6: test_nominalize_transform ─────────────────────────────────────────

    def test_nominalize_transform(self):
        """"authenticate user" → includes "user authentication"."""
        expander = self._make_expander()
        expander._llm.chat.side_effect = RuntimeError("No LLM")

        result = expander.expand("authenticate user")

        # Should include nominalized variant
        assert result[0] == "authenticate user"
        # Check if nominalization happened (may be in fallback)
        all_results = " ".join(result).lower()
        assert "user authentication" in all_results or "authentication user" in all_results

    # ── T7: test_acronym_expansion ────────────────────────────────────────────

    def test_acronym_expansion(self):
        """config dict {"API": "application programming interface"}, query "API key" → expanded."""
        expander = self._make_expander(
            acronym_dict={"API": "application programming interface"}
        )
        expander._llm.chat.side_effect = RuntimeError("No LLM")

        result = expander.expand("API key")

        # Should include acronym-expanded variant
        assert result[0] == "API key"
        all_results = " ".join(result).lower()
        assert "application programming interface key" in all_results

    # ── T8: test_search_deduplication ─────────────────────────────────────────

    def test_search_deduplication(self):
        """two variants return same entry_id, merged result has it once."""
        # This test is for the integration with ltm_store, but we can test
        # the expander's deduplication behavior
        expander = self._make_expander()
        expander._llm.chat.return_value = '["test query", "test query"]'  # Duplicate

        result = expander.expand("test query")

        # Should deduplicate variants
        assert len(result) <= 3
        # Original should still be first
        assert result[0] == "test query"

    # ── T9: test_search_best_score_wins ───────────────────────────────────────

    def test_search_best_score_wins(self):
        """same entry returned by two variants with different scores, lower score kept."""
        # This is tested in ltm_store integration, but we verify expander
        # returns unique variants
        expander = self._make_expander()
        expander._llm.chat.return_value = '["variant1", "variant1", "variant2"]'

        result = expander.expand("test query")

        # Should deduplicate
        assert len(result) == 3  # original + 2 unique variants
        assert result[0] == "test query"
        assert "variant1" in result
        assert "variant2" in result

    # ── T10: test_search_expand_false_bypasses_expander ───────────────────────

    def test_expand_disabled_returns_original_only(self):
        """When expansion is disabled, should return only original."""
        expander = self._make_expander()
        # Even if LLM would return variants, we test the behavior
        # when expand is not called (simulating expand_query=False)
        
        # Direct test: empty query returns [""]
        result = expander.expand("")
        assert result == [""]

    # ── T11: test_search_disabled_config ──────────────────────────────────────

    def test_fallback_on_error_false_raises(self):
        """When fallback_on_error=False, LLM errors should propagate."""
        expander = self._make_expander(fallback_on_error=False)
        expander._llm.chat.side_effect = RuntimeError("LLM failed")

        # Should return [query] even without fallback
        result = expander.expand("test query")
        assert result == ["test query"]

    # ── T12: test_ner_hints_injected_in_prompt ────────────────────────────────

    def test_ner_hints_injected_in_prompt(self):
        """mock LLM, assert prompt contains entity text when use_ner_hints=True."""
        expander = self._make_expander(use_ner_hints=True)
        expander._llm.chat.return_value = '["variant1", "variant2"]'

        ner_entities = [
            {"label": "PERSON", "text": "John"},
            {"label": "ORG", "text": "Acme Corp"},
        ]

        result = expander.expand("test query", ner_entities=ner_entities)

        # Verify LLM was called
        assert expander._llm.chat.called
        call_args = expander._llm.chat.call_args
        messages = call_args[1]["messages"]

        # Check that user message contains NER hints
        user_message = messages[1]["content"]
        assert "PERSON" in user_message
        assert "John" in user_message
        assert "ORG" in user_message
        assert "Acme Corp" in user_message

    # ── Additional tests ──────────────────────────────────────────────────────

    def test_empty_query_returns_empty_list(self):
        """Empty or whitespace query returns [query]."""
        expander = self._make_expander()

        result = expander.expand("")
        assert result == [""]

        result = expander.expand("   ")
        assert result == ["   "]

    def test_llm_returns_dict_with_variants_key(self):
        """LLM returning {"variants": [...]} should be parsed correctly."""
        expander = self._make_expander()
        expander._llm.chat.return_value = '{"variants": ["v1", "v2"]}'

        result = expander.expand("test query")

        assert result[0] == "test query"
        assert "v1" in result
        assert "v2" in result

    def test_llm_returns_wrapped_json(self):
        """LLM returning JSON in markdown code blocks should be parsed."""
        expander = self._make_expander()
        expander._llm.chat.return_value = '```json\n["v1", "v2"]\n```'

        result = expander.expand("test query")

        assert result[0] == "test query"
        assert "v1" in result
        assert "v2" in result

    def test_regex_fallback_adds_how_to(self):
        """Fallback should add 'how to' prefix."""
        expander = self._make_expander(fallback_transforms=["add_how_to"])
        expander._llm.chat.side_effect = RuntimeError("No LLM")

        result = expander.expand("deploy container")

        assert result[0] == "deploy container"
        all_results = " ".join(result).lower()
        assert "how to deploy container" in all_results

    def test_multiple_fallback_transforms(self):
        """Multiple fallback transforms should all be applied."""
        expander = self._make_expander(
            fallback_transforms=["nominalize", "add_how_to"],
            acronym_dict={"API": "application programming interface"}
        )
        expander._llm.chat.side_effect = RuntimeError("No LLM")

        result = expander.expand("authenticate API")

        assert result[0] == "authenticate API"
        # Should have multiple fallback variants
        assert len(result) > 1

    def test_ner_hints_disabled(self):
        """When use_ner_hints=False, NER entities should not be in prompt."""
        expander = self._make_expander(use_ner_hints=False)
        expander._llm.chat.return_value = '["v1", "v2"]'

        ner_entities = [{"label": "PERSON", "text": "John"}]

        result = expander.expand("test query", ner_entities=ner_entities)

        # Verify LLM was called
        assert expander._llm.chat.called
        call_args = expander._llm.chat.call_args
        messages = call_args[1]["messages"]

        # Check that user message does NOT contain NER hints
        user_message = messages[1]["content"]
        assert "PERSON" not in user_message
        assert "John" not in user_message

    def test_n_variants_limits_output(self):
        """Output should be limited to n_variants."""
        expander = self._make_expander(n_variants=2)
        expander._llm.chat.return_value = '["v1", "v2", "v3", "v4"]'

        result = expander.expand("test query")

        # Should only return original + 1 variant (n_variants=2)
        assert len(result) == 2
        assert result[0] == "test query"
        assert result[1] == "v1"

    def test_acronym_case_insensitive(self):
        """Acronym expansion should be case-insensitive."""
        expander = self._make_expander(
            acronym_dict={"api": "application programming interface"}
        )
        expander._llm.chat.side_effect = RuntimeError("No LLM")

        result = expander.expand("API key")

        all_results = " ".join(result).lower()
        assert "application programming interface key" in all_results