"""
Test empty response handling in LLM client.

This module tests the diagnostic and retry logic for empty LLM responses.
"""

import pytest
from unittest.mock import MagicMock, patch
from agents.llm_client import LLMClient, ChatResponseInfo


class TestEmptyResponseHandling:
    """Test handling of empty or whitespace-only responses."""

    def test_empty_response_returns_info(self):
        """Test that empty responses are detected and logged."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = ""  # Empty response
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=0)

        mock_client.chat.completions.create.return_value = mock_response

        llm = LLMClient(mock_client, default_model="test-model")
        result = llm.chat_with_info([{"role": "user", "content": "Hello"}])

        assert result.is_empty is True
        assert result.content == ""
        assert result.finish_reason == "stop"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 0

    def test_whitespace_response_is_empty(self):
        """Test that whitespace-only responses are detected as empty."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "   \n  "  # Whitespace only
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=1)

        mock_client.chat.completions.create.return_value = mock_response

        llm = LLMClient(mock_client, default_model="test-model")
        result = llm.chat_with_info([{"role": "user", "content": "Hello"}])

        assert result.is_empty is True
        assert result.content == "   \n  "

    def test_empty_response_retries(self):
        """Test that empty responses trigger retry logic."""
        mock_client = MagicMock()

        # First call returns empty, second returns content
        empty_response = MagicMock()
        empty_response.choices = [MagicMock()]
        empty_response.choices[0].message = MagicMock()
        empty_response.choices[0].message.content = ""
        empty_response.choices[0].finish_reason = "stop"
        empty_response.choices[0].message.tool_calls = None
        empty_response.usage = MagicMock(prompt_tokens=100, completion_tokens=0)

        good_response = MagicMock()
        good_response.choices = [MagicMock()]
        good_response.choices[0].message = MagicMock()
        good_response.choices[0].message.content = "Hello!"
        good_response.choices[0].finish_reason = "stop"
        good_response.choices[0].message.tool_calls = None
        good_response.usage = MagicMock(prompt_tokens=100, completion_tokens=5)

        mock_client.chat.completions.create.side_effect = [empty_response, good_response]

        llm = LLMClient(mock_client, default_model="test-model")
        result = llm.chat_with_info([{"role": "user", "content": "Hello"}])

        assert result.is_empty is False
        assert result.content == "Hello!"
        assert result.retries_used == 1
        assert mock_client.chat.completions.create.call_count == 2

    def test_empty_response_exhausts_retries(self):
        """Test that empty responses after all retries still return."""
        mock_client = MagicMock()

        # All calls return empty
        empty_response = MagicMock()
        empty_response.choices = [MagicMock()]
        empty_response.choices[0].message = MagicMock()
        empty_response.choices[0].message.content = ""
        empty_response.choices[0].finish_reason = "stop"
        empty_response.choices[0].message.tool_calls = None
        empty_response.usage = MagicMock(prompt_tokens=100, completion_tokens=0)

        mock_client.chat.completions.create.return_value = empty_response

        llm = LLMClient(mock_client, default_model="test-model")
        result = llm.chat_with_info([{"role": "user", "content": "Hello"}], retries=2)

        # Should return after exhausting retries (default 2 + 1 = 3 calls)
        assert result.is_empty is True
        assert result.content == ""
        assert result.retries_used == 2  # Last successful attempt index

    def test_normal_response_not_empty(self):
        """Test that normal responses are not flagged as empty."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "This is a response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=10)

        mock_client.chat.completions.create.return_value = mock_response

        llm = LLMClient(mock_client, default_model="test-model")
        result = llm.chat_with_info([{"role": "user", "content": "Hello"}])

        assert result.is_empty is False
        assert result.content == "This is a response"
        assert result.retries_used == 0

    def test_chat_method_empty_response_retry(self):
        """Test that the original chat() method also retries on empty responses."""
        mock_client = MagicMock()

        empty_response = MagicMock()
        empty_response.choices = [MagicMock()]
        empty_response.choices[0].message = MagicMock()
        empty_response.choices[0].message.content = ""
        empty_response.choices[0].finish_reason = "stop"
        empty_response.choices[0].message.tool_calls = None
        empty_response.usage = MagicMock(prompt_tokens=100, completion_tokens=0)

        good_response = MagicMock()
        good_response.choices = [MagicMock()]
        good_response.choices[0].message = MagicMock()
        good_response.choices[0].message.content = "Good response"
        good_response.choices[0].finish_reason = "stop"
        good_response.choices[0].message.tool_calls = None
        good_response.usage = MagicMock(prompt_tokens=100, completion_tokens=5)

        mock_client.chat.completions.create.side_effect = [empty_response, good_response]

        llm = LLMClient(mock_client, default_model="test-model")
        result = llm.chat([{"role": "user", "content": "Hello"}])

        assert result == "Good response"
        assert mock_client.chat.completions.create.call_count == 2


class TestChatResponseInfo:
    """Test the ChatResponseInfo dataclass."""

    def test_info_defaults(self):
        """Test default values for ChatResponseInfo."""
        info = ChatResponseInfo(content="test", model="test-model")
        assert info.content == "test"
        assert info.model == "test-model"
        assert info.finish_reason is None
        assert info.prompt_tokens == 0
        assert info.completion_tokens == 0
        assert info.total_tokens == 0
        assert info.is_empty is False
        assert info.retries_used == 0

    def test_info_with_all_fields(self):
        """Test ChatResponseInfo with all fields populated."""
        info = ChatResponseInfo(
            content="Hello",
            model="gpt-4",
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            is_empty=False,
            retries_used=1,
        )
        assert info.content == "Hello"
        assert info.model == "gpt-4"
        assert info.finish_reason == "stop"
        assert info.prompt_tokens == 100
        assert info.completion_tokens == 50
        assert info.total_tokens == 150
        assert info.is_empty is False
        assert info.retries_used == 1