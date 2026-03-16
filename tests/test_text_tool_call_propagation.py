"""
Test for TextToolCallResponse propagation in LLMClient.chat().

This test verifies the fix for the bug where TextToolCallResponse was
caught by the retry loop instead of propagating to the caller.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from agents.llm_client import LLMClient, ToolCallResponse, TextToolCallResponse, TextToolCall, detect_text_tool_calls


class TestTextToolCallPropagation:
    """Test that TextToolCallResponse propagates correctly through chat()."""

    def test_text_tool_call_response_propagates_not_retried(self):
        """
        TextToolCallResponse should propagate immediately, not trigger retry loop.

        Before the fix, the chat() method only checked for ToolCallResponse
        in its exception handler, causing TextToolCallResponse to be caught
        as a generic Exception and trigger retries, eventually failing with
        "LLM call failed after 3 attempts".
        """
        # Setup mock client
        mock_openai_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = '{"tool": "list_documents", "args": {}}'
        mock_message.tool_calls = None  # No native tool calls
        mock_response.choices = [Mock(message=mock_message)]

        # Mock usage to avoid int + Mock error
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_response.usage = mock_usage

        mock_openai_client.chat.completions.create.return_value = mock_response

        # Create LLMClient with the mock
        llm = LLMClient(client=mock_openai_client, default_model="test-model")

        # Define tools to enable text tool call detection
        tools = [{
            "type": "function",
            "function": {
                "name": "list_documents",
                "description": "List all documents",
                "parameters": {"type": "object", "properties": {}}
            }
        }]

        # The text contains a valid tool call JSON, which detect_text_tool_calls
        # will parse and raise TextToolCallResponse
        with pytest.raises(TextToolCallResponse) as exc_info:
            llm.chat(
                messages=[{"role": "user", "content": "List documents"}],
                tools=tools,
            )

        # Verify the exception contains the correct tool info
        assert exc_info.value.tool_call.function.name == "list_documents"
        assert exc_info.value.tool_call.function.arguments == {}

        # CRITICAL: Verify only ONE API call was made (no retries)
        # Before the fix, this would be 3 (or more) due to retry loop
        assert mock_openai_client.chat.completions.create.call_count == 1

    def test_tool_call_response_propagates_not_retried(self):
        """
        ToolCallResponse (native API tool call) should also propagate immediately.
        """
        # Setup mock client to return native tool call
        mock_openai_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = None

        # Mock usage
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_response.usage = mock_usage

        # Mock native tool call
        mock_tool_call = Mock()
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = '{"query": "test"}'
        mock_message.tool_calls = [mock_tool_call]

        mock_response.choices = [Mock(message=mock_message)]
        mock_openai_client.chat.completions.create.return_value = mock_response

        llm = LLMClient(client=mock_openai_client, default_model="test-model")

        tools = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}}
            }
        }]

        with pytest.raises(ToolCallResponse) as exc_info:
            llm.chat(
                messages=[{"role": "user", "content": "Search for test"}],
                tools=tools,
            )

        assert exc_info.value.tool_call.function.name == "search"
        assert mock_openai_client.chat.completions.create.call_count == 1

    def test_text_tool_call_with_retries_enabled_still_propagates(self):
        """
        Even with retries=5, TextToolCallResponse should propagate on first attempt.
        """
        mock_openai_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = '{"tool": "web_search", "args": {"query": "test"}}'
        mock_message.tool_calls = None

        # Mock usage
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_response.usage = mock_usage

        mock_response.choices = [Mock(message=mock_message)]
        mock_openai_client.chat.completions.create.return_value = mock_response

        llm = LLMClient(client=mock_openai_client, default_model="test-model")

        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Web search",
                "parameters": {"type": "object", "properties": {}}
            }
        }]

        with pytest.raises(TextToolCallResponse):
            llm.chat(
                messages=[{"role": "user", "content": "Search web"}],
                tools=tools,
                retries=5,  # Explicit high retry count
            )

        # Must still be 1 call, not 6
        assert mock_openai_client.chat.completions.create.call_count == 1

    def test_detect_text_tool_calls_parses_correctly(self):
        """
        Unit test for detect_text_tool_calls to verify it parses our format.
        """
        # Test the exact format from the trace
        text = '{"tool": "list_documents", "args": {}}'
        calls = detect_text_tool_calls(text)

        assert len(calls) == 1
        assert calls[0].function.name == "list_documents"
        assert calls[0].function.arguments == {}