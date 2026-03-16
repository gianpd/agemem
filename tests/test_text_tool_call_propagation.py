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

    def test_detect_text_tool_calls_rejects_metadata_json(self):
        """
        Format 4 heuristic should reject JSON that looks like metadata, not tool calls.

        This tests the fix for Bug 1: aggressive text-based tool call detection.
        JSON like {"web_search": {"description": "..."}} should NOT be detected as a tool call.
        """
        # This should NOT be detected as a tool call (it's metadata about web_search)
        text = '{"web_search": {"description": "A tool for searching the web", "last_used": "2024-01-01"}}'
        calls = detect_text_tool_calls(text)
        assert len(calls) == 0

        # This also should NOT be detected (has "results" key suggesting metadata)
        text = '{"search_metadata": {"results": 5, "status": "complete"}}'
        calls = detect_text_tool_calls(text)
        assert len(calls) == 0

        # But this SHOULD be detected (looks like actual tool arguments)
        text = '{"web_search": {"query": "test search"}}'
        calls = detect_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].function.name == "web_search"
        assert calls[0].function.arguments == {"query": "test search"}

    def test_detect_text_tool_calls_strips_thinking_blocks(self):
        """
        Tool calls inside thinking blocks should not be detected.

        This tests the fix for Bug 3: tool call detection in thinking blocks.
        """
        # Tool call inside thinking block should be ignored
        text = '<think>{"web_search": {"query": "test"}}</think>Some actual response text.'
        calls = detect_text_tool_calls(text)
        # The thinking block should be stripped, so no tool call should be found
        assert len(calls) == 0

        # Tool call outside thinking block should still be detected
        text = '<think>Let me think...</think>{"web_search": {"query": "actual query"}}'
        calls = detect_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].function.name == "web_search"

    def test_malformed_native_tool_call_skipped(self):
        """
        Native tool calls with empty/missing names should be skipped, not raised.

        This tests the fix for Bug 2: malformed native tool calls from the LLM.
        """
        mock_openai_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Here is my response text."
        mock_message.tool_calls = None  # Will be set below

        # Mock usage
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_response.usage = mock_usage

        # Mock malformed native tool call (empty name)
        mock_tool_call = Mock()
        mock_tool_call.function.name = ""  # Empty name - malformed
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

        # Should NOT raise ToolCallResponse - should return content instead
        result = llm.chat(
            messages=[{"role": "user", "content": "Search for test"}],
            tools=tools,
        )

        assert result == "Here is my response text."
        assert mock_openai_client.chat.completions.create.call_count == 1