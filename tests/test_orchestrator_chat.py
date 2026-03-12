"""
Tests for the Orchestrator.chat() method.

Validates tool call handling with focus on the tool_call.function.arguments attribute,
including edge cases where json.loads returns a string instead of a dict.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch, ANY
from pathlib import Path
import tempfile
import shutil

from agents.orchestrator import Orchestrator, ToolCall, ToolCallTracker
from agents.llm_client import LLMClient, ToolCallResponse, TextToolCallResponse
from agents.response_handler import ResponseHandler
from core.config import AgememConfig
from core.types import ContextMessage, MemoryOpResult, MemoryOp, TriggerKind


class MockToolCall:
    """Mock tool call for testing."""
    def __init__(self, name: str, arguments: any, id: str = "test_call_id"):
        self.id = id
        self.function = Mock()
        self.function.name = name
        self.function.arguments = arguments


class MockTextToolCall:
    """Mock text-based tool call for testing."""
    def __init__(self, name: str, arguments: dict, id: str = "test_call_id"):
        self.id = id
        self.function = Mock()
        self.function.name = name
        self.function.arguments = arguments


def create_mock_openai_client(response_text: str = "Mock response"):
    """
    Create a properly mocked OpenAI-compatible client.

    The LLMClient expects client.chat.completions.create() to return a response object.
    """
    mock_client = MagicMock()

    # Build the nested response structure
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client.chat.completions.create.return_value = mock_response

    return mock_client


def create_mock_llm_client_with_tool_response(tool_call):
    """
    Create a mock OpenAI client that returns a tool call response.
    """
    mock_client = MagicMock()

    def side_effect(**kwargs):
        # Check if tools are provided - if so, return tool call
        if kwargs.get("tools"):
            mock_message = MagicMock()
            mock_message.content = None
            mock_message.tool_calls = [tool_call]

            mock_choice = MagicMock()
            mock_choice.message = mock_message

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 20
            mock_usage.completion_tokens = 10

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage

            return mock_response
        else:
            # Return text response
            mock_message = MagicMock()
            mock_message.content = "Final response"
            mock_message.tool_calls = None

            mock_choice = MagicMock()
            mock_choice.message = mock_message

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 5

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage

            return mock_response

    mock_client.chat.completions.create.side_effect = side_effect
    return mock_client


def create_test_config(temp_dir: str) -> AgememConfig:
    """Create a test configuration with temp directory."""
    return AgememConfig(
        LTM_MAX_ENTRIES=100,
        STM_TOKEN_LIMIT=2000,
        STM_WARNING_THRESHOLD=0.75,
        STM_CRITICAL_THRESHOLD=0.90,
        STM_MIN_MESSAGES=4,
        PERSIST_DIR=temp_dir,
        DEFAULT_MAX_TOKENS=1024,
        DEFAULT_TEMPERATURE=0.7,
        ENABLE_SEMANTIC_SEARCH=False,
        ENABLE_QUERY_EXPANSION=False,
        SKILL_DETECTION_ENABLED=False,
    )


class TestToolCallArgumentsHandling:
    """Test tool_call.function.arguments handling edge cases."""

    def test_arguments_as_dict(self):
        """Test when tool_call.function.arguments is already a dict."""
        # Create tool call with dict arguments
        tool_call = MockToolCall("web_search", {"query": "test query"})

        # Create mock client that returns tool call then text
        mock_openai_client = create_mock_llm_client_with_tool_response(tool_call)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            response = orchestrator.chat("Search for something")

            # Should complete without error and return a response
            assert isinstance(response, str)
            assert len(response) > 0

            # Verify trace was recorded
            last_trace = orchestrator.last_trace()
            assert last_trace is not None
            assert last_trace.user_input == "Search for something"

    def test_arguments_as_json_string_valid_dict(self):
        """Test when tool_call.function.arguments is a JSON string parsing to dict."""
        # Arguments is a JSON-encoded string
        tool_call = MockToolCall("web_search", '{"query": "json string query"}')

        mock_openai_client = create_mock_llm_client_with_tool_response(tool_call)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            response = orchestrator.chat("Search for something")

            # Should complete without error
            assert isinstance(response, str)
            assert len(response) > 0

            # Verify trace was recorded
            last_trace = orchestrator.last_trace()
            assert last_trace is not None

    def test_arguments_as_json_string_parses_to_string(self):
        """
        Test the edge case where json.loads returns a string, not a dict.

        This happens when the LLM returns something like '"search_term"' which
        json.loads converts to just 'search_term' (a string).

        The issue: if json.loads returns a string, json.dumps will serialize
        it as "search_term" not {"query": ...}
        """
        # Arguments is a JSON-encoded string that parses to a string, not a dict
        # This simulates the edge case: json.loads('"search_term"') == 'search_term'
        tool_call = MockToolCall("web_search", '"search_term"')

        mock_openai_client = create_mock_llm_client_with_tool_response(tool_call)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            # Should not crash - the code handles this by treating it as empty args
            # or the string gets passed through
            response = orchestrator.chat("Search")

            # Verify it handled gracefully
            assert isinstance(response, str)

    def test_arguments_as_json_number(self):
        """Test when json.loads returns a number instead of dict."""
        # Arguments is a JSON number
        tool_call = MockToolCall("some_tool", "12345")

        mock_openai_client = create_mock_llm_client_with_tool_response(tool_call)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            # Should handle gracefully without crashing
            response = orchestrator.chat("Test")
            assert isinstance(response, str)

    def test_arguments_as_json_list(self):
        """Test when json.loads returns a list instead of dict."""
        # Arguments is a JSON array
        tool_call = MockToolCall("some_tool", '["item1", "item2"]')

        mock_openai_client = create_mock_llm_client_with_tool_response(tool_call)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            # Should handle gracefully
            response = orchestrator.chat("Test")
            assert isinstance(response, str)

    def test_arguments_empty_string(self):
        """Test when tool_call.function.arguments is an empty string."""
        tool_call = MockToolCall("web_search", "")

        mock_openai_client = create_mock_llm_client_with_tool_response(tool_call)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            # Should handle empty arguments gracefully
            response = orchestrator.chat("Search")
            assert isinstance(response, str)

    def test_arguments_none(self):
        """Test when tool_call.function.arguments is None."""
        tool_call = MockToolCall("web_search", None)

        mock_openai_client = create_mock_llm_client_with_tool_response(tool_call)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            # Should handle None arguments gracefully
            response = orchestrator.chat("Search")
            assert isinstance(response, str)

    def test_arguments_invalid_json(self):
        """Test when tool_call.function.arguments is invalid JSON."""
        tool_call = MockToolCall("web_search", 'not valid json {{{')

        mock_openai_client = create_mock_llm_client_with_tool_response(tool_call)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            # Should handle invalid JSON gracefully - falls back to empty dict
            response = orchestrator.chat("Search")
            assert isinstance(response, str)


class TestToolCallSerialization:
    """Test that tool arguments are properly serialized in STM."""

    def test_dict_arguments_serialization(self):
        """
        Test that dict arguments are properly handled in tool call flow.

        Verifies the fix for: when tool_args is a dict, the system handles it
        correctly without crashing.
        """
        tool_call = MockToolCall("web_search", {"query": "test", "num_results": 5})

        mock_openai_client = create_mock_llm_client_with_tool_response(tool_call)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            orchestrator.chat("Search")

            # Verify messages were added to STM (user + assistant + tool)
            messages = orchestrator._stm.messages()
            assert len(messages) >= 3  # system + user + assistant + tool

            # Verify user message was recorded
            user_msgs = [m for m in messages if m.role == "user" and m.content == "Search"]
            assert len(user_msgs) >= 1

            # Verify assistant message was recorded
            assistant_msgs = [m for m in messages if m.role == "assistant"]
            assert len(assistant_msgs) >= 1

    def test_string_arguments_serialization(self):
        """
        Test that string arguments (from json.loads returning string) are handled.

        Edge case: if json.loads returns a string, we need to ensure
        the code doesn't crash and handles it gracefully.
        """
        # This is the problematic case: arguments is a JSON string that
        # parses to a string value, not a dict
        tool_call = MockToolCall("web_search", '"just_a_string"')

        mock_openai_client = create_mock_llm_client_with_tool_response(tool_call)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            # Should not crash
            orchestrator.chat("Search")

            # Verify messages were still added
            messages = orchestrator._stm.messages()
            # Should have at least system, user, and assistant messages
            assert len(messages) >= 2


class TestTextToolCallResponse:
    """Test TextToolCallResponse handling (for models without API tool calling)."""

    def test_text_tool_call_with_dict_args(self):
        """Test TextToolCallResponse with dict arguments."""
        # Create mock that returns text containing a tool call
        mock_openai_client = MagicMock()

        # First call returns text with tool call JSON
        first_response = MagicMock()
        first_response.choices = [MagicMock()]
        first_response.choices[0].message = MagicMock()
        first_response.choices[0].message.content = '{"tool": "web_search", "args": {"query": "text tool query"}}'
        first_response.choices[0].message.tool_calls = None
        first_response.usage = MagicMock(prompt_tokens=20, completion_tokens=15)

        # Second call returns final text
        second_response = MagicMock()
        second_response.choices = [MagicMock()]
        second_response.choices[0].message = MagicMock()
        second_response.choices[0].message.content = "Done"
        second_response.choices[0].message.tool_calls = None
        second_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        mock_openai_client.chat.completions.create.side_effect = [
            first_response,
            second_response
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            response = orchestrator.chat("Search")
            assert isinstance(response, str)

    def test_text_tool_call_validation_failure(self):
        """Test TextToolCallResponse with invalid arguments triggers validation error."""
        mock_openai_client = MagicMock()

        # First call returns text with invalid tool call (no tool name)
        first_response = MagicMock()
        first_response.choices = [MagicMock()]
        first_response.choices[0].message = MagicMock()
        # Invalid: no known tool name
        first_response.choices[0].message.content = '{"some_key": "value"}'
        first_response.choices[0].message.tool_calls = None
        first_response.usage = MagicMock(prompt_tokens=20, completion_tokens=15)

        # Second call returns final text
        second_response = MagicMock()
        second_response.choices = [MagicMock()]
        second_response.choices[0].message = MagicMock()
        second_response.choices[0].message.content = "Done"
        second_response.choices[0].message.tool_calls = None
        second_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        mock_openai_client.chat.completions.create.side_effect = [
            first_response,
            second_response
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            response = orchestrator.chat("Search")
            assert isinstance(response, str)


class TestToolCallLoopGuard:
    """Test the LoopGuard duplicate tool call detection."""

    def test_duplicate_tool_call_detection(self):
        """Test that duplicate tool calls are detected and blocked."""
        # Same tool call used twice
        tool_call = MockToolCall("web_search", {"query": "duplicate"})

        mock_openai_client = MagicMock()

        # Build responses that return the same tool call twice
        def create_tool_response(tc):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message = MagicMock()
            response.choices[0].message.content = None
            response.choices[0].message.tool_calls = [tc]
            response.usage = MagicMock(prompt_tokens=20, completion_tokens=10)
            return response

        def create_text_response(text):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message = MagicMock()
            response.choices[0].message.content = text
            response.choices[0].message.tool_calls = None
            response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            return response

        mock_openai_client.chat.completions.create.side_effect = [
            create_tool_response(tool_call),  # First call
            create_tool_response(tool_call),  # Duplicate - should be blocked
            create_text_response("Final response after duplicate")
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            response = orchestrator.chat("Search")
            assert "Final response" in response

    def test_different_args_not_duplicate(self):
        """Test that tool calls with different args are not duplicates."""
        tool_call1 = MockToolCall("web_search", {"query": "first"})
        tool_call2 = MockToolCall("web_search", {"query": "second"})

        mock_openai_client = MagicMock()

        def create_tool_response(tc):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message = MagicMock()
            response.choices[0].message.content = None
            response.choices[0].message.tool_calls = [tc]
            response.usage = MagicMock(prompt_tokens=20, completion_tokens=10)
            return response

        def create_text_response(text):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message = MagicMock()
            response.choices[0].message.content = text
            response.choices[0].message.tool_calls = None
            response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            return response

        mock_openai_client.chat.completions.create.side_effect = [
            create_tool_response(tool_call1),
            create_tool_response(tool_call2),
            create_text_response("Done")
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            response = orchestrator.chat("Search")
            assert isinstance(response, str)
            # Both tool calls should be in ops
            last_trace = orchestrator.last_trace()
            web_search_ops = [op for op in last_trace.ops_applied if "web_search" in op.detail]
            assert len(web_search_ops) == 2

    def test_max_iterations_limit(self):
        """Test that max tool iterations prevents infinite loops."""
        # Always return a tool call (simulating infinite loop)
        tool_call = MockToolCall("web_search", {"query": "loop"})

        mock_openai_client = MagicMock()

        def create_tool_response(tc):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message = MagicMock()
            response.choices[0].message.content = None
            response.choices[0].message.tool_calls = [tc]
            response.usage = MagicMock(prompt_tokens=20, completion_tokens=10)
            return response

        # Return tool call many times
        mock_openai_client.chat.completions.create.side_effect = [
            create_tool_response(tool_call) for _ in range(25)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            response = orchestrator.chat("Search")
            # Should hit max iterations and return system message
            assert "Maximum tool call iterations" in response


class TestChatBasicFlow:
    """Test basic chat flow without tool calls."""

    def test_simple_chat_response(self):
        """Test a simple chat without tool calls."""
        mock_openai_client = create_mock_openai_client("Hello! How can I help you today?")

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            response = orchestrator.chat("Hi there")

            assert response == "Hello! How can I help you today?"
            # Verify turn was tracked
            assert orchestrator._stm.current_turn() == 1

    def test_chat_turn_counter_increment(self):
        """Test that chat increments the turn counter."""
        mock_openai_client = create_mock_openai_client("Response")

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            initial_turn = orchestrator._stm.current_turn()
            orchestrator.chat("Message 1")
            assert orchestrator._stm.current_turn() == initial_turn + 1
            orchestrator.chat("Message 2")
            assert orchestrator._stm.current_turn() == initial_turn + 2

    def test_chat_trace_recording(self):
        """Test that chat turns are recorded in traces."""
        mock_openai_client = create_mock_openai_client("Test response")

        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir)
            llm_client = LLMClient(mock_openai_client, default_model="test-model")
            orchestrator = Orchestrator(llm_client, config=config)

            orchestrator.chat("Test input")

            last_trace = orchestrator.last_trace()
            assert last_trace is not None
            assert last_trace.user_input == "Test input"
            assert last_trace.assistant_response == "Test response"
            assert last_trace.latency_ms >= 0


class TestToolCallTracker:
    """Unit tests for the ToolCallTracker class."""

    def test_record_new_call(self):
        """Test recording a new tool call."""
        tracker = ToolCallTracker()
        call = ToolCall(name="search", arguments={"query": "test"})

        is_duplicate = tracker.record(call)

        assert is_duplicate is False

    def test_record_duplicate_call(self):
        """Test detecting a duplicate tool call."""
        tracker = ToolCallTracker()
        call = ToolCall(name="search", arguments={"query": "test"})

        tracker.record(call)
        is_duplicate = tracker.record(call)

        assert is_duplicate is True

    def test_reset_clears_duplicates(self):
        """Test that reset clears the duplicate tracking."""
        tracker = ToolCallTracker()
        call = ToolCall(name="search", arguments={"query": "test"})

        tracker.record(call)
        tracker.reset()
        is_duplicate = tracker.record(call)

        assert is_duplicate is False

    def test_different_args_not_duplicate(self):
        """Test that calls with different args are not duplicates."""
        tracker = ToolCallTracker()
        call1 = ToolCall(name="search", arguments={"query": "first"})
        call2 = ToolCall(name="search", arguments={"query": "second"})

        tracker.record(call1)
        is_duplicate = tracker.record(call2)

        assert is_duplicate is False

    def test_different_names_not_duplicate(self):
        """Test that calls with different names are not duplicates."""
        tracker = ToolCallTracker()
        call1 = ToolCall(name="search", arguments={"query": "test"})
        call2 = ToolCall(name="fetch", arguments={"query": "test"})

        tracker.record(call1)
        is_duplicate = tracker.record(call2)

        assert is_duplicate is False


class TestToolCallKeyGeneration:
    """Test the ToolCall.key() method for deduplication."""

    def test_key_consistency(self):
        """Test that same args produce same key."""
        call1 = ToolCall(name="search", arguments={"query": "test", "limit": 5})
        call2 = ToolCall(name="search", arguments={"query": "test", "limit": 5})

        assert call1.key() == call2.key()

    def test_key_uniqueness(self):
        """Test that different args produce different keys."""
        call1 = ToolCall(name="search", arguments={"query": "first"})
        call2 = ToolCall(name="search", arguments={"query": "second"})

        assert call1.key() != call2.key()

    def test_key_order_independence(self):
        """Test that key order doesn't matter for dict arguments."""
        call1 = ToolCall(name="search", arguments={"a": 1, "b": 2})
        call2 = ToolCall(name="search", arguments={"b": 2, "a": 1})

        assert call1.key() == call2.key()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
