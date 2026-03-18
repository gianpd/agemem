"""
Tests for the enhanced response handler.

Validates tool call validation, retry logic, and metrics tracking.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from agents.response_handler import (
    ResponseHandler,
    ResponseType,
    ResponseMetrics,
    ToolCallValidation,
)
from agents.llm_client import LLMClient, ToolCallResponse, TextToolCallResponse, JSONParseError


class MockToolCall:
    """Mock tool call for testing."""
    def __init__(self, name: str, arguments: any, id: str = "test_id"):
        self.id = id
        self.function = Mock()
        self.function.name = name
        self.function.arguments = arguments


class TestToolCallValidation:
    """Test tool call validation functionality."""

    def test_valid_tool_call_with_dict_args(self):
        """Test validation of a valid tool call with dict arguments."""
        handler = ResponseHandler(Mock())
        tool_call = MockToolCall("web_search", {"query": "test"})
        
        validation = handler.validate_tool_call(tool_call)
        
        assert validation.is_valid
        assert validation.tool_name == "web_search"
        assert validation.arguments == {"query": "test"}
        assert len(validation.errors) == 0

    def test_valid_tool_call_with_json_string_args(self):
        """Test validation of a valid tool call with JSON string arguments."""
        handler = ResponseHandler(Mock())
        tool_call = MockToolCall("web_search", '{"query": "test"}')
        
        validation = handler.validate_tool_call(tool_call)
        
        assert validation.is_valid
        assert validation.tool_name == "web_search"
        assert validation.arguments == {"query": "test"}

    def test_missing_tool_name(self):
        """Test validation fails when tool name is missing."""
        handler = ResponseHandler(Mock())
        tool_call = MockToolCall("", {"query": "test"})
        
        validation = handler.validate_tool_call(tool_call)
        
        assert not validation.is_valid
        assert "Tool name is missing" in validation.errors

    def test_invalid_json_arguments(self):
        """Test validation fails with invalid JSON arguments."""
        handler = ResponseHandler(Mock())
        tool_call = MockToolCall("web_search", '{"query": "test"')  # Missing closing brace
        
        validation = handler.validate_tool_call(tool_call)
        
        assert not validation.is_valid
        assert any("Invalid JSON" in error for error in validation.errors)

    def test_repair_json_with_code_blocks(self):
        """Test JSON repair removes markdown code blocks."""
        handler = ResponseHandler(Mock())
        tool_call = MockToolCall("web_search", '```json\n{"query": "test"}\n```')

        validation = handler.validate_tool_call(tool_call)

        assert validation.is_valid
        assert validation.arguments == {"query": "test"}

    def test_repair_json_with_trailing_comma(self):
        """Test JSON repair fixes trailing commas."""
        handler = ResponseHandler(Mock())
        tool_call = MockToolCall("web_search", '{"query": "test",}')
        
        validation = handler.validate_tool_call(tool_call)
        
        assert validation.is_valid
        assert validation.arguments == {"query": "test"}

    def test_repair_json_with_unquoted_keys(self):
        """Test JSON repair quotes unquoted keys."""
        handler = ResponseHandler(Mock())
        tool_call = MockToolCall("web_search", '{query: "test"}')
        
        validation = handler.validate_tool_call(tool_call)
        
        assert validation.is_valid
        assert validation.arguments == {"query": "test"}

    def test_none_arguments(self):
        """Test validation handles None arguments."""
        handler = ResponseHandler(Mock())
        tool_call = MockToolCall("web_search", None)
        
        validation = handler.validate_tool_call(tool_call)
        
        assert validation.is_valid
        assert validation.arguments == {}
        assert any("No arguments provided" in warning for warning in validation.warnings)


class TestResponseMetrics:
    """Test response metrics tracking."""

    def test_metrics_initialization(self):
        """Test metrics are properly initialized."""
        metrics = ResponseMetrics(
            response_type=ResponseType.TEXT,
            latency_ms=100.0,
            token_count=50,
            has_tool_calls=False,
            json_valid=True,
        )
        
        assert metrics.response_type == ResponseType.TEXT
        assert metrics.latency_ms == 100.0
        assert metrics.token_count == 50
        assert metrics.has_tool_calls is False
        assert metrics.json_valid is True
        assert metrics.error_count == 0
        assert metrics.recovery_attempts == 0
        assert metrics.quality_score == 1.0

    def test_metrics_quality_score_calculation(self):
        """Test quality score is calculated correctly."""
        metrics = ResponseMetrics(
            response_type=ResponseType.TEXT,
            latency_ms=100.0,
            token_count=50,
            has_tool_calls=False,
            json_valid=True,
            error_count=1,
            recovery_attempts=2,
        )
        
        # Quality score should be reduced based on errors and recovery attempts
        assert metrics.quality_score < 1.0


class TestResponseHandlerRetry:
    """Test retry logic and error recovery."""

    def test_successful_first_attempt(self):
        """Test successful response on first attempt."""
        mock_llm = Mock()
        mock_llm.chat.return_value = "Hello, world!"
        
        handler = ResponseHandler(mock_llm, max_retries=3)
        response, metrics = handler.chat_with_recovery(
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert response == "Hello, world!"
        assert metrics.recovery_attempts == 0
        assert metrics.quality_score == 1.0
        mock_llm.chat.assert_called_once()

    def test_retry_on_transient_error(self):
        """Test retry logic on transient errors."""
        mock_llm = Mock()
        # Fail twice, then succeed
        mock_llm.chat.side_effect = [
            Exception("Temporary error"),
            Exception("Temporary error"),
            "Success!"
        ]
        
        handler = ResponseHandler(mock_llm, max_retries=3)
        response, metrics = handler.chat_with_recovery(
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert response == "Success!"
        assert metrics.recovery_attempts == 2
        assert mock_llm.chat.call_count == 3

    def test_max_retries_exceeded(self):
        """Test failure after max retries exceeded."""
        mock_llm = Mock()
        mock_llm.chat.side_effect = Exception("Persistent error")
        
        handler = ResponseHandler(mock_llm, max_retries=2)
        
        with pytest.raises(RuntimeError, match="LLM call failed after 3 attempts"):
            handler.chat_with_recovery(
                messages=[{"role": "user", "content": "Hello"}],
            )
        
        assert mock_llm.chat.call_count == 3

    def test_tool_call_validation_failure_recovery(self):
        """Test recovery from tool call validation failure."""
        mock_llm = Mock()
        # First call returns invalid tool call, second call returns valid text
        invalid_tool_call = MockToolCall("", {"query": "test"})  # Missing name
        mock_llm.chat.side_effect = [
            ToolCallResponse(invalid_tool_call),
            "Valid text response"
        ]
        
        handler = ResponseHandler(mock_llm, max_retries=2, enable_validation=True)
        response, metrics = handler.chat_with_recovery(
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert response == "Valid text response"
        assert metrics.recovery_attempts == 1
        assert mock_llm.chat.call_count == 2

    def test_json_parse_error_recovery(self):
        """Test recovery from JSON parse error."""
        mock_llm = Mock()
        # First call returns invalid JSON, second call returns valid JSON
        mock_llm.chat_json.side_effect = [
            JSONParseError("invalid json", "parse error"),
            {"score": 0.8, "rationale": "test"}
        ]
        
        handler = ResponseHandler(mock_llm, max_retries=2)
        result, metrics = handler.chat_json_with_recovery(
            messages=[{"role": "user", "content": "Return JSON"}],
        )
        
        assert result == {"score": 0.8, "rationale": "test"}
        assert metrics.recovery_attempts == 1
        assert mock_llm.chat_json.call_count == 2


class TestResponseHandlerMetrics:
    """Test metrics tracking functionality."""

    def test_metrics_history_tracking(self):
        """Test metrics are tracked in history."""
        mock_llm = Mock()
        mock_llm.chat.return_value = "Response 1"
        
        handler = ResponseHandler(mock_llm)
        
        # First call
        handler.chat_with_recovery(messages=[{"role": "user", "content": "Hello"}])
        
        # Second call
        mock_llm.chat.return_value = "Response 2"
        handler.chat_with_recovery(messages=[{"role": "user", "content": "World"}])
        
        metrics = handler.get_metrics()
        assert len(metrics) == 2
        assert metrics[0].token_count > 0
        assert metrics[1].token_count > 0

    def test_average_quality_calculation(self):
        """Test average quality score calculation."""
        mock_llm = Mock()
        mock_llm.chat.return_value = "Response"
        
        handler = ResponseHandler(mock_llm)
        
        # Make multiple calls
        for i in range(3):
            handler.chat_with_recovery(messages=[{"role": "user", "content": f"Message {i}"}])
        
        avg_quality = handler.get_average_quality()
        assert 0.0 <= avg_quality <= 1.0

    def test_metrics_reset(self):
        """Test metrics can be reset."""
        mock_llm = Mock()
        mock_llm.chat.return_value = "Response"
        
        handler = ResponseHandler(mock_llm)
        handler.chat_with_recovery(messages=[{"role": "user", "content": "Hello"}])
        
        assert len(handler.get_metrics()) == 1
        
        # Reset by creating new handler
        handler = ResponseHandler(mock_llm)
        assert len(handler.get_metrics()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])