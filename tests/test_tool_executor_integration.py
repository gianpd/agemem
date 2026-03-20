"""
Integration tests for ToolExecutor - comprehensive verification of RFC-001 changes.

These tests verify:
1. ToolExecutor works in isolation with mocked dependencies
2. Orchestrator correctly delegates to ToolExecutor
3. Side effects (MemoryOpResult) flow back correctly
4. STM injection happens for retrieval tools
5. Error handling is consistent
6. Tracing is properly integrated
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Any

from agents.tool_executor import ToolExecutor, ToolResult, ToolCategory
from agents.orchestrator import Orchestrator, ToolCall, ToolCallTracker
from core.types import MemoryOpResult, MemoryOp, TriggerKind, ContextMessage
from core.config import AgememConfig, DEFAULT_CONFIG


class TestToolExecutorStandalone:
    """
    Standalone tests for ToolExecutor with fully mocked dependencies.
    These verify the deep module works correctly in isolation.
    """

    def create_mock_dependencies(self) -> tuple[Any, ...]:
        """Create fully mocked dependencies for ToolExecutor."""
        mock_stm = Mock(spec=['messages', 'current_turn', 'retrieve', 'add_message'])
        mock_stm.messages.return_value = []
        mock_stm.current_turn.return_value = 5

        mock_ltm = Mock(spec=['search', 'add', 'all_entries'])
        mock_ltm.all_entries.return_value = []

        mock_llm = Mock(spec=['chat', 'complete'])
        mock_llm.chat.return_value = "test response"

        mock_config = Mock(spec=AgememConfig)
        mock_config.MEMORY_AGENT_MODEL = "test-model"

        mock_tracer = Mock(spec=['log_tool_call', 'log_introspection_trigger',
                                  'log_introspection_result', 'log_memory_op'])

        return mock_stm, mock_ltm, mock_llm, mock_config, mock_tracer

    def test_tool_executor_initialization(self):
        """ToolExecutor properly stores injected dependencies."""
        stm, ltm, llm, config, tracer = self.create_mock_dependencies()

        executor = ToolExecutor(
            stm=stm,
            ltm=ltm,
            llm=llm,
            config=config,
            tracer=tracer
        )

        assert executor._stm is stm
        assert executor._ltm is ltm
        assert executor._llm is llm
        assert executor._config is config
        assert executor._tracer is tracer

    def test_unknown_tool_error_format(self):
        """Unknown tools return consistent [TOOL ERROR] format."""
        stm, ltm, llm, config, tracer = self.create_mock_dependencies()
        executor = ToolExecutor(stm, ltm, llm, config, tracer)

        result = executor.execute("nonexistent_tool", {"arg": "value"})

        assert not result.success
        assert result.output.startswith("[TOOL ERROR]")
        assert "nonexistent_tool" in result.output
        assert result.tool_name == "nonexistent_tool"
        assert result.duration_ms >= 0

    def test_tracing_on_unknown_tool(self):
        """Tracer is called even for unknown tools."""
        stm, ltm, llm, config, tracer = self.create_mock_dependencies()
        executor = ToolExecutor(stm, ltm, llm, config, tracer)

        executor.execute("unknown", {"key": "value"})

        tracer.log_tool_call.assert_called_once()
        call_kwargs = tracer.log_tool_call.call_args.kwargs
        assert call_kwargs['tool_name'] == "unknown"
        assert call_kwargs['arguments'] == {"key": "value"}
        assert call_kwargs['success'] is False

    def test_web_search_with_various_async_scenarios(self):
        """web_search handles different async runtime scenarios."""
        stm, ltm, llm, config, tracer = self.create_mock_dependencies()
        executor = ToolExecutor(stm, ltm, llm, config, tracer)

        # Scenario 1: Normal async execution
        with patch("agents.tool_executor.asyncio.run") as mock_run:
            mock_run.return_value = "Search results: Python programming"

            result = executor.execute("web_search", {"query": "Python", "num_results": 5})

            assert result.success
            assert result.output == "Search results: Python programming"
            mock_run.assert_called_once()

        # Scenario 2: Exception returns [TOOL ERROR]
        with patch("agents.tool_executor.asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Network failure")

            result = executor.execute("web_search", {"query": "failing query"})

            assert not result.success
            assert "[TOOL ERROR]" in result.output
            assert "Network failure" in result.output

    def test_corpus_tools_integration(self):
        """Corpus tools delegate correctly with proper error handling."""
        stm, ltm, llm, config, tracer = self.create_mock_dependencies()
        executor = ToolExecutor(stm, ltm, llm, config, tracer)

        test_cases = [
            ("list_documents", {}, "doc1.md\ndoc2.md"),
            ("read_document", {"doc_id": "test"}, "# Test Document"),
            ("grep_corpus", {"pattern": "test", "context_lines": 3}, "test.py:1: def test():"),
            ("search_metadata", {"keyword": "python"}, "Found in: doc1.md"),
        ]

        for tool_name, args, expected_output in test_cases:
            with patch(f"tools.corpus.{tool_name}") as mock_tool:
                mock_tool.return_value = expected_output

                result = executor.execute(tool_name, args)

                assert result.success, f"{tool_name} should succeed"
                assert result.output == expected_output, f"{tool_name} output mismatch"
                mock_tool.assert_called_once()

    def test_introspection_tools_with_stm_access(self):
        """Introspection tools access STM for context."""
        stm, ltm, llm, config, tracer = self.create_mock_dependencies()

        # Simulate STM with messages
        mock_messages = [
            Mock(role="user", content="Hello", turn_index=1, timestamp=None),
            Mock(role="assistant", content="Hi there", turn_index=1, timestamp=None),
            Mock(role="user", content="What's Python?", turn_index=2, timestamp=None),
        ]
        stm.messages.return_value = mock_messages

        executor = ToolExecutor(stm, ltm, llm, config, tracer)

        with patch("memory.ltm_introspection.assess_conversation_drift") as mock_drift:
            mock_result = Mock()
            mock_result.to_dict.return_value = {"drift_detected": False}
            mock_result.confidence.value = "HIGH"
            mock_result.drift_type.value = "NONE"
            mock_result.topic_drift_score = 0.1
            mock_drift.return_value = mock_result

            result = executor.execute("assess_conversation_drift", {
                "current_query": "What's Python?"
            })

            assert result.success
            # Verify drift function was called with turns from STM
            call_args = mock_drift.call_args
            assert "recent_turns" in call_args.kwargs
            assert len(call_args.kwargs["recent_turns"]) <= 10  # Last 10 messages

            # Verify tracer was called
            tracer.log_introspection_trigger.assert_called()

    def test_trigger_retrieval_with_stm_injection(self):
        """trigger_contextual_ltm_retrieval injects into STM."""
        stm, ltm, llm, config, tracer = self.create_mock_dependencies()
        executor = ToolExecutor(stm, ltm, llm, config, tracer)

        with patch("memory.ltm_introspection.trigger_contextual_ltm_retrieval") as mock_trigger:
            # Simulate retrieved memories
            mock_memories = [
                Mock(
                    entry=Mock(content="Python is a programming language"),
                    retrieval_score=0.95,
                    source_query="Python"
                ),
                Mock(
                    entry=Mock(content="Python was created by Guido"),
                    retrieval_score=0.87,
                    source_query="Python"
                ),
            ]
            mock_result = Mock()
            mock_result.memories = mock_memories
            mock_result.to_dict.return_value = {"memories": [{"content": "m1"}, {"content": "m2"}]}
            mock_trigger.return_value = mock_result

            result = executor.execute("trigger_contextual_ltm_retrieval", {
                "query": "Python",
                "mode": "single_query",
                "top_k": 5
            })

            assert result.success
            assert result.should_inject_to_stm is True
            assert result.stm_injection_data is not None
            assert len(result.stm_injection_data) == 2
            assert result.stm_injection_data[0]["content"] == "Python is a programming language"

            # Verify introspection logging
            tracer.log_introspection_result.assert_called_once()

    def test_persistence_tools_with_side_effects(self):
        """Persistence tools return side effects for Orchestrator tracking."""
        stm, ltm, llm, config, tracer = self.create_mock_dependencies()
        executor = ToolExecutor(stm, ltm, llm, config, tracer)

        with patch("memory.ltm_introspection.force_memory_persistence") as mock_force:
            mock_result = Mock()
            mock_result.success = True
            mock_result.to_dict.return_value = {"success": True, "memory_id": "mem_123"}
            mock_force.return_value = mock_result

            result = executor.execute("force_memory_persistence", {
                "content": "Important memory to save",
                "learning_score": 0.95,
                "trigger": "user_command"
            })

            assert result.success
            assert len(result.side_effects) == 1
            side_effect = result.side_effects[0]
            assert side_effect.op == MemoryOp.ADD
            assert side_effect.success is True
            assert side_effect.trigger == TriggerKind.MAIN_AGENT

    def test_empty_content_for_persistence(self):
        """force_memory_persistence handles empty content gracefully."""
        stm, ltm, llm, config, tracer = self.create_mock_dependencies()
        executor = ToolExecutor(stm, ltm, llm, config, tracer)

        result = executor.execute("force_memory_persistence", {
            "content": "",  # Empty content
            "learning_score": 0.9
        })

        assert result.success  # Tool executed successfully
        result_data = json.loads(result.output)
        assert result_data["success"] is False
        assert "error" in result_data

    def test_error_handling_consistency(self):
        """All tools handle errors consistently with [TOOL ERROR] prefix."""
        stm, ltm, llm, config, tracer = self.create_mock_dependencies()
        executor = ToolExecutor(stm, ltm, llm, config, tracer)

        # Force errors in different tool categories
        error_test_cases = [
            ("web_search", lambda: patch("agents.tool_executor.asyncio.run", side_effect=Exception("Network error"))),
            ("list_documents", lambda: patch("tools.corpus.list_documents", side_effect=Exception("IO error"))),
            ("read_document", lambda: patch("tools.corpus.read_document", side_effect=Exception("File not found"))),
        ]

        for tool_name, patcher in error_test_cases:
            with patcher():
                result = executor.execute(tool_name, {"doc_id": "test"} if "document" in tool_name else {})

                assert not result.success, f"{tool_name} should fail"
                assert result.output.startswith("[TOOL ERROR]"), f"{tool_name} should use [TOOL ERROR] prefix"
                assert tool_name in result.output, f"{tool_name} should be mentioned in error"


class TestOrchestratorToolExecutorIntegration:
    """
    Integration tests verifying Orchestrator and ToolExecutor work together.
    """

    def create_minimal_orchestrator(self) -> Orchestrator:
        """Create an Orchestrator with fully mocked LLM."""
        mock_llm = Mock()
        mock_llm.chat.return_value = "test response"
        mock_llm.complete.return_value = "test completion"

        config = DEFAULT_CONFIG

        return Orchestrator(llm=mock_llm, config=config)

    def test_orchestrator_initializes_tool_executor(self):
        """Orchestrator creates ToolExecutor on initialization."""
        orch = self.create_minimal_orchestrator()

        assert hasattr(orch, '_tool_executor')
        assert isinstance(orch._tool_executor, ToolExecutor)
        assert orch._tool_executor._stm is orch._stm
        assert orch._tool_executor._ltm is orch._ltm
        assert orch._tool_executor._llm is orch._llm

    def test_execute_tool_delegates_to_executor(self):
        """_execute_tool delegates to ToolExecutor and returns output."""
        orch = self.create_minimal_orchestrator()

        # Mock the executor's execute method
        orch._tool_executor.execute = Mock(return_value=ToolResult(
            output="tool result",
            success=True,
            tool_name="test_tool"
        ))

        result = orch._execute_tool("test_tool", {"arg": "value"})

        assert result == "tool result"
        orch._tool_executor.execute.assert_called_once_with("test_tool", {"arg": "value"})

    def test_execute_tool_handles_stm_injection(self):
        """_execute_tool handles STM injection for retrieval tools."""
        orch = self.create_minimal_orchestrator()

        # Mock STM.retrieve to avoid the entry_id issue
        orch._stm.retrieve = Mock(return_value=MemoryOpResult(
            op=MemoryOp.RETRIEVE, success=True, trigger=TriggerKind.MAIN_AGENT
        ))

        injection_data = [{"content": "memory 1"}, {"content": "memory 2"}]
        orch._tool_executor.execute = Mock(return_value=ToolResult(
            output="retrieved memories",
            success=True,
            tool_name="trigger_contextual_ltm_retrieval",
            should_inject_to_stm=True,
            stm_injection_data=injection_data
        ))

        result = orch._execute_tool("trigger_contextual_ltm_retrieval", {"query": "test"})

        assert result == "retrieved memories"
        # Verify STM.retrieve was called with injection data
        orch._stm.retrieve.assert_called_once_with(injection_data, trigger=TriggerKind.MAIN_AGENT)

    def test_tool_call_loop_uses_executor(self):
        """Tool call loop in chat() uses ToolExecutor."""
        orch = self.create_minimal_orchestrator()

        # Track executor calls
        orch._tool_executor.execute = Mock(return_value=ToolResult(
            output="search results",
            success=True,
            tool_name="web_search"
        ))

        # Mock the response handler to simulate a tool call
        with patch.object(orch._response_handler, 'chat_with_recovery') as mock_chat:
            from agents.llm_client import ToolCallResponse
            from agents.response_handler import ResponseMetrics, ResponseType
            from dataclasses import dataclass

            @dataclass
            class MockFunction:
                name: str
                arguments: dict

            @dataclass
            class MockToolCall:
                id: str
                function: MockFunction

            # ResponseMetrics for successful response
            metrics = ResponseMetrics(
                response_type=ResponseType.TEXT,
                latency_ms=100.0,
                token_count=10,
                has_tool_calls=False,
                json_valid=True,
                quality_score=0.9
            )

            # First call raises ToolCallResponse, second call returns (text, metrics) tuple
            mock_chat.side_effect = [
                ToolCallResponse(MockToolCall(
                    id="call_123",
                    function=MockFunction("web_search", {"query": "test"})
                )),
                ("Final response based on search", metrics)
            ]

            orch.chat("Do a search")

            # Verify executor was called
            orch._tool_executor.execute.assert_called_once_with("web_search", {"query": "test"})


class TestToolCallTracker:
    """Tests for the LoopGuard pattern in tool call tracking."""

    def test_duplicate_detection(self):
        """ToolCallTracker correctly identifies duplicates."""
        tracker = ToolCallTracker()

        call1 = ToolCall(name="web_search", arguments={"query": "python"})
        call2 = ToolCall(name="web_search", arguments={"query": "python"})  # Same
        call3 = ToolCall(name="web_search", arguments={"query": "java"})    # Different

        # First call is new
        assert tracker.record(call1) is False

        # Second call with same args is duplicate
        assert tracker.record(call2) is True

        # Third call with different args is new
        assert tracker.record(call3) is False

    def test_reset_clears_tracker(self):
        """Reset clears all tracked calls."""
        tracker = ToolCallTracker()

        call = ToolCall(name="tool", arguments={"arg": "value"})
        tracker.record(call)

        # Should be duplicate
        assert tracker.record(call) is True

        # Reset
        tracker.reset()

        # Should be new after reset
        assert tracker.record(call) is False

    def test_key_generation_consistency(self):
        """ToolCall.key() generates consistent keys for same args."""
        call1 = ToolCall(name="test", arguments={"a": 1, "b": 2})
        call2 = ToolCall(name="test", arguments={"b": 2, "a": 1})  # Different order

        # Keys should be the same (sorted)
        assert call1.key() == call2.key()


class TestToolResultMetadata:
    """Tests for ToolResult metadata and side effects tracking."""

    def test_duration_tracking(self):
        """ToolResult accurately tracks execution duration."""
        import time

        result = ToolResult(output="test", duration_ms=150.5)

        assert result.duration_ms == 150.5

    def test_side_effects_collection(self):
        """ToolResult can collect multiple side effects."""
        effects = [
            MemoryOpResult(op=MemoryOp.ADD, success=True, trigger=TriggerKind.MAIN_AGENT),
            MemoryOpResult(op=MemoryOp.UPDATE, success=True, trigger=TriggerKind.SYSTEM_RULE),
        ]

        result = ToolResult(
            output="test",
            side_effects=effects
        )

        assert len(result.side_effects) == 2
        assert result.side_effects[0].op == MemoryOp.ADD
        assert result.side_effects[1].op == MemoryOp.UPDATE

    def test_stm_injection_data(self):
        """ToolResult correctly stores STM injection data."""
        injection_data = [
            {"content": "fact 1", "score": 0.9},
            {"content": "fact 2", "score": 0.8},
        ]

        result = ToolResult(
            output="retrieved",
            should_inject_to_stm=True,
            stm_injection_data=injection_data
        )

        assert result.should_inject_to_stm is True
        assert len(result.stm_injection_data) == 2
        assert result.stm_injection_data[0]["score"] == 0.9


class TestToolCategoryClassification:
    """Tests for tool category classification."""

    def test_category_values(self):
        """ToolCategory enum has expected values."""
        categories = {
            ToolCategory.EXTERNAL: "external",
            ToolCategory.CORPUS: "corpus",
            ToolCategory.INTROSPECTION: "introspection",
            ToolCategory.PERSISTENCE: "persistence",
        }

        for cat, value in categories.items():
            assert cat.value == value

    def test_category_comparison(self):
        """ToolCategory can be compared."""
        assert ToolCategory.EXTERNAL != ToolCategory.CORPUS
        assert ToolCategory.EXTERNAL == ToolCategory.EXTERNAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
