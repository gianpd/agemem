"""Tests for ToolExecutor - extracted from Orchestrator."""
import pytest
import json
from unittest.mock import Mock, patch

from agents.tool_executor import ToolExecutor, ToolResult, ToolCategory
from core.types import MemoryOpResult, MemoryOp, TriggerKind


class TestToolExecutor:
    """Test ToolExecutor in isolation."""

    def setup_method(self):
        """Create ToolExecutor with mocks."""
        self.mock_stm = Mock()
        self.mock_stm.messages.return_value = []
        self.mock_stm.current_turn.return_value = 0
        self.mock_ltm = Mock()
        self.mock_llm = Mock()
        self.mock_config = Mock()
        self.mock_config.MEMORY_AGENT_MODEL = "test-model"

        self.executor = ToolExecutor(
            stm=self.mock_stm,
            ltm=self.mock_ltm,
            llm=self.mock_llm,
            config=self.mock_config,
            tracer=Mock(),  # Explicit mock tracer
        )

    def test_execute_unknown_tool_returns_error(self):
        """Unknown tools return [TOOL ERROR]."""
        result = self.executor.execute("unknown_tool", {})

        assert not result.success
        assert "[TOOL ERROR]" in result.output
        assert "unknown_tool" in result.output
        assert result.tool_name == "unknown_tool"
        assert result.duration_ms >= 0

    def test_execute_includes_duration(self):
        """ToolResult includes execution duration."""
        result = self.executor.execute("unknown_tool", {})

        assert result.duration_ms >= 0

    def test_execute_logs_to_tracer(self):
        """Execution logs to tracer."""
        result = self.executor.execute("unknown_tool", {})

        self.executor._tracer.log_tool_call.assert_called_once()
        call_args = self.executor._tracer.log_tool_call.call_args
        assert call_args.kwargs['tool_name'] == "unknown_tool"
        assert call_args.kwargs['success'] is False

    def test_tool_result_detects_error_prefix(self):
        """ToolResult.success is False when output starts with [TOOL ERROR]."""
        result = ToolResult(output="[TOOL ERROR] something failed")
        assert not result.success

    def test_tool_result_default_success(self):
        """ToolResult.success is True by default."""
        result = ToolResult(output="normal output")
        assert result.success

    def test_tool_category_enum(self):
        """ToolCategory enum values are correct."""
        assert ToolCategory.EXTERNAL.value == "external"
        assert ToolCategory.CORPUS.value == "corpus"
        assert ToolCategory.INTROSPECTION.value == "introspection"
        assert ToolCategory.PERSISTENCE.value == "persistence"

    def test_execute_web_search_delegates_correctly(self):
        """web_search tool calls web_tools module."""
        with patch("agents.tool_executor.asyncio.run") as mock_run:
            mock_run.return_value = "Search results"

            result = self.executor.execute("web_search", {"query": "test"})

            assert result.success
            assert result.output == "Search results"
            mock_run.assert_called_once()

    def test_execute_corpus_tools(self):
        """Corpus tools are dispatched correctly."""
        with patch("tools.corpus.list_documents") as mock_list:
            mock_list.return_value = "doc1\ndoc2"

            result = self.executor.execute("list_documents", {})

            assert result.success
            assert result.output == "doc1\ndoc2"
            mock_list.assert_called_once()

    def test_execute_read_document(self):
        """read_document tool works."""
        with patch("tools.corpus.read_document") as mock_read:
            mock_read.return_value = "document content"

            result = self.executor.execute("read_document", {"doc_id": "test_doc"})

            assert result.success
            assert result.output == "document content"
            mock_read.assert_called_once_with("test_doc", corpus_path=None)

    def test_execute_with_side_effects(self):
        """Tools can return side effects."""
        # force_memory_persistence returns side effects
        with patch("memory.ltm_introspection.force_memory_persistence") as mock_force:
            mock_result = Mock()
            mock_result.success = True
            mock_result.to_dict.return_value = {"success": True}
            mock_force.return_value = mock_result

            result = self.executor.execute("force_memory_persistence", {
                "content": "test memory"
            })

            assert result.success
            assert len(result.side_effects) == 1
            assert result.side_effects[0].op == MemoryOp.ADD

    def test_execute_trigger_retrieval_injection(self):
        """trigger_contextual_ltm_retrieval sets injection flag."""
        with patch("memory.ltm_introspection.trigger_contextual_ltm_retrieval") as mock_trigger:
            mock_result = Mock()
            mock_result.memories = [
                Mock(entry=Mock(content="memory1"), retrieval_score=0.9, source_query="q")
            ]
            mock_result.to_dict.return_value = {"memories": [{"content": "memory1"}]}
            mock_trigger.return_value = mock_result

            result = self.executor.execute("trigger_contextual_ltm_retrieval", {
                "query": "test"
            })

            assert result.success
            assert result.should_inject_to_stm is True
            assert result.stm_injection_data is not None
            assert len(result.stm_injection_data) == 1

    def test_execute_assess_persistence_need(self):
        """assess_persistence_need tool works."""
        with patch("memory.ltm_introspection.assess_persistence_need") as mock_assess:
            mock_result = Mock()
            mock_result.should_persist = True
            mock_result.urgency.value = "high"
            mock_result.to_dict.return_value = {"should_persist": True}
            mock_assess.return_value = mock_result

            result = self.executor.execute("assess_persistence_need", {
                "content": "important fact"
            })

            assert result.success
            assert "should_persist" in result.output

    def test_get_definitions_returns_empty_list(self):
        """get_definitions returns empty list (placeholder for future)."""
        definitions = self.executor.get_definitions()
        assert definitions == []

    def test_get_definitions_with_category(self):
        """get_definitions with category returns empty list."""
        definitions = self.executor.get_definitions(ToolCategory.EXTERNAL)
        assert definitions == []


class TestToolResult:
    """Test ToolResult dataclass."""

    def test_default_values(self):
        """ToolResult has correct defaults."""
        result = ToolResult(output="test")
        assert result.output == "test"
        assert result.success is True
        assert result.tool_name == ""
        assert result.duration_ms == 0.0
        assert result.side_effects == []
        assert result.should_inject_to_stm is False
        assert result.stm_injection_data is None

    def test_explicit_values(self):
        """ToolResult accepts explicit values."""
        side_effects = [MemoryOpResult(op=MemoryOp.ADD, success=True, trigger=TriggerKind.MAIN_AGENT)]
        result = ToolResult(
            output="test output",
            success=False,
            tool_name="test_tool",
            duration_ms=100.5,
            side_effects=side_effects,
            should_inject_to_stm=True,
            stm_injection_data=["data"],
        )
        assert result.output == "test output"
        assert result.success is False
        assert result.tool_name == "test_tool"
        assert result.duration_ms == 100.5
        assert result.side_effects == side_effects
        assert result.should_inject_to_stm is True
        assert result.stm_injection_data == ["data"]
