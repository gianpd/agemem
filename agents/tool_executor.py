"""Deep module for tool execution - hides dispatch and dependencies."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import time
import json
import asyncio
import logging

from core.types import MemoryOpResult, TriggerKind
from core.config import AgememConfig
from core.tracing import get_tracer
from memory.ltm_store import LTMStore
from memory.stm_context import STMContext
from agents.llm_client import LLMClient


class ToolCategory(Enum):
    """Tool classification for filtering and organization."""
    EXTERNAL = "external"           # web_search, fetch_url, file I/O
    CORPUS = "corpus"               # grep_corpus, read_document, etc.
    INTROSPECTION = "introspection" # assess_drift, readiness_check, etc.
    PERSISTENCE = "persistence"     # force_persistence, validate_commit, etc.


@dataclass
class ToolResult:
    """Result of tool execution with metadata for tracing."""
    output: str                           # String result for LLM context
    success: bool = True                  # False if [TOOL ERROR] prefix
    tool_name: str = ""                   # For logging
    duration_ms: float = 0.0              # For tracing
    side_effects: list[MemoryOpResult] = field(default_factory=list)
    should_inject_to_stm: bool = False    # For retrieval tools
    stm_injection_data: Optional[list] = None

    def __post_init__(self):
        if self.output.startswith("[TOOL ERROR]"):
            self.success = False


class ToolExecutor:
    """
    Deep module for tool execution.

    Hides:
    - Tool dispatch (17 tools, 4 categories)
    - Dependency injection (STM, LTM, LLM access)
    - Async/sync complexity (event loops, thread pools)
    - Error wrapping (consistent [TOOL ERROR] format)
    - Tracing (automatic logging via injected tracer)
    - STM injection (retrieval tools auto-inject)

    The Orchestrator calls execute() and gets back a ToolResult.
    Everything else is implementation detail.
    """

    def __init__(
        self,
        stm: STMContext,
        ltm: LTMStore,
        llm: LLMClient,
        config: AgememConfig,
        tracer: Optional[Any] = None,
    ):
        """
        Inject all dependencies at construction.

        Args:
            stm: Short-term memory (read/write for retrieval injection)
            ltm: Long-term memory (read/write for persistence tools)
            llm: LLM client (for introspection tools that need LLM calls)
            config: AgeMem configuration (model names, timeouts, etc.)
            tracer: Optional tracer for logging (uses get_tracer() if None)
        """
        self._stm = stm
        self._ltm = ltm
        self._llm = llm
        self._config = config
        self._tracer = tracer  # None means use get_tracer() lazily

        # Lazy-loaded tool modules (cached after first import)
        self._web_tools = None
        self._corpus_tools = None
        self._introspection = None

    def _get_tracer(self):
        """Get tracer, lazily loading from get_tracer() if not injected."""
        return self._tracer or get_tracer()

    # === Tool Handler Methods (moved from Orchestrator) ===

    # --- External Tools ---

    def _execute_web_search(self, arguments: dict) -> str:
        """Execute web_search tool."""
        try:
            from tools.web_tools import web_search
            query = arguments.get("query", "")
            num_results = arguments.get("num_results", 5)

            try:
                result = asyncio.run(web_search(query, num_results))
            except RuntimeError:
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    result = asyncio.run(web_search(query, num_results))
                except ImportError:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, web_search(query, num_results))
                            result = future.result()
                    else:
                        result = loop.run_until_complete(web_search(query, num_results))
            return result
        except Exception as e:
            return f"[TOOL ERROR] web_search failed: {e}"

    def _execute_write_file(self, arguments: dict) -> str:
        """Execute write_file tool."""
        try:
            from tools.web_tools import write_file
            path = arguments.get("path", "")
            content = arguments.get("content", "")
            return write_file(path, content)
        except Exception as e:
            return f"[TOOL ERROR] write_file failed: {e}"

    def _execute_ingest_document(self, arguments: dict) -> str:
        """Execute ingest_document tool."""
        try:
            from tools.corpus import ingest_document
            path = arguments.get("path", "")
            return ingest_document(path)
        except Exception as e:
            return f"[TOOL ERROR] ingest_document failed: {e}"

    def _execute_fetch_url(self, arguments: dict) -> str:
        """Execute fetch_url tool."""
        try:
            from tools.web_tools import fetch_url_tool
            url = arguments.get("url", "")
            max_length = arguments.get("max_length", 10000)
            save_path = arguments.get("save_path")
            return fetch_url_tool(url, max_length, save_path)
        except Exception as e:
            return f"[TOOL ERROR] fetch_url failed: {e}"

    def _execute_browser_navigate(self, arguments: dict) -> str:
        """Execute browser_navigate tool."""
        try:
            from tools.web_tools import browser_navigate_tool
            return browser_navigate_tool(
                url=arguments.get("url", ""),
                action=arguments.get("action", "navigate"),
                full_page=arguments.get("full_page", True),
                wait_ms=arguments.get("wait_ms", 1000),
                headless=arguments.get("headless", True),
                use_cdp=arguments.get("use_cdp", False),
                wait_until=arguments.get("wait_until", "networkidle"),
                output_dir=arguments.get("output_dir", "screenshots"),
                keep_session=arguments.get("keep_session", False),
            )
        except Exception as e:
            return f"[TOOL ERROR] browser_navigate failed: {e}"

    # --- Browser Automation Tools ---

    def _execute_browser_click(self, arguments: dict) -> str:
        """Execute browser_click tool."""
        try:
            from tools.browser_tools import browser_click_tool
            return browser_click_tool(
                x=arguments.get("x", 0),
                y=arguments.get("y", 0),
                button=arguments.get("button", "left"),
                double=arguments.get("double", False),
            )
        except Exception as e:
            return f"[TOOL ERROR] browser_click failed: {e}"

    def _execute_browser_scroll(self, arguments: dict) -> str:
        """Execute browser_scroll tool."""
        try:
            from tools.browser_tools import browser_scroll_tool
            return browser_scroll_tool(
                direction=arguments.get("direction", "down"),
                amount=arguments.get("amount", 500),
            )
        except Exception as e:
            return f"[TOOL ERROR] browser_scroll failed: {e}"

    def _execute_browser_type(self, arguments: dict) -> str:
        """Execute browser_type tool."""
        try:
            from tools.browser_tools import browser_type_tool
            return browser_type_tool(
                text=arguments.get("text", ""),
                clear_first=arguments.get("clear_first", False),
            )
        except Exception as e:
            return f"[TOOL ERROR] browser_type failed: {e}"

    def _execute_browser_press(self, arguments: dict) -> str:
        """Execute browser_press tool."""
        try:
            from tools.browser_tools import browser_press_tool
            return browser_press_tool(
                key=arguments.get("key", "Enter"),
            )
        except Exception as e:
            return f"[TOOL ERROR] browser_press failed: {e}"

    def _execute_browser_read_page(self, arguments: dict) -> str:
        """Execute browser_read_page tool."""
        try:
            from tools.browser_tools import browser_read_page_tool
            return browser_read_page_tool(
                selector=arguments.get("selector"),
                max_length=arguments.get("max_length", 10000),
            )
        except Exception as e:
            return f"[TOOL ERROR] browser_read_page failed: {e}"

    def _execute_browser_screenshot(self, arguments: dict) -> str:
        """Execute browser_screenshot tool."""
        try:
            from tools.browser_tools import browser_screenshot_tool
            return browser_screenshot_tool(
                full_page=arguments.get("full_page", True),
                output_dir=arguments.get("output_dir", "screenshots"),
            )
        except Exception as e:
            return f"[TOOL ERROR] browser_screenshot failed: {e}"

    def _execute_browser_close(self, arguments: dict) -> str:
        """Execute browser_close tool."""
        try:
            from tools.browser_tools import browser_close_tool
            return browser_close_tool()
        except Exception as e:
            return f"[TOOL ERROR] browser_close failed: {e}"

    # --- Corpus Tools ---

    def _execute_list_documents(self, arguments: dict) -> str:
        """Execute list_documents tool."""
        try:
            from tools.corpus import list_documents
            return list_documents()
        except Exception as e:
            return f"[TOOL ERROR] list_documents failed: {e}"

    def _execute_search_metadata(self, arguments: dict) -> str:
        """Execute search_metadata tool."""
        try:
            from tools.corpus import search_metadata
            keyword = arguments.get("keyword", "")
            return search_metadata(keyword)
        except Exception as e:
            return f"[TOOL ERROR] search_metadata failed: {e}"

    def _execute_grep_corpus(self, arguments: dict) -> str:
        """Execute grep_corpus tool."""
        try:
            from tools.corpus import grep_corpus
            pattern = arguments.get("pattern", "")
            context_lines = arguments.get("context_lines", 3)
            return grep_corpus(pattern, context_lines)
        except Exception as e:
            return f"[TOOL ERROR] grep_corpus failed: {e}"

    def _execute_read_document(self, arguments: dict) -> str:
        """Execute read_document tool."""
        try:
            from tools.corpus import read_document
            doc_id = arguments.get("doc_id", "")
            return read_document(doc_id)
        except Exception as e:
            return f"[TOOL ERROR] read_document failed: {e}"

    def _execute_read_lines(self, arguments: dict) -> str:
        """Execute read_lines tool."""
        try:
            from tools.corpus import read_lines
            doc_id = arguments.get("doc_id", "")
            start_line = arguments.get("start_line", 1)
            end_line = arguments.get("end_line", 75)
            return read_lines(doc_id, start_line, end_line)
        except Exception as e:
            return f"[TOOL ERROR] read_lines failed: {e}"

    # --- Introspection Tools ---

    def _execute_assess_drift(self, arguments: dict) -> str:
        """Execute assess_conversation_drift with access to current state."""
        from memory.ltm_introspection import assess_conversation_drift
        from memory.ltm_introspection_types import Turn, ConfidenceLevel

        current_query = arguments.get("current_query", "")
        recent_context = arguments.get("recent_context", "")

        # Get recent messages from STM if not provided
        if not recent_context:
            recent_msgs = self._stm.messages()[-4:]  # Last 4 messages
            recent_context = "\n".join([
                f"{m.role}: {m.content[:200]}"
                for m in recent_msgs
            ])

        # Build turns from STM history
        turns = [
            Turn(
                role=m.role,
                content=m.content[:500],  # Truncate for efficiency
                turn_index=getattr(m, 'turn_index', 0),
                timestamp=getattr(m, 'timestamp', None)
            )
            for m in self._stm.messages()[-10:]
        ]

        result = assess_conversation_drift(
            current_query=current_query,
            recent_turns=turns,
        )

        # Map ConfidenceLevel to float for tracing
        confidence_map = {
            ConfidenceLevel.HIGH: 0.9,
            ConfidenceLevel.MEDIUM: 0.6,
            ConfidenceLevel.LOW: 0.3,
        }
        confidence_float = confidence_map.get(result.confidence, 0.5)

        # Log to tracing
        self._get_tracer().log_introspection_trigger(
            trigger_type="drift_assessment",
            confidence=confidence_float,
            context_summary=f"drift_type={result.drift_type.value}, score={result.topic_drift_score:.2f}",
        )

        return json.dumps(result.to_dict(), indent=2)

    def _execute_readiness_check(self, arguments: dict) -> str:
        """Execute are_you_ready_to_get_in_context_ltm."""
        from memory.ltm_introspection import are_you_ready_to_get_in_context_ltm

        query = arguments.get("current_query", "")
        urgency = arguments.get("urgency", "helpful")

        result = are_you_ready_to_get_in_context_ltm(
            query=query,
            urgency=urgency,
            current_messages=self._stm.messages()[-5:] if self._stm else None,
            current_turn=self._stm.current_turn() if self._stm else 0,
            ltm_store=self._ltm,
        )

        # Log readiness assessment
        confidence_score = result.confidence_report.overall_score if result.confidence_report else 0.5
        self._get_tracer().log_introspection_trigger(
            trigger_type="readiness_check",
            confidence=confidence_score,
            context_summary=f"should_retrieve={result.should_retrieve}, strategy={result.suggested_retrieval_strategy}",
        )

        return json.dumps(result.to_dict(), indent=2)

    def _execute_paraphrase(self, arguments: dict) -> str:
        """Execute paraphrase_for_coverage."""
        from memory.ltm_introspection import paraphrase_for_coverage

        query = arguments.get("query", "")
        n_variants = arguments.get("n_variants", 3)

        result = paraphrase_for_coverage(
            query=query,
            llm_client=self._llm,
            model=self._config.MEMORY_AGENT_MODEL,
            n_variants=min(n_variants, 5),
        )

        return json.dumps(result.to_dict() if hasattr(result, 'to_dict') else result, indent=2)

    def _execute_trigger_retrieval(self, arguments: dict) -> tuple[str, list[MemoryOpResult], list]:
        """Execute trigger_contextual_ltm_retrieval."""
        from memory.ltm_introspection import trigger_contextual_ltm_retrieval

        query = arguments.get("query", "")
        mode = arguments.get("mode", "single_query")
        top_k = arguments.get("top_k", 5)

        result = trigger_contextual_ltm_retrieval(
            query_or_concept=query,
            llm_client=self._llm,
            model=self._config.MEMORY_AGENT_MODEL,
            retrieval_mode=mode,
            top_k=top_k,
            ltm_store=self._ltm,
        )

        # Build injection data and side effects
        side_effects: list[MemoryOpResult] = []
        injection_data = None

        if result.memories:
            entries = [
                {
                    "content": m.entry.content if hasattr(m.entry, 'content') else str(m.entry),
                    "score": m.retrieval_score,
                    "source": m.source_query,
                }
                for m in result.memories
            ]
            injection_data = entries

            # Log introspection result
            self._get_tracer().log_introspection_result(
                action="ltm_retrieval",
                target_memory=f"{len(result.memories)} memories via {mode}",
                success=True,
                detail=f"Retrieved {len(result.memories)} memories, injected into STM",
            )

        return json.dumps(result.to_dict(), indent=2), side_effects, injection_data

    def _execute_validate(self, arguments: dict) -> str:
        """Execute validate_ltm_relevance."""
        from memory.ltm_introspection import validate_ltm_relevance
        from memory.ltm_introspection_types import RetrievedMemory, MemoryEntry

        memories = arguments.get("retrieved_memories", [])
        current_query = arguments.get("current_query", "")

        # Convert strings to RetrievedMemory objects
        retrieved = []
        for i, mem in enumerate(memories):
            if isinstance(mem, str):
                entry = MemoryEntry(content=mem, entry_id=f"val_{i}")
                retrieved.append(RetrievedMemory(
                    entry=entry,
                    retrieval_score=0.5,
                    source_query=current_query,
                    rank=i,
                ))
            else:
                retrieved.append(mem)

        result = validate_ltm_relevance(
            retrieved_memories=retrieved,
            current_turn_content=current_query,
            llm_client=self._llm,
            model=self._config.MEMORY_AGENT_MODEL,
        )

        # Log validation result
        self._get_tracer().log_introspection_result(
            action="validate_ltm",
            target_memory=f"{result.total_count} memories",
            success=result.coverage_sufficient,
            detail=f"{result.relevant_count}/{result.total_count} relevant, coverage={result.coverage_score:.2f}",
        )

        return json.dumps(result.to_dict(), indent=2)

    def _execute_refine(self, arguments: dict) -> str:
        """Execute refine_retrieval_target."""
        from memory.ltm_introspection import refine_retrieval_target
        from memory.ltm_introspection_types import FailureMode, RetrievalAttempt

        original_query = arguments.get("original_query", "")
        failure_mode_str = arguments.get("failure_mode", "TOO_NARROW")

        # Map string to enum
        try:
            failure_mode = FailureMode(failure_mode_str)
        except ValueError:
            failure_mode = FailureMode.TOO_NARROW

        attempt = RetrievalAttempt(
            query=original_query,
            retrieval_mode="single_query",
            results_count=0,
        )
        attempt.failure_mode = failure_mode

        result = refine_retrieval_target(
            failed_attempt=attempt,
            llm_client=self._llm,
            model=self._config.MEMORY_AGENT_MODEL,
        )

        return json.dumps(result.to_dict(), indent=2)

    def _execute_log_decision(self, arguments: dict) -> str:
        """Execute log_retrieval_decision."""
        from memory.ltm_introspection import log_retrieval_decision
        from memory.ltm_introspection_types import RetrievalDecision

        decision_chain = arguments.get("decision_chain", [])
        utility_score = arguments.get("utility_score", 0.5)
        was_skipped = arguments.get("was_retrieval_skipped", False)

        decision = RetrievalDecision(
            trigger="->".join(decision_chain),
            utility_score=utility_score,
            was_retrieved=not was_skipped,
            strategy_used=decision_chain[-1] if decision_chain else "unknown",
        )

        result = log_retrieval_decision(decision)

        # Also log to tracer for consistency
        self._get_tracer().log_memory_op(
            op_type="RETRIEVAL_DECISION_LOGGED",
            detail=f"chain={'->'.join(decision_chain)}, utility={utility_score:.2f}",
            success=True,
        )

        return json.dumps(result, indent=2)

    # --- Persistence Assurance Tool Handlers ---

    def _execute_assess_persistence_need(self, arguments: dict) -> str:
        """Execute assess_persistence_need with access to current state."""
        from memory.ltm_introspection import assess_persistence_need

        # Accept both 'user_input' and 'content' as parameter aliases
        user_input = arguments.get("user_input") or arguments.get("content", "")
        check_patterns = arguments.get("check_patterns")

        result = assess_persistence_need(
            user_input=user_input,
            recent_context=None,
            check_patterns=check_patterns,
        )

        # Log the assessment
        self._get_tracer().log_memory_op(
            op_type="PERSISTENCE_ASSESSMENT",
            detail=f"should_persist={result.should_persist}, urgency={result.urgency.value}",
            success=True,
            trigger="MAIN_AGENT",
        )

        return json.dumps(result.to_dict(), indent=2)

    def _execute_force_memory_persistence(self, arguments: dict) -> tuple[str, list[MemoryOpResult]]:
        """Execute force_memory_persistence - CRITICAL for memory integrity."""
        from memory.ltm_introspection import force_memory_persistence

        content = arguments.get("content", "")
        learning_score = arguments.get("learning_score", 0.9)
        trigger = arguments.get("trigger", "user_command")
        bypass_scoring = arguments.get("bypass_scoring", True)

        if not content:
            return json.dumps({
                "success": False,
                "error": "No content provided for persistence",
            }, indent=2), []

        # Execute with LTM store access
        result = force_memory_persistence(
            content=content,
            ltm_store=self._ltm,
            learning_score=learning_score,
            source_turn=self._stm.current_turn(),
            trigger=trigger,
            bypass_scoring=bypass_scoring,
        )

        # Log the persistence attempt
        self._get_tracer().log_memory_op(
            op_type="FORCE_PERSISTENCE",
            detail=f"content_preview={content[:100]}..., success={result.success}",
            success=result.success,
            trigger="MAIN_AGENT",
        )

        # Build side effects
        side_effects: list[MemoryOpResult] = []
        if result.success:
            from core.types import MemoryOp
            side_effects.append(MemoryOpResult(
                op=MemoryOp.ADD,
                success=True,
                trigger=TriggerKind.MAIN_AGENT,
                detail=f"Forced persistence: {content[:50]}...",
            ))

        return json.dumps(result.to_dict(), indent=2), side_effects

    def _execute_validate_memory_commit(self, arguments: dict) -> str:
        """Execute validate_memory_commit to verify persistence."""
        from memory.ltm_introspection import validate_memory_commit

        memory_id = arguments.get("memory_id")
        expected_content = arguments.get("expected_content", "")

        result = validate_memory_commit(
            memory_id=memory_id,
            expected_content=expected_content,
            ltm_store=self._ltm,
        )

        # Log validation result
        self._get_tracer().log_memory_op(
            op_type="PERSISTENCE_VALIDATION",
            detail=f"memory_id={memory_id}, validated={result.is_validated}",
            success=result.is_validated,
            trigger="MAIN_AGENT",
        )

        return json.dumps(result.to_dict(), indent=2)

    def _execute_log_persistence_failure(self, arguments: dict) -> str:
        """Execute log_persistence_failure for debugging."""
        from memory.ltm_introspection import log_persistence_failure

        content = arguments.get("content", "")
        error_message = arguments.get("error_message", "Unknown error")
        retry_count = arguments.get("retry_count", 0)
        context = arguments.get("context", {})

        # Convert error_message string to Exception for the function signature
        error = Exception(error_message)

        result = log_persistence_failure(
            content=content,
            error=error,
            retry_count=retry_count,
            context=context,
        )

        # Log failure to tracer
        self._get_tracer().log_memory_op(
            op_type="PERSISTENCE_FAILURE_LOGGED",
            detail=f"error={error_message}, retry_count={retry_count}",
            success=False,
            trigger="MAIN_AGENT",
        )

        return json.dumps(result.to_dict(), indent=2)

    # === Main Execute Method ===

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """
        Execute a tool by name with arguments.

        This is the primary method. It:
        1. Dispatches to the correct tool implementation
        2. Handles async/sync complexity internally
        3. Wraps errors in consistent [TOOL ERROR] format
        4. Logs to tracer automatically
        5. Returns ToolResult with output and metadata

        Args:
            name: Tool name (e.g., "web_search", "assess_conversation_drift")
            arguments: Tool arguments from LLM tool call

        Returns:
            ToolResult with output string and execution metadata
        """
        t0 = time.time()
        side_effects: list[MemoryOpResult] = []
        should_inject = False
        injection_data = None

        try:
            # Dispatch by tool name
            if name == "web_search":
                output = self._execute_web_search(arguments)
            elif name == "fetch_url":
                output = self._execute_fetch_url(arguments)
            elif name == "browser_navigate":
                output = self._execute_browser_navigate(arguments)
            elif name == "browser_click":
                output = self._execute_browser_click(arguments)
            elif name == "browser_scroll":
                output = self._execute_browser_scroll(arguments)
            elif name == "browser_type":
                output = self._execute_browser_type(arguments)
            elif name == "browser_press":
                output = self._execute_browser_press(arguments)
            elif name == "browser_read_page":
                output = self._execute_browser_read_page(arguments)
            elif name == "browser_screenshot":
                output = self._execute_browser_screenshot(arguments)
            elif name == "browser_close":
                output = self._execute_browser_close(arguments)
            elif name == "write_file":
                output = self._execute_write_file(arguments)
            elif name == "ingest_document":
                output = self._execute_ingest_document(arguments)
            elif name == "list_documents":
                output = self._execute_list_documents(arguments)
            elif name == "search_metadata":
                output = self._execute_search_metadata(arguments)
            elif name == "grep_corpus":
                output = self._execute_grep_corpus(arguments)
            elif name == "read_document":
                output = self._execute_read_document(arguments)
            elif name == "read_lines":
                output = self._execute_read_lines(arguments)
            elif name == "assess_conversation_drift":
                output = self._execute_assess_drift(arguments)
            elif name == "are_you_ready_to_get_in_context_ltm":
                output = self._execute_readiness_check(arguments)
            elif name == "paraphrase_for_coverage":
                output = self._execute_paraphrase(arguments)
            elif name == "trigger_contextual_ltm_retrieval":
                output, side_effects, injection_data = self._execute_trigger_retrieval(arguments)
                should_inject = True
            elif name == "validate_ltm_relevance":
                output = self._execute_validate(arguments)
            elif name == "refine_retrieval_target":
                output = self._execute_refine(arguments)
            elif name == "log_retrieval_decision":
                output = self._execute_log_decision(arguments)
            elif name == "assess_persistence_need":
                output = self._execute_assess_persistence_need(arguments)
            elif name == "force_memory_persistence":
                output, side_effects = self._execute_force_memory_persistence(arguments)
            elif name == "validate_memory_commit":
                output = self._execute_validate_memory_commit(arguments)
            elif name == "log_persistence_failure":
                output = self._execute_log_persistence_failure(arguments)
            else:
                output = f"[TOOL ERROR] Unknown tool: {name}"

        except Exception as e:
            output = f"[TOOL ERROR] {name} failed: {e}"

        # Build result
        result = ToolResult(
            output=output,
            tool_name=name,
            duration_ms=(time.time() - t0) * 1000,
            side_effects=side_effects,
            should_inject_to_stm=should_inject,
            stm_injection_data=injection_data,
        )

        # Tracing (lazy get_tracer if not injected)
        tracer = self._get_tracer()
        tracer.log_tool_call(
            tool_name=name,
            arguments=arguments,
            duration_ms=result.duration_ms,
            result=output[:500] if output else None,
            success=result.success,
        )

        return result

    def get_definitions(self, category: Optional[ToolCategory] = None) -> list[dict]:
        """
        Get OpenAI tool definitions for LLM.

        Args:
            category: Filter to specific category (None = all)

        Returns:
            List of tool definition dicts for OpenAI API
        """
        # Tool definitions will be moved here from Orchestrator
        # For now, return empty list - Orchestrator still manages this
        return []
