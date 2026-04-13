"""
agents/orchestrator.py
───────────────────────
The central coordinator of the AgeMem-hybrid system.

Turn lifecycle
──────────────
  1. Pre-turn:
     a. STM.force_fit()             — guarantee no overflow before LLM call
     b. LTM.search(user_query)      — retrieve relevant memories into STM

  2. Main turn:
     a. Build message list (system prompt + STM + retrieved memories)
     b. Call main LLM → get assistant response
     c. Append response to STM

  3. Post-turn:
     a. LearningScorer.collect()    — get agent self-rating (every N turns)
     b. SystemRules.evaluate()      — check deterministic triggers
     c. Execute triggered ops       — SUMMARY / FILTER if needed
     d. MemoryAgent.review()        — if periodic review or learning spike
     e. Apply MemoryAgent decisions — ADD / UPDATE / DELETE on LTM, score STM
     f. Increment turn counter

The Orchestrator is the *only* place that writes to LTM or STM.  All other
components are pure analysis / computation.

Acceptance criteria mapping
────────────────────────────
  AC-1 (no context explosion)  → step 1a + SystemRules R1/R2 + STMContext.force_fit
  AC-2 (learning score drives memory) → LearningScorer + SystemRules R4 +
                                         MemoryAgent.review on spike/periodic
"""

from __future__ import annotations

import re
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from core.types import (
    ContextMessage,
    ContextStats,
    LearningFeedback,
    MemoryOp,
    MemoryOpResult,
    TriggerKind,
    Skill,
)
from core.config import AgememConfig, DEFAULT_CONFIG
from memory.ltm_store import LTMStore
from memory.stm_context import STMContext
from memory.context_retrieval import (
    ContextAwareRetriever,
    ContextRetrievalConfig,
)
from memory.ltm_introspection import assess_persistence_need, PersistenceUrgency

from triggers.system_rules import SystemRules, RuleID
from triggers.memory_trigger_engine import MemoryTriggerEngine
from agents.llm_client import LLMClient, ToolCallResponse, TextToolCallResponse
from agents.memory_agent import MemoryAgent
from agents.learning_scorer import LearningScorer
from agents.response_handler import ResponseHandler
from agents.tool_executor import ToolExecutor, ToolResult, ToolCategory
from skills.manager import SkillManager
from tools.query_expansion import QueryExpander

import logging

# ── Tracing System ──────────────────────────────────────────────────────────
from core.tracing import get_tracer

# ── Tool Call Tracking (LoopGuard) ───────────────────────────────────────────

@dataclass
class ToolCall:
    """Represents a tool call with name and arguments."""
    name: str
    arguments: dict
    
    def key(self) -> str:
        """Unique key for deduplication."""
        args_str = str(sorted(self.arguments.items()))
        return f"{self.name}:{args_str}"


class ToolCallTracker:
    """Per-turn tracker to prevent duplicate tool calls (LoopGuard pattern)."""
    
    def __init__(self):
        self._calls: set[str] = set()
    
    def record(self, call: ToolCall) -> bool:
        """
        Record a tool call. Returns True if this is a duplicate.
        """
        key = call.key()
        if key in self._calls:
            return True
        self._calls.add(key)
        return False
    
    def reset(self):
        """Reset the tracker for a new turn."""
        self._calls.clear()


@dataclass
class ToolCallTrace:
    """Trace of a single tool call."""
    name: str
    arguments: dict
    result: str
    duration_ms: float
    success: bool


@dataclass
class TurnTrace:
    """
    Full audit record for one conversation turn.
    Useful for debugging and offline analysis.
    """
    turn_index: int
    user_input: str
    assistant_response: str
    stm_stats_before: ContextStats
    stm_stats_after: ContextStats
    ops_applied: list[MemoryOpResult] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    feedback: Optional[LearningFeedback] = None
    memory_agent_rationale: str = ""
    latency_ms: float = 0.0
    prompt_versions: dict[str, str] = field(default_factory=dict) # Map of prompt_id -> version used during this turn for audit trail.


class Orchestrator:

    def __init__(
        self,
        llm: LLMClient,
        config: AgememConfig = DEFAULT_CONFIG,
        ltm_store: Optional[LTMStore] = None,
        stm_context: Optional[STMContext] = None,
        learning_scorer_llm: Optional[LLMClient] = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._learning_scorer_llm = learning_scorer_llm or llm  # Fallback to main LLM if not provided

        # Initialize prompt registry
        self._prompt_versions: dict[str, str] = {}
        self._init_prompt_registry()

        # Resolve persistence paths
        self._persist_dir: Optional[Path] = None
        if config.PERSIST_DIR:
            self._persist_dir = Path(config.PERSIST_DIR)
            self._persist_dir.mkdir(parents=True, exist_ok=True)

        ltm_path = (
            self._persist_dir / config.LTM_PERSIST_FILENAME
            if self._persist_dir else None
        )
        stm_path = (
            self._persist_dir / config.STM_PERSIST_FILENAME
            if self._persist_dir else None
        )

        # SEMANTIC_SEARCH: Configure semantic search for LTM if enabled
        semantic_db_path = None
        if config.ENABLE_SEMANTIC_SEARCH and self._persist_dir:
            semantic_db_path = self._persist_dir / config.SEMANTIC_DB_FILENAME

        # LTM — pass persist_path so _maybe_persist() is active
        # SEMANTIC_SEARCH: Pass semantic_db_path if semantic search is enabled
        # QUERY_EXPANSION: Pass llm_client for query expansion
        self._ltm = ltm_store or LTMStore(
            config,
            persist_path=ltm_path,
            semantic_db_path=semantic_db_path,
            enable_semantic_search=config.ENABLE_SEMANTIC_SEARCH,
            llm_client=self._llm,
        )

        # CONTEXT_AWARE_RETRIEVAL: Initialize context-aware retriever if enabled
        self._context_retriever: Optional[ContextAwareRetriever] = None
        if config.CONTEXT_AWARE_RETRIEVAL:
            ctx_config = ContextRetrievalConfig.from_agemem_config(config)
            self._context_retriever = ContextAwareRetriever(self._ltm, ctx_config)

        # Memory agent and scorer - use dedicated learning scorer LLM if available
        self._memory_agent = MemoryAgent(self._learning_scorer_llm, config)
        self._scorer = LearningScorer(self._learning_scorer_llm, config)
        self._rules = SystemRules(config)

        # Response handler for enhanced error recovery
        self._response_handler = ResponseHandler(llm, max_retries=2, enable_validation=True)

        # STM — restore from disk if available
        self._stm = stm_context or STMContext(
            config=config,
            summary_fn=self._memory_agent.summarise_context,
        )
        if stm_context and stm_context._summary_fn is None:
            stm_context._summary_fn = self._memory_agent.summarise_context

        if stm_path:
            self._stm.load(stm_path)
            # Revalidate context size against current config
            # (config may have changed between sessions)
            self._stm.force_fit()

        # MemoryTriggerEngine — unified entry point for memory triggers
        # Created after stm and ltm are initialized
        # Use learning scorer LLM for memory operations
        self._trigger_engine = MemoryTriggerEngine(
            config=config,
            llm=self._learning_scorer_llm,
            stm=self._stm,
            ltm=self._ltm,
        )

        # Ensure pinned system prompt is up-to-date with registry
        # Always update on startup to pick up prompt changes
        has_system = any(
            m.role == "system" and m.is_pinned
            for m in self._stm.messages()
        )
        current_prompt = config.SYSTEM_PROMPT_HEADER
        if has_system:
            # Update existing pinned system message with fresh content
            self._stm.update_pinned_system_message(current_prompt)
        else:
            # Add new pinned system message
            self._stm.add_message(
                role="system",
                content=current_prompt,
                is_pinned=True,
            )

        self._stm_persist_path = stm_path
        self._traces: list[TurnTrace] = []
        
        # Tool support
        self._tools: list[dict] = []
        self._tool_tracker = ToolCallTracker()

        # Skill manager for dynamic capability hints
        self._skill_manager = SkillManager(config)
        self._skill_manager.load_skills()

        # Query expander for corpus search - use learning scorer LLM
        # Note: Use the learning scorer's model name, not MEMORY_AGENT_MODEL,
        # because the learning scorer LLM uses OpenRouter with a different model
        self._query_expander: Optional[QueryExpander] = None
        if getattr(self._config, 'ENABLE_QUERY_EXPANSION', False):
            self._query_expander = QueryExpander(
                llm_client=self._learning_scorer_llm,
                model=self._learning_scorer_llm._model,  # Use the client's model, not config
                n_variants=getattr(self._config, 'QUERY_EXPANSION_N_VARIANTS', 3),
                use_ner_hints=getattr(self._config, 'QUERY_EXPANSION_USE_NER_HINTS', False),
                timeout_ms=getattr(self._config, 'QUERY_EXPANSION_TIMEOUT_MS', 1500),
                fallback_transforms=getattr(self._config, 'QUERY_EXPANSION_FALLBACK_TRANSFORMS', ["nominalize", "add_how_to"]),
                acronym_dict=getattr(self._config, 'QUERY_EXPANSION_ACRONYM_DICT', {}),
            )

        # Derive per-user corpus path from persist_dir.
        # Convention: persist_dir=agent_memory/users/X → corpus_path=corpus/users/X
        self._corpus_path = None
        if self._persist_dir and "users" in self._persist_dir.parts:
            idx = self._persist_dir.parts.index("users")
            user_parts = self._persist_dir.parts[idx:]  # e.g. ("users", "alice")
            self._corpus_path = Path("corpus").joinpath(*user_parts)

        # Tool executor - encapsulates all tool execution logic
        self._tool_executor = ToolExecutor(
            stm=self._stm,
            ltm=self._ltm,
            llm=self._llm,
            config=self._config,
            tracer=None,  # Uses get_tracer() lazily
            corpus_path=self._corpus_path,
        )

    def _init_prompt_registry(self) -> None:
        """Initialize prompt registry and capture current prompt versions for audit."""
        try:
            from prompts import get_active_version, list_prompts
            # Capture active versions for audit trail
            for meta in list_prompts():
                version = get_active_version(meta.prompt_id)
                self._prompt_versions[meta.prompt_id] = version
        except Exception as e:
            # Registry is optional - don't fail if prompts aren't configured
            print(f"[Orchestrator] Prompt registry not initialized: {e}")

    def reload_prompts(self) -> dict[str, str]:
        """
        Reload prompts from the registry and return current versions.

        Also updates the pinned system message in STM with the new prompt content.

        Returns:
            Dictionary mapping prompt_id -> active_version
        """
        # Actually reload prompts from disk to pick up changes
        try:
            from prompts import reload as reload_prompts_registry
            reload_prompts_registry()
        except Exception as e:
            print(f"[Orchestrator] Failed to reload prompts: {e}")

        self._init_prompt_registry()

        # Update the pinned system message in STM with the new prompt
        try:
            new_system_prompt = self._config.SYSTEM_PROMPT_HEADER
            updated = self._stm.update_pinned_system_message(new_system_prompt)
            if updated:
                print(f"[Orchestrator] Updated system prompt to version: {self._prompt_versions.get('main-system', 'unknown')}")
        except Exception as e:
            print(f"[Orchestrator] Failed to update STM system message: {e}")

        return dict(self._prompt_versions)

    def get_prompt_versions(self) -> dict[str, str]:
        """
        Get the prompt versions used in this session.

        Returns:
            Dictionary mapping prompt_id -> active_version
        """
        return dict(self._prompt_versions)

    # ── Tool Configuration ────────────────────────────────────────────────────

    def set_tools(self, tools: list[dict]) -> None:
        """Set the available tools for the LLM."""
        self._tools = tools

    def get_available_tools(self) -> list[dict]:
        """Return the list of available tools."""
        return self._tools

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """
        Execute a tool by name - delegates to ToolExecutor.

        Currently supported:
        - web_search: Search the web for current information
        - fetch_url: Fetch content from a URL (HTML, JSON, PDF, etc.)
        - write_file: Write content to a file
        - ingest_document: Ingest a markdown file into the corpus
        - list_documents: List all ingested documents
        - search_metadata: Search document metadata for a keyword
        - grep_corpus: Full-text search across all documents
        - read_document: Read full document content
        - read_lines: Read specific lines from a document

        Returns the tool result as a string.
        """
        result = self._tool_executor.execute(name, arguments)

        # Handle STM injection for retrieval tools
        if result.should_inject_to_stm and result.stm_injection_data:
            self._stm.retrieve(result.stm_injection_data, trigger=TriggerKind.MAIN_AGENT)

        return result.output

    # === Introspection Tool Execution Handlers ===

    # ── Main public API ───────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """
        Process one user turn end-to-end.
        Returns the assistant's response text.
        """
        t0 = time.time()
        turn = self._stm.current_turn()
        stats_before = self._stm.stats()

        # Get tracer instance
        tracer = get_tracer()

        # Reset tool tracker for this turn
        self._tool_tracker.reset()

        ops: list[MemoryOpResult] = []

        # ── 1a. Prevent overflow before any LLM call ──────────────────────────
        overflow_ops = self._stm.force_fit()
        ops.extend(overflow_ops)

        # Log overflow operations
        for op in overflow_ops:
            tracer.log_memory_op(
                op_type="STM_OVERFLOW_GUARD",
                detail=op.detail or "",
                success=op.success,
                trigger=op.trigger.value if hasattr(op.trigger, 'value') else str(op.trigger),
            )

        # ── 1b. Retrieve relevant LTM entries into STM ───────────────────────
         # ── 1b. Check for explicit memory commands and persist immediately ────
        persistence_need = assess_persistence_need(user_input)
        if persistence_need.urgency == PersistenceUrgency.IMMEDIATE:
            from memory.ltm_introspection import force_memory_persistence, validate_memory_commit
            persist_result = force_memory_persistence(
                content=persistence_need.suggested_content or user_input,
                ltm_store=self._ltm,
                learning_score=0.9,
                source_turn=self._stm.current_turn(),
                trigger="explicit_command",
            )
            if persist_result.success:
                # Validate the commit
                validation = validate_memory_commit(
                    memory_id=persist_result.memory_id,
                    expected_content=persistence_need.suggested_content,
                    ltm_store=self._ltm,
                )
                # Notify agent via system message
                self._stm.add_message(
                    role="system",
                    content=f"[MEMORY STORED] {persistence_need.suggested_content[:200]}",
                    relevance_score=1.0,
                    is_pinned=False,
                )
                tracer.log_memory_op(
                    op_type="EXPLICIT_PERSISTENCE",
                    detail=f"User command persisted: {persistence_need.suggested_content[:100]}...",
                    success=True,
                    trigger="USER_COMMAND",
                )
            else:
                tracer.log_memory_op(
                    op_type="EXPLICIT_PERSISTENCE",
                    detail=f"Failed to persist: {persistence_need.suggested_content[:100]}...",
                    success=False,
                    trigger="USER_COMMAND",
                )
        # CONTEXT_AWARE_RETRIEVAL: Use context-aware retrieval if enabled
        if self._context_retriever is not None:
            relevant = self._context_retriever.retrieve(
                current_query=user_input,
                recent_messages=self._stm.messages(),
                current_turn=self._stm.current_turn(),
                top_k=5,
            )
        else:
            relevant = self._ltm.search(user_input, top_k=5)
        if relevant:
            retrieve_op = self._stm.retrieve(relevant, trigger=TriggerKind.SYSTEM_RULE)
            ops.append(retrieve_op)
            tracer.log_memory_op(
                op_type="LTM_RETRIEVE",
                detail=f"Retrieved {len(relevant)} entries",
                success=True,
                trigger="SYSTEM_RULE",
            )
        else:
            # LTM empty — search corpus and inject directly into STM (no duplication)
            corpus_context = self._search_corpus_for_context(user_input)
            if corpus_context:
                self._stm.add_message(
                    role="system",
                    content=f"[CORPUS CONTEXT]\n{corpus_context}",
                    relevance_score=0.9,
                    is_pinned=False,
                )
                tracer.log_memory_op(
                    op_type="CORPUS_INJECT",
                    detail=f"Injected corpus context ({len(corpus_context)} chars)",
                    success=True,
                )

        # ── 1c. Detect and inject relevant skills ──────────────────────────────
        skills = self._skill_manager.detect_skills(user_input)
        for skill in skills:
            self._stm.add_message(
                role="system",
                content=f"[SKILL HINT: {skill.name}] {skill.hint_message} - {skill.source_doc_id} - {skill.description}",
                relevance_score=self._config.SKILL_DEFAULT_RELEVANCE,
                is_pinned=False,
            )

        # ── 2. Main LLM call with tool support ────────────────────────────────
        self._stm.add_message(role="user", content=user_input, relevance_score=1.0)

        # Tool call loop - protected by LoopGuard and max iteration cap
        assistant_text = ""
        max_tool_iterations = 30  # Prevent infinite tool call loops
        tool_iterations = 0
        turn_tool_calls: list[ToolCallTrace] = []  # Track tool calls for this turn

        while tool_iterations < max_tool_iterations:
            messages = self._stm.openai_messages()
            tool_iterations += 1

            # Log LLM call
            llm_call_id = tracer.log_llm_call(
                messages=messages,
                model=self._config.DEFAULT_MODEL,
                max_tokens=self._config.DEFAULT_MAX_TOKENS,
                temperature=self._config.DEFAULT_TEMPERATURE,
                has_tools=bool(self._tools),
            )
            llm_call_start = time.time()

            try:
                # Call LLM with tools if available using enhanced response handler
                assistant_text, metrics = self._response_handler.chat_with_recovery(
                    messages=messages,
                    max_tokens=self._config.DEFAULT_MAX_TOKENS,
                    temperature=self._config.DEFAULT_TEMPERATURE,
                    tools=self._tools if self._tools else None,
                )

                # Log successful LLM response
                tracer.log_llm_response(
                    call_id=llm_call_id,
                    response=assistant_text,
                    latency_ms=(time.time() - llm_call_start) * 1000,
                    token_count=metrics.token_count,
                )

                # If we get here, no tool call was made - we have a text response
                # Log response quality metrics
                if metrics.quality_score < 0.8:
                    print(f"[DEBUG] Low quality response detected: score={metrics.quality_score:.2f}, type={metrics.response_type.value}", flush=True)
                break

            except RuntimeError as e:
                # LLM call failed after retries - provide fallback response
                tracer.log_llm_response(
                    call_id=llm_call_id,
                    response="[ERROR] LLM call failed",
                    latency_ms=(time.time() - llm_call_start) * 1000,
                    token_count=0,
                )
                # Check if this looks like a tool loop scenario
                if "LLM call failed" in str(e) and tool_iterations > 5:
                    assistant_text = "[SYSTEM] Maximum tool call iterations reached. Providing final response based on available information."
                else:
                    assistant_text = f"[SYSTEM] LLM call failed after retries: {str(e)[:100]}. Providing response based on available context."
                break

            except ToolCallResponse as e:
                # Log tool call
                tool_call_start = time.time()

                # Check iteration limit before processing tool call
                if tool_iterations >= max_tool_iterations:
                    assistant_text = "[SYSTEM] Maximum tool call iterations reached. Providing final response based on available information."
                    break

                # Parse tool call
                tool_call = e.tool_call
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id  # Unique ID for this tool call
                try:
                    # Some servers return arguments as dict, others as JSON string
                    if isinstance(tool_call.function.arguments, dict):
                        tool_args = tool_call.function.arguments
                    else:
                        tool_args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}

                # Check for duplicate calls (LoopGuard)
                call = ToolCall(name=tool_name, arguments=tool_args)
                if self._tool_tracker.record(call):
                    # Duplicate detected - inject user message and continue
                    self._stm.add_message(
                        role="user",
                        content="[SYSTEM] Duplicate tool call detected. You already called this tool with the same arguments. Please try a different approach or provide a final answer.",
                        relevance_score=0.0,
                    )
                    continue

                # Execute via ToolExecutor
                result = self._tool_executor.execute(tool_name, tool_args)
                tool_duration = (time.time() - tool_call_start) * 1000

                # Track tool call for TurnTrace
                turn_tool_calls.append(ToolCallTrace(
                    name=tool_name,
                    arguments=tool_args,
                    result=result.output[:500] if result.output else "",
                    duration_ms=result.duration_ms,
                    success=result.success,
                ))

                # Handle STM injection for retrieval
                if result.should_inject_to_stm and result.stm_injection_data:
                    self._stm.retrieve(result.stm_injection_data, trigger=TriggerKind.MAIN_AGENT)

                # Collect side effects
                ops.extend(result.side_effects)

                # Add assistant message with proper tool call structure
                tool_calls_data = [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
                    }
                }]
                self._stm.add_message(
                    role="assistant",
                    content=None,  # No content when making tool calls
                    relevance_score=0.2,
                    tool_calls=tool_calls_data,
                )
                # Add the tool result with proper role and tool_call_id
                self._stm.add_message(
                    role="tool",
                    content=result.output,
                    relevance_score=0.9,
                    tool_call_id=tool_call_id,
                )

                # Loop back to LLM call with the tool result
                continue

            except TextToolCallResponse as e:
                # Handle text-based tool calls (for models that don't use API tool calling)
                if tool_iterations >= max_tool_iterations:
                    assistant_text = "[SYSTEM] Maximum tool call iterations reached. Providing final response based on available information."
                    break

                tool_call = e.tool_call
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id
                tool_args = tool_call.function.arguments

                # Validate tool call arguments
                validation = self._response_handler.validate_tool_call(tool_call)
                if not validation.is_valid:
                    self._stm.add_message(
                        role="user",
                        content=f"[SYSTEM] Tool call validation failed: {'; '.join(validation.errors)}. Please try again with valid arguments.",
                        relevance_score=0.0,
                    )
                    continue

                # Check for duplicate calls (LoopGuard)
                call = ToolCall(name=tool_name, arguments=tool_args)
                if self._tool_tracker.record(call):
                    self._stm.add_message(
                        role="user",
                        content="[SYSTEM] Duplicate tool call detected. You already called this tool with the same arguments. Please try a different approach or provide a final answer.",
                        relevance_score=0.0,
                    )
                    continue

                # Execute via ToolExecutor
                result = self._tool_executor.execute(tool_name, tool_args)

                # Track tool call for TurnTrace
                turn_tool_calls.append(ToolCallTrace(
                    name=tool_name,
                    arguments=tool_args,
                    result=result.output[:500] if result.output else "",
                    duration_ms=result.duration_ms,
                    success=result.success,
                ))

                # Handle STM injection for retrieval
                if result.should_inject_to_stm and result.stm_injection_data:
                    self._stm.retrieve(result.stm_injection_data, trigger=TriggerKind.MAIN_AGENT)

                # Collect side effects
                ops.extend(result.side_effects)

                tool_calls_data = [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
                    }
                }]
                self._stm.add_message(
                    role="assistant",
                    content=None,
                    relevance_score=0.2,
                    tool_calls=tool_calls_data,
                )
                self._stm.add_message(
                    role="tool",
                    content=result.output,
                    relevance_score=0.9,
                    tool_call_id=tool_call_id,
                )
                continue

        self._stm.add_message(role="assistant", content=assistant_text, relevance_score=0.8)

        # Post-response overflow guard: the assistant reply may have pushed us over
        post_ops = self._stm.force_fit()
        ops.extend(post_ops)

        # Log post-response overflow
        for op in post_ops:
            tracer.log_memory_op(
                op_type="POST_OVERFLOW_GUARD",
                detail=op.detail or "",
                success=op.success,
            )

        self._stm.increment_turn()
        turn_after = self._stm.current_turn()

        # ── 3a. Collect learning feedback ─────────────────────────────────────
        feedback: Optional[LearningFeedback] = None
        if self._scorer.should_collect(turn_after):
            feedback = self._scorer.collect(
                context_messages=self._stm.openai_messages(),
                turn_index=turn_after,
            )
            if feedback:
                feedback.turn_index = turn_after

        # ── 3b–e. Memory Trigger Engine ───────────────────────────────────────
        # Unified entry point for all memory trigger logic (RFC-001)
        report = self._trigger_engine.process_turn(
            turn_index=turn_after,
            feedback=feedback,
            assistant_response=assistant_text,
        )
        ops.extend(report.operations)
        ma_rationale = report.agent_rationale or ""

        stats_after = report.stm_stats or self._stm.stats()

        # ── Persist STM after every turn ──────────────────────────────────────
        # LTM already persists inside LTMStore._maybe_persist() on every write
        self._persist_stm()

        # ── Trace ─────────────────────────────────────────────────────────────
        turn_latency = (time.time() - t0) * 1000
        self._traces.append(TurnTrace(
            turn_index=turn_after,
            user_input=user_input,
            assistant_response=assistant_text,
            stm_stats_before=stats_before,
            stm_stats_after=stats_after,
            ops_applied=ops,
            tool_calls=turn_tool_calls,
            feedback=feedback,
            memory_agent_rationale=ma_rationale,
            latency_ms=turn_latency,
            prompt_versions=dict(self._prompt_versions),  # Copy for audit trail
        ))

        # ── Seed logging ─────────────────────────────────────────────────────
        # Write complete turn as a seed entry for fine-tuning dataset generation.
        # Only logs turns with tool calls (direct responses have no tool training signal).
        if turn_tool_calls:
            self._write_seed(
                user_input=user_input,
                tool_calls=turn_tool_calls,
                final_response=assistant_text,
                turn_index=turn_after,
                latency_ms=turn_latency,
            )

        return assistant_text

    # ── Seed logging helper ──────────────────────────────────────────────────

    def _write_seed(
        self,
        user_input: str,
        tool_calls: list[ToolCallTrace],
        final_response: str,
        turn_index: int,
        latency_ms: float,
    ) -> None:
        """Write a seed entry to the seed log for fine-tuning dataset generation."""
        # Get system prompt from STM's pinned message
        system_prompt = ""
        for msg in self._stm.messages():
            if msg.role == "system" and msg.is_pinned:
                system_prompt = msg.content
                break

        # Convert ToolCallTrace to plain dicts for the tracer
        tc_dicts = [
            {
                "name": tc.name,
                "arguments": tc.arguments,
                "result": tc.result,
                "duration_ms": tc.duration_ms,
                "success": tc.success,
            }
            for tc in tool_calls
        ]

        tracer = get_tracer()
        tracer.log_seed(
            system_prompt=system_prompt,
            user_message=user_input,
            tool_calls=tc_dicts,
            final_response=final_response,
            model=self._config.DEFAULT_MODEL,
            turn_index=turn_index,
            latency_ms=latency_ms,
        )

    # ── Inspection helpers ────────────────────────────────────────────────────

    def last_trace(self) -> Optional[TurnTrace]:
        return self._traces[-1] if self._traces else None

    def all_traces(self) -> list[TurnTrace]:
        return list(self._traces)

    def ltm_snapshot(self) -> list[dict]:
        return [e.to_dict() for e in self._ltm.all_entries()]

    def stm_stats(self) -> ContextStats:
        return self._stm.stats()

    def reset_stm(self) -> None:
        """Reset STM context, clearing all messages except the pinned system prompt."""
        # Find and keep the pinned system message
        pinned_system = None
        for msg in self._stm.messages():
            if msg.role == "system" and msg.is_pinned:
                pinned_system = msg
                break
        
        # Clear all messages
        self._stm._messages.clear()
        self._stm._turn_index = 0
        
        # Re-add the pinned system prompt if it existed
        if pinned_system:
            self._stm.add_message(
                role="system",
                content=pinned_system.content,
                is_pinned=True,
            )
        
        # Persist the cleared state
        self._persist_stm()

    def clear_ltm(self) -> None:
        """Clear all LTM entries from memory and disk."""
        self._ltm._entries.clear()
        self._ltm._maybe_persist()

    def _search_corpus_for_context(self, query: str) -> Optional[str]:
        """
        Search corpus files and return matching content for STM injection.

        This method searches the corpus without duplicating data into LTM.
        Results are injected directly into STM as ephemeral context.
        Uses query expansion when ENABLE_QUERY_EXPANSION=True for better recall
        on paraphrase queries.

        Args:
            query: The user query to search for

        Returns:
            Formatted corpus content or None if no matches
        """
        from tools.corpus import grep_corpus, read_document

        tracer = get_tracer()
        logger = logging.getLogger("agemem")

        # Track timing for query expansion
        expansion_start = time.time()

        # Step 1: Generate query variants
        if self._query_expander is not None:
            queries = self._query_expander.expand(query)
            method = "llm" if self._config.ENABLE_QUERY_EXPANSION else "regex"
            tracer.log_memory_op(
                op_type="QUERY_EXPANSION_INIT",
                detail=f"Query expansion enabled with {len(queries)} variants",
                success=True,
            )
        else:
            queries = [query]
            method = "disabled"
            tracer.log_memory_op(
                op_type="QUERY_EXPANSION_INIT",
                detail="Query expansion disabled",
                success=True,
            )

        # Step 2: Run grep_corpus for each variant, collect all result lines
        all_lines: list[str] = []
        hits_per_variant: list[int] = []

        for q in queries:
            variant_hits = 0
            try:
                result = grep_corpus(q, context_lines=3, corpus_path=self._corpus_path)
            except Exception as e:
                logger.debug(f"[query_expansion] grep_corpus failed for '{q[:40]}...': {e}")
                hits_per_variant.append(0)
                continue

            if result and "No matches found" not in result:
                lines = [line for line in result.split('\n') if line.strip()]
                variant_hits = len(lines)
                all_lines.extend(lines)

            hits_per_variant.append(variant_hits)

        # Step 3: Deduplicate lines by exact string, preserving first-occurrence order
        seen_lines: set[str] = set()
        deduplicated_lines: list[str] = []
        for line in all_lines:
            if line not in seen_lines:
                seen_lines.add(line)
                deduplicated_lines.append(line)

        dedup_count = len(all_lines) - len(deduplicated_lines)

        # Step 4: Extract doc_ids from deduplicated lines
        doc_ids: set[str] = set()
        for line in deduplicated_lines:
            if ': ' in line and not line.startswith('[TOOL'):
                # Extract doc_id from path like "corpus/test1_0922ba.md:..."
                match = re.search(r'corpus/([^.]+)\.md:', line)
                if match:
                    doc_ids.add(match.group(1))

        expansion_duration_ms = (time.time() - expansion_start) * 1000

        # Log query expansion results
        tracer.log_query_expansion(
            original_query=query,
            variants=queries,
            hit_counts=hits_per_variant,
            duration_ms=expansion_duration_ms,
            method=method,
            success=True,
        )

        # Log deduplication stats
        if dedup_count > 0:
            tracer.log_memory_op(
                op_type="QUERY_EXPANSION_DEDUP",
                detail=f"Deduplicated {dedup_count} duplicate lines",
                success=True,
            )

        if not doc_ids:
            tracer.log_memory_op(
                op_type="QUERY_EXPANSION_NO_HITS",
                detail=f"No documents found for query variants",
                success=True,
            )
            return None

        # Log which variants produced hits
        variants_with_hits = sum(1 for h in hits_per_variant if h > 0)
        logger.debug(
            f"[query_expansion] query='{query[:60]}...' method={method} "
            f"variants={len(queries)} hits={variants_with_hits}/{len(queries)} "
            f"docs={len(doc_ids)} dedup_removed={dedup_count}"
        )

        # Step 5: Build context from matching documents (cap at 3)
        context_parts = []
        docs_used = 0
        for doc_id in list(doc_ids)[:3]:
            try:
                content = read_document(doc_id, corpus_path=self._corpus_path)
                if not content or content.startswith("Error:"):
                    continue

                # Strip YAML frontmatter
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        content_body = parts[2].strip()
                    else:
                        content_body = content
                else:
                    content_body = content

                # Truncate if too long
                if len(content_body) > 3000:
                    content_body = content_body[:3000] + "\n... [truncated]"

                context_parts.append(f"--- Document: {doc_id} ---\n{content_body}")
                docs_used += 1

            except Exception as e:
                logger.debug(f"[query_expansion] Failed to read document {doc_id}: {e}")
                continue

        if not context_parts:
            tracer.log_memory_op(
                op_type="QUERY_EXPANSION_EMPTY_CONTEXT",
                detail="No valid context extracted from documents",
                success=False,
            )
            return None

        # Log final context built
        total_chars = sum(len(p) for p in context_parts)
        tracer.log_memory_op(
            op_type="QUERY_EXPANSION_CONTEXT_BUILT",
            detail=f"Built context from {docs_used} docs, {total_chars} chars",
            success=True,
        )

        return "\n\n".join(context_parts)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _persist_stm(self) -> None:
        """Persist STM to disk if persistence is enabled."""
        if self._stm_persist_path:
            self._stm.save(self._stm_persist_path)
