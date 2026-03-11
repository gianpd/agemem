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

import json
import time
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
from triggers.system_rules import SystemRules, RuleID
from agents.llm_client import LLMClient, ToolCallResponse
from agents.memory_agent import MemoryAgent
from agents.learning_scorer import LearningScorer
from skills.manager import SkillManager


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
    ) -> None:
        self._config = config
        self._llm = llm

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
        self._ltm = ltm_store or LTMStore(
            config,
            persist_path=ltm_path,
            semantic_db_path=semantic_db_path,
            enable_semantic_search=config.ENABLE_SEMANTIC_SEARCH,
        )

        # Memory agent and scorer
        self._memory_agent = MemoryAgent(llm, config)
        self._scorer = LearningScorer(llm, config)
        self._rules = SystemRules(config)

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

        # Only add the pinned system prompt if STM was empty after load
        # (avoids duplicating it on every restart)
        has_system = any(
            m.role == "system" and m.is_pinned
            for m in self._stm.messages()
        )
        if not has_system:
            self._stm.add_message(
                role="system",
                content=config.SYSTEM_PROMPT_HEADER,
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

        Returns:
            Dictionary mapping prompt_id -> active_version
        """
        self._init_prompt_registry()
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
        Execute a tool by name.

        Currently supported:
        - web_search: Search the web for current information
        - write_file: Write content to a file
        - ingest_document: Ingest a markdown file into the corpus
        - list_documents: List all ingested documents
        - search_metadata: Search document metadata for a keyword
        - grep_corpus: Full-text search across all documents
        - read_document: Read full document content
        - read_lines: Read specific lines from a document
        - commodity_price: Fetch current commodity prices (gold, silver, oil, etc.)
        - commodity_history: Fetch historical commodity price data

        Returns the tool result as a string.
        """
        if name == "web_search":
            # Import here to avoid circular dependencies
            try:
                from tools.web_tools import web_search
                query = arguments.get("query", "")
                num_results = arguments.get("num_results", 5)
                # Run async function synchronously using asyncio.run()
                import asyncio
                try:
                    result = asyncio.run(web_search(query, num_results))
                except RuntimeError:
                    # If already in an event loop, use nest_asyncio or get existing loop
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        result = asyncio.run(web_search(query, num_results))
                    except ImportError:
                        # Fallback: try to get the running loop
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Create a new loop in a thread
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, web_search(query, num_results))
                                result = future.result()
                        else:
                            result = loop.run_until_complete(web_search(query, num_results))
                return result
            except Exception as e:
                return f"[TOOL ERROR] web_search failed: {e}"

        if name == "write_file":
            try:
                from tools.web_tools import write_file
                path = arguments.get("path", "")
                content = arguments.get("content", "")
                return write_file(path, content)
            except Exception as e:
                return f"[TOOL ERROR] write_file failed: {e}"

        if name == "ingest_document":
            try:
                from tools.web_tools import ingest_document
                path = arguments.get("path", "")
                return ingest_document(path)
            except Exception as e:
                return f"[TOOL ERROR] ingest_document failed: {e}"

        # Corpus tools
        if name == "list_documents":
            try:
                from tools.corpus import list_documents
                return list_documents()
            except Exception as e:
                return f"[TOOL ERROR] list_documents failed: {e}"

        if name == "search_metadata":
            try:
                from tools.corpus import search_metadata
                keyword = arguments.get("keyword", "")
                return search_metadata(keyword)
            except Exception as e:
                return f"[TOOL ERROR] search_metadata failed: {e}"

        if name == "grep_corpus":
            try:
                from tools.corpus import grep_corpus
                pattern = arguments.get("pattern", "")
                context_lines = arguments.get("context_lines", 3)
                return grep_corpus(pattern, context_lines)
            except Exception as e:
                return f"[TOOL ERROR] grep_corpus failed: {e}"

        if name == "read_document":
            try:
                from tools.corpus import read_document
                doc_id = arguments.get("doc_id", "")
                return read_document(doc_id)
            except Exception as e:
                return f"[TOOL ERROR] read_document failed: {e}"

        if name == "read_lines":
            try:
                from tools.corpus import read_lines
                doc_id = arguments.get("doc_id", "")
                start_line = arguments.get("start_line", 1)
                end_line = arguments.get("end_line", 75)
                return read_lines(doc_id, start_line, end_line)
            except Exception as e:
                return f"[TOOL ERROR] read_lines failed: {e}"

        # Commodity tools
        if name == "commodity_price":
            try:
                from tools.commodity_tool import get_commodity_price_tool
                tag = arguments.get("tag", "")
                location = arguments.get("location", None)
                tool = get_commodity_price_tool()
                # Run async function synchronously
                import asyncio
                try:
                    result = asyncio.run(tool.execute(tag=tag, location=location))
                except RuntimeError:
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        result = asyncio.run(tool.execute(tag=tag, location=location))
                    except ImportError:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, tool.execute(tag=tag, location=location))
                                result = future.result()
                        else:
                            result = loop.run_until_complete(tool.execute(tag=tag, location=location))
                return result
            except Exception as e:
                return f"[TOOL ERROR] commodity_price failed: {e}"

        if name == "commodity_history":
            try:
                from tools.commodity_tool import get_commodity_history_tool
                tag = arguments.get("tag", "")
                period_months = arguments.get("period_months", 12)
                tool = get_commodity_history_tool()
                # Run async function synchronously
                import asyncio
                try:
                    result = asyncio.run(tool.execute(tag=tag, period_months=period_months))
                except RuntimeError:
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        result = asyncio.run(tool.execute(tag=tag, period_months=period_months))
                    except ImportError:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, tool.execute(tag=tag, period_months=period_months))
                                result = future.result()
                        else:
                            result = loop.run_until_complete(tool.execute(tag=tag, period_months=period_months))
                return result
            except Exception as e:
                return f"[TOOL ERROR] commodity_history failed: {e}"

        return f"[TOOL ERROR] Unknown tool: {name}"

    # ── Main public API ───────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """
        Process one user turn end-to-end.
        Returns the assistant's response text.
        """
        t0 = time.time()
        turn = self._stm.current_turn()
        stats_before = self._stm.stats()
        
        # Reset tool tracker for this turn
        self._tool_tracker.reset()

        ops: list[MemoryOpResult] = []

        # ── 1a. Prevent overflow before any LLM call ──────────────────────────
        overflow_ops = self._stm.force_fit()
        ops.extend(overflow_ops)

        # ── 1b. Retrieve relevant LTM entries into STM ───────────────────────
        relevant = self._ltm.search(user_input, top_k=3)
        if relevant:
            retrieve_op = self._stm.retrieve(relevant, trigger=TriggerKind.SYSTEM_RULE)
            ops.append(retrieve_op)

        # ── 1c. Detect and inject relevant skills ──────────────────────────────
        skills = self._skill_manager.detect_skills(user_input)
        for skill in skills:
            self._stm.add_message(
                role="system",
                content=f"[SKILL HINT: {skill.name}] {skill.hint_message}",
                relevance_score=self._config.SKILL_DEFAULT_RELEVANCE,
                is_pinned=False,
            )

        # ── 2. Main LLM call with tool support ────────────────────────────────
        self._stm.add_message(role="user", content=user_input, relevance_score=1.0)
        
        # Tool call loop - protected by LoopGuard and max iteration cap
        assistant_text = ""
        max_tool_iterations = 20  # Prevent infinite tool call loops
        tool_iterations = 0

        while tool_iterations < max_tool_iterations:
            messages = self._stm.openai_messages()
            tool_iterations += 1

            try:
                # Call LLM with tools if available
                assistant_text = self._llm.chat(
                    messages=messages,
                    max_tokens=self._config.DEFAULT_MAX_TOKENS,
                    temperature=self._config.DEFAULT_TEMPERATURE,
                    tools=self._tools if self._tools else None,
                )
                # If we get here, no tool call was made - we have a text response
                break

            except ToolCallResponse as e:
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
                        relevance_score=1.0,
                    )
                    continue
                
                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)
                
                # Record the tool call in ops_applied with TriggerKind.MAIN_AGENT
                ops.append(MemoryOpResult(
                    op=MemoryOp.RETRIEVE,  # Using RETRIEVE as the closest match for tool execution
                    success=True,
                    trigger=TriggerKind.MAIN_AGENT,
                    detail=f"Tool '{tool_name}' executed with args: {tool_args}",
                ))
                
                # Add assistant message with proper tool call structure
                # This is required for the LLM to understand its own tool call
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
                    relevance_score=0.8,
                    tool_calls=tool_calls_data,
                )
                # Add the tool result with proper role and tool_call_id
                self._stm.add_message(
                    role="tool",
                    content=tool_result,
                    relevance_score=0.9,
                    tool_call_id=tool_call_id,
                )
                
                # Loop back to LLM call with the tool result
                continue
        
        self._stm.add_message(role="assistant", content=assistant_text, relevance_score=0.8)
        self._stm.increment_turn()
        turn_after = self._stm.current_turn()

        # Post-response overflow guard: the assistant reply may have pushed us over
        post_ops = self._stm.force_fit()
        ops.extend(post_ops)

        # ── 3a. Collect learning feedback ─────────────────────────────────────
        feedback: Optional[LearningFeedback] = None
        if self._scorer.should_collect(turn_after):
            feedback = self._scorer.collect(
                context_messages=self._stm.openai_messages(),
                turn_index=turn_after,
            )
            if feedback:
                feedback.turn_index = turn_after

        # ── 3b. Evaluate system rules ─────────────────────────────────────────
        current_stats = self._stm.stats()
        decisions = self._rules.evaluate(current_stats, turn_after, feedback)

        ma_rationale = ""
        should_run_memory_agent = False

        for decision in decisions:
            if decision.rule_id in (RuleID.OVERFLOW_CRITICAL,):
                filter_op = self._stm.filter(trigger=TriggerKind.SYSTEM_RULE)
                ops.append(filter_op)
                summary_op = self._stm.summary(trigger=TriggerKind.SYSTEM_RULE)
                ops.append(summary_op)

            elif decision.rule_id == RuleID.OVERFLOW_WARN:
                summary_op = self._stm.summary(trigger=TriggerKind.SYSTEM_RULE)
                ops.append(summary_op)

            elif decision.rule_id in (RuleID.PERIODIC_REVIEW, RuleID.LEARNING_SPIKE):
                should_run_memory_agent = True

        # ── 3c. Immediately promote to LTM on learning spike ─────────────────
        if feedback and feedback.score >= self._config.LTM_PROMOTE_THRESHOLD:
            # Use affected_content if provided, otherwise extract from recent assistant message
            content_to_store = feedback.affected_content
            if not content_to_store:
                # Find the most recent assistant message as fallback
                for msg in reversed(self._stm.messages()):
                    if msg.role == "assistant" and msg.content:
                        content_to_store = msg.content[:500]  # Limit to 500 chars
                        break
            if content_to_store:
                add_op = self._ltm.add(
                    content=content_to_store,
                    learning_score=feedback.score,
                    source_turn=turn_after,
                    trigger=TriggerKind.LEARNING_SCORE,
                )
                ops.append(add_op)

        # ── 3d–e. Memory agent review ─────────────────────────────────────────
        if should_run_memory_agent:
            decision_obj = self._memory_agent.review(
                recent_messages=self._stm.messages(),
                ltm_entries=self._ltm.all_entries(),
                feedback=feedback,
            )
            ma_rationale = decision_obj.rationale
            ops.extend(self._apply_memory_agent_decision(decision_obj, turn_after, feedback))

        stats_after = self._stm.stats()

        # ── Persist STM after every turn ──────────────────────────────────────
        # LTM already persists inside LTMStore._maybe_persist() on every write
        self._persist_stm()

        # ── Trace ─────────────────────────────────────────────────────────────
        self._traces.append(TurnTrace(
            turn_index=turn_after,
            user_input=user_input,
            assistant_response=assistant_text,
            stm_stats_before=stats_before,
            stm_stats_after=stats_after,
            ops_applied=ops,
            feedback=feedback,
            memory_agent_rationale=ma_rationale,
            latency_ms=(time.time() - t0) * 1000,
            prompt_versions=dict(self._prompt_versions),  # Copy for audit trail
        ))

        return assistant_text

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

    # ── Internal ──────────────────────────────────────────────────────────────

    def _persist_stm(self) -> None:
        """Persist STM to disk if persistence is enabled."""
        if self._stm_persist_path:
            self._stm.save(self._stm_persist_path)

    def _apply_memory_agent_decision(
        self,
        decision,
        turn_index: int,
        feedback: Optional[LearningFeedback],
    ) -> list[MemoryOpResult]:
        ops: list[MemoryOpResult] = []

        for ltm_op in decision.ltm_operations:
            # Only apply high-confidence operations
            if ltm_op.confidence < 0.6:
                continue

            score = feedback.score if feedback else ltm_op.confidence

            if ltm_op.op == MemoryOp.ADD:
                result = self._ltm.add(
                    content=ltm_op.content,
                    learning_score=score,
                    tags=ltm_op.tags,
                    source_turn=turn_index,
                    trigger=TriggerKind.MEMORY_AGENT,
                )
                ops.append(result)

            elif ltm_op.op == MemoryOp.UPDATE and ltm_op.entry_id:
                result = self._ltm.update(
                    entry_id=ltm_op.entry_id,
                    content=ltm_op.content,
                    learning_score=score,
                    trigger=TriggerKind.MEMORY_AGENT,
                )
                ops.append(result)

            elif ltm_op.op == MemoryOp.DELETE and ltm_op.entry_id:
                result = self._ltm.delete(
                    entry_id=ltm_op.entry_id,
                    trigger=TriggerKind.MEMORY_AGENT,
                )
                ops.append(result)

        # Apply context relevance scores from MemoryAgent to STM messages
        if decision.context_relevance:
            for msg in self._stm.messages():
                if msg.turn_index in decision.context_relevance:
                    msg.relevance_score = decision.context_relevance[msg.turn_index]

        # SUMMARY if MemoryAgent requested it
        if decision.summary_needed:
            ops.append(self._stm.summary(trigger=TriggerKind.MEMORY_AGENT))

        return ops
