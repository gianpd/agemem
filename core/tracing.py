"""
core/tracing.py
───────────────
Comprehensive logging and tracing system for AgeMem.

Provides:
- InteractionLogger: Traces user input to agent response
- Rotating file logs with 1-month retention
- Debug-level logging for all interactions
- Raw response capture before any processing

Usage:
    from core.tracing import get_tracer, init_tracing

    # Initialize at startup
    init_tracing(log_dir="logs", debug=True)

    # Get tracer instance
    tracer = get_tracer()

    # Trace user input
    tracer.log_user_input(user_input)

    # Trace raw LLM response (before processing)
    tracer.log_raw_response(response)

    # Trace processed response
    tracer.log_final_response(processed_response)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Optional


# ── Log Record Types ─────────────────────────────────────────────────────────

@dataclass
class InteractionRecord:
    """Represents a complete user-to-agent interaction trace."""
    trace_id: str
    timestamp: str
    user_input: str
    raw_response: Optional[str] = None
    final_response: Optional[str] = None
    processing_time_ms: float = 0.0
    turn_index: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    memory_ops: list[dict] = field(default_factory=list)
    stm_stats: dict = field(default_factory=dict)
    ltm_entries: int = 0
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass
class ToolCallRecord:
    """Represents a tool call trace."""
    trace_id: str
    timestamp: str
    tool_name: str
    arguments: dict
    result: Optional[str] = None
    success: bool = True
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class LLMCallRecord:
    """Represents an LLM API call trace."""
    trace_id: str
    timestamp: str
    model: str
    message_count: int
    max_tokens: int
    temperature: float
    has_tools: bool
    response_preview: Optional[str] = None
    latency_ms: float = 0.0
    token_count: int = 0
    error: Optional[str] = None


# ── Custom Formatter ─────────────────────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured log entries.
    Supports both human-readable and JSON formats.
    """

    def __init__(self, json_format: bool = False):
        self.json_format = json_format
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "trace_id"):
            entry["trace_id"] = record.trace_id
        if hasattr(record, "interaction_type"):
            entry["interaction_type"] = record.interaction_type
        if hasattr(record, "duration_ms"):
            entry["duration_ms"] = record.duration_ms
        if hasattr(record, "turn_index"):
            entry["turn_index"] = record.turn_index

        # Add exception info if present
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        # Add any additional extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename",
                           "funcName", "levelname", "levelno", "lineno",
                           "module", "msecs", "pathname", "process",
                           "processName", "relativeCreated", "stack_info",
                           "exc_info", "exc_text", "thread", "threadName",
                           "message", "asctime"]:
                try:
                    # Try to serialize, skip if not JSON-serializable
                    json.dumps({key: value})
                    entry[key] = value
                except (TypeError, ValueError):
                    entry[key] = str(value)

        if self.json_format:
            return json.dumps(entry, ensure_ascii=False)
        else:
            # Human-readable format
            parts = [f"[{entry['timestamp']}] [{entry['level']:5}] [{entry['logger']}]"]
            if "trace_id" in entry and entry["trace_id"]:
                parts.append(f"[{entry['trace_id'][:8]}]")
            parts.append(entry["message"])
            if "duration_ms" in entry:
                parts.append(f"({entry['duration_ms']:.1f}ms)")
            return " ".join(parts)


# ── Seed Domain/Difficulty Helpers ────────────────────────────────────────────

_CORPUS_TOOLS = {
    "list_documents", "search_metadata", "grep_corpus",
    "read_document", "read_lines", "ingest_document",
}
_PERSISTENCE_TOOLS = {
    "force_memory_persistence", "validate_memory_commit",
    "assess_persistence_need", "log_persistence_failure",
}
_INTROSPECTION_TOOLS = {
    "assess_conversation_drift", "are_you_ready_to_get_in_context_ltm",
    "paraphrase_for_coverage", "trigger_contextual_ltm_retrieval",
    "validate_ltm_relevance", "refine_retrieval_target",
    "log_retrieval_decision",
}
_EXTERNAL_TOOLS = {"web_search", "fetch_url", "write_file"}
_BROWSER_TOOLS = {
    "browser_navigate", "browser_click", "browser_scroll",
    "browser_type", "browser_press", "browser_read_page",
    "browser_screenshot", "browser_close",
}


def _map_seed_domain(tool_calls: list[dict]) -> str:
    """Map tool call sequence to a domain ID for the seed."""
    if not tool_calls:
        return "DIRECT-RESPONSE"

    categories = set()
    has_error_recovery = False
    for i, tc in enumerate(tool_calls):
        name = tc.get("name", "")
        if name in _CORPUS_TOOLS:
            categories.add("CORPUS")
        elif name in _PERSISTENCE_TOOLS:
            categories.add("PERSISTENCE")
        elif name in _INTROSPECTION_TOOLS:
            categories.add("INTROSPECTION")
        elif name in _EXTERNAL_TOOLS:
            categories.add("EXTERNAL")
        elif name in _BROWSER_TOOLS:
            categories.add("BROWSER")
        if not tc.get("success", True) and i + 1 < len(tool_calls):
            has_error_recovery = True

    if has_error_recovery:
        return "TOOL-RECOVER"
    if len(categories) >= 2:
        return "TOOL-CHAIN"
    if categories == {"PERSISTENCE"}:
        return "MEMORY-PERSIST"
    if categories == {"CORPUS"}:
        return "CORPUS-CHAIN" if len(tool_calls) >= 2 else "CORPUS-QUERY"
    if categories == {"EXTERNAL"}:
        return "EXTERNAL-SEARCH"
    if categories == {"BROWSER"}:
        return "BROWSER-AUTOMATION"
    if categories == {"INTROSPECTION"}:
        return "MEMORY-INTROSPECT"
    return "TOOL-CHAIN"


def _estimate_seed_difficulty(tool_calls: list[dict], user_message: str) -> int:
    """Estimate difficulty 1-5."""
    n = len(tool_calls)
    categories = set()
    has_errors = False
    for tc in tool_calls:
        name = tc.get("name", "")
        if name in _CORPUS_TOOLS:
            categories.add("CORPUS")
        elif name in _PERSISTENCE_TOOLS:
            categories.add("PERSISTENCE")
        elif name in _INTROSPECTION_TOOLS:
            categories.add("INTROSPECTION")
        elif name in _EXTERNAL_TOOLS:
            categories.add("EXTERNAL")
        elif name in _BROWSER_TOOLS:
            categories.add("BROWSER")
        if not tc.get("success", True):
            has_errors = True

    if n >= 5 or (has_errors and n >= 3):
        return 5
    if n >= 3 and len(categories) >= 2:
        return 4
    if n >= 2:
        return 3
    if n == 1 and (len(user_message) > 200 or has_errors):
        return 2
    if n == 1:
        return 1
    return 2


def _get_seed_meta_skills(domain: str, tool_calls: list[dict]) -> list[str]:
    """Assign meta skill tags."""
    skills = ["META-REASON"]
    if len(tool_calls) >= 2:
        skills.append("META-TOOL-CHAIN")
    if domain == "TOOL-RECOVER":
        skills.append("META-TOOL-RECOVER")
    if domain == "MEMORY-PERSIST":
        skills.append("META-MEMORY-OPS")
    if any(not tc.get("success", True) for tc in tool_calls):
        skills.append("META-ERROR-HANDLING")
    return skills


# ── Interaction Logger ───────────────────────────────────────────────────────

class InteractionLogger:
    """
    Central logger for tracing all interactions in AgeMem.

    Features:
    - Traces user input to agent response
    - Captures raw LLM responses before processing
    - Logs tool calls and memory operations
    - Rotating file logs with configurable retention
    - Debug-level detailed logging
    """

    _instance: Optional["InteractionLogger"] = None

    def __init__(
        self,
        log_dir: str = "logs",
        debug: bool = False,
        json_format: bool = False,
        retention_days: int = 30,
        max_file_size_mb: int = 50,
    ):
        self._log_dir = Path(log_dir).resolve()
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._debug = debug
        self._json_format = json_format
        self._retention_days = retention_days
        self._current_trace: Optional[InteractionRecord] = None
        self._trace_start_time: float = 0.0

        # State tracking to reduce log noise
        self._last_message_count: int = 0
        self._raw_response_logged: bool = False

        # Seed file for direct dataset capture (append-only JSONL)
        self._seed_file = None
        self._seed_lock = threading.Lock()

        # Set up loggers
        self._setup_loggers()

    def _setup_loggers(self):
        """Configure all loggers with rotating file handlers."""
        # Main interaction logger
        self._logger = logging.getLogger("agemem.interaction")
        self._logger.setLevel(logging.DEBUG if self._debug else logging.INFO)
        self._logger.handlers.clear()

        # LLM call logger
        self._llm_logger = logging.getLogger("agemem.llm")
        self._llm_logger.setLevel(logging.DEBUG if self._debug else logging.INFO)
        self._llm_logger.handlers.clear()

        # Tool call logger
        self._tool_logger = logging.getLogger("agemem.tools")
        self._tool_logger.setLevel(logging.DEBUG if self._debug else logging.INFO)
        self._tool_logger.handlers.clear()

        # Memory operations logger
        self._memory_logger = logging.getLogger("agemem.memory")
        self._memory_logger.setLevel(logging.DEBUG if self._debug else logging.INFO)
        self._memory_logger.handlers.clear()

        # Add rotating file handlers (1 month rotation, keep 1 backup = 2 months total)
        for logger, filename in [
            (self._logger, "interactions.log"),
            (self._llm_logger, "llm_calls.log"),
            (self._tool_logger, "tool_calls.log"),
            (self._memory_logger, "memory_ops.log"),
        ]:
            # Time-based rotation: rotate monthly
            time_handler = TimedRotatingFileHandler(
                filename=self._log_dir / filename,
                when="MIDNIGHT",  # Rotate at midnight
                interval=1,       # Every day
                backupCount=self._retention_days,  # Keep 30 days
                encoding="utf-8",
            )
            time_handler.suffix = "%Y-%m-%d"
            time_handler.setFormatter(StructuredFormatter(json_format=self._json_format))
            logger.addHandler(time_handler)

        # Console handler for errors (always show)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(StructuredFormatter(json_format=False))

        for logger in [self._logger, self._llm_logger, self._tool_logger, self._memory_logger]:
            logger.addHandler(console_handler)

        # Also create a combined debug log
        self._debug_logger = logging.getLogger("agemem.debug")
        self._debug_logger.setLevel(logging.DEBUG)
        self._debug_logger.handlers.clear()

        debug_handler = TimedRotatingFileHandler(
            filename=self._log_dir / "debug.log",
            when="MIDNIGHT",
            interval=1,
            backupCount=self._retention_days,
            encoding="utf-8",
        )
        debug_handler.suffix = "%Y-%m-%d"
        debug_handler.setFormatter(StructuredFormatter(json_format=False))
        self._debug_logger.addHandler(debug_handler)

        # Seed file: append-only JSONL for distilled-pipeline fine-tuning
        # Each line is a complete seed in OpenAI chat format with meta.
        # Opened lazily on first write, stays open for the session.
        self._seed_file_path = self._log_dir / "seeds.jsonl"

    # ── Trace Management ──────────────────────────────────────────────────────

    def start_trace(self, user_input: str, turn_index: int = 0) -> str:
        """
        Start a new interaction trace.

        Returns the trace_id for this interaction.
        """
        trace_id = str(uuid.uuid4())

        self._current_trace = InteractionRecord(
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            turn_index=turn_index,
        )
        self._trace_start_time = time.time()
        self._last_message_count = 0
        self._raw_response_logged = False

        # Build input preview
        input_preview = user_input[:150] + "..." if len(user_input) > 150 else user_input

        # Log to structured logger
        self._logger.debug(
            f"TRACE_START: id={trace_id[:8]} turn={turn_index} input_len={len(user_input)}",
            extra={
                "trace_id": trace_id,
                "interaction_type": "trace_start",
                "turn_index": turn_index,
                "input_length": len(user_input),
            }
        )

        # Log to debug with clear visual separation
        self._debug_logger.debug(
            f"\n{'═' * 60}\n"
            f"[TRACE_START] trace_id={trace_id} turn={turn_index}\n"
            f"{'─' * 60}\n"
            f"USER_INPUT: {input_preview}\n"
            f"{'═' * 60}"
        )

        return trace_id

    def end_trace(self, final_response: Optional[str] = None, error: Optional[str] = None):
        """End the current interaction trace and log the complete record."""
        if not self._current_trace:
            return

        self._current_trace.processing_time_ms = (time.time() - self._trace_start_time) * 1000
        self._current_trace.final_response = final_response
        self._current_trace.error = error

        # Build memory operations summary
        mem_summary = self._build_memory_summary()
        tool_summary = self._build_tool_summary()

        # Log to main interaction logger
        self._logger.debug(
            f"TRACE_END: id={self._current_trace.trace_id[:8]} "
            f"duration={self._current_trace.processing_time_ms:.1f}ms "
            f"tools={len(self._current_trace.tool_calls)} "
            f"memory_ops={len(self._current_trace.memory_ops)}",
            extra={
                "trace_id": self._current_trace.trace_id,
                "interaction_type": "trace_end",
                "duration_ms": self._current_trace.processing_time_ms,
                "tool_calls_count": len(self._current_trace.tool_calls),
                "memory_ops_count": len(self._current_trace.memory_ops),
                "error": error,
            }
        )

        # Log complete trace summary to debug log
        self._debug_logger.debug(
            f"\n{'═' * 60}\n"
            f"[TRACE_END] trace_id={self._current_trace.trace_id}\n"
            f"{'─' * 60}\n"
            f"Duration: {self._current_trace.processing_time_ms:.1f}ms | Turn: {self._current_trace.turn_index}\n"
            f"{tool_summary}\n"
            f"{mem_summary}\n"
            f"{'═' * 60}"
        )

        # Reset state for next trace
        self._current_trace = None
        self._last_message_count = 0
        self._raw_response_logged = False

    def _build_tool_summary(self) -> str:
        """Build a concise summary of tool calls."""
        if not self._current_trace or not self._current_trace.tool_calls:
            return "Tools: none"

        tools = self._current_trace.tool_calls
        by_name = {}
        for t in tools:
            name = t["tool_name"]
            by_name[name] = by_name.get(name, 0) + 1

        tool_list = ", ".join([f"{name}({count})" for name, count in by_name.items()])
        return f"Tools ({len(tools)} total): {tool_list}"

    def _build_memory_summary(self) -> str:
        """Build a clear summary of memory operations."""
        if not self._current_trace or not self._current_trace.memory_ops:
            return "Memory: no operations"

        ops = self._current_trace.memory_ops
        by_type = {}
        for op in ops:
            op_type = op.get("op_type", "unknown")
            by_type[op_type] = by_type.get(op_type, 0) + 1

        # Format each operation type
        summary_parts = []

        # STM
        if "stm_snapshot" in by_type:
            stm_tokens = self._current_trace.stm_stats.get("total_tokens", 0) if self._current_trace else 0
            summary_parts.append(f"STM: {by_type['stm_snapshot']} snapshots ({stm_tokens} tokens)")

        # LTM Retrieval
        if "ltm_retrieval" in by_type:
            summary_parts.append(f"LTM: {by_type['ltm_retrieval']} retrievals")

        # LTM Storage
        if "ltm_storage" in by_type:
            summary_parts.append(f"LTM: {by_type['ltm_storage']} stored")

        # Introspection
        intro_count = by_type.get("introspection_trigger", 0) + by_type.get("introspection_result", 0)
        if intro_count:
            summary_parts.append(f"Introspection: {intro_count} ops")

        # Consolidation
        if "consolidation" in by_type:
            summary_parts.append(f"Consolidation: {by_type['consolidation']} ops")

        return "Memory: " + " | ".join(summary_parts) if summary_parts else "Memory: operations logged"

    # ── Raw Response Logging ──────────────────────────────────────────────────

    def log_raw_response(self, response: str, model: str = "unknown"):
        """
        Log the RAW LLM response BEFORE any processing or parsing.

        This is the key method for testing the first agent response.
        """
        if not self._current_trace:
            return

        self._current_trace.raw_response = response
        self._raw_response_logged = True  # Flag to prevent duplication

        # Determine response type for better categorization
        is_tool_call = "function" in response or "tool_calls" in response
        response_type = "TOOL_CALL" if is_tool_call else "TEXT"

        # Log to debug with clear boundaries
        self._debug_logger.debug(
            f"[RAW_RESPONSE:{response_type}] trace_id={self._current_trace.trace_id[:8]}\n"
            f"Model: {model} | Length: {len(response)} chars\n"
            f"{'─' * 50}\n"
            f"{response}\n"
            f"{'─' * 50}"
        )

        # Also log summary to main logger
        self._logger.debug(
            f"RAW_RESPONSE: type={response_type} model={model} length={len(response)}",
            extra={
                "trace_id": self._current_trace.trace_id,
                "interaction_type": "raw_response",
                "model": model,
                "response_length": len(response),
                "is_tool_call": is_tool_call,
            }
        )

    # ── User Input Logging ────────────────────────────────────────────────────

    def log_user_input(self, user_input: str, turn_index: int = 0) -> str:
        """Log user input and start a trace. Returns trace_id."""
        return self.start_trace(user_input, turn_index)

    # ── LLM Call Logging ──────────────────────────────────────────────────────

    def log_llm_call(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
        has_tools: bool = False,
    ) -> str:
        """
        Log an LLM API call. Returns a call_id for matching with response.
        Only shows NEW messages added since last call (reduces noise).
        """
        call_id = str(uuid.uuid4())[:8]

        # Calculate what's new (only log messages beyond previous count)
        prev_count = getattr(self, '_last_message_count', 0)
        new_count = len(messages)
        new_messages = messages[prev_count:] if prev_count > 0 else messages
        self._last_message_count = new_count

        # Log to LLM logger (structured)
        self._llm_logger.debug(
            f"LLM_CALL: model={model} total_messages={new_count} "
            f"new_messages={len(new_messages)} max_tokens={max_tokens}",
            extra={
                "call_id": call_id,
                "trace_id": self._current_trace.trace_id if self._current_trace else None,
                "model": model,
                "total_messages": new_count,
                "new_messages": len(new_messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "has_tools": has_tools,
            }
        )

        # Build concise summary of NEW messages only
        if new_messages:
            msg_summaries = []
            for i, msg in enumerate(new_messages):
                actual_idx = prev_count + i
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "tool":
                    # Tool results - show tool name and result size
                    tool_name = msg.get("name", "unknown")
                    result_preview = content[:80] + "..." if len(content) > 80 else content
                    msg_summaries.append(f"  [{actual_idx}] tool:{tool_name} → {result_preview}")
                elif role == "assistant" and msg.get("tool_calls"):
                    # Assistant requesting tool calls - list tool names
                    tool_calls = msg.get("tool_calls", [])
                    tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                    msg_summaries.append(f"  [{actual_idx}] assistant: calls [{', '.join(tool_names)}]")
                elif content:
                    # Regular message - content preview
                    preview = content[:100] + "..." if len(content) > 100 else content
                    msg_summaries.append(f"  [{actual_idx}] {role}: {preview}")
                else:
                    msg_summaries.append(f"  [{actual_idx}] {role}: [empty]")

            new_msg_info = f"\nNew messages ({len(new_messages)} of {new_count} total):\n" + "\n".join(msg_summaries)
        else:
            new_msg_info = "\nNo new messages (all messages previously logged)"

        self._debug_logger.debug(
            f"[LLM_CALL] call_id={call_id} model={model}\n"
            f"Config: max_tokens={max_tokens} temp={temperature} tools={has_tools}\n"
            f"Context: {new_count} messages total"
            f"{new_msg_info}"
        )

        return call_id

    def log_llm_response(
        self,
        call_id: str,
        response: str,
        latency_ms: float,
        token_count: int = 0,
        error: Optional[str] = None,
        model: str = "unknown",
        finish_reason: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        """
        Log an LLM API response.
        NOTE: Raw response content is logged separately via log_raw_response()
        """
        self._llm_logger.debug(
            f"LLM_RESPONSE: call_id={call_id} latency={latency_ms:.1f}ms "
            f"tokens={completion_tokens} finish={finish_reason}",
            extra={
                "call_id": call_id,
                "trace_id": self._current_trace.trace_id if self._current_trace else None,
                "latency_ms": latency_ms,
                "token_count": completion_tokens,
                "response_length": len(response) if response else 0,
                "error": error,
                "model": model,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

        # Log response metadata only (content goes to raw_response)
        is_empty = not response or not response.strip()
        has_tool_calls = "tool_calls" in (response or "")

        self._debug_logger.debug(
            f"[LLM_RESPONSE] call_id={call_id} model={model}\n"
            f"Latency: {latency_ms:.1f}ms | Tokens: {prompt_tokens}→{completion_tokens}\n"
            f"Finish: {finish_reason} | Has tool_calls: {has_tool_calls} | Empty: {is_empty}"
        )

        # Raw response logged separately - don't duplicate here
        if response and not getattr(self, '_raw_response_logged', False):
            self.log_raw_response(response, model=model)

    # ── Tool Call Logging ─────────────────────────────────────────────────────

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        duration_ms: float = 0.0,
        result: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        validation_errors: Optional[list] = None,
        validation_warnings: Optional[list] = None,
    ):
        """
        Log a tool call execution with consolidated information.
        Replaces the scattered TOOL_CALL_DETECTED + TOOL_CALL logs.
        """
        if self._current_trace:
            self._current_trace.tool_calls.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "duration_ms": duration_ms,
                "success": success,
                "error": error,
            })

        # Build single-line summary for structured logger
        status = "OK" if success else "FAIL"
        result_preview = (result or "None")[:100]
        if result and len(result) > 100:
            result_preview += "..."

        self._tool_logger.debug(
            f"TOOL:{tool_name} status={status} duration={duration_ms:.1f}ms",
            extra={
                "trace_id": self._current_trace.trace_id if self._current_trace else None,
                "tool_name": tool_name,
                "arguments": arguments,
                "duration_ms": duration_ms,
                "success": success,
                "error": error,
                "result_preview": result_preview,
            }
        )

        # Build detailed debug log with validation info
        validation_info = ""
        if validation_errors:
            validation_info = f"\n  Validation ERRORS: {validation_errors}"
        elif validation_warnings:
            validation_info = f"\n  Validation WARNINGS: {validation_warnings}"

        self._debug_logger.debug(
            f"[TOOL_EXEC] {tool_name} | status={status} | duration={duration_ms:.1f}ms\n"
            f"  Args: {json.dumps(arguments, ensure_ascii=False)[:150]}\n"
            f"  Result: {result_preview}"
            f"{validation_info}"
            f"{f' Error: {error}' if error else ''}"
        )

    # ── Memory Operation Logging ──────────────────────────────────────────────

    def log_memory_op(
        self,
        op_type: str,
        detail: str,
        success: bool = True,
        trigger: Optional[str] = None,
    ):
        """Log a memory operation (LTM/STM)."""
        if self._current_trace:
            self._current_trace.memory_ops.append({
                "op_type": op_type,
                "detail": detail,
                "success": success,
                "trigger": trigger,
            })

    def log_stm_snapshot(self, stats: dict, trigger: str = "turn_start"):
        """
        Log STM state snapshot at key moments.
        Use this instead of scattered STM_STATS logs.
        """
        if self._current_trace:
            self._current_trace.stm_stats = stats

        self._debug_logger.debug(
            f"[MEMORY:STM] trigger={trigger} "
            f"tokens={stats.get('total_tokens', 0)} "
            f"util={stats.get('utilisation_ratio', 0):.1%} "
            f"messages={stats.get('message_count', 0)}"
        )

    def log_ltm_retrieval(
        self,
        query: str,
        hits: list[dict],
        duration_ms: float,
        search_method: str = "semantic",
        trigger: str = "user_input",
    ):
        """
        Log LTM retrieval operation with clear hit/miss visibility.

        Args:
            query: The search query used
            hits: List of retrieved memory entries
            duration_ms: Time taken for retrieval
            search_method: How the search was performed (semantic, keyword, hybrid)
            trigger: What triggered the retrieval (user_input, tool_result, etc.)
        """
        if self._current_trace:
            self._current_trace.memory_ops.append({
                "op_type": "ltm_retrieval",
                "query": query[:100],
                "hits_count": len(hits),
                "duration_ms": duration_ms,
                "trigger": trigger,
            })

        # Format hit summary
        if hits:
            hit_summaries = [f"{h.get('doc_id', 'unknown')[:8]}:{h.get('score', 0):.2f}" for h in hits[:3]]
            hit_detail = f"hits={len(hits)} top=[{', '.join(hit_summaries)}]"
        else:
            hit_detail = "hits=0"

        self._debug_logger.debug(
            f"[MEMORY:LTM:RETRIEVE] trigger={trigger} method={search_method} "
            f"duration={duration_ms:.1f}ms {hit_detail}"
        )

    def log_ltm_storage(
        self,
        doc_id: str,
        content_type: str,
        token_count: int,
        labels: Optional[list] = None,
        trigger: str = "ingestion",
    ):
        """Log LTM storage operation."""
        if self._current_trace:
            self._current_trace.memory_ops.append({
                "op_type": "ltm_storage",
                "doc_id": doc_id,
                "content_type": content_type,
                "token_count": token_count,
                "trigger": trigger,
            })

        self._debug_logger.debug(
            f"[MEMORY:LTM:STORE] doc_id={doc_id[:16]} type={content_type} "
            f"tokens={token_count} labels={labels or []} trigger={trigger}"
        )

    def log_introspection_trigger(
        self,
        trigger_type: str,
        confidence: float,
        context_summary: str,
    ):
        """
        Log when self-introspection is triggered.

        Args:
            trigger_type: Why introspection was triggered (memory_threshold, user_request, etc.)
            confidence: Model confidence in triggering introspection
            context_summary: Brief summary of what triggered it
        """
        if self._current_trace:
            self._current_trace.memory_ops.append({
                "op_type": "introspection_trigger",
                "trigger_type": trigger_type,
                "confidence": confidence,
            })

        self._debug_logger.debug(
            f"[MEMORY:INTROSPECTION:TRIGGER] type={trigger_type} "
            f"confidence={confidence:.2f} context='{context_summary[:60]}'"
        )

    def log_introspection_result(
        self,
        action: str,
        target_memory: str,
        success: bool,
        detail: str,
    ):
        """
        Log the result of a self-introspection operation.

        Args:
            action: What action was taken (consolidate, prune, summarize, etc.)
            target_memory: Which memory was affected (stm, ltm_doc_xxx, etc.)
            success: Whether the operation succeeded
            detail: Brief description of the result
        """
        if self._current_trace:
            self._current_trace.memory_ops.append({
                "op_type": "introspection_result",
                "action": action,
                "target": target_memory,
                "success": success,
            })

        status = "OK" if success else "FAIL"
        self._debug_logger.debug(
            f"[MEMORY:INTROSPECTION:RESULT] status={status} action={action} "
            f"target={target_memory[:30]} detail='{detail[:60]}'"
        )

    def log_memory_consolidation(
        self,
        source_count: int,
        target_count: int,
        consolidation_type: str,
        duration_ms: float,
    ):
        """Log memory consolidation operation (STM→LTM or LTM merge)."""
        if self._current_trace:
            self._current_trace.memory_ops.append({
                "op_type": "consolidation",
                "source_count": source_count,
                "target_count": target_count,
            })

        self._debug_logger.debug(
            f"[MEMORY:CONSOLIDATION] type={consolidation_type} "
            f"{source_count}→{target_count} duration={duration_ms:.1f}ms"
        )

    def log_query_expansion(
        self,
        original_query: str,
        variants: list[str],
        hit_counts: list[int],
        duration_ms: float,
        method: str = "llm",
        success: bool = True,
    ):
        """
        Log query expansion details for corpus search.

        Parameters
        ----------
        original_query : str
            The original user query.
        variants : list[str]
            All query variants including the original (first element).
        hit_counts : list[int]
            Number of hits for each variant (parallel to variants).
        duration_ms : float
            Time taken for expansion and search.
        method : str
            Expansion method used: "llm", "regex", or "disabled".
        success : bool
            Whether expansion succeeded.
        """
        num_variants = len(variants)
        num_hits = sum(1 for h in hit_counts if h > 0)
        total_matches = sum(hit_counts)

        # Build detailed log message
        variant_details = []
        for i, (variant, hits) in enumerate(zip(variants, hit_counts)):
            status = "HIT" if hits > 0 else "no hit"
            marker = "[ORIG]" if i == 0 else f"[VAR{i}]"
            variant_details.append(f"  {marker} '{variant[:60]}{'...' if len(variant) > 60 else ''}' ({status}: {hits} matches)")

        self._debug_logger.debug(
            f"[QUERY_EXPANSION] original='{original_query[:80]}{'...' if len(original_query) > 80 else ''}'\n"
            f"Method: {method}\n"
            f"Variants generated: {num_variants}\n"
            f"Variants with hits: {num_hits}/{num_variants}\n"
            f"Total unique matches: {total_matches}\n"
            f"Duration: {duration_ms:.1f}ms\n"
            f"Success: {success}\n"
            f"Variant details:\n" + "\n".join(variant_details)
        )

        # Also log to main logger for structured output
        self._logger.debug(
            f"QUERY_EXPANSION: method={method} variants={num_variants} hits={num_hits}/{num_variants} duration={duration_ms:.1f}ms",
            extra={
                "trace_id": self._current_trace.trace_id if self._current_trace else None,
                "interaction_type": "query_expansion",
                "original_query": original_query[:200],
                "num_variants": num_variants,
                "variants_with_hits": num_hits,
                "total_matches": total_matches,
                "duration_ms": duration_ms,
                "method": method,
                "success": success,
                "variant_list": [v[:100] for v in variants],  # Truncated for log size
            }
        )

    def log_stm_stats(self, stats: dict):
        """Log STM statistics."""
        if self._current_trace:
            self._current_trace.stm_stats = stats

        self._debug_logger.debug(
            f"[STM_STATS] tokens={stats.get('total_tokens', 0)} "
            f"utilisation={stats.get('utilisation_ratio', 0):.1%}"
        )

    def log_ltm_count(self, count: int):
        """Log LTM entry count."""
        if self._current_trace:
            self._current_trace.ltm_entries = count

    # ── Seed Logging ────────────────────────────────────────────────────────────

    def log_seed(
        self,
        system_prompt: str,
        user_message: str,
        tool_calls: list[dict],
        final_response: str,
        model: str = "unknown",
        turn_index: int = 0,
        latency_ms: float = 0.0,
    ):
        """
        Write a complete seed entry to seeds.jsonl.

        Called once per conversation turn, after the orchestrator completes.
        The output is directly consumable by the distilled-pipeline's SeedLoader.

        Args:
            system_prompt: The pinned system prompt from STM.
            user_message: The user's input to chat().
            tool_calls: List of dicts with keys: name, arguments (dict),
                        result (str), duration_ms (float), success (bool).
            final_response: The assistant's text response.
            model: LLM model identifier.
            turn_index: Turn number within the session.
            latency_ms: Total turn duration.
        """
        # Build reasoning from tool call chain
        reasoning_steps = []
        for i, tc in enumerate(tool_calls, 1):
            args = tc.get("arguments", {})
            if isinstance(args, dict):
                args_display = ", ".join(
                    f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                    for k, v in args.items()
                )
            else:
                args_display = str(args)[:100]

            result = tc.get("result", "")
            success = tc.get("success", True)
            if not success:
                result_summary = "ERROR"
            elif result:
                result_summary = result.split("\n")[0][:120]
            else:
                result_summary = "(no result)"

            reasoning_steps.append(
                f"Step {i}: Call {tc['name']}({args_display}) -> {result_summary}"
            )

        reasoning = "\n".join(reasoning_steps)
        answer = final_response if final_response else "(tool call sequence completed)"

        # Map domain from tool categories
        domain = _map_seed_domain(tool_calls)
        difficulty = _estimate_seed_difficulty(tool_calls, user_message)
        meta_skills = _get_seed_meta_skills(domain, tool_calls)
        tool_names = [tc["name"] for tc in tool_calls if tc.get("name")]

        # Build seed ID from timestamp
        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        seed_id = f"agemem_{date_str}_{turn_index:04d}"

        seed = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
                {
                    "role": "assistant",
                    "content": f"[Reasoning]\n{reasoning}\n[/Reasoning]\n[Answer]\n{answer}\n[/Answer]",
                },
            ],
            "meta": {
                "domain": domain,
                "difficulty": difficulty,
                "seed_id": seed_id,
                "meta_skills": meta_skills,
                "source_model": model,
                "tools_used": tool_names,
                "tool_count": len(tool_calls),
                "turn_duration_ms": round(latency_ms, 1),
            },
            "status": "seed",
        }

        # Write to seed file (open lazily, keep open, thread-safe)
        try:
            with self._seed_lock:
                if self._seed_file is None:
                    self._seed_file = open(
                        self._seed_file_path, "a", encoding="utf-8"
                    )
                self._seed_file.write(json.dumps(seed, ensure_ascii=False) + "\n")
                self._seed_file.flush()
        except Exception:
            # Seed logging should never break the main flow
            pass

    # ── Context Manager ───────────────────────────────────────────────────────

    @contextmanager
    def trace_interaction(self, user_input: str, turn_index: int = 0):
        """
        Context manager for tracing an interaction.

        Usage:
            with tracer.trace_interaction(user_input) as trace_id:
                response = orch.chat(user_input)
                tracer.log_final_response(response)
        """
        trace_id = self.start_trace(user_input, turn_index)
        error = None
        try:
            yield trace_id
        except Exception as e:
            error = str(e)
            raise
        finally:
            self.end_trace(error=error)

    # ── Utility Methods ───────────────────────────────────────────────────────

    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID if a trace is active."""
        return self._current_trace.trace_id if self._current_trace else None

    def set_debug(self, debug: bool):
        """Enable or disable debug mode."""
        self._debug = debug
        level = logging.DEBUG if debug else logging.INFO
        for logger in [self._logger, self._llm_logger, self._tool_logger, self._memory_logger]:
            logger.setLevel(level)


# ── Singleton Management ─────────────────────────────────────────────────────

_tracer: Optional[InteractionLogger] = None


def init_tracing(
    log_dir: str = "logs",
    debug: bool = False,
    json_format: bool = False,
    retention_days: int = 30,
) -> InteractionLogger:
    """
    Initialize the global tracer instance.

    Args:
        log_dir: Directory for log files
        debug: Enable debug-level logging
        json_format: Use JSON format for log entries
        retention_days: Number of days to keep log files (default 30)

    Returns:
        The global InteractionLogger instance
    """
    global _tracer
    _tracer = InteractionLogger(
        log_dir=log_dir,
        debug=debug,
        json_format=json_format,
        retention_days=retention_days,
    )
    return _tracer


def get_tracer() -> InteractionLogger:
    """
    Get the global tracer instance.

    Initializes with defaults if not already initialized.
    """
    global _tracer
    if _tracer is None:
        _tracer = InteractionLogger()
    return _tracer


def shutdown_tracing():
    """Shutdown and flush all log handlers."""
    global _tracer
    if _tracer:
        for handler in _tracer._logger.handlers:
            handler.close()
        for handler in _tracer._llm_logger.handlers:
            handler.close()
        for handler in _tracer._tool_logger.handlers:
            handler.close()
        for handler in _tracer._memory_logger.handlers:
            handler.close()
        for handler in _tracer._debug_logger.handlers:
            handler.close()
        if _tracer._seed_file:
            _tracer._seed_file.close()
    _tracer = None