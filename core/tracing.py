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
            if "trace_id" in entry:
                parts.append(f"[{entry['trace_id'][:8]}]")
            parts.append(entry["message"])
            if "duration_ms" in entry:
                parts.append(f"({entry['duration_ms']:.1f}ms)")
            return " ".join(parts)


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
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._debug = debug
        self._json_format = json_format
        self._retention_days = retention_days
        self._current_trace: Optional[InteractionRecord] = None
        self._trace_start_time: float = 0.0

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

        # Log the user input
        self._logger.debug(
            f"USER_INPUT: {user_input[:200]}{'...' if len(user_input) > 200 else ''}",
            extra={
                "trace_id": trace_id,
                "interaction_type": "user_input",
                "turn_index": turn_index,
                "input_length": len(user_input),
            }
        )
        self._debug_logger.debug(
            f"[TRACE_START] trace_id={trace_id} turn={turn_index}\n"
            f"USER_INPUT:\n{user_input}\n"
            f"{'─' * 40}"
        )

        return trace_id

    def end_trace(self, final_response: Optional[str] = None, error: Optional[str] = None):
        """End the current interaction trace and log the complete record."""
        if not self._current_trace:
            return

        self._current_trace.processing_time_ms = (time.time() - self._trace_start_time) * 1000
        self._current_trace.final_response = final_response
        self._current_trace.error = error

        # Log the final response
        self._logger.debug(
            f"FINAL_RESPONSE: {final_response[:200] if final_response else 'None'}{'...' if final_response and len(final_response) > 200 else ''}",
            extra={
                "trace_id": self._current_trace.trace_id,
                "interaction_type": "final_response",
                "duration_ms": self._current_trace.processing_time_ms,
                "response_length": len(final_response) if final_response else 0,
                "error": error,
            }
        )

        # Log complete interaction to debug log
        self._debug_logger.debug(
            f"[TRACE_END] trace_id={self._current_trace.trace_id}\n"
            f"Duration: {self._current_trace.processing_time_ms:.1f}ms\n"
            f"Turn: {self._current_trace.turn_index}\n"
            f"Tool Calls: {len(self._current_trace.tool_calls)}\n"
            f"Memory Ops: {len(self._current_trace.memory_ops)}\n"
            f"FINAL_RESPONSE:\n{final_response or 'None'}\n"
            f"{'─' * 40}"
        )

        self._current_trace = None

    # ── Raw Response Logging ──────────────────────────────────────────────────

    def log_raw_response(self, response: str, model: str = "unknown"):
        """
        Log the RAW LLM response BEFORE any processing or parsing.

        This is the key method for testing the first agent response.
        """
        if not self._current_trace:
            return

        self._current_trace.raw_response = response

        # Always log raw response to debug log (this is the key acceptance criteria)
        self._debug_logger.debug(
            f"[RAW_RESPONSE] trace_id={self._current_trace.trace_id}\n"
            f"Model: {model}\n"
            f"Length: {len(response)} chars\n"
            f"{'─' * 40}\n"
            f"{response}\n"
            f"{'─' * 40}"
        )

        # Also log summary to main logger
        self._logger.debug(
            f"RAW_RESPONSE: model={model} length={len(response)}",
            extra={
                "trace_id": self._current_trace.trace_id,
                "interaction_type": "raw_response",
                "model": model,
                "response_length": len(response),
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
        """Log an LLM API call. Returns a call_id for matching with response."""
        call_id = str(uuid.uuid4())[:8]

        self._llm_logger.debug(
            f"LLM_CALL: model={model} messages={len(messages)} max_tokens={max_tokens}",
            extra={
                "call_id": call_id,
                "trace_id": self._current_trace.trace_id if self._current_trace else None,
                "model": model,
                "message_count": len(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "has_tools": has_tools,
            }
        )

        # Log message summary to debug
        # Include message role and content preview for each message
        msg_summaries = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                preview = content[:100] + "..." if len(content) > 100 else content
            else:
                # Check for tool_calls in assistant message
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    preview = f"[tool_calls: {len(tool_calls)}]"
                else:
                    preview = "[empty]"
            msg_summaries.append(f"  [{i}] {role}: {preview}")

        self._debug_logger.debug(
            f"[LLM_CALL] call_id={call_id} model={model}\n"
            f"Messages: {len(messages)}\n"
            f"Max tokens: {max_tokens}\n"
            f"Temperature: {temperature}\n"
            f"Tools: {has_tools}\n"
            f"Message details:\n" + "\n".join(msg_summaries)
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
        """Log an LLM API response."""
        self._llm_logger.debug(
            f"LLM_RESPONSE: call_id={call_id} latency={latency_ms:.1f}ms tokens={token_count}",
            extra={
                "call_id": call_id,
                "trace_id": self._current_trace.trace_id if self._current_trace else None,
                "latency_ms": latency_ms,
                "token_count": token_count,
                "response_length": len(response) if response else 0,
                "error": error,
                "model": model,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

        # Log detailed response info to debug log
        is_empty = not response or not response.strip()
        self._debug_logger.debug(
            f"[LLM_RESPONSE] call_id={call_id} model={model}\n"
            f"Latency: {latency_ms:.1f}ms\n"
            f"Tokens: prompt={prompt_tokens} completion={completion_tokens}\n"
            f"Finish reason: {finish_reason}\n"
            f"Response length: {len(response) if response else 0} chars\n"
            f"Is empty: {is_empty}"
        )

        # Log raw response for testing
        if response:
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
    ):
        """Log a tool call execution."""
        if self._current_trace:
            self._current_trace.tool_calls.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "duration_ms": duration_ms,
                "success": success,
                "error": error,
            })

        self._tool_logger.debug(
            f"TOOL_CALL: {tool_name} success={success}",
            extra={
                "trace_id": self._current_trace.trace_id if self._current_trace else None,
                "tool_name": tool_name,
                "arguments": arguments,
                "duration_ms": duration_ms,
                "success": success,
                "error": error,
            }
        )

        self._debug_logger.debug(
            f"[TOOL_CALL] tool={tool_name} success={success}\n"
            f"Arguments: {json.dumps(arguments, ensure_ascii=False)[:200]}\n"
            f"Duration: {duration_ms:.1f}ms\n"
            f"Result: {(result or 'None')[:200]}{'...' if result and len(result) > 200 else ''}\n"
            f"Error: {error or 'None'}"
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

        self._memory_logger.debug(
            f"MEMORY_OP: {op_type} success={success}",
            extra={
                "trace_id": self._current_trace.trace_id if self._current_trace else None,
                "op_type": op_type,
                "detail": detail[:200] if detail else None,
                "success": success,
                "trigger": trigger,
            }
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
    _tracer = None