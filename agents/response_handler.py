"""
Enhanced response handler for AgeMem agent.

Provides robust handling of LLM responses including:
- Tool call validation and sanitization
- JSON response parsing with fallbacks
- Error recovery mechanisms
- Response quality monitoring
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from agents.llm_client import LLMClient, ToolCallResponse, TextToolCallResponse, JSONParseError
from core.tracing import get_tracer
from core.json_utils import repair_json, find_json_string, strip_wrappers


class ResponseType(Enum):
    """Types of LLM responses."""
    TEXT = "text"
    JSON = "json"
    TOOL_CALL = "tool_call"
    ERROR = "error"


@dataclass
class ResponseMetrics:
    """Metrics for response quality monitoring."""
    response_type: ResponseType
    latency_ms: float
    token_count: int
    has_tool_calls: bool
    json_valid: bool
    error_count: int = 0
    recovery_attempts: int = 0
    quality_score: float = 1.0
    # Additional diagnostic fields
    model: str = "unknown"
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    is_empty_response: bool = False

    def __post_init__(self):
        """Calculate quality score based on errors and recovery attempts."""
        if self.error_count > 0 or self.recovery_attempts > 0:
            # Reduce quality score based on errors and recovery attempts
            penalty = (self.error_count * 0.3) + (self.recovery_attempts * 0.2)
            self.quality_score = max(0.0, 1.0 - penalty)
        # Further reduce quality for empty responses
        if self.is_empty_response:
            self.quality_score = min(self.quality_score, 0.1)


@dataclass
class ToolCallValidation:
    """Validation result for a tool call."""
    is_valid: bool
    tool_name: str
    arguments: dict
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ResponseHandler:
    """
    Enhanced response handler with robust error recovery and validation.
    
    Features:
    - Validates tool call arguments before execution
    - Provides fallback mechanisms for malformed responses
    - Tracks response quality metrics
    - Implements retry logic with exponential backoff
    """

    def __init__(
        self,
        llm: LLMClient,
        max_retries: int = 3,
        enable_validation: bool = True,
    ):
        self._llm = llm
        self._max_retries = max_retries
        self._enable_validation = enable_validation
        self._metrics_history: list[ResponseMetrics] = []

    def get_metrics(self) -> list[ResponseMetrics]:
        """Get response metrics history."""
        return list(self._metrics_history)

    def get_average_quality(self) -> float:
        """Get average response quality score."""
        if not self._metrics_history:
            return 1.0
        return sum(m.quality_score for m in self._metrics_history) / len(self._metrics_history)

    def validate_tool_call(self, tool_call: Any) -> ToolCallValidation:
        """
        Validate a tool call and its arguments.
        
        Args:
            tool_call: The tool call object from the LLM
            
        Returns:
            ToolCallValidation with validation results
        """
        errors = []
        warnings = []
        
        # Extract tool name
        tool_name = getattr(tool_call.function, 'name', None)
        if not tool_name:
            errors.append("Tool name is missing")
            return ToolCallValidation(
                is_valid=False,
                tool_name="",
                arguments={},
                errors=errors,
            )
        
        # Extract and validate arguments
        raw_args = getattr(tool_call.function, 'arguments', None)
        arguments = {}
        
        if raw_args is None:
            warnings.append("No arguments provided")
        elif isinstance(raw_args, dict):
            arguments = raw_args
        elif isinstance(raw_args, str):
            # Try to parse JSON string
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as e:
                # Try to repair common JSON issues
                repaired = self._repair_json_arguments(raw_args)
                if repaired:
                    try:
                        arguments = json.loads(repaired)
                        warnings.append(f"Arguments JSON was repaired: {str(e)[:100]}")
                    except json.JSONDecodeError:
                        errors.append(f"Invalid JSON in arguments: {str(e)[:200]}")
                else:
                    errors.append(f"Invalid JSON in arguments: {str(e)[:200]}")
        else:
            errors.append(f"Arguments must be dict or string, got {type(raw_args)}")
        
        # Validate argument types
        if arguments and self._enable_validation:
            for key, value in arguments.items():
                if not isinstance(key, str):
                    errors.append(f"Argument key must be string, got {type(key)}")
                if value is None:
                    warnings.append(f"Argument '{key}' is None")
        
        is_valid = len(errors) == 0
        
        return ToolCallValidation(
            is_valid=is_valid,
            tool_name=tool_name,
            arguments=arguments,
            errors=errors,
            warnings=warnings,
        )

    def _repair_json_arguments(self, json_str: str) -> Optional[str]:
        """
        Attempt to repair common JSON formatting issues in tool arguments.

        Uses core.json_utils for consistent JSON repair across the codebase.

        Args:
            json_str: The potentially malformed JSON string

        Returns:
            Repaired JSON string or None if repair failed
        """
        if not json_str or not json_str.strip():
            return None

        # Strip wrappers (markdown code blocks, etc.)
        cleaned = strip_wrappers(json_str)

        # Find the JSON object in the cleaned text
        json_obj_str = find_json_string(cleaned)
        if not json_obj_str:
            return None

        # Apply comprehensive JSON repair
        repaired = repair_json(json_obj_str)

        # Clean up extra whitespace
        repaired = ' '.join(repaired.split())

        return repaired.strip()

    def chat_with_recovery(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        tools: Optional[list[dict]] = None,
        timeout: float = 300.0,
    ) -> tuple[str, ResponseMetrics]:
        """
        Send a chat request with enhanced error recovery.
        
        Returns:
            Tuple of (response_text, metrics)
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self._max_retries + 1):
            try:
                response = self._llm.chat(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=json_mode,
                    tools=tools,
                    timeout=timeout,
                )
                
                # Calculate metrics
                latency_ms = (time.time() - start_time) * 1000
                metrics = ResponseMetrics(
                    response_type=ResponseType.JSON if json_mode else ResponseType.TEXT,
                    latency_ms=latency_ms,
                    token_count=len(response.split()) if response else 0,
                    has_tool_calls=False,
                    json_valid=True,
                    recovery_attempts=attempt,
                )
                
                self._metrics_history.append(metrics)
                return response, metrics
                
            except ToolCallResponse as e:
                # Validate the tool call
                validation = self.validate_tool_call(e.tool_call)

                # TRACE: Log raw tool call details for debugging
                tracer = get_tracer()
                raw_args = getattr(e.tool_call.function, 'arguments', None)
                tracer._debug_logger.debug(
                    f"[TOOL_CALL_DETECTED] type=ToolCallResponse\n"
                    f"Tool name: {getattr(e.tool_call.function, 'name', 'UNKNOWN')}\n"
                    f"Raw arguments: {raw_args!r}\n"
                    f"Validation errors: {validation.errors}\n"
                    f"Validation warnings: {validation.warnings}\n"
                    f"Is valid: {validation.is_valid}"
                )

                if not validation.is_valid:
                    # Tool call is invalid, try to recover
                    if attempt < self._max_retries:
                        # Add error context to messages and retry
                        error_msg = (
                            f"[SYSTEM] Tool call validation failed: {'; '.join(validation.errors)}. "
                            f"Please provide a valid tool call or respond with text instead."
                        )
                        messages.append({"role": "user", "content": error_msg})
                        continue
                    else:
                        # Max retries reached, return error
                        latency_ms = (time.time() - start_time) * 1000
                        metrics = ResponseMetrics(
                            response_type=ResponseType.ERROR,
                            latency_ms=latency_ms,
                            token_count=0,
                            has_tool_calls=True,
                            json_valid=False,
                            error_count=1,
                            recovery_attempts=attempt,
                            quality_score=0.0,
                        )
                        self._metrics_history.append(metrics)
                        raise RuntimeError(f"Tool call validation failed after {attempt + 1} attempts: {'; '.join(validation.errors)}")
                
                # Tool call is valid, re-raise for orchestrator to handle
                latency_ms = (time.time() - start_time) * 1000
                metrics = ResponseMetrics(
                    response_type=ResponseType.TOOL_CALL,
                    latency_ms=latency_ms,
                    token_count=0,
                    has_tool_calls=True,
                    json_valid=True,
                    recovery_attempts=attempt,
                )
                self._metrics_history.append(metrics)
                raise
                
            except TextToolCallResponse as e:
                # Similar handling for text-based tool calls
                validation = self.validate_tool_call(e.tool_call)

                # TRACE: Log raw tool call details for debugging
                tracer = get_tracer()
                raw_args = getattr(e.tool_call.function, 'arguments', None)
                tracer._debug_logger.debug(
                    f"[TOOL_CALL_DETECTED] type=TextToolCallResponse\n"
                    f"Tool name: {getattr(e.tool_call.function, 'name', 'UNKNOWN')}\n"
                    f"Raw arguments: {raw_args!r}\n"
                    f"Validation errors: {validation.errors}\n"
                    f"Validation warnings: {validation.warnings}\n"
                    f"Is valid: {validation.is_valid}"
                )

                if not validation.is_valid:
                    if attempt < self._max_retries:
                        error_msg = (
                            f"[SYSTEM] Tool call validation failed: {'; '.join(validation.errors)}. "
                            f"Please provide a valid tool call or respond with text instead."
                        )
                        messages.append({"role": "user", "content": error_msg})
                        continue
                    else:
                        latency_ms = (time.time() - start_time) * 1000
                        metrics = ResponseMetrics(
                            response_type=ResponseType.ERROR,
                            latency_ms=latency_ms,
                            token_count=0,
                            has_tool_calls=True,
                            json_valid=False,
                            error_count=1,
                            recovery_attempts=attempt,
                            quality_score=0.0,
                        )
                        self._metrics_history.append(metrics)
                        raise RuntimeError(f"Text tool call validation failed after {attempt + 1} attempts: {'; '.join(validation.errors)}")
                
                latency_ms = (time.time() - start_time) * 1000
                metrics = ResponseMetrics(
                    response_type=ResponseType.TOOL_CALL,
                    latency_ms=latency_ms,
                    token_count=0,
                    has_tool_calls=True,
                    json_valid=True,
                    recovery_attempts=attempt,
                )
                self._metrics_history.append(metrics)
                raise
                
            except JSONParseError as e:
                # JSON parsing failed
                last_error = e

                # TRACE: Log the raw response that caused JSON parse error
                tracer = get_tracer()
                tracer._debug_logger.debug(
                    f"[JSON_PARSE_ERROR] reason={e.reason}\n"
                    f"Raw response: {e.raw!r}"
                )

                if attempt < self._max_retries:
                    # Try to recover by asking for valid JSON
                    error_msg = (
                        f"[SYSTEM] JSON parsing failed: {e.reason}. "
                        f"Please respond with valid JSON only."
                    )
                    messages.append({"role": "user", "content": error_msg})
                    continue
                else:
                    latency_ms = (time.time() - start_time) * 1000
                    metrics = ResponseMetrics(
                        response_type=ResponseType.ERROR,
                        latency_ms=latency_ms,
                        token_count=0,
                        has_tool_calls=False,
                        json_valid=False,
                        error_count=1,
                        recovery_attempts=attempt,
                        quality_score=0.0,
                    )
                    self._metrics_history.append(metrics)
                    raise
                    
            except Exception as e:
                # Other errors
                last_error = e
                if attempt < self._max_retries:
                    time.sleep(1.5 ** attempt)  # Exponential backoff
                    continue
                else:
                    latency_ms = (time.time() - start_time) * 1000
                    metrics = ResponseMetrics(
                        response_type=ResponseType.ERROR,
                        latency_ms=latency_ms,
                        token_count=0,
                        has_tool_calls=False,
                        json_valid=False,
                        error_count=1,
                        recovery_attempts=attempt,
                        quality_score=0.0,
                    )
                    self._metrics_history.append(metrics)
                    raise RuntimeError(f"LLM call failed after {attempt + 1} attempts: {last_error}")
        
        # Should not reach here
        raise RuntimeError("Unexpected error in chat_with_recovery")

    def chat_json_with_recovery(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 512,
        repair: bool = True,
    ) -> tuple[dict, ResponseMetrics]:
        """
        Request JSON output with enhanced error recovery.
        
        Returns:
            Tuple of (parsed_json, metrics)
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self._max_retries + 1):
            try:
                result = self._llm.chat_json(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    repair=repair,
                )
                
                latency_ms = (time.time() - start_time) * 1000
                metrics = ResponseMetrics(
                    response_type=ResponseType.JSON,
                    latency_ms=latency_ms,
                    token_count=len(json.dumps(result)) if result else 0,
                    has_tool_calls=False,
                    json_valid=True,
                    recovery_attempts=attempt,
                )
                
                self._metrics_history.append(metrics)
                return result, metrics
                
            except JSONParseError as e:
                last_error = e
                if attempt < self._max_retries:
                    # Try to recover by asking for valid JSON
                    error_msg = (
                        f"[SYSTEM] JSON parsing failed: {e.reason}. "
                        f"Please respond with valid JSON only, no other text."
                    )
                    messages.append({"role": "user", "content": error_msg})
                    continue
                else:
                    latency_ms = (time.time() - start_time) * 1000
                    metrics = ResponseMetrics(
                        response_type=ResponseType.ERROR,
                        latency_ms=latency_ms,
                        token_count=0,
                        has_tool_calls=False,
                        json_valid=False,
                        error_count=1,
                        recovery_attempts=attempt,
                        quality_score=0.0,
                    )
                    self._metrics_history.append(metrics)
                    raise
                    
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    time.sleep(1.5 ** attempt)
                    continue
                else:
                    latency_ms = (time.time() - start_time) * 1000
                    metrics = ResponseMetrics(
                        response_type=ResponseType.ERROR,
                        latency_ms=latency_ms,
                        token_count=0,
                        has_tool_calls=False,
                        json_valid=False,
                        error_count=1,
                        recovery_attempts=attempt,
                        quality_score=0.0,
                    )
                    self._metrics_history.append(metrics)
                    raise RuntimeError(f"JSON request failed after {attempt + 1} attempts: {last_error}")
        
        raise RuntimeError("Unexpected error in chat_json_with_recovery")