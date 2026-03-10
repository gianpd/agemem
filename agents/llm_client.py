"""
agents/llm_client.py
─────────────────────
Thin wrapper around any OpenAI-compatible HTTP endpoint.

Assumptions
───────────
* The caller supplies a pre-configured `openai.OpenAI` (or compatible) client.
* This module contains zero business logic — just sends messages, returns text.
* Structured JSON responses are requested via response_format when needed.

Why a wrapper and not direct openai calls everywhere?
──────────────────────────────────────────────────────
Centralises retry logic, logging, and token-budget enforcement.  Makes it
easy to swap the backend (local Ollama, Azure OpenAI, Anthropic via compat
layer) without touching agent code.

JSON Parsing Support
────────────────────
Handles multiple LLM output formats:
* OpenAI-compatible: native JSON mode with response_format
* llama.cpp: may wrap JSON in code blocks or mix with prose
* Reasoning models (DeepSeek-R1, Qwen3): <think>...</think> tags before JSON
* Models with output tags: <output>...</output> or similar wrappers
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional


class ToolCallResponse(Exception):
    """Raised when the LLM returns a tool call instead of text."""

    def __init__(self, tool_call):
        self.tool_call = tool_call
        super().__init__(f"Tool call: {tool_call.function.name}")


class JSONParseError(Exception):
    """Raised when JSON cannot be extracted from LLM output."""

    def __init__(self, raw: str, reason: str):
        self.raw = raw
        self.reason = reason
        super().__init__(f"JSON parse failed ({reason}): {raw[:200]}...")


def extract_json(text: str, repair: bool = True) -> dict | list:
    """
    Extract and parse JSON from LLM output that may contain mixed content.

    Handles:
    - Plain JSON response
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON preceded by thinking/reasoning tags (<think>...</think>)
    - JSON wrapped in output tags (<output>...</output>)
    - JSON with minor syntax errors (trailing commas, unquoted keys)

    Parameters
    ----------
    text : str
        Raw LLM output text
    repair : bool
        Whether to attempt JSON repair on parse failures

    Returns
    -------
    dict | list
        Parsed JSON object or array

    Raises
    ------
    JSONParseError
        If no valid JSON could be extracted
    """
    if not text or not text.strip():
        raise JSONParseError(text, "empty input")

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip common wrappers and try again
    cleaned = _strip_wrappers(text)

    # Try to find JSON in the cleaned text
    json_str = _find_json_string(cleaned)

    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            if repair:
                try:
                    repaired = _repair_json(json_str)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

    # Last resort: try repairing the whole cleaned text
    if repair:
        try:
            repaired = _repair_json(cleaned)
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    raise JSONParseError(text, "no valid JSON found")


def _strip_wrappers(text: str) -> str:
    """Remove common LLM output wrappers."""
    result = text.strip()

    # Remove thinking/reasoning tags (DeepSeek-R1, Qwen3, etc.)
    # Pattern: <think>...</think> or <thinking>...</thinking>
    think_patterns = [
        r'<think>.*?</think>\s*',
        r'<thinking>.*?</thinking>\s*',
        r'<reasoning>.*?</reasoning>\s*',
    ]
    for pattern in think_patterns:
        result = re.sub(pattern, '', result, flags=re.DOTALL | re.IGNORECASE)

    # Remove output wrapper tags
    output_patterns = [
        r'<output>\s*(.*?)\s*</output>',
        r'<response>\s*(.*?)\s*</response>',
        r'<json>\s*(.*?)\s*</json>',
    ]
    for pattern in output_patterns:
        match = re.search(pattern, result, re.DOTALL | re.IGNORECASE)
        if match:
            result = match.group(1)
            break

    # Strip markdown code blocks
    # Pattern: ```json ... ``` or ``` ... ```
    code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result, re.IGNORECASE)
    if code_block:
        result = code_block.group(1)

    return result.strip()


def _find_json_string(text: str) -> str | None:
    """
    Find a JSON object or array in text that may contain other content.

    Uses brace/bracket matching to handle nested structures.
    """
    text = text.strip()

    # Look for object or array start
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue

        # Track nesting depth
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]

    return None


def _repair_json(text: str) -> str:
    """
    Attempt to repair common JSON syntax errors.

    Handles:
    - Trailing commas before ] or }
    - Unquoted property names
    - Single quotes instead of double quotes
    - Missing quotes around string values
    - Comments (// and /* */)
    """
    result = text

    # Remove JavaScript-style comments
    result = re.sub(r'//.*$', '', result, flags=re.MULTILINE)
    result = re.sub(r'/\*.*?\*/', '', result, flags=re.DOTALL)

    # Remove trailing commas before ] or }
    result = re.sub(r',\s*([}\]])', r'\1', result)

    # Quote unquoted property names
    # Pattern: {name: or ,name: or [name: (where name is not already quoted)
    def quote_unquoted_key(match):
        prefix = match.group(1)  # The { or , before the key
        key = match.group(2)     # The unquoted key name
        return f'{prefix}"{key}":'

    result = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', quote_unquoted_key, result)

    # Convert single quotes to double quotes (carefully)
    # This is a simplified approach - may not handle all edge cases
    def fix_quotes(match):
        content = match.group(1)
        # Escape any double quotes inside
        content = content.replace('\\"', '"').replace('"', '\\"')
        return f'"{content}"'

    # Match single-quoted strings (simplified)
    result = re.sub(r"'([^']*(?:\\'[^']*)*)'", fix_quotes, result)

    return result


class LLMClient:

    # Models known to emit thinking/reasoning tags that break JSON grammar
    THINKING_MODELS = {
        "qwen",      # Qwen2.5, Qwen3, Qwen3.5
        "deepseek",  # DeepSeek-R1, DeepSeek-V3
        "r1",        # DeepSeek R1
    }

    def __init__(
        self,
        client: Any,
        default_model: str,
        default_temperature: float = 0.2,
        disable_thinking_for_json: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        client : openai.OpenAI or any compatible object with
                 client.chat.completions.create(**kwargs)
        default_model : str
        default_temperature : float
        disable_thinking_for_json : bool
            For models with thinking mode (Qwen3, DeepSeek-R1), automatically
            inject /no_think directive when requesting JSON output. This prevents
            thinking tags from breaking llama.cpp's JSON grammar constraint.
        """
        self._client = client
        self._model = default_model
        self._temperature = default_temperature
        self._disable_thinking_for_json = disable_thinking_for_json
        self._total_calls = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    def _is_thinking_model(self, model: str) -> bool:
        """Check if model is known to emit thinking tags."""
        model_lower = model.lower()
        return any(t in model_lower for t in self.THINKING_MODELS)

    def _inject_no_think(self, messages: list[dict]) -> list[dict]:
        """
        Inject /no_think directive to disable thinking mode for JSON responses.

        For Qwen models, adding '/no_think' to the last user message disables
        the 󿰌...󿿿 thinking blocks that would otherwise break JSON grammar.
        """
        if not messages:
            return messages

        # Create a copy to avoid mutating the original
        messages = [m.copy() for m in messages]

        # Find the last user message and append /no_think
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                content = messages[i].get("content", "")
                if "/no_think" not in content:
                    messages[i]["content"] = f"{content} /no_think"
                break

        return messages

    # ── Core call ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        retries: int = 2,
        tools: Optional[list[dict]] = None,
        timeout: float = 300.0,
    ) -> str:
        """
        Send a chat completion request.

        Returns the assistant text content as a plain string.
        Raises RuntimeError after exhausting retries.

        If tools are provided and the model returns a tool call, raises
        ToolCallResponse with the tool call details.
        """
        model = model or self._model
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
            "timeout": timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

            # Disable thinking mode for models that emit 󿰌...󿿿 tags
            # This prevents grammar conflicts in llama.cpp when using JSON mode
            if self._disable_thinking_for_json and self._is_thinking_model(model):
                kwargs["messages"] = self._inject_no_think(messages)

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                self._total_calls += 1
                usage = getattr(response, "usage", None)
                if usage:
                    self._total_tokens_in  += getattr(usage, "prompt_tokens",     0)
                    self._total_tokens_out += getattr(usage, "completion_tokens",  0)
                
                message = response.choices[0].message

                # Check for tool calls
                tool_calls = getattr(message, "tool_calls", None)
                if isinstance(tool_calls, list) and len(tool_calls) > 0:
                    raise ToolCallResponse(tool_calls[0])
                
                return message.content or ""
            except Exception as exc:
                # Don't retry on tool calls - let them propagate
                if isinstance(exc, ToolCallResponse):
                    raise
                last_exc = exc
                if attempt < retries:
                    time.sleep(1.5 ** attempt)

        raise RuntimeError(f"LLM call failed after {retries+1} attempts: {last_exc}")

    def chat_json(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 512,
        repair: bool = True,
    ) -> dict:
        """
        Request JSON output and parse with robust multi-format support.

        Handles output from:
        - OpenAI-compatible APIs (native JSON mode)
        - llama.cpp / Ollama (may wrap in code blocks)
        - Reasoning models (DeepSeek-R1, Qwen3) with 󿰌...󿿿 tags
        - Models using <output> or similar wrapper tags

        For thinking models (Qwen3, DeepSeek-R1), this automatically injects
        /no_think to prevent thinking tags from breaking llama.cpp's JSON
        grammar constraint. Set disable_thinking_for_json=False in constructor
        to disable this behavior.

        Parameters
        ----------
        messages : list[dict]
            Chat messages for the LLM
        model : str, optional
            Model override
        max_tokens : int
            Maximum tokens for response
        repair : bool
            Whether to attempt JSON repair on parse failures (default: True)

        Returns
        -------
        dict
            Parsed JSON object

        Raises
        ------
        JSONParseError
            If JSON cannot be extracted from the response
        RuntimeError
            If the LLM call itself fails after retries
        """
        raw = self.chat(messages, model=model, max_tokens=max_tokens, json_mode=True)
        try:
            return extract_json(raw, repair=repair)
        except JSONParseError:
            # Re-raise with more context
            raise

    # ── Stats ─────────────────────────────────────────────────────────────────

    def usage_stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
        }
