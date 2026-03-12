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


class TextToolCall:
    """Represents a tool call parsed from text (for models that don't use API tool calling)."""

    def __init__(self, name: str, arguments: dict):
        self.id = f"text_tool_{int(time.time() * 1000)}"
        self.function = type('Function', (), {
            'name': name,
            'arguments': arguments if isinstance(arguments, dict) else json.loads(arguments) if isinstance(arguments, str) else {}
        })()


class TextToolCallResponse(Exception):
    """Raised when a tool call is detected in text output (not via API)."""

    def __init__(self, tool_call: TextToolCall):
        self.tool_call = tool_call
        super().__init__(f"Text tool call: {tool_call.function.name}")


# Known tool names that the model might call
KNOWN_TOOL_NAMES = {
    "web_search", "fetch_url", "write_file", "ingest_document",
    "list_documents", "search_metadata", "grep_corpus", "read_document", "read_lines"
}


def parse_text_tool_call(text: str) -> TextToolCall | None:
    """
    Parse a tool call from text output (for models that don't use API tool calling).

    Detects various formats:
    - {"tool": "name", "args": {...}}
    - {"name": "tool_name", "arguments": {...}}
    - {"function": {"name": "...", "arguments": {...}}}
    - Code blocks with tool call JSON

    Returns TextToolCall if found, None otherwise.
    """
    if not text or not text.strip():
        return None

    # Try to extract JSON from the text
    try:
        # First try direct parse
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting from code blocks or wrappers
        cleaned = _strip_wrappers(text)
        json_str = _find_json_string(cleaned)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try repair
            try:
                repaired = _repair_json(json_str)
                data = json.loads(repaired)
            except (json.JSONDecodeError, TypeError):
                return None

    if not isinstance(data, dict):
        return None

    # Extract tool name and arguments from various formats
    tool_name = None
    tool_args = {}

    # Format 1: {"tool": "name", "args": {...}} or {"tool": "name", "arguments": {...}}
    if "tool" in data:
        tool_name = data.get("tool")
        tool_args = data.get("args") or data.get("arguments") or {}

    # Format 2: {"name": "tool_name", "arguments": {...}}
    elif "name" in data:
        tool_name = data.get("name")
        tool_args = data.get("arguments") or {}

    # Format 3: {"function": {"name": "...", "arguments": {...}}}
    elif "function" in data and isinstance(data["function"], dict):
        func = data["function"]
        tool_name = func.get("name")
        tool_args = func.get("arguments") or {}
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {}

    # Format 4: {"tool_name": {...}} where key is a known tool name
    else:
        for key in data:
            if key in KNOWN_TOOL_NAMES:
                tool_name = key
                tool_args = data[key] if isinstance(data[key], dict) else {}
                break

    # Validate tool name
    if not tool_name or not isinstance(tool_name, str):
        return None

    # Normalize tool name
    tool_name = tool_name.lower().strip()

    # Check if it's a known tool
    if tool_name not in KNOWN_TOOL_NAMES:
        return None

    # Ensure arguments is a dict
    if not isinstance(tool_args, dict):
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {}
        else:
            tool_args = {}

    return TextToolCall(name=tool_name, arguments=tool_args)


def detect_text_tool_calls(text: str) -> list[TextToolCall]:
    """
    Detect all tool calls in text output.

    Handles multiple tool calls in a single response.
    Returns a list of TextToolCall objects found.
    """
    if not text or not text.strip():
        return []

    calls = []

    # First, try to find all JSON objects using brace matching
    # This handles nested objects correctly
    def find_all_json_objects(s: str) -> list[str]:
        """Find all JSON objects in a string using brace matching."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                # Found start of object, find matching end
                depth = 0
                in_string = False
                escape_next = False
                start = i
                for j in range(i, len(s)):
                    c = s[j]
                    if escape_next:
                        escape_next = False
                        continue
                    if c == '\\' and in_string:
                        escape_next = True
                        continue
                    if c == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            objects.append(s[start:j + 1])
                            i = j + 1
                            break
                else:
                    i += 1
            else:
                i += 1
        return objects

    # Find all JSON objects in the text
    json_objects = find_all_json_objects(text)

    for json_str in json_objects:
        call = parse_text_tool_call(json_str)
        if call:
            calls.append(call)

    # Also check for arrays of tool calls
    if '"tool_calls"' in text or '"tools"' in text:
        try:
            cleaned = _strip_wrappers(text)
            json_str = _find_json_string(cleaned)
            if json_str:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    tc_list = data.get("tool_calls") or data.get("tools") or []
                    if isinstance(tc_list, list):
                        for tc in tc_list:
                            if isinstance(tc, dict):
                                call = parse_text_tool_call(json.dumps(tc))
                                if call:
                                    calls.append(call)
        except (json.JSONDecodeError, TypeError):
            pass

    return calls


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

                content = message.content or ""

                # Check for text-based tool calls (for models that don't use API tool calling)
                # This handles models like GLM-5 that output tool calls as JSON in text
                if tools:
                    text_tool_calls = detect_text_tool_calls(content)
                    if text_tool_calls:
                        # Raise for the first detected tool call
                        raise TextToolCallResponse(text_tool_calls[0])

                return content
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
