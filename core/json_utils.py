"""
core/json_utils.py
──────────────────
JSON extraction, parsing, and repair utilities.

Handles JSON from LLM outputs that may contain mixed content,
wrappers, markdown blocks, or minor syntax errors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class JSONParseError(Exception):
    """Raised when JSON cannot be extracted from LLM output."""

    raw: str
    reason: str

    def __str__(self) -> str:
        return f"JSON parse failed ({self.reason}): {self.raw[:200]}..."


def strip_wrappers(text: str) -> str:
    """
    Remove common LLM output wrappers.

    Handles:
    - Thinking/reasoning tags (DeepSeek-R1, Qwen3, etc.)
    - Output wrapper tags (<output>, <response>, <json>)
    - Markdown code blocks (```json ... ```)
    """
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


def find_json_string(text: str) -> str | None:
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


def repair_json(text: str) -> str:
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
    cleaned = strip_wrappers(text)

    # Try to find JSON in the cleaned text
    json_str = find_json_string(cleaned)

    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            if repair:
                try:
                    repaired = repair_json(json_str)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

    # Last resort: try repairing the whole cleaned text
    if repair:
        try:
            repaired = repair_json(cleaned)
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    raise JSONParseError(text, "no valid JSON found")


def safe_parse_json(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with a fallback default.

    Parameters
    ----------
    text : str
        JSON string to parse
    default : Any
        Default value to return on parse failure

    Returns
    -------
    Any
        Parsed JSON or default value
    """
    if not text or not text.strip():
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def find_all_json_objects(text: str) -> list[str]:
    """
    Find all JSON objects in a string using brace matching.

    Handles nested objects correctly.
    """
    objects = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Found start of object, find matching end
            depth = 0
            in_string = False
            escape_next = False
            start = i
            for j in range(i, len(text)):
                c = text[j]
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
                        objects.append(text[start:j + 1])
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return objects


def is_valid_json(text: str) -> bool:
    """Check if text is valid JSON without raising exceptions."""
    if not text or not text.strip():
        return False
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False
