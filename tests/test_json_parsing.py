"""
tests/test_json_parsing.py
──────────────────────────
Unit tests for robust JSON extraction from various LLM output formats.

Coverage
────────
J01  Plain JSON response (OpenAI native)
J02  JSON in markdown code block (llama.cpp style)
J03  JSON preceded by thinking tags (DeepSeek-R1, Qwen3)
J04  JSON in output wrapper tags
J05  Trailing comma repair
J06  Unquoted property names repair
J07  Single quotes to double quotes repair
J08  Comments in JSON
J09  Nested JSON structures
J10  JSON array extraction
J11  Empty or invalid input raises error
J12  Mixed content with JSON at end
J13  /no_think injection for thinking models
J14  Non-thinking models are not modified
J15  Already has /no_think is not duplicated
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import directly from core.json_utils - the single source of truth
from core.json_utils import extract_json, JSONParseError
from agents.llm_client import LLMClient


class TestExtractJSON(unittest.TestCase):

    # ── J01: Plain JSON ────────────────────────────────────────────────────────

    def test_J01_plain_json(self):
        """OpenAI-style native JSON response."""
        text = '{"name": "Alice", "age": 30, "active": true}'
        result = extract_json(text)
        self.assertEqual(result["name"], "Alice")
        self.assertEqual(result["age"], 30)
        self.assertTrue(result["active"])

    # ── J02: Markdown code blocks ──────────────────────────────────────────────

    def test_J02_json_in_code_block(self):
        """llama.cpp style: JSON wrapped in markdown code block."""
        text = '''Here's the response:
```json
{"status": "ok", "count": 42}
```
Hope that helps!'''
        result = extract_json(text)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 42)

    def test_J02_code_block_no_language(self):
        """Code block without language specifier."""
        text = '''```
{"result": "success"}
```'''
        result = extract_json(text)
        self.assertEqual(result["result"], "success")

    # ── J03: Thinking/reasoning tags ───────────────────────────────────────────

    def test_J03_thinking_tags_deepseek(self):
        """DeepSeek-R1 style: 󿰌...󿿿 tags before JSON."""
        text = '''Let me analyze this...
First, I need to check the data structure.
The answer is clear.

{"answer": "Paris", "confidence": 0.95}'''
        result = extract_json(text)
        self.assertEqual(result["answer"], "Paris")

    def test_J03_thinking_tag_variant(self):
        """Alternative thinking tag format."""
        text = '''<thinking>
Analyzing the question...
The response should be JSON.
</thinking>
{"value": 123}'''
        result = extract_json(text)
        self.assertEqual(result["value"], 123)

    def test_J03_reasoning_tags(self):
        """Reasoning tags variant."""
        text = '''<reasoning>
Step 1: Parse input
Step 2: Calculate result
</reasoning>
{"steps": 2, "result": "done"}'''
        result = extract_json(text)
        self.assertEqual(result["steps"], 2)

    # ── J04: Output wrapper tags ────────────────────────────────────────────────

    def test_J04_output_wrapper(self):
        """Output tag wrapper."""
        text = '''<output>
{"data": "wrapped"}
</output>'''
        result = extract_json(text)
        self.assertEqual(result["data"], "wrapped")

    def test_J04_response_wrapper(self):
        """Response tag wrapper."""
        text = '''<response>{"status": "complete"}</response>'''
        result = extract_json(text)
        self.assertEqual(result["status"], "complete")

    def test_J04_json_wrapper(self):
        """JSON tag wrapper."""
        text = '''Some preamble text
<json>
{"wrapped": true}
</json>
Some trailing text'''
        result = extract_json(text)
        self.assertTrue(result["wrapped"])

    # ── J05-J08: JSON repair ───────────────────────────────────────────────────

    def test_J05_trailing_comma(self):
        """Trailing comma before closing brace/bracket."""
        text = '{"items": [1, 2, 3,], "name": "test",}'
        result = extract_json(text)
        self.assertEqual(result["items"], [1, 2, 3])
        self.assertEqual(result["name"], "test")

    def test_J06_unquoted_keys(self):
        """Unquoted property names."""
        text = '{name: "John", age: 25, active: true}'
        result = extract_json(text)
        self.assertEqual(result["name"], "John")
        self.assertEqual(result["age"], 25)

    def test_J07_single_quotes(self):
        """Single quotes instead of double quotes."""
        text = "{'name': 'Alice', 'value': 42}"
        result = extract_json(text)
        self.assertEqual(result["name"], "Alice")
        self.assertEqual(result["value"], 42)

    def test_J08_javascript_comments(self):
        """JavaScript-style comments in JSON."""
        text = '''{
    // This is a comment
    "name": "test",  /* inline comment */
    "value": 123
}'''
        result = extract_json(text)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 123)

    # ── J09: Nested structures ─────────────────────────────────────────────────

    def test_J09_nested_objects(self):
        """Deeply nested JSON objects."""
        text = '{"outer": {"middle": {"inner": {"value": "deep"}}}}'
        result = extract_json(text)
        self.assertEqual(result["outer"]["middle"]["inner"]["value"], "deep")

    def test_J09_nested_with_strings_containing_braces(self):
        """Strings containing brace characters."""
        text = '{"code": "function() { return { x: 1 }; }", "name": "test"}'
        result = extract_json(text)
        self.assertIn("{ return { x: 1 }", result["code"])
        self.assertEqual(result["name"], "test")

    # ── J10: JSON arrays ────────────────────────────────────────────────────────

    def test_J10_json_array(self):
        """JSON array as root."""
        text = '[{"id": 1}, {"id": 2}, {"id": 3}]'
        result = extract_json(text)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1]["id"], 2)

    def test_J10_array_in_mixed_content(self):
        """Array extraction from mixed content."""
        text = '''Here are the results:
```json
[1, 2, 3, 4, 5]
```
That's all.'''
        result = extract_json(text)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    # ── J11: Error cases ───────────────────────────────────────────────────────

    def test_J11_empty_input(self):
        """Empty input raises error."""
        with self.assertRaises(JSONParseError) as ctx:
            extract_json("")
        self.assertIn("empty", ctx.exception.reason)

    def test_J11_no_json_found(self):
        """Text with no JSON raises error."""
        with self.assertRaises(JSONParseError) as ctx:
            extract_json("This is just plain text with no JSON structure.")
        self.assertIn("no valid JSON", ctx.exception.reason)

    def test_J11_whitespace_only(self):
        """Whitespace only raises error."""
        with self.assertRaises(JSONParseError):
            extract_json("   \n\t  ")

    # ── J12: Mixed content ──────────────────────────────────────────────────────

    def test_J12_json_after_prose(self):
        """JSON appears after explanatory text."""
        text = '''I've analyzed your request and here's what I found.

Based on the input parameters, the result is:
{"found": true, "count": 5, "items": ["a", "b"]}'''
        result = extract_json(text)
        self.assertTrue(result["found"])
        self.assertEqual(result["count"], 5)

    def test_J12_json_with_surrounding_text(self):
        """JSON surrounded by text on both sides."""
        text = '''Before JSON.
{"key": "value"}
After JSON.'''
        result = extract_json(text)
        self.assertEqual(result["key"], "value")

    # ── Edge cases ──────────────────────────────────────────────────────────────

    def test_escaped_quotes_in_strings(self):
        """Strings with escaped quotes."""
        text = '{"message": "He said \\"hello\\"", "count": 1}'
        result = extract_json(text)
        self.assertIn('hello', result["message"])

    def test_unicode_content(self):
        """Unicode characters in JSON."""
        text = '{"greeting": "Привет мир", "emoji": "🎉"}'
        result = extract_json(text)
        self.assertEqual(result["greeting"], "Привет мир")

    def test_multiline_string_values(self):
        """JSON with newline in string value."""
        text = '''{"text": "line1\\nline2\\nline3", "ok": true}'''
        result = extract_json(text)
        self.assertIn("line1", result["text"])
        self.assertTrue(result["ok"])

    def test_disabling_repair(self):
        """Repair can be disabled."""
        # This would normally be repaired
        text = '{"items": [1, 2, 3,],}'
        # With repair=False, this should fail
        with self.assertRaises(JSONParseError):
            extract_json(text, repair=False)

    # ── J13-J15: /no_think injection ───────────────────────────────────────────

    def test_J13_no_think_injection_for_qwen(self):
        """Qwen models get /no_think injected for JSON mode."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"ok": true}'))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5)
        )
        llm = LLMClient(mock_client, default_model="Qwen2.5-7B-Instruct")

        llm.chat([{"role": "user", "content": "Return JSON"}], json_mode=True)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # Should have /no_think appended
        self.assertIn("/no_think", messages[0]["content"])

    def test_J13_no_think_injection_for_deepseek(self):
        """DeepSeek models get /no_think injected for JSON mode."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"ok": true}'))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5)
        )
        llm = LLMClient(mock_client, default_model="deepseek-r1-70b")

        llm.chat([{"role": "user", "content": "Give me JSON"}], json_mode=True)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        self.assertIn("/no_think", messages[0]["content"])

    def test_J14_non_thinking_model_not_modified(self):
        """Non-thinking models don't get /no_think injected."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"ok": true}'))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5)
        )
        llm = LLMClient(mock_client, default_model="llama-3.1-70b")

        llm.chat([{"role": "user", "content": "Return JSON"}], json_mode=True)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # Should NOT have /no_think appended
        self.assertNotIn("/no_think", messages[0]["content"])

    def test_J15_no_duplicate_no_think(self):
        """If /no_think is already present, don't add it again."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"ok": true}'))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5)
        )
        llm = LLMClient(mock_client, default_model="Qwen3-8B")

        llm.chat([{"role": "user", "content": "Return JSON /no_think"}], json_mode=True)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # Should only appear once
        self.assertEqual(messages[0]["content"].count("/no_think"), 1)

    def test_J15_disable_thinking_injection(self):
        """Can disable automatic /no_think injection via constructor."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"ok": true}'))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5)
        )
        llm = LLMClient(
            mock_client,
            default_model="Qwen3-8B",
            disable_thinking_for_json=False
        )

        llm.chat([{"role": "user", "content": "Return JSON"}], json_mode=True)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # Should NOT have /no_think since we disabled it
        self.assertNotIn("/no_think", messages[0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)