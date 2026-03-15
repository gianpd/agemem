"""
Tests for text cleaning utilities in cli_text_utils.py
"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli_text_utils import (
    clean_pasted_text,
    is_likely_paste,
    get_cleaning_summary,
    INVISIBLE_CHARS,
    SMART_QUOTES,
    DASHES,
    OTHER_PUNCT,
)


class TestTextCleaning(unittest.TestCase):
    """Test the text cleaning utilities for paste handling."""

    def test_basic_text_unchanged(self):
        """Basic text without special characters should remain unchanged."""
        text = "Hello world, this is a test."
        result = clean_pasted_text(text)
        self.assertEqual(result, text)

    def test_empty_and_whitespace_input(self):
        """Empty or whitespace-only input should return empty string."""
        self.assertEqual(clean_pasted_text(""), "")
        self.assertEqual(clean_pasted_text("   "), "")
        self.assertEqual(clean_pasted_text("\n\n\n"), "")
        self.assertEqual(clean_pasted_text("\t\t\t"), "")

    def test_zero_width_characters_removed(self):
        """Zero-width and invisible characters should be removed."""
        # Zero-width space
        text = "Hello\u200bWorld"
        self.assertEqual(clean_pasted_text(text), "HelloWorld")

        # Zero-width non-joiner
        text = "Hello\u200cWorld"
        self.assertEqual(clean_pasted_text(text), "HelloWorld")

        # BOM (Byte Order Mark)
        text = "\ufeffHello World"
        self.assertEqual(clean_pasted_text(text), "Hello World")

        # Multiple invisible chars
        text = "Hello\u200b\u200c\u200dWorld"
        self.assertEqual(clean_pasted_text(text), "HelloWorld")

    def test_smart_quotes_normalized(self):
        """Smart quotes should be normalized to ASCII equivalents."""
        # Left and right double quotes
        text = '\u201cHello World\u201d'
        self.assertEqual(clean_pasted_text(text), '"Hello World"')

        # Left and right single quotes
        text = "\u2018Hello World\u2019"
        self.assertEqual(clean_pasted_text(text), "'Hello World'")

        # Mixed quotes
        text = '\u201cIt\u2019s a \\"test\\"\u201d'
        self.assertEqual(clean_pasted_text(text), '"It\'s a \\"test\\""')

    def test_dashes_normalized(self):
        """Various dash types should be normalized."""
        # En dash
        text = "2020\u20132021"
        self.assertEqual(clean_pasted_text(text), "2020--2021")

        # Em dash
        text = "Hello\u2014World"
        self.assertEqual(clean_pasted_text(text), "Hello---World")

    def test_ellipsis_normalized(self):
        """Ellipsis character should be normalized to three dots."""
        text = "And so on\u2026"
        self.assertEqual(clean_pasted_text(text), "And so on...")

    def test_excessive_empty_lines_collapsed(self):
        """Excessive empty lines should be collapsed to at most 3."""
        text = "Line 1\n\n\n\n\n\nLine 2"
        result = clean_pasted_text(text)
        # Should collapse to at most 3 newlines (4 lines max)
        self.assertEqual(result, "Line 1\n\n\nLine 2")

    def test_trailing_whitespace_removed(self):
        """Trailing whitespace on each line should be removed."""
        text = "Line 1   \nLine 2\t\t\nLine 3"
        result = clean_pasted_text(text)
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    def test_leading_trailing_whitespace_stripped(self):
        """Leading and trailing whitespace on entire text should be stripped."""
        text = "\n\n  Hello World  \n\n"
        result = clean_pasted_text(text)
        self.assertEqual(result, "Hello World")

    def test_carriage_return_normalized(self):
        """Windows-style line endings should be normalized."""
        text = "Line 1\r\nLine 2\r\nLine 3"
        result = clean_pasted_text(text)
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

        # Old Mac style
        text = "Line 1\rLine 2\rLine 3"
        result = clean_pasted_text(text)
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    def test_excessive_punctuation_collapsed(self):
        """Excessive punctuation should be limited."""
        text = "Wow!!!!!"
        result = clean_pasted_text(text)
        self.assertEqual(result, "Wow!!!")

        text = "Really...."
        result = clean_pasted_text(text)
        self.assertEqual(result, "Really...")

        text = "What???"
        result = clean_pasted_text(text)
        self.assertEqual(result, "What???")  # 3 is preserved

    def test_code_blocks_preserved(self):
        """Code blocks with intentional formatting should be preserved."""
        text = """Here's some code:

```python
def hello():
    print("Hello")

    return True
```

More text."""
        result = clean_pasted_text(text)
        # Code structure should be preserved
        self.assertIn("```python", result)
        self.assertIn("def hello():", result)
        self.assertIn("    print", result)  # Indentation preserved

    def test_non_breaking_space_normalized(self):
        """Non-breaking spaces should be normalized to regular spaces."""
        text = "Hello\u00a0World"
        result = clean_pasted_text(text)
        self.assertEqual(result, "Hello World")

    def test_complex_paste_scenario(self):
        """A realistic complex paste with multiple issues."""
        # Simulates text copied from a PDF or web page
        text = """\ufeff\u200b
"Introduction to Python"

by John Doe\u20142024

This is a \u201ctest\u201d document\u2026



With excessive spacing\u00a0above.

"""
        result = clean_pasted_text(text)

        # Should have normalized quotes
        self.assertIn('"Introduction to Python"', result)
        self.assertIn('"test"', result)

        # Should have normalized dash
        self.assertIn("John Doe---2024", result)

        # Should have normalized ellipsis
        self.assertIn("document...", result)

        # Should not have BOM or zero-width chars
        self.assertNotIn("\ufeff", result)
        self.assertNotIn("\u200b", result)

        # Should not have non-breaking space
        self.assertNotIn("\u00a0", result)

        # Should not have trailing whitespace
        self.assertNotIn("   ", result)

        # Should have collapsed excessive newlines
        self.assertNotIn("\n\n\n\n", result)

    def test_is_likely_paste_detects_multiline(self):
        """is_likely_paste should detect multiline text."""
        self.assertTrue(is_likely_paste("Line 1\nLine 2"))
        self.assertTrue(is_likely_paste("Line 1\r\nLine 2"))
        self.assertFalse(is_likely_paste("Single line"))

    def test_is_likely_paste_detects_invisible_chars(self):
        """is_likely_paste should detect invisible characters."""
        self.assertTrue(is_likely_paste("Hello\u200bWorld"))
        self.assertTrue(is_likely_paste("\ufeffTest"))

    def test_is_likely_paste_detects_smart_quotes(self):
        """is_likely_paste should detect smart quotes."""
        self.assertTrue(is_likely_paste('\u201cHello\u201d'))
        self.assertTrue(is_likely_paste("\u2018Hello\u2019"))

    def test_is_likely_paste_empty(self):
        """is_likely_paste should handle empty/None input."""
        self.assertFalse(is_likely_paste(""))
        self.assertFalse(is_likely_paste(None))

    def test_get_cleaning_summary(self):
        """get_cleaning_summary should return accurate statistics."""
        original = "Hello\u200b World\u00a0  "
        cleaned = clean_pasted_text(original)
        summary = get_cleaning_summary(original, cleaned)

        self.assertEqual(summary["original_length"], len(original))
        self.assertEqual(summary["cleaned_length"], len(cleaned))
        self.assertEqual(summary["chars_removed"], len(original) - len(cleaned))
        self.assertTrue(summary["was_cleaned"])

    def test_get_cleaning_summary_no_changes(self):
        """get_cleaning_summary should handle unchanged text."""
        text = "Hello World"
        cleaned = clean_pasted_text(text)
        summary = get_cleaning_summary(text, cleaned)

        self.assertEqual(summary["chars_removed"], 0)
        self.assertFalse(summary["was_cleaned"])


class TestTextCleaningEdgeCases(unittest.TestCase):
    """Edge cases for text cleaning."""

    def test_only_whitespace_lines(self):
        """Text with only whitespace lines should be handled."""
        text = "   \n\n   \n   "
        result = clean_pasted_text(text)
        self.assertEqual(result, "")

    def test_unicode_beyond_basic(self):
        """Other Unicode characters should be preserved."""
        text = "Hello 世界 🌍 émojis"
        result = clean_pasted_text(text)
        self.assertEqual(result, text)

    def test_tab_characters_preserved(self):
        """Tab characters should be preserved (indentation)."""
        text = "def test():\n\tpass"
        result = clean_pasted_text(text)
        self.assertEqual(result, text)

    def test_mixed_line_endings(self):
        """Mixed line endings should all be normalized."""
        text = "Line 1\r\nLine 2\nLine 3\rLine 4"
        result = clean_pasted_text(text)
        self.assertEqual(result, "Line 1\nLine 2\nLine 3\nLine 4")

    def test_invisible_chars_regex(self):
        """INVISIBLE_CHARS regex should match expected characters."""
        for char in ['\u200b', '\u200c', '\u200d', '\ufeff', '\u2060']:
            self.assertTrue(INVISIBLE_CHARS.search(char), f"Should match {repr(char)}")

    def test_smart_quotes_mapping(self):
        """SMART_QUOTES should contain expected mappings."""
        self.assertEqual(SMART_QUOTES['\u201c'], '"')
        self.assertEqual(SMART_QUOTES['\u201d'], '"')
        self.assertEqual(SMART_QUOTES['\u2018'], "'")
        self.assertEqual(SMART_QUOTES['\u2019'], "'")

    def test_dashes_mapping(self):
        """DASHES should contain expected mappings."""
        self.assertEqual(DASHES['\u2013'], '--')  # En dash
        self.assertEqual(DASHES['\u2014'], '---')  # Em dash

    def test_other_punct_mapping(self):
        """OTHER_PUNCT should contain expected mappings."""
        self.assertEqual(OTHER_PUNCT['\u2026'], '...')
        self.assertEqual(OTHER_PUNCT['\u00a0'], ' ')


if __name__ == "__main__":
    unittest.main(verbosity=2)
