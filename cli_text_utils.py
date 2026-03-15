"""
cli_text_utils.py
─────────────────
Text cleaning and normalization utilities for CLI input handling.

This module provides functions to clean and normalize text pasted into
the CLI, handling common issues like:
- Invisible/zero-width characters from PDFs/web pages
- Smart quotes and special punctuation
- Excessive whitespace and line endings
- Mixed line ending styles
"""

from __future__ import annotations

import re


# Invisible/zero-width characters that should be removed
INVISIBLE_CHARS = re.compile(
    '['
    '\u200b'  # Zero-width space
    '\u200c'  # Zero-width non-joiner
    '\u200d'  # Zero-width joiner
    '\ufeff'  # Zero-width no-break space (BOM)
    '\u2060'  # Word joiner
    '\u2061'  # Function application
    '\u2062'  # Invisible times
    '\u2063'  # Invisible separator
    '\u2064'  # Invisible plus
    '\u206a'  # Inhibit symmetric swapping
    '\u206b'  # Activate symmetric swapping
    '\u206c'  # Inhibit arabic form shaping
    '\u206d'  # Activate arabic form shaping
    '\u206e'  # National digit shapes
    '\u206f'  # Nominal digit shapes
    ']+',
    re.UNICODE
)

# Smart quotes and apostrophes to normalize
SMART_QUOTES = {
    '\u2018': "'",  # Left single quotation mark
    '\u2019': "'",  # Right single quotation mark (also apostrophe)
    '\u201c': '"',  # Left double quotation mark
    '\u201d': '"',  # Right double quotation mark
    '\u201a': ",",  # Single low-9 quotation mark
    '\u201e': '"',  # Double low-9 quotation mark
    '\u2032': "'",  # Prime
    '\u2033': '"',  # Double prime
    # Note: \u0060 (grave accent/backtick) is NOT included
    # because backticks are used for Markdown code blocks
    '\u00b4': "'",  # Acute accent
}

# Dashes and hyphens to normalize
DASHES = {
    '\u2010': '-',  # Hyphen
    '\u2011': '-',  # Non-breaking hyphen
    '\u2012': '--',  # Figure dash
    '\u2013': '--',  # En dash
    '\u2014': '---',  # Em dash
    '\u2015': '---',  # Horizontal bar
}

# Ellipsis and other punctuation
OTHER_PUNCT = {
    '\u2026': '...',  # Horizontal ellipsis
    '\u00a0': ' ',    # Non-breaking space
}


def clean_pasted_text(text: str) -> str:
    """
    Clean and normalize pasted text for LLM processing.

    This function handles common issues when pasting text from various sources
    (web pages, PDFs, documents) including:
    - Removing invisible/zero-width characters
    - Normalizing smart quotes to ASCII equivalents
    - Normalizing dashes and special punctuation
    - Collapsing excessive whitespace and empty lines
    - Preserving intentional formatting (code blocks, lists)

    Args:
        text: Raw input text from user paste or typing

    Returns:
        Clean, normalized text ready for the agent
    """
    if not text or not text.strip():
        return ""

    # Step 1: Remove invisible/zero-width characters
    text = INVISIBLE_CHARS.sub('', text)

    # Step 2: Normalize smart quotes and special characters
    for char, replacement in SMART_QUOTES.items():
        text = text.replace(char, replacement)
    for char, replacement in DASHES.items():
        text = text.replace(char, replacement)
    for char, replacement in OTHER_PUNCT.items():
        text = text.replace(char, replacement)

    # Step 3: Normalize line endings to Unix style
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Step 4: Collapse excessive empty lines (3+ newlines -> 2 newlines)
    # This preserves paragraph breaks but removes excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Step 5: Remove trailing whitespace from each line while preserving structure
    lines = text.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    text = '\n'.join(cleaned_lines)

    # Step 6: Trim leading/trailing whitespace from entire text
    text = text.strip()

    # Step 7: Ensure text doesn't end with excessive punctuation
    # (but preserve intentional use like "..." or "!!!")
    text = re.sub(r'([.!?]){4,}', r'\1\1\1', text)

    return text


def is_likely_paste(text: str | None) -> bool:
    """
    Detect if input appears to be a paste operation based on characteristics.

    Returns True if the text shows signs of being pasted content:
    - Contains multiple lines
    - Has unusual character patterns
    - Contains invisible characters
    """
    if not text:
        return False

    # Multiple lines suggests paste
    if '\n' in text or '\r' in text:
        return True

    # Presence of invisible characters suggests paste
    if INVISIBLE_CHARS.search(text):
        return True

    # Smart quotes or other special chars suggest paste from formatted source
    for char in SMART_QUOTES:
        if char in text:
            return True

    return False


def get_cleaning_summary(original: str, cleaned: str) -> dict:
    """
    Get a summary of what was cleaned from the text.

    Returns a dict with:
    - original_length: Character count before cleaning
    - cleaned_length: Character count after cleaning
    - chars_removed: Number of characters removed
    - was_cleaned: Whether any cleaning occurred
    """
    return {
        "original_length": len(original),
        "cleaned_length": len(cleaned),
        "chars_removed": len(original) - len(cleaned),
        "was_cleaned": original != cleaned,
    }
