"""
tools package for AgeMem agent.

This package provides various tools for the agent including:
- Web search and fetch tools (web_tools.py)
- Browser automation tools (browser_tools.py)
- Corpus/document management tools (corpus.py)
- Query expansion utilities (query_expansion.py)

Note:
- tool_registry.py is currently empty and reserved for future use
- web_tools.py contains substantial functionality (~44KB) and may be split in future
"""

from .corpus import (
    list_documents,
    search_metadata,
    grep_corpus,
    read_document,
    read_lines,
    ingest_document,
    tool_definitions as corpus_tool_definitions,
)

from .query_expansion import QueryExpander

# Lazy imports for web_tools to avoid dependency issues
# Use: from tools.web_tools import web_search
def __getattr__(name):
    """Lazy load web_tools to avoid heavy dependencies on import."""
    web_exports = {
        'web_search',
        'fetch_url',
        'fetch_url_tool',
        'write_file',
        'register_conversation_urls',
        'validate_url_for_fetch',
        'web_tool_definitions',
    }
    browser_exports = {
        'browser_click',
        'browser_click_tool',
        'browser_scroll',
        'browser_scroll_tool',
        'browser_type',
        'browser_type_tool',
        'browser_press',
        'browser_press_tool',
        'browser_read_page',
        'browser_read_page_tool',
        'browser_screenshot',
        'browser_screenshot_tool',
        'browser_close',
        'browser_close_tool',
        'browser_navigate',
        'browser_navigate_tool',
        'BrowserSession',
        'browser_tool_definitions',
    }
    config_exports = {
        'BROWSER_CDP_ENDPOINT',
        'BROWSER_CONNECT_OVER_CDP',
    }
    if name in web_exports:
        from . import web_tools
        return getattr(web_tools, name)
    if name in browser_exports:
        from . import browser_tools
        return getattr(browser_tools, name)
    if name in config_exports:
        from core.config import BROWSER_CDP_ENDPOINT, BROWSER_CONNECT_OVER_CDP
        if name == 'BROWSER_CDP_ENDPOINT':
            return BROWSER_CDP_ENDPOINT
        if name == 'BROWSER_CONNECT_OVER_CDP':
            return BROWSER_CONNECT_OVER_CDP
    raise AttributeError(f"module 'tools' has no attribute '{name}'")


__all__ = [
    # Web tools (lazy loaded)
    "web_search",
    "fetch_url",
    "fetch_url_tool",
    "write_file",
    "register_conversation_urls",
    "validate_url_for_fetch",
    "web_tool_definitions",
    # Browser tools (lazy loaded)
    "browser_click",
    "browser_click_tool",
    "browser_scroll",
    "browser_scroll_tool",
    "browser_type",
    "browser_type_tool",
    "browser_press",
    "browser_press_tool",
    "browser_read_page",
    "browser_read_page_tool",
    "browser_screenshot",
    "browser_screenshot_tool",
    "browser_close",
    "browser_close_tool",
    "browser_navigate",
    "browser_navigate_tool",
    "BrowserSession",
    "browser_tool_definitions",
    # Browser CDP config
    "BROWSER_CDP_ENDPOINT",
    "BROWSER_CONNECT_OVER_CDP",
    # Corpus tools
    "list_documents",
    "search_metadata",
    "grep_corpus",
    "read_document",
    "read_lines",
    "ingest_document",
    "corpus_tool_definitions",
    # Query expansion
    "QueryExpander",
]
