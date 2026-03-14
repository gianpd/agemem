"""
tools package for AgeMem agent.

This package provides various tools for the agent including:
- Web search and fetch tools (web_tools.py)
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
        'ingest_document',
        'register_conversation_urls',
        'validate_url_for_fetch',
        'web_tool_definitions',
    }
    if name in web_exports:
        from . import web_tools
        return getattr(web_tools, name)
    raise AttributeError(f"module 'tools' has no attribute '{name}'")


__all__ = [
    # Web tools (lazy loaded)
    "web_search",
    "fetch_url",
    "fetch_url_tool",
    "write_file",
    "ingest_document",
    "register_conversation_urls",
    "validate_url_for_fetch",
    "web_tool_definitions",
    # Corpus tools
    "list_documents",
    "search_metadata",
    "grep_corpus",
    "read_document",
    "read_lines",
    "corpus_tool_definitions",
    # Query expansion
    "QueryExpander",
]
