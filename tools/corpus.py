"""
Corpus-related tool implementations.

Tools for interacting with the local document corpus.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from core.config import CORPUS, MAX_READ_LINES

logger = logging.getLogger("agemem")


tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List all ingested documents in the corpus with their metadata. "
                "Returns document IDs, titles, types, and dates. "
                "Use this to see what documents are available before searching or reading."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_metadata",
            "description": (
                "Search document metadata (titles, types, tags) for a keyword. "
                "Returns matching documents with their metadata. "
                "Use this to find documents by title, type, or keywords in frontmatter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "The keyword to search for in document metadata."}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_corpus",
            "description": (
                "Search the full text of all documents using a keyword or regex pattern. "
                "Returns matching lines with context. "
                "For best results, use pipe-separated alternatives rather than natural language: "
                "GOOD: grep_corpus('operating loss|breakeven|profitability') "
                "GOOD: grep_corpus('Siemens|partnership|industrial') "
                "BAD: grep_corpus('which company is closer to profitability') "
                "Extract the most distinctive nouns and domain terms from the question "
                "and combine them with | before calling this tool. "
                "Results are capped at 4000 characters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The search pattern (keyword or regex)."},
                    "context_lines": {"type": "integer", "description": "Number of context lines around matches (default 3, max 5)."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Read the full content of a specific document by its doc_id. "
                "Use list_documents first to find available doc_ids. "
                "Returns the document content, truncated if too long (max ~8000 chars). "
                "For large documents, use read_lines to read specific portions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The document ID (filename without extension)."}
                },
                "required": ["doc_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_lines",
            "description": (
                "Read a specific line range from a document. "
                "Useful for reading large documents in sections. "
                "Line numbers are 1-indexed. Maximum 75 lines per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The document ID."},
                    "start_line": {"type": "integer", "description": "Start line number (1-indexed)."},
                    "end_line": {"type": "integer", "description": "End line number (1-indexed)."}
                },
                "required": ["doc_id", "start_line", "end_line"]
            }
        }
    }
]


def list_documents() -> str:
    """
    List all ingested documents with metadata.

    Returns:
        JSON string with list of documents
    """
    docs = []
    for md_file in sorted(CORPUS.glob("*.md")):
        try:
            with open(md_file, "r") as f:
                content = f.read()

            # Parse YAML frontmatter
            meta = {}
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    import yaml
                    try:
                        meta = yaml.safe_load(parts[1]) or {}
                    except yaml.YAMLError:
                        pass

            docs.append({
                "doc_id": md_file.stem,
                "title": meta.get("title", md_file.stem),
                "type": meta.get("type", "unknown"),
                "date": meta.get("date", ""),
                "path": str(md_file)
            })
        except IOError as e:
            logger.warning(f"[list_documents] Failed to read {md_file}: {e}")

    return json.dumps({"documents": docs, "count": len(docs)}, indent=2)


def search_metadata(keyword: str) -> str:
    """
    Search document metadata for a keyword.

    Args:
        keyword: The keyword to search for

    Returns:
        JSON string with matching documents
    """
    keyword_lower = keyword.lower()
    matches = []

    for md_file in sorted(CORPUS.glob("*.md")):
        try:
            with open(md_file, "r") as f:
                content = f.read()

            # Check frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1].lower()
                    if keyword_lower in frontmatter:
                        import yaml
                        try:
                            meta = yaml.safe_load(parts[1]) or {}
                        except yaml.YAMLError:
                            meta = {}

                        matches.append({
                            "doc_id": md_file.stem,
                            "title": meta.get("title", md_file.stem),
                            "type": meta.get("type", "unknown"),
                            "matched_in": "frontmatter"
                        })
        except IOError:
            continue

    return json.dumps({"matches": matches, "count": len(matches)}, indent=2)


def grep_corpus(pattern: str, context_lines: int = 3) -> str:
    """
    Search document body text using ripgrep.

    Args:
        pattern: The search pattern (single keyword recommended)
        context_lines: Lines of context around matches

    Returns:
        Search results as formatted text
    """
    if context_lines > 5:
        context_lines = 5

    try:
        result = subprocess.run(
            ["rg", "-i", "-C", str(context_lines), "--no-heading", pattern, str(CORPUS)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            output = result.stdout
            if len(output) > 4000:
                output = output[:4000] + "\n... [truncated]"
            return output
        else:
            return f"No matches found for pattern: {pattern}"

    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except FileNotFoundError:
        return "Error: ripgrep (rg) not installed"
    except Exception as e:
        return f"Error: {e}"


def read_document(doc_id: str) -> str:
    """
    Read the full content of a document by doc_id.

    Args:
        doc_id: The document ID (filename without extension)

    Returns:
        Document content or error message
    """
    doc_path = CORPUS / f"{doc_id}.md"

    if not doc_path.exists():
        return f"Error: Document '{doc_id}' not found"

    try:
        with open(doc_path, "r") as f:
            content = f.read()

        # Truncate large documents
        if len(content) > 8000:
            content = content[:8000] + "\n\n... [TRUNCATED - use read_lines for the rest]"

        return content
    except IOError as e:
        return f"Error reading document: {e}"


def read_lines(doc_id: str, start_line: int, end_line: int) -> str:
    """
    Read a specific line range from a document.

    Args:
        doc_id: The document ID
        start_line: 1-indexed start line
        end_line: 1-indexed end line

    Returns:
        The requested lines or error message
    """
    doc_path = CORPUS / f"{doc_id}.md"

    if not doc_path.exists():
        return f"Error: Document '{doc_id}' not found"

    # Enforce line limit
    if end_line - start_line > MAX_READ_LINES:
        end_line = start_line + MAX_READ_LINES

    try:
        with open(doc_path, "r") as f:
            lines = f.readlines()

        # Convert to 0-indexed
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        selected = lines[start_idx:end_idx]
        result = "".join(selected)

        return f"Lines {start_line}-{end_line} of {doc_id}:\n\n{result}"
    except IOError as e:
        return f"Error reading document: {e}"