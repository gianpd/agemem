"""
Corpus-related tool implementations.

Tools for interacting with the local document corpus.
"""

import json
import logging
import re
import os
from pathlib import Path
from typing import Optional

from core.config import CORPUS, MAX_READ_LINES

logger = logging.getLogger("agemem")

tool_definitions =[
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
                "Search document metadata (titles, types, tags, entities) for a keyword. "
                "Returns matching documents with their metadata. "
                "Use this to find documents by title, type, or entities in frontmatter."
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
                "Search the body text of all documents using a keyword or regex pattern. "
                "Automatically skips YAML frontmatter to reduce noise. "
                "CRITICAL: Use pipe-separated words for multiple concepts, NOT spaces! "
                "GOOD: grep_corpus('operating loss|breakeven|profitability') "
                "GOOD: grep_corpus('Siemens|partnership|industrial') "
                "BAD: grep_corpus('which company is closer to profitability') "
                "Results are intelligently grouped and capped at ~4000 characters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The search pattern (use | for OR, e.g., 'referendum|PM')."},
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
                "Returns the document content, truncated if too long. "
                "For large documents, use read_lines to read specific portions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The document ID (from list_documents)."}
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
                "required":["doc_id", "start_line", "end_line"]
            }
        }
    }
]


def _parse_frontmatter(content: str) -> tuple[dict, str, str]:
    """Safely parse frontmatter and return (metadata_dict, body_text, raw_frontmatter)."""
    meta = {}
    body_text = content
    raw_frontmatter = ""
    
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            raw_frontmatter = parts[1]
            body_text = parts[2]
            try:
                import yaml
                meta = yaml.safe_load(raw_frontmatter) or {}
            except Exception as e:
                logger.warning(f"Failed to parse YAML frontmatter: {e}")
                
    return meta, body_text, raw_frontmatter


def _get_doc_info(md_file: Path, meta: dict) -> dict:
    """Extract standard fields from possibly varying metadata schemas."""
    return {
        "doc_id": meta.get("doc_id", md_file.stem),
        "title": meta.get("doc_title", meta.get("title", md_file.stem)),
        "type": meta.get("doc_type", meta.get("type", "unknown")),
        "date": meta.get("doc_date", meta.get("date", "")),
        "path": str(md_file)
    }


def list_documents() -> str:
    """List all ingested documents with metadata."""
    docs =[]
    for md_file in sorted(CORPUS.glob("*.md")):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            meta, _, _ = _parse_frontmatter(content)
            docs.append(_get_doc_info(md_file, meta))
        except IOError as e:
            logger.warning(f"[list_documents] Failed to read {md_file}: {e}")

    return json.dumps({"documents": docs, "count": len(docs)}, indent=2)


def search_metadata(keyword: str) -> str:
    """Search document metadata for a keyword anywhere in the frontmatter."""
    keyword_lower = keyword.lower()
    matches =[]

    for md_file in sorted(CORPUS.glob("*.md")):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            meta, _, raw_frontmatter = _parse_frontmatter(content)
            
            # Search the raw text of the frontmatter (handles nested lists/entities safely)
            if keyword_lower in raw_frontmatter.lower():
                doc_info = _get_doc_info(md_file, meta)
                doc_info["matched_in"] = "frontmatter"
                matches.append(doc_info)
        except IOError:
            continue

    return json.dumps({"matches": matches, "count": len(matches)}, indent=2)


def grep_corpus(pattern: str, context_lines: int = 3) -> str:
    """
    Search document body text using pure Python. 
    Intelligently skips YAML frontmatter and groups context.
    """
    context_lines = min(context_lines, 5)
    
    # AUTO-FIX: If the LLM mistakenly uses spaces instead of pipes, and no regex operators are present
    if not any(c in pattern for c in "|.*+?()[]{}") and " " in pattern:
        pattern = "|".join(pattern.split())
        
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regular expression - {e}"

    all_results =[]
    total_matches = 0
    char_count = 0
    MAX_CHARS = 4000

    for md_file in sorted(CORPUS.rglob("*.md")):
        if char_count > MAX_CHARS:
            break

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            meta, body_text, _ = _parse_frontmatter(content)
            doc_id = meta.get("doc_id", md_file.stem)

            lines = body_text.splitlines()
            match_indices =[i for i, line in enumerate(lines) if regex.search(line)]

            if not match_indices:
                continue

            total_matches += len(match_indices)
            
            file_header = f"\n📄[{doc_id}] ({len(match_indices)} matches)"
            all_results.append(file_header)
            char_count += len(file_header)

            MAX_MATCHES_PER_FILE = 5
            snippets = []
            current_snippet =[]
            last_idx = -100

            for i in match_indices[:MAX_MATCHES_PER_FILE]:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                if start <= last_idx:
                    current_snippet.extend(lines[last_idx:end])
                else:
                    if current_snippet:
                        snippets.append("\n".join(current_snippet))
                    current_snippet = lines[start:end]

                last_idx = end

            if current_snippet:
                snippets.append("\n".join(current_snippet))

            for snip in snippets:
                formatted_snip = f"...\n{snip.strip()}\n..."
                all_results.append(formatted_snip)
                char_count += len(formatted_snip)

            if len(match_indices) > MAX_MATCHES_PER_FILE:
                overflow_msg = f"  *(+{len(match_indices) - MAX_MATCHES_PER_FILE} more matches not shown)*"
                all_results.append(overflow_msg)
                char_count += len(overflow_msg)

        except Exception as e:
            logger.warning(f"Error searching {md_file}: {e}")

    if not total_matches:
        return f"No matches found for pattern: {pattern}"

    result_str = "\n".join(all_results)
    if len(result_str) > MAX_CHARS:
        result_str = result_str[:MAX_CHARS] + "\n\n[TRUNCATED: Maximum output length reached]"
        
    return result_str


def _find_file_by_doc_id(doc_id: str) -> Optional[Path]:
    """Resolve a doc_id to its actual file path."""
    # First try exact filename match
    direct_path = CORPUS / f"{doc_id}.md"
    if direct_path.exists():
        return direct_path
        
    # Then fallback to parsing frontmatter to find the true ID
    for md_file in CORPUS.glob("*.md"):
        if md_file.stem == doc_id:
            return md_file
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                meta, _, _ = _parse_frontmatter(f.read())
                if meta.get("doc_id") == doc_id:
                    return md_file
        except Exception:
            pass
            
    return None


def read_document(doc_id: str) -> str:
    """Read the full content of a document by doc_id."""
    target_file = _find_file_by_doc_id(doc_id)
    
    if not target_file:
        return f"Error: Document '{doc_id}' not found"

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        if len(content) > 8000:
            content = content[:8000] + f"\n\n...[TRUNCATED at 8000 chars. Use read_lines(doc_id='{doc_id}', start_line=..., end_line=...) for the rest]"

        return content
    except IOError as e:
        return f"Error reading document: {e}"


def read_lines(doc_id: str, start_line: int, end_line: int) -> str:
    """Read a specific line range from a document."""
    target_file = _find_file_by_doc_id(doc_id)

    if not target_file:
        return f"Error: Document '{doc_id}' not found"

    if end_line - start_line > MAX_READ_LINES:
        end_line = start_line + MAX_READ_LINES

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        selected = lines[start_idx:end_idx]
        result = "".join(selected)

        return f"Lines {start_line}-{end_line} of {doc_id}:\n\n{result}"
    except IOError as e:
        return f"Error reading document: {e}"
    
if __name__ == "__main__":
    r1 = grep_corpus("referendum giustizia italia")
    print(r1)