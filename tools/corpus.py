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
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_document",
            "description": (
                "Ingest a document into the corpus with NER entity extraction. "
                "Supports .md, .pdf, and .docx files. "
                "For .pdf/.docx: converts to markdown via Docling, extracts entities via GLiNER, adds to corpus. "
                "For .md: adds to corpus with entity extraction. "
                "Returns doc_id on success."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (.md, .pdf, or .docx)"
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Document type: document, contract, research, cronoprogramma, etc. (PDF/DOCX only, default: document)"
                    },
                    "labels": {
                        "type": "string",
                        "description": "Label set for entity extraction: edilizia, legal, research (PDF/DOCX only, default: edilizia)"
                    }
                },
                "required": ["path"]
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
    from datetime import date as date_type

    raw_date = meta.get("doc_date", meta.get("date", ""))
    # Convert date object to ISO string if needed
    if isinstance(raw_date, date_type):
        raw_date = raw_date.isoformat()

    return {
        "doc_id": meta.get("doc_id", md_file.stem),
        "title": meta.get("doc_title", meta.get("title", md_file.stem)),
        "type": meta.get("doc_type", meta.get("type", "unknown")),
        "date": raw_date,
        "path": str(md_file)
    }


def _get_corpus_path(custom_path: Optional[Path] = None) -> Path:
    """Get corpus path, using custom path if provided or default CORPUS."""
    if custom_path is not None:
        return custom_path
    return CORPUS


def list_documents(corpus_path: Optional[Path] = None) -> str:
    """List all ingested documents with metadata. Supports custom corpus path."""
    corpus = _get_corpus_path(corpus_path)
    docs = []
    for md_file in sorted(corpus.glob("*.md")):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            meta, _, _ = _parse_frontmatter(content)
            docs.append(_get_doc_info(md_file, meta))
        except IOError as e:
            logger.warning(f"[list_documents] Failed to read {md_file}: {e}")

    return json.dumps({"documents": docs, "count": len(docs)}, indent=2)


def search_metadata(keyword: str, corpus_path: Optional[Path] = None) -> str:
    """Search document metadata for a keyword anywhere in the frontmatter. Supports custom corpus path."""
    corpus = _get_corpus_path(corpus_path)
    keyword_lower = keyword.lower()
    matches = []

    for md_file in sorted(corpus.glob("*.md")):
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


def grep_corpus(pattern: str, context_lines: int = 3, corpus_path: Optional[Path] = None) -> str:
    """
    Search document body text using pure Python.
    Intelligently skips YAML frontmatter and groups context.
    Supports custom corpus path.
    """
    corpus = _get_corpus_path(corpus_path)
    context_lines = min(context_lines, 5)

    # AUTO-FIX: If the LLM mistakenly uses spaces instead of pipes, and no regex operators are present
    if not any(c in pattern for c in "|.*+?()[]{}") and " " in pattern:
        pattern = "|".join(pattern.split())

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regular expression - {e}"

    all_results = []
    total_matches = 0
    char_count = 0
    MAX_CHARS = 4000

    for md_file in sorted(corpus.rglob("*.md")):
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


def _find_file_by_doc_id(doc_id: str, corpus_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve a doc_id to its actual file path. Supports custom corpus path."""
    corpus = _get_corpus_path(corpus_path)
    # First try exact filename match
    direct_path = corpus / f"{doc_id}.md"
    if direct_path.exists():
        return direct_path

    # Then fallback to parsing frontmatter to find the true ID
    for md_file in corpus.glob("*.md"):
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


def read_document(doc_id: str, corpus_path: Optional[Path] = None) -> str:
    """Read the full content of a document by doc_id. Supports custom corpus path."""
    target_file = _find_file_by_doc_id(doc_id, corpus_path)
    
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


def read_lines(doc_id: str, start_line: int, end_line: int, corpus_path: Optional[Path] = None) -> str:
    """Read a specific line range from a document. Supports custom corpus path."""
    target_file = _find_file_by_doc_id(doc_id, corpus_path)

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


def ingest_document(path: str, doc_type: str = "document", labels: str = "edilizia") -> str:
    """
    Ingest a document into the corpus.

    Supports .md, .pdf, and .docx files:
    - .md files: processed directly with entity extraction
    - .pdf files: converted via Docling using uv run ingest/ingest.py
    - .docx files: converted via Docling using uv run ingest/ingest.py

    Args:
        path: Path to the file (.md, .pdf, or .docx)
        doc_type: Document type for PDFs/DOCX (default: document)
        labels: Label set for PDFs/DOCX (default: edilizia)

    Returns:
        Success message with doc_id or error
    """
    import subprocess

    file_path = Path(path)

    if not file_path.exists():
        return f"Error: File not found: {path}"

    suffix = file_path.suffix.lower()

    if suffix == ".md":
        # Markdown ingestion - import and call ingest function directly
        try:
            from ingest.ingest import ingest
            doc_id = ingest(str(file_path))
            logger.info(f"[ingest_document] Ingested markdown {path} as {doc_id}")
            return f"Successfully ingested markdown. doc_id: {doc_id}"
        except Exception as e:
            return f"Error ingesting markdown: {e}"

    elif suffix in (".pdf", ".docx"):
        # PDF/DOCX ingestion - use uv run ingest/ingest.py
        cmd = [
            "uv", "run", "ingest/ingest.py",
            str(file_path),
            doc_type,
            "--labels", labels
        ]

        try:
            logger.info(f"[ingest_document] Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for large documents
                cwd=str(Path(__file__).parent.parent)  # Run from project root
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return f"Error ingesting {suffix}: {error_msg}"

            # Extract doc_id from output (last line usually contains it)
            output = result.stdout.strip()
            import re
            doc_id_match = re.search(r'doc_id\s*[:=]\s*(\S+)', output)
            doc_id = doc_id_match.group(1) if doc_id_match else "unknown"

            logger.info(f"[ingest_document] Ingested {suffix} {path} as {doc_id}")
            return f"Successfully ingested {suffix}.\n\n{output}"

        except subprocess.TimeoutExpired:
            return f"Error: {suffix} ingestion timed out (after 10 minutes)"
        except FileNotFoundError:
            return "Error: 'uv' command not found. Make sure uv is installed and in PATH."
        except Exception as e:
            return f"Error ingesting {suffix}: {e}"

    else:
        return f"Error: Unsupported file type '{suffix}'. Only .md, .pdf, and .docx files are supported."


def ingest_document_to_corpus(
    path: str,
    target_corpus: Path,
    doc_type: str = "document",
    labels: str = "edilizia",
    original_filename: str | None = None,
) -> str:
    """
    Ingest a document into a specific corpus directory.

    This is similar to ingest_document but allows specifying a custom corpus path,
    enabling per-user isolation.

    Supports .md, .pdf, and .docx files.

    Args:
        path: Path to the file (.md, .pdf, or .docx)
        target_corpus: Target corpus directory Path
        doc_type: Document type for PDFs/DOCX (default: document)
        labels: Label set for PDFs/DOCX (default: edilizia)
        original_filename: Original filename to use for doc_id/title (e.g. from upload)

    Returns:
        Success message with doc_id or error
    """
    import subprocess

    file_path = Path(path)

    if not file_path.exists():
        return f"Error: File not found: {path}"

    # Ensure target corpus exists
    target_corpus.mkdir(parents=True, exist_ok=True)

    suffix = file_path.suffix.lower()

    if suffix == ".md":
        # Markdown ingestion - process directly with entity extraction
        try:
            from ingest.ingest import write_document, extract_entities, load_labels

            # Read markdown
            markdown = file_path.read_text(encoding="utf-8")
            sections = re.findall(r'^#{1,2}\s+(.+)$', markdown, re.MULTILINE)

            # Load labels and extract entities
            label_config = load_labels(labels)
            entities = extract_entities(markdown, label_config)

            # Write directly to target corpus, using original_filename for doc_id/title
            out_path = write_document(
                file_path, markdown, sections, entities, doc_type, label_config,
                display_name=original_filename,
                target_corpus=target_corpus,
            )

            doc_id = out_path.stem
            logger.info(f"[ingest_document_to_corpus] Ingested markdown {path} to {target_corpus} as {doc_id}")
            return f"Successfully ingested markdown. doc_id: {doc_id}"

        except Exception as e:
            logger.error(f"[ingest_document_to_corpus] Error: {e}")
            return f"Error ingesting markdown: {e}"

    elif suffix in (".pdf", ".docx"):
        # PDF/DOCX ingestion - use uv run ingest/ingest.py with corpus path
        cmd = [
            "uv", "run", "ingest/ingest.py",
            str(file_path),
            doc_type,
            "--labels", labels,
            "--corpus", str(target_corpus),
        ]
        if original_filename:
            cmd.extend(["--original-filename", original_filename])

        try:
            logger.info(f"[ingest_document_to_corpus] Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for large documents
                cwd=str(Path(__file__).parent.parent)  # Run from project root
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return f"Error ingesting {suffix}: {error_msg}"

            # Extract doc_id from output
            output = result.stdout.strip()
            doc_id_match = re.search(r'doc_id\s*[:=]\s*(\S+)', output)
            doc_id = doc_id_match.group(1) if doc_id_match else "unknown"

            logger.info(f"[ingest_document_to_corpus] Ingested {suffix} {path} as {doc_id} to {target_corpus}")
            return f"Successfully ingested {suffix}.\n\ndoc_id: {doc_id}"

        except subprocess.TimeoutExpired:
            return f"Error: {suffix} ingestion timed out (after 10 minutes)"
        except FileNotFoundError:
            return "Error: 'uv' command not found. Make sure uv is installed and in PATH."
        except Exception as e:
            return f"Error ingesting {suffix}: {e}"

    else:
        return f"Error: Unsupported file type '{suffix}'. Only .md, .pdf, and .docx files are supported."


def search_corpus_structured(
    query: str,
    corpus_path: Path,
    max_results: int = 10,
    search_type: str = "keyword",
) -> list[dict]:
    """
    Search corpus and return structured results for API responses.

    Args:
        query: Search query string
        corpus_path: Corpus directory to search
        max_results: Maximum number of results to return
        search_type: 'keyword' (regex), 'semantic' (embedding), or 'hybrid'

    Returns:
        List of dicts with doc_id, title, snippet, score, source_type
    """
    results = []

    # Keyword search (regex-based)
    if search_type in ("keyword", "hybrid"):
        keyword_results = _keyword_search_structured(query, corpus_path, max_results)
        for r in keyword_results:
            if search_type == "hybrid":
                r["source_type"] = "keyword"
            else:
                r["source_type"] = "keyword"
            results.append(r)

    # Semantic search (embedding-based) - placeholder for future implementation
    if search_type == "semantic":
        # For now, fall back to keyword search with a note
        semantic_results = _keyword_search_structured(query, corpus_path, max_results)
        for r in semantic_results:
            r["source_type"] = "semantic"  # Mark as semantic even though using keyword for now
            results.append(r)

    # Hybrid: combine keyword and semantic results
    if search_type == "hybrid":
        # Deduplicate by doc_id, preferring higher scores
        seen = {}
        for r in results:
            doc_id = r["doc_id"]
            if doc_id not in seen or r["score"] > seen[doc_id]["score"]:
                seen[doc_id] = r
        results = list(seen.values())

    # Sort by score descending and limit
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def _keyword_search_structured(query: str, corpus_path: Path, max_results: int) -> list[dict]:
    """Perform keyword-based search and return structured results."""
    results = []
    query_lower = query.lower()
    query_tokens = set(query_lower.split())

    for md_file in sorted(corpus_path.glob("*.md")):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            meta, body_text, _ = _parse_frontmatter(content)
            doc_id = meta.get("doc_id", md_file.stem)
            title = meta.get("doc_title", meta.get("title", md_file.stem))

            # Score by token overlap in body text
            body_lower = body_text.lower()
            body_tokens = set(body_lower.split())

            # Jaccard-like overlap
            intersection = query_tokens & body_tokens
            if not intersection:
                continue

            # Calculate relevance score
            overlap_ratio = len(intersection) / len(query_tokens) if query_tokens else 0
            score = overlap_ratio

            # Extract snippet around first matching token
            snippet = _extract_snippet(body_text, query_tokens, max_chars=500)

            if snippet:
                results.append({
                    "doc_id": doc_id,
                    "title": title,
                    "snippet": snippet,
                    "score": score,
                })

        except IOError as e:
            logger.warning(f"[search_corpus_structured] Failed to read {md_file}: {e}")
            continue

    return results


def _extract_snippet(text: str, query_tokens: set[str], max_chars: int = 500) -> str:
    """Extract a snippet around the first matching query token."""
    text_lower = text.lower()
    lines = text.splitlines()

    # Find the first line containing a query token
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for token in query_tokens:
            if token in line_lower:
                # Extract context around this line
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                snippet_lines = lines[start:end]

                # Join and truncate
                snippet = "\n".join(snippet_lines).strip()
                if len(snippet) > max_chars:
                    snippet = snippet[:max_chars] + "..."

                return snippet

    return ""


if __name__ == "__main__":
    r1 = grep_corpus("referendum giustizia italia")
    print(r1)