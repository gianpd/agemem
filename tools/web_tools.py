"""
Web-related tool implementations.

Tools for web search, file writing, and document ingestion.
"""

import json
import logging
import httpx
from pathlib import Path
from typing import Optional

from core.config import (
    UWOT_SEARCH_ENABLED,
    UWOT_SEARCH_SERVICE_URL,
    UWOT_API_KEY,
)


logger = logging.getLogger("ask-swarm")

tool_definitions = [
{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Returns top results with title, URL, snippet. "
                "Use 3-5 distinct queries per topic to get comprehensive coverage. "
                "Results are capped at 4000 chars. "
                "RESEARCH MODE: this is your primary source — call before read_document or grep_corpus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Search query string."},
                    "num_results": {"type": "integer", "description": "Number of results (default 5, max 10)."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file at the specified path. "
                "Requires BOTH 'path' (where to write) and 'content' (what to write). "
                "Creates parent directories automatically. "
                "Use for saving notes, reports, or any text output. "
                "Example: path='output/report.md', content='# Report\\n\\nText here...'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Relative file path to write."},
                    "content": {"type": "string", "description": "Full file content as a string."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_document",
            "description": (
                "Ingest a document into the corpus with NER entity extraction. "
                "Supports both .md and .pdf files. "
                "For .pdf: converts to markdown via Docling, extracts entities via GLiNER, adds to corpus. "
                "For .md: adds to corpus with entity extraction. "
                "Returns doc_id on success."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (.md or .pdf)"
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Document type: document, contract, research, cronoprogramma, etc. (PDF only, default: document)"
                    },
                    "labels": {
                        "type": "string",
                        "description": "Label set for entity extraction: edilizia, legal, research (PDF only, default: edilizia)"
                    }
                },
                "required": ["path"]
            }
        }
    }
]


def sanitize_for_llm(text: str) -> str:
    """Sanitize text to prevent llama.cpp parser failures."""
    if not text:
        return ""
    # Remove control characters except newlines and tabs
    text = ''.join(ch for ch in text if ch == '\n' or ch == '\t' or ch == '\r' or (ord(ch) >= 32 and ord(ch) < 127) or ord(ch) > 159)
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.strip()


def format_web_search_results(
    query: str,
    results: list,
    enable_scrape: bool = True,
) -> str:
    """
    Format web search results into a readable string for the agent.
    
    This function handles the post-processing of search results, including:
    - Formatting titles, URLs, and snippets
    - Truncating long content to prevent context bloat
    - Including scraped content when available
    
    Args:
        query: The original search query
        results: List of search result dictionaries with keys:
                 - title: result title
                 - url or link: result URL
                 - snippet or description: result snippet
                 - scraped_content or content: scraped page content (optional)
        enable_scrape: Whether scraped content should be included
        
    Returns:
        Formatted search results string, capped via cap_tool_result()
    """
    if not results:
        return f"[WEB SEARCH] No results found for: '{sanitize_for_llm(query)}'"

    lines = [
        f"[WEB SEARCH RESULTS for '{sanitize_for_llm(query)}' — {len(results)} result(s)]",
        "=" * 60,
    ]

    for i, r in enumerate(results, 1):
        title = sanitize_for_llm(r.get("title", "No title"))
        url = sanitize_for_llm(r.get("url", r.get("link", "")))
        snippet = sanitize_for_llm(r.get("snippet", r.get("description", "")))

        lines.append(f"\n{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   Snippet: {snippet[:300]}{'...' if len(snippet) > 300 else ''}")

        # Include scraped content if available
        scraped = sanitize_for_llm(r.get("scraped_content", r.get("content", "")))
        if scraped and enable_scrape:
            # Truncate scraped content to prevent context bloat
            max_scrape_chars = 1500
            if len(scraped) > max_scrape_chars:
                scraped = scraped[:max_scrape_chars] + "... [truncated]"
            lines.append(f"   Content: {scraped}")

    lines.append("\n" + "=" * 60)
    result_text = "\n".join(lines)

    return result_text


async def web_search(
    query: str,
    num_results: int = 10,
    enable_scrape: bool = True,
    scrape_count: int = 3,
    language: str = "en",
) -> str:
    """
    Search the web using the uWOT Search Service.
    
    This function integrates with the uWOT search_web tool from the agent service,
    which enables DB persistence of retrieved context via session_id + db_session.
    
    Args:
        query: The search query string
        num_results: Maximum number of results (1-50, default 10)
        enable_scrape: Whether to scrape top results for content (default True)
        scrape_count: Number of results to scrape (1-10, default 3)
        language: Language code for results (default 'en')
        
    Returns:
        Formatted search results with titles, URLs, snippets, and optionally scraped content.
    """
    if not UWOT_SEARCH_ENABLED:
        return (
            "[WEB SEARCH DISABLED] Set UWOT_SEARCH_ENABLED=true to enable web search. "
            "The uWOT Search Service must be running and accessible."
        )
    
    # Ensure num_results is int
    try:
        num_results = int(num_results)
    except (TypeError, ValueError):
        num_results = 10
    
    # Clamp values
    num_results = max(1, min(50, num_results))
    scrape_count = max(1, min(10, scrape_count))
    
    logger.info(f"[web_search] Searching: '{query}' (num_results={num_results}, scrape={enable_scrape})")

    payload = {
        "query": query,
        "num_results": num_results,
        "enable_scrape": enable_scrape,
        "scrape_count": scrape_count,
        "language": language,
        "region": "wt-wt",
        "safesearch": "moderate",
        "enable_cache": True,
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{UWOT_SEARCH_SERVICE_URL}/api/v1/search",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            
            if response.status_code != 200:
                error_msg = f"Search service returned {response.status_code}: {response.text[:200]}"
                logger.error(f"[web_search] {error_msg}")
                return f"[WEB SEARCH ERROR] {error_msg}"
            
            data = response.json()
            
        # Extract results and format using shared function
        results = data.get("results", [])
        logger.info(f"[web_search] Found {len(results)} results for '{query}' (via direct HTTP)")
        return format_web_search_results(query, results, enable_scrape)
        
    except httpx.ConnectError as e:
        error_msg = f"Cannot connect to uWOT Search Service at {UWOT_SEARCH_SERVICE_URL}"
        logger.error(f"[web_search] {error_msg}: {e}")
        return (
            f"[WEB SEARCH ERROR] {error_msg}.\n"
            f"Ensure the search service is running: docker-compose up search"
        )
    except httpx.TimeoutException:
        error_msg = f"Timeout connecting to uWOT Search Service at {UWOT_SEARCH_SERVICE_URL}"
        logger.error(f"[web_search] {error_msg}")
        return f"[WEB SEARCH ERROR] {error_msg}"
    except Exception as e:
        logger.error(f"[web_search] Unexpected error: {e}")
        return f"[WEB SEARCH ERROR] {e}"


async def web_search_tool(
    query: str,
    num_results: int = 5
) -> str:
    """
    Wrapper for web_search to maintain compatibility with existing tool interface.
    
    Integrates with the uWOT search_web tool for DB persistence of retrieved context.
    
    Args:
        query: The search query string
        num_results: Number of results (default 5, max 10 for tool interface)
        
    Returns:
        Formatted search results.
    """
    # Clamp num_results for tool interface (max 10)
    num_results = min(max(1, num_results), 10)
    
    return await web_search(
        query=query,
        num_results=num_results,
        enable_scrape=True,
        scrape_count=3,
        language="en",
    )


def write_file(path: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        path: The file path to write to
        content: The content to write
        
    Returns:
        Success message or error
    """
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            f.write(content)
        
        byte_count = len(content.encode("utf-8"))
        logger.info(f"[write_file] Wrote {byte_count} bytes to {path}")
        
        return f"Successfully wrote {byte_count} bytes to {path}"
        
    except IOError as e:
        return f"Error writing file: {e}"
    except Exception as e:
        return f"Error: {e}"


def ingest_document(path: str, doc_type: str = "document", labels: str = "edilizia") -> str:
    """
    Ingest a document into the corpus.

    Supports both .md and .pdf files:
    - .md files: processed directly with entity extraction
    - .pdf files: converted via Docling using uv run ingest/ingest.py

    Args:
        path: Path to the file (.md or .pdf)
        doc_type: Document type for PDFs (default: document)
        labels: Label set for PDFs (default: edilizia)

    Returns:
        Success message with doc_id or error
    """
    import subprocess
    import re

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

    elif suffix == ".pdf":
        # PDF ingestion - use uv run ingest/ingest.py
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
                timeout=600,  # 10 minutes for large PDFs
                cwd=str(Path(__file__).parent.parent)  # Run from project root
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return f"Error ingesting PDF: {error_msg}"

            # Extract doc_id from output (last line usually contains it)
            output = result.stdout.strip()
            doc_id_match = re.search(r'doc_id\s*[:=]\s*(\S+)', output)
            doc_id = doc_id_match.group(1) if doc_id_match else "unknown"

            logger.info(f"[ingest_document] Ingested PDF {path} as {doc_id}")
            return f"Successfully ingested PDF.\n\n{output}"

        except subprocess.TimeoutExpired:
            return "Error: PDF ingestion timed out (after 10 minutes)"
        except FileNotFoundError:
            return "Error: 'uv' command not found. Make sure uv is installed and in PATH."
        except Exception as e:
            return f"Error ingesting PDF: {e}"

    else:
        return f"Error: Unsupported file type '{suffix}'. Only .md and .pdf files are supported."