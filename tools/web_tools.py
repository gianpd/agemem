"""
Web-related tool implementations.

Tools for web search, file writing, and document ingestion.
"""

import json
import logging
import httpx
import re
import unicodedata
import socket
import ipaddress
import asyncio
import time
from pathlib import Path
from typing import Optional, Set, List
from urllib.parse import urlparse, urlunparse
from dataclasses import dataclass
from bs4 import BeautifulSoup, Comment

from core.config import (
    UWOT_SEARCH_ENABLED,
    UWOT_SEARCH_SERVICE_URL,
    FETCH_ONLY_MENTIONED_URLS,
    BROWSER_CDP_ENDPOINT,
    BROWSER_CONNECT_OVER_CDP,
)

# Optional Playwright import - graceful degradation if not installed
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Security configuration for fetch_url
FETCH_URL_MAX_CONTENT_LENGTH = 5_500_000  # 5.5MB max
FETCH_URL_TIMEOUT_SECONDS = 30
FETCH_URL_MAX_REDIRECTS = 3

# Domain security configuration
# ASCII-only allowed domains (prevent homograph attacks)
ALLOWED_DOMAINS: Set[str] = set()  # Empty = no restrictions beyond HTTPS

# Blocked hostnames - exact matches and suffix matches
BLOCKED_HOSTNAMES: Set[str] = {
    "localhost",
    "127.0.0.1", "::1", "0.0.0.0",
    "169.254.169.254",  # AWS/GCP/Azure metadata
    "169.254.170.2",    # ECS metadata
    "metadata.google.internal",
    "metadata.compute.internal",
    "instance-data",    # AWS EC2 legacy
}

# Blocked IP networks (CIDR notation)
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / cloud metadata
    ipaddress.ip_network("100.64.0.0/10"),   # Carrier-grade NAT
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
    ipaddress.ip_network("fd00:ec2::/32"),     # AWS IPv6 metadata
]

# Suspicious Unicode ranges for homograph detection
# Cyrillic, Greek, Armenian, Hebrew, Arabic, etc. that look like Latin
SUSPICIOUS_RANGES = [
    (0x0400, 0x04FF, "Cyrillic"),      # Cyrillic
    (0x0500, 0x052F, "Cyrillic Supplement"),
    (0x2DE0, 0x2DFF, "Cyrillic Extended-A"),
    (0xA640, 0xA69F, "Cyrillic Extended-B"),
    (0x0370, 0x03FF, "Greek"),          # Greek
    (0x1F00, 0x1FFF, "Greek Extended"),
    (0x0530, 0x058F, "Armenian"),       # Armenian
    (0x0590, 0x05FF, "Hebrew"),         # Hebrew
    (0x0600, 0x06FF, "Arabic"),         # Arabic
    (0x0750, 0x077F, "Arabic Supplement"),
    (0x0900, 0x097F, "Devanagari"),     # Devanagari
    (0x3040, 0x309F, "Hiragana"),       # Hiragana
    (0x30A0, 0x30FF, "Katakana"),       # Katakana
]

# URL context tracker - URLs from conversation context
# This is session-level storage for validated URLs
_conversation_urls: Set[str] = set()


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
            "name": "fetch_url",
            "description": (
                "Fetch content from a URL. SECURITY RESTRICTED: Only fetches URLs that appeared "
                "in previous web_search results, user messages, or tool/skills outputs in this conversation. "
                "HTTPS only. Content is sanitized to prevent prompt injection. "
                "Returns text content for HTML/JSON/API responses, or saves binary files (PDFs) to disk. "
                "Max 500KB. Use this AFTER web_search to retrieve full paper content or API data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch. Must be from conversation context (web_search results, previous tool outputs, skills, or user messages). HTTPS only."
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 10000, max 50000)."
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Optional: save binary content (PDFs, images) to this path instead of returning text."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file. CRITICAL: Both 'path' and 'content' parameters are REQUIRED "
                "and must be non-empty strings. The path must include a filename (e.g., 'docs/report.md', not just 'docs/'). "
                "Creates parent directories automatically. "
                "Use for saving notes, reports, or any text output. "
                "Example: path='output/report.md', content='# Report\\n\\nYour content here...'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Relative file path including filename (e.g., 'output/notes.md'). Must be non-empty."},
                    "content": {"type": "string", "description": "Full file content as a string. Must be non-empty."}
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
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": (
                "Navigate to a URL using Playwright browser automation and capture a screenshot. "
                "Useful for: verifying page content visually, checking UI state, capturing proof of actions, "
                "debugging web issues, or archiving page state. "
                "Supports connecting to an existing browser via CDP (Chrome DevTools Protocol) - "
                "set BROWSER_CDP_ENDPOINT=http://localhost:9222 to use your logged-in browser session. "
                "Returns the path to the saved screenshot. "
                "Requires playwright to be installed: pip install playwright && playwright install chromium"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to. Must be HTTPS. Should be from conversation context or web_search results."
                    },
                    "action": {
                        "type": "string",
                        "description": "Short description of what you're doing (used in filename). Example: 'check pricing page'",
                        "default": "navigate"
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": "Capture full page screenshot (default: true). If false, captures viewport only.",
                        "default": True
                    },
                    "wait_ms": {
                        "type": "integer",
                        "description": "Additional wait time in milliseconds after page load for JS-heavy pages (default: 1000)",
                        "default": 1000
                    },
                    "wait_until": {
                        "type": "string",
                        "description": "When to consider navigation complete: 'networkidle' (default), 'domcontentloaded', 'load', or 'commit'. Use 'domcontentloaded' for JS-heavy sites that timeout.",
                        "default": "networkidle",
                        "enum": ["networkidle", "domcontentloaded", "load", "commit"]
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save screenshots (default: 'screenshots').",
                        "default": "screenshots"
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "Run browser in headless mode (default: true). Set false to see browser window. Ignored when using CDP.",
                        "default": True
                    },
                    "use_cdp": {
                        "type": "boolean",
                        "description": "Connect to existing browser via CDP instead of launching new. Auto-enabled if BROWSER_CDP_ENDPOINT is set.",
                        "default": False
                    }
                },
                "required": ["url"]
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


# =============================================================================
# URL SECURITY VALIDATION
# =============================================================================

def _normalize_url(url: str) -> str:
    """Normalize URL for comparison and validation."""
    # Strip whitespace and trailing slashes
    url = url.strip().rstrip('/')
    # Ensure protocol is present
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


def _is_homograph_attack(domain: str) -> tuple[bool, str]:
    """
    Detect potential homograph attacks (Unicode lookalike domains).

    Returns (is_attack, reason) tuple.
    """
    # Normalize the domain
    try:
        # Convert to NFC normalization
        normalized = unicodedata.normalize('NFC', domain)
        # Check for punycode encoding
        if normalized.startswith('xn--'):
            # Punycode encoded international domain - decode and check
            try:
                import idna
                decoded = idna.decode(normalized)
                normalized = decoded
            except (ImportError, Exception):
                # idna not available or decode failed - treat with caution
                return True, "Punycode domain could not be verified"
    except Exception:
        return True, "Domain normalization failed"

    # Check each character for suspicious ranges
    latin_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.')
    found_scripts = set()

    for char in normalized:
        if char in latin_chars:
            continue

        code_point = ord(char)

        # Check against suspicious Unicode ranges
        for start, end, script_name in SUSPICIOUS_RANGES:
            if start <= code_point <= end:
                found_scripts.add(script_name)
                break

    # If we found non-Latin scripts mixed with Latin, it's suspicious
    # Also check if there are confusable characters
    if found_scripts:
        # Check for confusable characters (characters that look like Latin)
        confusables = _check_confusable_chars(normalized)
        if confusables:
            return True, f"Potential homograph attack: mixed scripts ({', '.join(found_scripts)}) with confusable characters: {confusables}"

    return False, ""


def _check_confusable_chars(domain: str) -> List[str]:
    """
    Check for specific confusable characters that look like Latin letters.
    Returns list of suspicious characters found.
    """
    # Common confusables: Cyrillic а (U+0430) looks like Latin a (U+0061)
    confusable_map = {
        '\u0430': 'Cyrillic а (looks like a)',  # а
        '\u0435': 'Cyrillic е (looks like e)',  # е
        '\u043e': 'Cyrillic о (looks like o)',  # о
        '\u0440': 'Cyrillic р (looks like p)',  # р
        '\u0441': 'Cyrillic с (looks like c)',  # с
        '\u0445': 'Cyrillic х (looks like x)',  # х
        '\u0456': 'Cyrillic і (looks like i)',  # і
        '\u0458': 'Cyrillic ј (looks like j)',  # ј
        '\u03b1': 'Greek α (looks like a)',     # α
        '\u03bf': 'Greek ο (looks like o)',     # ο
        '\u03c1': 'Greek ρ (looks like p)',     # ρ
        '\u03b5': 'Greek ε (looks like e)',     # ε
    }

    found = []
    for char in domain:
        if char in confusable_map:
            found.append(confusable_map[char])

    return found


def _is_ip_private(ip_str: str) -> tuple[bool, str]:
    """
    Check if an IP address is in a private/internal range.

    Args:
        ip_str: A string that is ALREADY VALIDATED as an IP address literal.

    Returns (is_private, reason) tuple.
    Raises:
        ValueError: If ip_str is not a valid IP address (caller must ensure valid input).
    """
    addr = ipaddress.ip_address(ip_str)  # Let ValueError propagate to caller
    for network in BLOCKED_IP_NETWORKS:
        if addr in network:
            return True, f"IP {ip_str} is in blocked network {network}"
    return False, ""


def _resolve_and_validate_host(hostname: str) -> tuple[bool, str]:
    """
    Resolve hostname and validate all resulting IPs are public.

    This prevents DNS rebinding attacks where a domain initially resolves
    to a public IP but then resolves to a private IP at request time.

    Returns (is_valid, error_message) tuple.
    """
    # Check for cloud metadata hostnames first (no DNS needed)
    hostname_lower = hostname.lower()
    if hostname_lower in BLOCKED_HOSTNAMES:
        return False, f"Hostname '{hostname}' is blocked"

    # Try to parse as IP address directly
    try:
        # If this succeeds, hostname is an IP literal
        ipaddress.ip_address(hostname)
        is_private, reason = _is_ip_private(hostname)
        if is_private:
            return False, reason
        # It's a valid public IP
        return True, ""
    except ValueError:
        pass  # Not an IP literal, continue with DNS resolution

    # Resolve hostname to IPs
    try:
        results = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        if not results:
            return False, f"No DNS records found for '{hostname}'"

        for result in results:
            ip = result[4][0]
            is_private, reason = _is_ip_private(ip)
            if is_private:
                return False, f"Hostname '{hostname}' resolves to blocked IP: {reason}"
        return True, ""
    except socket.gaierror as e:
        return False, f"DNS resolution failed for '{hostname}': {e}"
    except Exception as e:
        return False, f"Error resolving '{hostname}': {e}"


def _extract_urls_from_text(text: str) -> Set[str]:
    """Extract all URLs from a text blob."""
    urls = set()
    # Match http/https URLs
    url_pattern = r'https?://[^\s<>"\']+(?:\.[^\s<>"\']+)*'
    for match in re.finditer(url_pattern, text):
        urls.add(match.group(0))
    return urls


def register_conversation_urls(*text_sources: str):
    """
    Register URLs from conversation context for fetch_url validation.
    Call this with user messages, tool results, etc.
    """
    for text in text_sources:
        if not text:
            continue
        urls = _extract_urls_from_text(text)
        _conversation_urls.update(urls)
        logger.debug(f"[fetch_url] Registered {len(urls)} URLs from conversation context")


def _is_url_from_context(url: str) -> bool:
    """
    Check if URL was previously seen in conversation context.

    SECURITY: Only matches exact URLs or same-origin (scheme+host) URLs.
    Does NOT allow arbitrary path prefix matching which could enable SSRF.
    """
    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    request_origin = f"{parsed.scheme}://{parsed.netloc}"

    # Direct match
    if normalized in _conversation_urls:
        return True

    for context_url in _conversation_urls:
        # Normalize context URL for comparison
        context_url_normalized = _normalize_url(context_url)

        # Exact URL match (with/without trailing slash)
        if normalized.rstrip('/') == context_url_normalized.rstrip('/'):
            return True

        # Same-origin check (scheme + host must match exactly)
        # This allows fetching different paths on the same domain that appeared
        # in search results, but not arbitrary subpaths that weren't registered
        ctx_parsed = urlparse(context_url_normalized)
        ctx_origin = f"{ctx_parsed.scheme}://{ctx_parsed.netloc}"
        if request_origin == ctx_origin:
            # Both URLs are on the same origin - allow if the path is a subpath
            # of a registered URL (e.g., if /blog/ is registered, allow /blog/post)
            # SECURITY: Prevent path traversal (../) and ensure proper subpath match
            ctx_path = ctx_parsed.path.rstrip('/')  # Normalize trailing slash
            req_path = parsed.path.rstrip('/')
            if ctx_path != '' and req_path.startswith(ctx_path):
                # Ensure it's a proper subpath: either exact match or continues with /
                next_char = req_path[len(ctx_path):len(ctx_path)+1]
                if next_char in ('', '/'):  # Exact match or proper subpath
                    # Block path traversal attempts
                    if '..' not in req_path:
                        return True

    return False


def _html_to_safe_text(html: str) -> str:
    """
    Convert HTML to plain text, removing all tags and hidden content.

    This prevents indirect prompt injection (IDPI) attacks where malicious
    instructions are hidden in HTML elements, CSS, or invisible text.

    Removes:
    - Script, style, meta, noscript, iframe, object elements
    - Elements with display:none, visibility:hidden, font-size:0
    - HTML comments
    - Zero-width and invisible Unicode characters
    - Suspicious instruction-like patterns
    """
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        # If parsing fails, return the raw text with basic sanitization
        return _sanitize_fetched_content(html, "")

    # Remove script, style, meta, noscript, iframe, object elements
    for tag in soup(["script", "style", "meta", "noscript", "iframe", "object", "embed", "applet"]):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove elements with suspicious CSS (hidden content)
    for tag in soup.find_all():
        # Defensive check: tag or attrs might be None with malformed HTML
        if tag is None or tag.attrs is None:
            continue
        style = tag.get('style', '')
        if style:
            style_normalized = style.replace(' ', '').lower()
            if any(pattern in style_normalized for pattern in [
                'display:none',
                'visibility:hidden',
                'font-size:0',
                'opacity:0',
                'height:0',
                'width:0',
                'position:absolute;left:-',
                'text-indent:-',
            ]):
                tag.decompose()
                continue

        # Remove elements with aria-hidden="true" (screen reader hidden)
        if tag.get('aria-hidden') == 'true':
            tag.decompose()
            continue

        # Remove elements with hidden attribute
        if tag.get('hidden') is not None:
            tag.decompose()
            continue

    # Get text content
    text = soup.get_text(separator='\n')

    # Strip zero-width and invisible Unicode characters
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060\u180e]', '', text)

    # Flag suspicious instruction-like patterns
    text = re.sub(
        r'(?i)(ignore\s+(previous|prior|all)\s+instructions?|'
        r'system\s*:|new\s+instructions?:|\[INST\]|\[/INST\]|'
        r'<\s*system\s*>|</\s*system\s*>|'
        r'you\s+are\s+now\s+|\bSYSTEM\b.*?\bINSTRUCTION\b)',
        '[POTENTIAL INJECTION ATTEMPT REMOVED]',
        text
    )

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)

    return text


def _sanitize_fetched_content(content: str, source_url: str) -> str:
    """
    Sanitize fetched web content to prevent prompt injection attacks.

    Removes or neutralizes:
    - XML/external entity declarations
    - Suspicious markdown that could break out of context
    - Very long lines that could be prompt stuffing
    - Control characters
    - Zero-width characters
    - HTML tags (via _html_to_safe_text for HTML content)
    """
    if not content:
        return ""

    original_length = len(content)

    # Detect if content is HTML
    is_html = bool(re.search(r'<\s*(html|head|body|div|script|style)[\s>]', content[:5000], re.IGNORECASE))

    if is_html:
        # Use HTML-to-text conversion for better IDPI protection
        content = _html_to_safe_text(content)
    else:
        # For non-HTML content, still sanitize
        # Remove null bytes and control characters (except newlines/tabs)
        content = ''.join(ch for ch in content if ch == '\n' or ch == '\t' or ch == '\r' or (ord(ch) >= 32 and ord(ch) < 127) or ord(ch) > 159)

        # Strip zero-width and invisible Unicode characters
        content = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060\u180e]', '', content)

    # Remove XML declarations and DOCTYPE (XXE prevention)
    content = re.sub(r'<\?xml[^?]*\?>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<!DOCTYPE[^>]*>', '', content, flags=re.IGNORECASE | re.DOTALL)

    # Neutralize suspicious markdown sequences that could break context
    # Replace triple backticks with markers to prevent code block injection
    content = content.replace('```', "'''")

    # Limit line length to prevent horizontal stuffing attacks
    lines = content.split('\n')
    sanitized_lines = []
    for line in lines:
        if len(line) > 2000:
            # Truncate very long lines with indicator
            line = line[:2000] + "... [truncated long line]"
        sanitized_lines.append(line)
    content = '\n'.join(sanitized_lines)

    # Remove any remaining suspicious patterns
    # Jinja/template injection attempts
    content = re.sub(r'\{\{.*?\}\}', '[template removed]', content)
    # XML external entity references
    content = re.sub(r'&[a-zA-Z]+;', '[entity removed]', content)

    sanitized_length = len(content)
    if original_length != sanitized_length:
        logger.info(f"[fetch_url] Sanitized content from {source_url}: {original_length} -> {sanitized_length} chars")

    return content


def validate_url_for_fetch(url: str, require_context: bool = True) -> tuple[bool, str, Optional[str]]:
    """
    Comprehensive URL validation for fetch_url security.

    Args:
        url: The URL to validate
        require_context: If True, URL must be from conversation context

    Returns:
        (is_valid, error_message, normalized_url)
    """
    if not url or not isinstance(url, str):
        return False, "URL is required and must be a string", None

    url = url.strip()
    if not url:
        return False, "URL cannot be empty", None

    # Normalize URL
    normalized = _normalize_url(url)

    # Parse URL
    try:
        parsed = urlparse(normalized)
    except Exception as e:
        return False, f"Invalid URL format: {e}", None

    # Check scheme - HTTPS only
    if parsed.scheme != 'https':
        return False, f"Only HTTPS URLs are allowed (got: {parsed.scheme})", None

    # Check for empty hostname
    if not parsed.hostname:
        return False, "URL must have a valid hostname", None

    hostname = parsed.hostname.lower()

    # DNS resolution and IP validation (CRIT-1: DNS rebinding protection)
    # This resolves the hostname and validates all resulting IPs
    is_valid_ip, ip_error = _resolve_and_validate_host(hostname)
    if not is_valid_ip:
        return False, f"Security violation: {ip_error}", None

    # Check for homograph attacks
    is_homograph, reason = _is_homograph_attack(hostname)
    if is_homograph:
        return False, f"Security violation: {reason}", None

    # Check if URL is from conversation context
    if require_context and not _is_url_from_context(url):
        return False, (
            "URL not found in conversation context. "
            "For security, fetch_url can only access URLs that appeared in: "
            "(1) previous web_search results, (2) user messages, or (3) previous tool outputs. "
            "Use web_search first to discover URLs, then fetch_url to retrieve content."
        ), None

    # Reconstruct normalized URL without fragment (HIGH-1: HashJack prevention)
    clean_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        ''  # Remove fragment
    ))

    return True, "", clean_url


async def fetch_url(
    url: str,
    max_length: int = 10000,
    save_path: Optional[str] = None,
) -> str:
    """
    Securely fetch content from a URL with comprehensive security guardrails.

    SECURITY FEATURES:
    - HTTPS only (no HTTP)
    - URL context validation (configurable via FETCH_ONLY_MENTIONED_URLS)
    - Homograph attack detection (Cyrillic lookalikes, etc.)
    - Internal IP blocking with DNS resolution (prevents DNS rebinding)
    - Cloud metadata endpoint blocking (169.254.x.x)
    - IPv6 private range blocking
    - Redirect validation (each hop is validated, not just the original URL)
    - Content sanitization (XXE prevention, prompt injection protection, HTML stripping)
    - Size limits (500KB max)
    - Timeout protection (30s)
    - Redirect limit (3 hops)

    Args:
        url: URL to fetch. Context validation depends on FETCH_ONLY_MENTIONED_URLS config.
        max_length: Maximum characters to return (default 10000, max 50000).
        save_path: If provided, save binary content to this path instead of returning text.

    Returns:
        Fetched and sanitized content, or error message.
    """
    # Validate URL
    is_valid, error_msg, clean_url = validate_url_for_fetch(url, require_context=FETCH_ONLY_MENTIONED_URLS)
    if not is_valid:
        logger.warning(f"[fetch_url] Validation failed for '{url}': {error_msg}")
        return f"[FETCH URL ERROR] {error_msg}"

    # Clamp max_length
    max_length = min(max(100, max_length), 50000)

    logger.info(f"[fetch_url] Fetching: {clean_url}")

    # Track redirect chain for validation
    redirect_count = 0
    current_url = clean_url
    visited_urls = {clean_url}  # Track to prevent loops

    try:
        async with httpx.AsyncClient(
            timeout=FETCH_URL_TIMEOUT_SECONDS,
            follow_redirects=False,  # CRIT-4: Handle redirects manually for validation
        ) as client:

            while redirect_count <= FETCH_URL_MAX_REDIRECTS:
                response = await client.get(
                    current_url,
                    headers={
                        "User-Agent": "AgeMem-Agent/1.0 (Research Tool)",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        "Accept-Encoding": "identity",  # Don't request compression (simpler handling)
                    },
                )

                # Handle redirects manually with validation
                if response.status_code in (301, 302, 303, 307, 308):
                    redirect_count += 1
                    if redirect_count > FETCH_URL_MAX_REDIRECTS:
                        return f"[FETCH URL ERROR] Too many redirects (max {FETCH_URL_MAX_REDIRECTS})"

                    # Defensive check for headers
                    headers = getattr(response, 'headers', None) or {}
                    location = headers.get("location", "")
                    if not location:
                        return f"[FETCH URL ERROR] Redirect response missing Location header"

                    # Resolve relative URLs
                    if location.startswith('/'):
                        parsed_current = urlparse(current_url)
                        location = f"{parsed_current.scheme}://{parsed_current.netloc}{location}"
                    elif not location.startswith(('http://', 'https://')):
                        # Relative path without leading slash
                        parsed_current = urlparse(current_url)
                        base_path = parsed_current.path.rsplit('/', 1)[0] if '/' in parsed_current.path else ''
                        location = f"{parsed_current.scheme}://{parsed_current.netloc}{base_path}/{location}"

                    # Validate the redirect destination (CRIT-4)
                    is_valid, redirect_error, validated_url = validate_url_for_fetch(
                        location, require_context=False  # Redirects don't need context check
                    )
                    if not is_valid:
                        logger.warning(f"[fetch_url] Redirect to unsafe URL blocked: {redirect_error}")
                        return f"[FETCH URL ERROR] Redirect to unsafe URL blocked: {redirect_error}"

                    # Check for redirect loops
                    if validated_url in visited_urls:
                        return "[FETCH URL ERROR] Redirect loop detected"
                    visited_urls.add(validated_url)

                    current_url = validated_url
                    logger.debug(f"[fetch_url] Following redirect to: {current_url}")
                    continue

                # Not a redirect, process the response
                break

            # Check final status
            if response.status_code != 200:
                return f"[FETCH URL ERROR] HTTP {response.status_code}"

            # Check content length
            content_length = len(response.content)
            if content_length > FETCH_URL_MAX_CONTENT_LENGTH:
                return (
                    f"[FETCH URL ERROR] Content too large ({content_length} bytes, "
                    f"max {FETCH_URL_MAX_CONTENT_LENGTH} bytes)"
                )

            # Handle binary content (PDFs, images) if save_path provided
            # Defensive check for headers
            headers = getattr(response, 'headers', None) or {}
            content_type = headers.get('content-type', '').lower()
            is_binary = any(ct in content_type for ct in [
                'application/pdf', 'image/', 'application/octet-stream',
                'application/zip', 'application/gzip'
            ])

            if is_binary and save_path:
                # Save binary content
                file_path = Path(save_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"[fetch_url] Saved binary content ({content_length} bytes) to {save_path}")
                return f"Successfully saved binary content ({content_length} bytes) to {save_path}"

            # Handle text content
            try:
                text_content = response.text
            except Exception as e:
                return f"[FETCH URL ERROR] Could not decode content as text: {e}"

            # Sanitize content
            sanitized = _sanitize_fetched_content(text_content, clean_url)

            # Truncate if needed
            if len(sanitized) > max_length:
                sanitized = sanitized[:max_length] + f"\n\n... [truncated, {len(sanitized) - max_length} more chars]"

            logger.info(f"[fetch_url] Successfully fetched {len(sanitized)} chars from {clean_url}")
            return sanitized

    except httpx.TimeoutException:
        logger.error(f"[fetch_url] Timeout after {FETCH_URL_TIMEOUT_SECONDS}s: {clean_url}")
        return "[FETCH URL ERROR] Request timed out"
    except httpx.ConnectError as e:
        logger.error(f"[fetch_url] Connection error: {clean_url}: {e}")
        return "[FETCH URL ERROR] Could not connect to server"
    except Exception as e:
        import traceback
        logger.error(f"[fetch_url] Unexpected error fetching {clean_url}: {e}")
        logger.error(f"[fetch_url] Traceback:\n{traceback.format_exc()}")
        return "[FETCH URL ERROR] Request failed"


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
    - Registering URLs for fetch_url context validation

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

    # Collect URLs for context registration
    urls_found = []

    for i, r in enumerate(results, 1):
        # Skip None results defensively
        if r is None:
            continue
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

        # Collect URL for context registration
        if url:
            urls_found.append(url)

    lines.append("\n" + "=" * 60)
    result_text = "\n".join(lines)

    # Register URLs from search results for fetch_url validation
    if urls_found:
        _conversation_urls.update(urls_found)
        logger.info(f"[web_search] Registered {len(urls_found)} URLs for fetch_url context")

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


def fetch_url_tool(
    url: str,
    max_length: int = 10000,
    save_path: Optional[str] = None,
) -> str:
    """
    Synchronous wrapper for fetch_url tool interface.

    Fetches content from a URL with comprehensive security guardrails:
    - HTTPS only
    - URL must be from conversation context
    - Homograph attack detection
    - Content sanitization (XXE prevention, prompt injection protection)
    - Size limits (500KB max)

    Args:
        url: URL to fetch. Must be from web_search results or user messages.
        max_length: Maximum characters to return (default 10000, max 50000).
        save_path: Optional: save binary content (PDFs) to this path.

    Returns:
        Fetched and sanitized content, or error message.
    """
    import asyncio

    # Clamp max_length
    max_length = min(max(100, max_length), 50000)

    # Run async function (handle both sync and async contexts)
    try:
        loop = asyncio.get_running_loop()
        # We're inside an async context (e.g., FastAPI, Jupyter)
        # Use nest_asyncio if available, otherwise run in thread
        try:
            import nest_asyncio
            nest_asyncio.apply(loop)
            return loop.run_until_complete(fetch_url(url, max_length, save_path))
        except ImportError:
            # Run in a separate thread to avoid event loop conflicts
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    lambda: asyncio.run(fetch_url(url, max_length, save_path))
                )
                return future.result()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(fetch_url(url, max_length, save_path))
    except Exception as e:
        logger.error(f"[fetch_url_tool] Error: {e}")
        return f"[FETCH URL ERROR] {type(e).__name__}: {str(e)[:200]}"


def write_file(path: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        path: The file path to write to (required, cannot be empty)
        content: The content to write (required)

    Returns:
        Success message or error
    """
    # Validate arguments
    if not path or not isinstance(path, str) or not path.strip():
        return "[TOOL ERROR] write_file: 'path' is required and cannot be empty. Example: path='output/report.md'"

    if content is None:
        return "[TOOL ERROR] write_file: 'content' is required and cannot be None."

    try:
        file_path = Path(path)

        # Prevent writing to directories or dangerous paths
        if file_path.is_dir():
            return f"[TOOL ERROR] write_file: '{path}' is a directory, not a file. Please specify a file path."

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(content)

        byte_count = len(content.encode("utf-8"))
        logger.info(f"[write_file] Wrote {byte_count} bytes to {path}")

        return f"Successfully wrote {byte_count} bytes to {path}"

    except IOError as e:
        return f"[TOOL ERROR] write_file failed: {e}"
    except Exception as e:
        return f"[TOOL ERROR] write_file unexpected error: {e}"


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


# =============================================================================
# BROWSER AUTOMATION TOOLS
# =============================================================================

async def browser_navigate(
    url: str,
    action: str = "navigate",
    full_page: bool = True,
    wait_ms: int = 1000,
    headless: bool = True,
    output_dir: str = "screenshots",
    use_cdp: bool = False,
    cdp_endpoint: Optional[str] = None,
    wait_until: str = "networkidle",
) -> str:
    """
    Navigate to a URL using Playwright and capture a screenshot.

    SECURITY FEATURES:
    - HTTPS only (same validation as fetch_url)
    - URL context validation (configurable via FETCH_ONLY_MENTIONED_URLS)
    - Internal IP blocking
    - Homograph attack detection
    - Cloud metadata endpoint blocking

    CDP MODE (connect to existing browser):
    - Set use_cdp=True or BROWSER_CONNECT_OVER_CDP=true
    - Set cdp_endpoint or BROWSER_CDP_ENDPOINT (default: http://localhost:9222)
    - Preserves your logged-in sessions, cookies, and saved passwords
    - Does NOT close the browser when done (your browser stays open)

    Args:
        url: URL to navigate to. Must be HTTPS and from conversation context.
        action: Short description for filename (default: "navigate")
        full_page: Capture full page vs viewport only (default: True)
        wait_ms: Additional wait time after load for JS-heavy pages (default: 1000)
        headless: Run browser in headless mode (default: True). Ignored when using CDP.
        output_dir: Directory to save screenshots (default: "screenshots")
        use_cdp: Connect to existing browser via CDP instead of launching new
        cdp_endpoint: CDP endpoint URL (default: http://localhost:9222)
        wait_until: When to consider navigation complete: 'networkidle' (default),
                   'domcontentloaded' (faster, for JS-heavy sites), or 'load'

    Returns:
        Path to saved screenshot or error message.
    """
    # Validate wait_until parameter
    VALID_WAIT_STRATEGIES = {"networkidle", "domcontentloaded", "load", "commit"}
    if wait_until not in VALID_WAIT_STRATEGIES:
        return f"[BROWSER ERROR] Invalid wait_until value: '{wait_until}'. Must be one of {VALID_WAIT_STRATEGIES}"

    if not PLAYWRIGHT_AVAILABLE:
        return (
            "[BROWSER ERROR] Playwright not installed. "
            "Install with: pip install playwright && playwright install chromium"
        )

    # Validate URL (same security as fetch_url)
    is_valid, error_msg, clean_url = validate_url_for_fetch(
        url, require_context=FETCH_ONLY_MENTIONED_URLS
    )
    if not is_valid:
        logger.warning(f"[browser_navigate] Validation failed for '{url}': {error_msg}")
        return f"[BROWSER ERROR] {error_msg}"

    # Clamp wait time
    wait_ms = max(0, min(wait_ms, 30000))  # 0-30 seconds

    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build filename from action
    safe_action = re.sub(r"[^a-zA-Z0-9_-]", "_", action)[:50]
    timestamp = int(time.time())
    filename = f"browser_{safe_action}_{timestamp}.png"
    screenshot_path = output_path / filename

    logger.info(f"[browser_navigate] Launching browser for: {clean_url}")

    # Determine if we should use CDP mode
    cdp_url = cdp_endpoint or BROWSER_CDP_ENDPOINT or "http://localhost:9222"
    should_use_cdp = use_cdp or BROWSER_CONNECT_OVER_CDP

    try:
        async with async_playwright() as pw:
            if should_use_cdp:
                # CDP MODE: Connect to existing browser
                logger.info(f"[browser_navigate] Connecting via CDP to {cdp_url}")
                try:
                    browser = await pw.chromium.connect_over_cdp(cdp_url)
                except Exception as e:
                    logger.error(f"[browser_navigate] Failed to connect via CDP: {e}")
                    return (
                        f"[BROWSER ERROR] Could not connect to browser at {cdp_url}. "
                        f"Make sure Chrome/Chromium is running with --remote-debugging-port=9222. "
                        f"Error: {str(e)[:100]}"
                    )

                # Reuse existing context (your logged-in session) or create new if none exists
                contexts = browser.contexts
                if contexts:
                    context = contexts[0]
                    logger.info("[browser_navigate] Reusing existing browser context (with your sessions/cookies)")
                else:
                    context = await browser.new_context(
                        viewport={"width": 1280, "height": 800},
                    )
                    logger.info("[browser_navigate] Created new context in connected browser")

                page = await context.new_page()

                try:
                    # Navigate and wait for network idle
                    await page.goto(
                        clean_url,
                        wait_until=wait_until,
                        timeout=30000,
                    )

                    # Additional wait for JS-heavy pages
                    if wait_ms > 0:
                        await page.wait_for_timeout(wait_ms)

                    # Capture screenshot
                    await page.screenshot(
                        path=str(screenshot_path),
                        full_page=full_page,
                    )

                    logger.info(f"[browser_navigate] Screenshot saved: {screenshot_path}")

                finally:
                    await page.close()
                    # NOTE: Do NOT close browser in CDP mode - it would kill your browser!
                    logger.info("[browser_navigate] Page closed (browser left running)")

            else:
                # STANDARD MODE: Launch fresh browser
                logger.info("[browser_navigate] Launching fresh browser instance")
                browser = await pw.chromium.launch(
                    headless=headless,
                    args=["--no-sandbox", "--disable-setuid-sandbox"],
                )
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                )
                page = await context.new_page()

                try:
                    # Navigate and wait for network idle
                    await page.goto(
                        clean_url,
                        wait_until=wait_until,
                        timeout=30000,
                    )

                    # Additional wait for JS-heavy pages
                    if wait_ms > 0:
                        await page.wait_for_timeout(wait_ms)

                    # Capture screenshot
                    await page.screenshot(
                        path=str(screenshot_path),
                        full_page=full_page,
                    )

                    logger.info(f"[browser_navigate] Screenshot saved: {screenshot_path}")

                finally:
                    await context.close()
                    await browser.close()

        mode_info = " (CDP mode - used your logged-in browser)" if should_use_cdp else ""
        return f"Successfully captured screenshot of {clean_url}{mode_info}\nSaved to: {screenshot_path}"

    except Exception as e:
        logger.error(f"[browser_navigate] Error: {e}")
        return f"[BROWSER ERROR] {type(e).__name__}: {str(e)[:200]}"


def browser_navigate_tool(
    url: str,
    action: str = "navigate",
    full_page: bool = True,
    wait_ms: int = 1000,
    headless: bool = True,
    use_cdp: bool = False,
    cdp_endpoint: Optional[str] = None,
    wait_until: str = "networkidle",
    output_dir: str = "screenshots",
) -> str:
    """
    Synchronous wrapper for browser_navigate tool interface.

    CDP MODE:
    - Set use_cdp=True to connect to an existing browser via CDP
    - Or set BROWSER_CDP_ENDPOINT environment variable
    - Preserves your logged-in sessions, cookies, and saved passwords

    Args:
        url: URL to navigate to. Must be HTTPS and from conversation context.
        action: Short description for filename (default: "navigate")
        full_page: Capture full page vs viewport only (default: True)
        wait_ms: Additional wait time after load for JS-heavy pages (default: 1000)
        headless: Run browser in headless mode (default: True). Ignored when using CDP.
        use_cdp: Connect to existing browser via CDP instead of launching new
        cdp_endpoint: Custom CDP endpoint URL (default: http://localhost:9222)
        wait_until: When to consider navigation complete: 'networkidle' (default),
                   'domcontentloaded' (faster, for JS-heavy sites), or 'load'
        output_dir: Directory to save screenshots (default: "screenshots")

    Returns:
        Path to saved screenshot or error message.
    """
    # Run async function (handle both sync and async contexts)
    try:
        loop = asyncio.get_running_loop()
        try:
            import nest_asyncio
            nest_asyncio.apply(loop)
            return loop.run_until_complete(
                browser_navigate(url, action, full_page, wait_ms, headless, output_dir, use_cdp, cdp_endpoint, wait_until)
            )
        except ImportError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    lambda: asyncio.run(
                        browser_navigate(url, action, full_page, wait_ms, headless, output_dir, use_cdp, cdp_endpoint, wait_until)
                    )
                )
                return future.result()
    except RuntimeError:
        return asyncio.run(
            browser_navigate(url, action, full_page, wait_ms, headless, output_dir, use_cdp, cdp_endpoint, wait_until)
        )
    except Exception as e:
        logger.error(f"[browser_navigate_tool] Error: {e}")
        return f"[BROWSER ERROR] {type(e).__name__}: {str(e)[:200]}"
    