"""
Browser automation tools for interactive web browsing.

Provides tools for:
- Clicking at coordinates
- Scrolling pages
- Typing text
- Pressing keys
- Reading page content
- Capturing screenshots
- Managing browser sessions

Uses Playwright for browser automation with CDP support for
connecting to existing browser sessions.
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ask-swarm")

# Optional Playwright import - graceful degradation if not installed
try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = None
    Browser = None
    BrowserContext = None

from core.config import (
    BROWSER_CDP_ENDPOINT,
    BROWSER_CONNECT_OVER_CDP,
)


# =============================================================================
# BROWSER SESSION MANAGER
# =============================================================================

class BrowserSession:
    """
    Manages browser lifecycle across tool calls.

    Singleton pattern ensures browser persists between tool calls,
    enabling multi-step automation workflows.

    Supports two modes:
    - CDP mode: Connect to external Chrome (preserves sessions/cookies)
    - Standard mode: Launch fresh Chromium instance

    Usage:
        session = BrowserSession.get_instance()
        page = await session.get_page(use_cdp=True)
        # ... use page ...
        await session.close()  # Optional - explicit cleanup
    """

    _instance: Optional["BrowserSession"] = None

    def __init__(self):
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._playwright = None
        self._cdp_mode = False
        self._headless = True
        self._cdp_endpoint: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "BrowserSession":
        """Get singleton instance of BrowserSession."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing or cleanup)."""
        if cls._instance is not None:
            asyncio.create_task(cls._instance.close())
        cls._instance = None

    async def get_page(
        self,
        use_cdp: bool = False,
        cdp_endpoint: Optional[str] = None,
        headless: bool = True,
    ) -> Page:
        """
        Get or create a browser page.

        Args:
            use_cdp: Connect to existing browser via CDP
            cdp_endpoint: CDP endpoint URL (default: from config)
            headless: Run headless (ignored in CDP mode)

        Returns:
            Playwright Page object ready for interaction
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright not installed. "
                "Install with: pip install playwright && playwright install chromium"
            )

        # Check if existing page is still valid
        if self._page and not self._page.is_closed():
            return self._page

        # Need to create new session
        self._headless = headless
        self._cdp_mode = use_cdp or BROWSER_CONNECT_OVER_CDP
        self._cdp_endpoint = cdp_endpoint or BROWSER_CDP_ENDPOINT or "http://localhost:9222"

        if self._playwright is None:
            self._playwright = await async_playwright().start()

        if self._cdp_mode:
            await self._connect_cdp()
        else:
            await self._launch_browser()

        return self._page

    async def _connect_cdp(self):
        """Connect to existing browser via CDP."""
        logger.info(f"[BrowserSession] Connecting via CDP to {self._cdp_endpoint}")

        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(self._cdp_endpoint)
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to browser at {self._cdp_endpoint}. "
                f"Make sure Chrome is running with --remote-debugging-port=9222. "
                f"Error: {str(e)[:100]}"
            )

        # Reuse existing context (your logged-in session) or create new
        contexts = self._browser.contexts
        if contexts:
            self._context = contexts[0]
            logger.info("[BrowserSession] Reusing existing browser context")
        else:
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 800},
            )
            logger.info("[BrowserSession] Created new context in connected browser")

        self._page = await self._context.new_page()
        logger.info("[BrowserSession] Connected via CDP")

    async def _launch_browser(self):
        """Launch fresh browser instance."""
        logger.info("[BrowserSession] Launching fresh browser instance")

        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()
        logger.info("[BrowserSession] Browser launched")

    async def navigate(
        self,
        url: str,
        wait_until: str = "networkidle",
        timeout: int = 30000,
    ) -> str:
        """
        Navigate to URL and return current page title.

        Args:
            url: URL to navigate to
            wait_until: Navigation wait strategy
            timeout: Timeout in milliseconds

        Returns:
            Page title after navigation
        """
        page = await self.get_page()
        await page.goto(url, wait_until=wait_until, timeout=timeout)
        return await page.title()

    async def close(self):
        """Close browser session completely."""
        if self._cdp_mode:
            # In CDP mode, only close the page, not the browser
            if self._page:
                try:
                    await self._page.close()
                except Exception:
                    pass
            logger.info("[BrowserSession] Page closed (browser kept running)")
        else:
            # In standard mode, close everything
            if self._context:
                try:
                    await self._context.close()
                except Exception:
                    pass
            if self._browser:
                try:
                    await self._browser.close()
                except Exception:
                    pass
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
            logger.info("[BrowserSession] Browser closed")

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    @property
    def has_page(self) -> bool:
        """Check if there's an active page."""
        if self._page is None:
            return False
        try:
            # is_closed() may fail if page is from different event loop
            return not self._page.is_closed()
        except Exception:
            # Page is likely from a different event loop, treat as invalid
            self._page = None
            return False


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": (
                "Click at specific coordinates on the current browser page. "
                "Use after browser_navigate to interact with page elements. "
                "Coordinates are in pixels from top-left corner (0,0). "
                "Returns success confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "X coordinate in pixels from left edge"
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y coordinate in pixels from top edge"
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "default": "left",
                        "description": "Mouse button to use"
                    },
                    "double": {
                        "type": "boolean",
                        "default": False,
                        "description": "Perform double-click"
                    }
                },
                "required": ["x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": (
                "Scroll the browser page up or down. "
                "Use to read long content that extends beyond the viewport. "
                "Returns approximate scroll position after action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "default": "down",
                        "description": "Direction to scroll"
                    },
                    "amount": {
                        "type": "integer",
                        "default": 500,
                        "description": "Number of pixels to scroll (default: 500)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": (
                "Type text into the currently focused element. "
                "Use after clicking on an input field or textarea. "
                "Optionally clears existing content before typing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to type"
                    },
                    "clear_first": {
                        "type": "boolean",
                        "default": False,
                        "description": "Clear existing content before typing"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_press",
            "description": (
                "Press a keyboard key in the browser. "
                "Use for actions like Enter (submit forms), Escape (close modals), "
                "Tab (navigate between elements), or key combinations like 'Control+a'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "Key to press. Examples: 'Enter', 'Escape', 'Tab', "
                            "'Backspace', 'ArrowDown', 'Control+a', 'Control+c'"
                        )
                    }
                },
                "required": ["key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_read_page",
            "description": (
                "Extract text content from the current browser page. "
                "Returns readable text from the page body. "
                "Use to read posts, articles, or any page content. "
                "Optionally filter to a specific element using CSS selector."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to read specific element (e.g., 'article', '.post-content')"
                    },
                    "max_length": {
                        "type": "integer",
                        "default": 10000,
                        "description": "Maximum characters to return (default: 10000, max: 50000)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": (
                "Capture a screenshot of the current browser page. "
                "Use to verify page state, capture visual content, or debug. "
                "Returns path to saved screenshot file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "full_page": {
                        "type": "boolean",
                        "default": True,
                        "description": "Capture full page (true) or viewport only (false)"
                    },
                    "output_dir": {
                        "type": "string",
                        "default": "screenshots",
                        "description": "Directory to save screenshot"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_close",
            "description": (
                "Close the browser session. "
                "Call when done with browser automation to free resources. "
                "In CDP mode, only closes the page (your browser keeps running)."
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
            "name": "browser_navigate",
            "description": (
                "Navigate to a URL using Playwright browser automation and capture a screenshot. "
                "Useful for: verifying page content visually, checking UI state, capturing proof of actions, "
                "debugging web issues, or archiving page state. "
                "Supports connecting to an existing browser via CDP (Chrome DevTools Protocol) - "
                "set BROWSER_CDP_ENDPOINT=http://localhost:9222 to use your logged-in browser session. "
                "Set keep_session=true to keep the browser open for subsequent browser_* tools (click, scroll, type, etc.). "
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
                    },
                    "keep_session": {
                        "type": "boolean",
                        "description": "Keep browser session open for subsequent browser_* tools (click, scroll, type, read_page). Call browser_close when done.",
                        "default": False
                    }
                },
                "required": ["url"]
            }
        }
    },
]


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

async def browser_click(
    x: int,
    y: int,
    button: str = "left",
    double: bool = False,
) -> str:
    """
    Click at specific coordinates on the page.

    Args:
        x: X coordinate in pixels
        y: Y coordinate in pixels
        button: Mouse button ('left', 'right', 'middle')
        double: Perform double-click

    Returns:
        Success message
    """
    session = BrowserSession.get_instance()

    try:
        page = await session.get_page()

        if double:
            await page.mouse.dblclick(x, y, button=button)
        else:
            await page.mouse.click(x, y, button=button)

        # Wait briefly for any JS handlers
        await page.wait_for_timeout(500)

        logger.info(f"[browser_click] Clicked at ({x}, {y}) with {button} button")
        return f"Clicked at ({x}, {y}) with {button} button"

    except Exception as e:
        logger.error(f"[browser_click] Error: {e}")
        return f"[BROWSER ERROR] Click failed: {str(e)[:200]}. For multi-step workflows, use CDP mode (BROWSER_CDP_ENDPOINT=http://localhost:9222)."


async def browser_scroll(
    direction: str = "down",
    amount: int = 500,
) -> str:
    """
    Scroll the page.

    Args:
        direction: 'up' or 'down'
        amount: Pixels to scroll

    Returns:
        Message with new scroll position
    """
    session = BrowserSession.get_instance()

    try:
        page = await session.get_page()

        delta = amount if direction == "down" else -amount
        await page.mouse.wheel(0, delta)

        # Wait for scroll to complete
        await page.wait_for_timeout(300)

        # Get current scroll position
        scroll_y = await page.evaluate("() => window.scrollY")
        scroll_height = await page.evaluate("() => document.body.scrollHeight")
        viewport_height = await page.evaluate("() => window.innerHeight")

        logger.info(f"[browser_scroll] Scrolled {direction} by {amount}px")
        return (
            f"Scrolled {direction} by {amount}px. "
            f"Current position: {scroll_y}px of {scroll_height}px "
            f"(viewport: {viewport_height}px)"
        )

    except Exception as e:
        logger.error(f"[browser_scroll] Error: {e}")
        return f"[BROWSER ERROR] Scroll failed: {str(e)[:200]}. For multi-step workflows, use CDP mode (BROWSER_CDP_ENDPOINT=http://localhost:9222)."


async def browser_type(
    text: str,
    clear_first: bool = False,
) -> str:
    """
    Type text into focused element.

    Args:
        text: Text to type
        clear_first: Clear existing content first

    Returns:
        Success message
    """
    session = BrowserSession.get_instance()

    try:
        page = await session.get_page()

        if clear_first:
            # Select all and delete
            await page.keyboard.press("ControlOrMeta+a")
            await page.keyboard.press("Backspace")
            await page.wait_for_timeout(100)

        await page.keyboard.type(text, delay=50)
        await page.wait_for_timeout(200)

        logger.info(f"[browser_type] Typed {len(text)} characters")
        return f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}"

    except Exception as e:
        logger.error(f"[browser_type] Error: {e}")
        return f"[BROWSER ERROR] Type failed: {str(e)[:200]}. For multi-step workflows, use CDP mode (BROWSER_CDP_ENDPOINT=http://localhost:9222)."


async def browser_press(key: str) -> str:
    """
    Press a keyboard key.

    Args:
        key: Key name (Enter, Escape, Tab, etc.)

    Returns:
        Success message
    """
    session = BrowserSession.get_instance()

    try:
        page = await session.get_page()

        await page.keyboard.press(key)

        # Wait for action to process
        await page.wait_for_timeout(500)

        logger.info(f"[browser_press] Pressed: {key}")
        return f"Pressed key: {key}"

    except Exception as e:
        logger.error(f"[browser_press] Error: {e}")
        return f"[BROWSER ERROR] Key press failed: {str(e)[:200]}. For multi-step workflows, use CDP mode (BROWSER_CDP_ENDPOINT=http://localhost:9222)."


async def browser_read_page(
    selector: Optional[str] = None,
    max_length: int = 10000,
) -> str:
    """
    Extract text content from the page.

    Args:
        selector: Optional CSS selector to target specific element
        max_length: Maximum characters to return

    Returns:
        Extracted text content
    """
    session = BrowserSession.get_instance()

    # Clamp max_length
    max_length = min(max(100, max_length), 50000)

    try:
        # Try to get or create a page
        # Note: For reliable multi-step workflows, use CDP mode (connect to external browser)
        page = await session.get_page()

        if selector:
            # Get text from specific element
            element = page.locator(selector).first
            text = await element.inner_text()
        else:
            # Get all text from body
            text = await page.evaluate("""
                () => {
                    // Get all text nodes, excluding script/style
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT,
                        {
                            acceptNode: (node) => {
                                const parent = node.parentElement;
                                if (!parent) return NodeFilter.FILTER_REJECT;
                                const tag = parent.tagName.toLowerCase();
                                if (['script', 'style', 'noscript', 'svg'].includes(tag)) {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                if (getComputedStyle(parent).display === 'none') {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                return NodeFilter.FILTER_ACCEPT;
                            }
                        }
                    );

                    const chunks = [];
                    while (walker.nextNode()) {
                        const txt = walker.currentNode.textContent.trim();
                        if (txt) chunks.push(txt);
                    }
                    return chunks.join('\\n');
                }
            """)

        # Clean up text
        text = text.strip()
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + f"\n\n... [truncated, {len(text) - max_length} more chars]"

        logger.info(f"[browser_read_page] Extracted {len(text)} characters")
        return text

    except Exception as e:
        logger.error(f"[browser_read_page] Error: {e}")
        return f"[BROWSER ERROR] Read page failed: {str(e)[:200]}. For multi-step workflows, use CDP mode (BROWSER_CDP_ENDPOINT=http://localhost:9222)."


async def browser_screenshot(
    full_page: bool = True,
    output_dir: str = "screenshots",
) -> str:
    """
    Capture a screenshot of the current page.

    Args:
        full_page: Capture full page or viewport only
        output_dir: Directory to save screenshot

    Returns:
        Path to saved screenshot
    """
    session = BrowserSession.get_instance()

    try:
        page = await session.get_page()

        # Prepare output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build filename
        timestamp = int(time.time())
        filename = f"browser_screenshot_{timestamp}.png"
        screenshot_path = output_path / filename

        await page.screenshot(
            path=str(screenshot_path),
            full_page=full_page,
        )

        logger.info(f"[browser_screenshot] Saved to {screenshot_path}")
        return f"Screenshot saved to: {screenshot_path}"

    except Exception as e:
        logger.error(f"[browser_screenshot] Error: {e}")
        return f"[BROWSER ERROR] Screenshot failed: {str(e)[:200]}. For multi-step workflows, use CDP mode (BROWSER_CDP_ENDPOINT=http://localhost:9222)."


async def browser_close() -> str:
    """
    Close the browser session.

    Returns:
        Success message
    """
    session = BrowserSession.get_instance()

    try:
        await session.close()
        logger.info("[browser_close] Browser session closed")
        return "Browser session closed"

    except Exception as e:
        logger.error(f"[browser_close] Error: {e}")
        return f"[BROWSER ERROR] Close failed: {str(e)[:200]}"


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
    keep_session: bool = False,
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

    SESSION PERSISTENCE:
    - Set keep_session=True to keep browser open for subsequent browser_* tools
    - When True, the browser session is shared and can be used by browser_click,
      browser_scroll, browser_type, browser_read_page, etc.
    - Call browser_close when done with the session.

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
        keep_session: Keep browser open for subsequent browser_* tools (default: False)

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
    from tools.web_tools import validate_url_for_fetch
    from core.config import FETCH_ONLY_MENTIONED_URLS

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

    # If keep_session is True, use shared BrowserSession for multi-step workflows
    if keep_session:
        try:
            session = BrowserSession.get_instance()
            page = await session.get_page(use_cdp=should_use_cdp, cdp_endpoint=cdp_url, headless=headless)

            # Navigate
            await page.goto(clean_url, wait_until=wait_until, timeout=30000)
            if wait_ms > 0:
                await page.wait_for_timeout(wait_ms)

            # Capture screenshot
            await page.screenshot(path=str(screenshot_path), full_page=full_page)

            mode_info = " (CDP mode - used your logged-in browser)" if should_use_cdp else " (session kept open for browser_* tools)"
            logger.info(f"[browser_navigate] Screenshot saved: {screenshot_path}")
            return f"Successfully captured screenshot of {clean_url}{mode_info}\nSaved to: {screenshot_path}"

        except Exception as e:
            logger.error(f"[browser_navigate] Error with keep_session: {e}")
            return f"[BROWSER ERROR] {type(e).__name__}: {str(e)[:200]}"

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


# =============================================================================
# SYNC WRAPPERS (for ToolExecutor)
# =============================================================================

def _run_async(coro):
    """Run async coroutine, handling event loop complexity."""
    try:
        loop = asyncio.get_running_loop()
        # We're inside an async context
        try:
            import nest_asyncio
            nest_asyncio.apply(loop)
            return loop.run_until_complete(coro)
        except ImportError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coro)


def browser_click_tool(
    x: int,
    y: int,
    button: str = "left",
    double: bool = False,
) -> str:
    """Synchronous wrapper for browser_click."""
    return _run_async(browser_click(x, y, button, double))


def browser_scroll_tool(
    direction: str = "down",
    amount: int = 500,
) -> str:
    """Synchronous wrapper for browser_scroll."""
    return _run_async(browser_scroll(direction, amount))


def browser_type_tool(
    text: str,
    clear_first: bool = False,
) -> str:
    """Synchronous wrapper for browser_type."""
    return _run_async(browser_type(text, clear_first))


def browser_press_tool(key: str) -> str:
    """Synchronous wrapper for browser_press."""
    return _run_async(browser_press(key))


def browser_read_page_tool(
    selector: Optional[str] = None,
    max_length: int = 10000,
) -> str:
    """Synchronous wrapper for browser_read_page."""
    return _run_async(browser_read_page(selector, max_length))


def browser_screenshot_tool(
    full_page: bool = True,
    output_dir: str = "screenshots",
) -> str:
    """Synchronous wrapper for browser_screenshot."""
    return _run_async(browser_screenshot(full_page, output_dir))


def browser_close_tool() -> str:
    """Synchronous wrapper for browser_close."""
    return _run_async(browser_close())


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
    keep_session: bool = False,
) -> str:
    """Synchronous wrapper for browser_navigate tool interface."""
    return _run_async(
        browser_navigate(url, action, full_page, wait_ms, headless, output_dir, use_cdp, cdp_endpoint, wait_until, keep_session)
    )