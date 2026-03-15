"""
main.py
───────
AgeMem Interactive Chat — REPL entry point for the AgeMem-hybrid system.

Usage:
    python main.py

Keybindings:
    Enter          Send message
    Alt+Enter      Insert newline (for multiline messages)
    Escape+Enter   Insert newline (alternative)
    Ctrl+C         Cancel current input or exit

Environment variables (all optional, with defaults):
    BASE_URL            Base URL for LLM API (default: http://localhost:8080)
    BASE_MODEL          Model name (default: qwen3-4b)
    BASE_MAX_TOKENS     Max tokens (default: 2048)
    BASE_TEMPERATURE    Temperature (default: 0.2)
    API_KEY             API key for non-local endpoints
    OPENAI_API_KEY      Fallback API key for OpenAI-compatible endpoints

Deprecated (still supported for backward compatibility):
    LLAMA_HOST          Use BASE_URL instead
    LLAMA_MODEL         Use BASE_MODEL instead
    LLAMA_MAX_TOKENS    Use BASE_MAX_TOKENS instead
    LLAMA_TEMPERATURE   Use BASE_TEMPERATURE instead

Other settings:
    WEB_SEARCH_MAX_RESULTS  default: 5
    TOOL_RESULT_MAX_CHARS   default: 4000
    PERSIST_DIR             default: agent_memory  (for LTM + STM storage)

Notes on persistence:
    Both LTM (long-term memory) and STM (short-term context) are stored in
    the PERSIST_DIR directory:
    - {PERSIST_DIR}/ltm_store.json  (persists across sessions)
    - {PERSIST_DIR}/stm_context.json (persists across sessions)

    The old LTM_PERSIST_PATH environment variable is deprecated. Use PERSIST_DIR
    to configure the storage location for both memory systems.
"""

from __future__ import annotations

import os
import re
import sys
import signal
import time
import warnings
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import markup as rich_markup
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML

# ── Tracing System ──────────────────────────────────────────────────────────
from core.tracing import init_tracing, get_tracer, shutdown_tracing

# ── Text Cleaning Utilities ─────────────────────────────────────────────────
from cli_text_utils import clean_pasted_text, is_likely_paste


# ── Configuration from environment ───────────────────────────────────────────

def get_env_with_fallback(new_name: str, old_name: str, default: str) -> str:
    """Get env var with fallback to old name for backward compatibility."""
    value = os.getenv(new_name) or os.getenv(old_name, default)
    if os.getenv(old_name) and not os.getenv(new_name):
        warnings.warn(
            f"Environment variable '{old_name}' is deprecated. "
            f"Please use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
    return value

# New generic env vars with fallback to old LLAMA_* names
BASE_URL = get_env_with_fallback("BASE_URL", "LLAMA_HOST", "http://localhost:8080")
BASE_MODEL = get_env_with_fallback("BASE_MODEL", "LLAMA_MODEL", "qwen3-9b")
BASE_MAX_TOKENS = int(get_env_with_fallback("BASE_MAX_TOKENS", "LLAMA_MAX_TOKENS", "2048"))
BASE_TEMPERATURE = float(get_env_with_fallback("BASE_TEMPERATURE", "LLAMA_TEMPERATURE", "0.2"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "4000"))

# Persistence directory - MUST be consistent between LTM and STM
# Both memories use this directory:
#   - {PERSIST_DIR}/ltm_store.json  (LTM entries)
#   - {PERSIST_DIR}/stm_context.json (STM context)
PERSIST_DIR = os.getenv("PERSIST_DIR", "agent_memory")

# Deprecated: LTM_PERSIST_PATH is no longer used. LTM and STM both use PERSIST_DIR.
# Kept for backward compatibility check only.
_LEGACY_LTM_PATH = os.getenv("LTM_PERSIST_PATH")
if _LEGACY_LTM_PATH:
    warnings.warn(
        "LTM_PERSIST_PATH is deprecated. Both LTM and STM now use PERSIST_DIR. "
        f"Set PERSIST_DIR='{Path(_LEGACY_LTM_PATH).parent}' instead.",
        DeprecationWarning,
        stacklevel=2
    )

STM_TOKEN_LIMIT = int(os.getenv("STM_TOKEN_LIMIT", "6000"))

# Tracing configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
TRACE_LOG_DIR = os.getenv("TRACE_LOG_DIR", "logs")
TRACE_RETENTION_DAYS = int(os.getenv("TRACE_RETENTION_DAYS", "30"))


# ── Imports from AgeMem package ──────────────────────────────────────────────

from core.config import AgememConfig
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator


# ── Tool Definitions ─────────────────────────────────────────────────────────

WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Returns top results with title, URL, and snippet. "
            "Use this when the user asks about current events, news, or information that may not be "
            "in your training data. Results are capped at {} characters.".format(TOOL_RESULT_MAX_CHARS)
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
                "num_results": {"type": "integer", "description": f"Number of results (default {WEB_SEARCH_MAX_RESULTS}, max 10)."}
            },
            "required": ["query"]
        }
    }
}

# Import all tool definitions
from tools.corpus import tool_definitions as CORPUS_TOOL_DEFINITIONS
from tools.web_tools import tool_definitions as WEB_TOOL_DEFINITIONS
from memory import introspection_tool_definitions

def get_llm_client() -> OpenAI:
    """Get the OpenAI-compatible client with automatic API key handling.

    For local endpoints (localhost/127.0.0.1), no API key is required.
    For remote endpoints, API_KEY or OPENAI_API_KEY environment variable is required.
    """
    base_url = BASE_URL

    # Determine if this is a local endpoint
    is_local = "localhost" in base_url or "127.0.0.1" in base_url

    if is_local:
        api_key = "not-needed"
    else:
        # Check API_KEY first, then fall back to OPENAI_API_KEY
        api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"API_KEY or OPENAI_API_KEY environment variable is required "
                f"for non-local endpoint: {base_url}"
            )

    # Ensure base_url ends with /v1 for OpenAI compatibility
    base_url_normalized = base_url.rstrip("/")
    if not base_url_normalized.endswith("/v1"):
        base_url_normalized = f"{base_url_normalized}/v1"

    return OpenAI(api_key=api_key, base_url=base_url_normalized)

# ── Build Orchestrator ───────────────────────────────────────────────────────

def build_orchestrator() -> Orchestrator:
    """
    Wire up AgeMem-hybrid with the configured LLM provider.
    """

    client = get_llm_client()

    cfg = AgememConfig(
        DEFAULT_MODEL=BASE_MODEL,
        MEMORY_AGENT_MODEL=BASE_MODEL,
        STM_TOKEN_LIMIT=STM_TOKEN_LIMIT,
        STM_WARNING_THRESHOLD=0.75,
        STM_CRITICAL_THRESHOLD=0.90,
        LTM_PROMOTE_THRESHOLD=0.65,
        LEARNING_SCORE_PROMPT_EVERY_N=5,
        TRIGGER_EVERY_N_TURNS=10,
        DEFAULT_MAX_TOKENS=BASE_MAX_TOKENS,
        DEFAULT_TEMPERATURE=BASE_TEMPERATURE,
        PERSIST_DIR=PERSIST_DIR,
    )

    llm = LLMClient(client, default_model=cfg.DEFAULT_MODEL)

    # Orchestrator handles both LTM and STM persistence via config.PERSIST_DIR
    # This ensures both memories use the SAME directory (coherence)
    orch = Orchestrator(llm=llm, config=cfg)

    # Set up tools - combine all tool definitions
    all_tools = (
        WEB_TOOL_DEFINITIONS +
        CORPUS_TOOL_DEFINITIONS +
        introspection_tool_definitions
    )
    orch.set_tools(all_tools)

    return orch


# ── Rich Console Setup ───────────────────────────────────────────────────────

console = Console()


def get_history_path() -> Path:
    """Get path to command history file."""
    # Use XDG_CACHE_HOME if set, otherwise ~/.cache
    cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "agemem"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "history"


def create_prompt_session() -> PromptSession:
    """Create a PromptSession with multiline support, history, and paste handling.

    Keybindings:
    - Enter: submit message
    - Alt+Enter / Escape+Enter: insert newline
    - Bracketed paste: automatically handles multi-line paste
    """
    bindings = KeyBindings()

    @bindings.add('enter')
    def _(event):
        """Enter submits the message (even in multiline mode)."""
        event.current_buffer.validate_and_handle()

    @bindings.add('escape', 'enter')
    def _(event):
        """Escape followed by Enter inserts a newline."""
        event.current_buffer.insert_text('\n')

    @bindings.add('c-j')  # Ctrl+J = Meta+Enter on some terminals
    def _(event):
        """Alt+Enter inserts a newline (terminal-dependent)."""
        event.current_buffer.insert_text('\n')

    history_path = get_history_path()

    session = PromptSession(
        history=FileHistory(str(history_path)),
        key_bindings=bindings,
        multiline=True,
        mouse_support=False,
        prompt_continuation="... ",
    )

    return session


# ── Diagnostics Display ──────────────────────────────────────────────────────

def format_progress_bar(ratio: float, width: int = 10) -> str:
    """Format a progress bar string."""
    filled = int(ratio * width)
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


def print_diagnostics(orch: Orchestrator):
    """Print per-turn diagnostics after assistant response."""
    trace = orch.last_trace()
    if not trace:
        return

    stats = trace.stm_stats_after
    ltm_count = len(orch.ltm_snapshot())
    turn = trace.turn_index

    # Progress bar for STM
    bar = format_progress_bar(stats.utilisation_ratio)

    # Use dimmed color for diagnostics
    console.print()
    diag_text = Text()
    diag_text.append("  [")
    diag_text.append(f"STM {bar} {stats.utilisation_ratio:.0%} ~{stats.total_tokens}tok", style="dim")
    diag_text.append(" | ")
    diag_text.append(f"LTM {ltm_count} entries", style="dim")
    diag_text.append(" | ")
    diag_text.append(f"turn {turn}", style="dim")
    diag_text.append("]")
    console.print(diag_text)

    # Memory ops
    for op in trace.ops_applied:
        if op.success:
            trigger_name = op.trigger.value if hasattr(op.trigger, 'value') else str(op.trigger)
            console.print(f"  [dim][MEM] {op.op.value.upper()} triggered by {trigger_name} — {op.detail}[/dim]")

    # Learning feedback
    if trace.feedback:
        console.print(f"  [dim][LEARN] score={trace.feedback.score:.2f} — {trace.feedback.rationale[:150]}...[/dim]")

    # Memory Agent rationale
    if trace.memory_agent_rationale:
        console.print(f"  [dim][AGENT] {trace.memory_agent_rationale[:100]}...[/dim]")


# ── REPL Commands ────────────────────────────────────────────────────────────

def print_help():
    """Print help message."""
    help_text = """
[bold]Available commands:[/bold]
  [cyan]/tools[/cyan]    List all available tools the agent can use.
  [cyan]/clear[/cyan]    Reset STM context. LTM is retained.
  [cyan]/memory[/cyan]   Print current LTM snapshot.
  [cyan]/stats[/cyan]    Print current STM stats.
  [cyan]/forget[/cyan]   Wipe LTM store from disk and memory.
  [cyan]/help[/cyan]     Show this help message.
  [cyan]Ctrl+C[/cyan]    Cancel current input or exit.

[bold]Multiline input (for long messages):[/bold]
  [cyan]Enter[/cyan]             Send message
  [cyan]Alt+Enter[/cyan]         Insert newline (may be Escape+Enter on some terminals)
  [cyan]Escape, Enter[/cyan]     Insert newline (two separate keypresses)

[bold]Paste handling:[/bold]
  [dim]Paste large text directly - it will be automatically cleaned:[/dim]
  [dim]• Removes invisible characters (zero-width spaces, BOM)[/dim]
  [dim]• Normalizes smart quotes: " " ' ' → ASCII equivalents[/dim]
  [dim]• Collapses excessive empty lines[/dim]
  [dim]• Preserves code blocks and intentional formatting[/dim]
"""
    console.print(Panel(help_text, border_style="dim", padding=(0, 1)))


def print_tools(orch: Orchestrator):
    """Print available tools."""
    tools = orch.get_available_tools()
    if not tools:
        console.print("  [yellow]No tools available[/yellow]")
        return

    tool_lines = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "No description available.")
        # Get first line of description only
        desc_first_line = desc.split(".")[0] + "." if "." in desc else desc
        tool_lines.append(f"  [cyan]/{name}[/cyan]")
        tool_lines.append(f"    {desc_first_line[:70]}{'...' if len(desc_first_line) > 70 else ''}")

    console.print(Panel("\n".join(tool_lines), title=f"[bold]Available Tools ({len(tools)})[/bold]", border_style="dim"))

    console.print("""
[dim]Use these tools by mentioning them in your message:[/dim]
  "Search the web for Python tutorials"  → uses web_search
  "List all documents"                   → uses list_documents
  "Write this to output.txt"             → uses write_file
""")


def print_memory(orch: Orchestrator):
    """Print LTM snapshot."""
    entries = orch.ltm_snapshot()
    if not entries:
        console.print("  [yellow]LTM is empty[/yellow]")
        return

    lines = []
    for entry in entries:
        score = entry.get('learning_score', 0)
        content = entry.get('content', '')[:200]
        entry_id = entry.get('entry_id', 'unknown')
        lines.append(f"  [{entry_id}] score={score:.2f}: {content}{'...' if len(entry.get('content', '')) > 80 else ''}")

    console.print(Panel("\n".join(lines), title=f"[bold]LTM Store: {len(entries)} entries[/bold]", border_style="dim"))


def print_stats(orch: Orchestrator):
    """Print STM stats."""
    stats = orch.stm_stats()
    content = f"""[bold]Tokens:[/bold] {stats.total_tokens}
[bold]Utilisation:[/bold] {stats.utilisation_ratio:.1%}
[bold]Messages:[/bold] {stats.message_count}
[bold]Pinned:[/bold] {stats.pinned_count}"""
    console.print(Panel(content, title="[bold]STM Stats[/bold]", border_style="dim"))


def cmd_clear(orch: Orchestrator) -> bool:
    """Reset STM context."""
    orch.reset_stm()
    console.print("  [green]STM cleared. LTM retained.[/green]")
    return True


def cmd_forget(orch: Orchestrator) -> bool:
    """Wipe LTM store."""
    confirm = console.input("  [yellow]Are you sure you want to wipe all LTM entries? [y/N]:[/yellow] ").strip().lower()
    if confirm == 'y':
        orch.clear_ltm()
        console.print("  [green]LTM wiped.[/green]")
    else:
        console.print("  [dim]Cancelled.[/dim]")
    return True


# ── Main REPL ────────────────────────────────────────────────────────────────

def print_banner(orch: Orchestrator):
    """Print startup banner."""
    ltm_count = len(orch.ltm_snapshot())
    tools_count = len(orch.get_available_tools())

    banner = f"""[bold]AgeMem Chat[/bold]
  [dim]Model   :[/dim] {BASE_MODEL} @ {BASE_URL}
  [dim]STM     :[/dim] {STM_TOKEN_LIMIT} token limit
  [dim]LTM     :[/dim] {PERSIST_DIR}/ltm_store.json ({ltm_count} entries loaded)
  [dim]Memory  :[/dim] STM resets on /clear — LTM persists across sessions
  [dim]Tools   :[/dim] {tools_count} available (type /tools to list)

[dim]Enter sends, Escape+Enter for newline, paste large text for auto-clean[/dim]"""

    console.print(Panel(banner, border_style="blue", padding=(0, 1)))


def main():
    """Main entry point."""
    # Initialize tracing system
    init_tracing(
        log_dir=TRACE_LOG_DIR,
        debug=DEBUG_MODE,
        retention_days=TRACE_RETENTION_DAYS,
    )
    tracer = get_tracer()

    # Build orchestrator
    try:
        orch = build_orchestrator()
    except Exception as e:
        console.print(f"[red bold]ERROR:[/red bold] Failed to initialize orchestrator: {e}")
        sys.exit(1)

    # Print banner
    print_banner(orch)

    # Create prompt session with history
    session = create_prompt_session()

    console.print("\nType [cyan]/help[/cyan] for commands or start chatting!\n")

    # Turn counter for tracing
    turn_counter = 0

    while True:
        try:
            # Use prompt_toolkit for multiline input
            user_input = session.prompt(
                HTML('<ansicyan><b>You:</b></ansicyan> '),
            )

        except KeyboardInterrupt:
            # Ctrl+C: exit cleanly
            console.print("\n[bold]Goodbye![/bold]")
            shutdown_tracing()
            break
        except EOFError:
            # Ctrl+D: exit cleanly
            console.print("\n[bold]Goodbye![/bold]")
            shutdown_tracing()
            break

        if not user_input or not user_input.strip():
            continue

        # Handle commands (check before cleaning to preserve command intent)
        stripped_input = user_input.strip()

        if stripped_input == "/help":
            print_help()
            continue

        if stripped_input == "/tools":
            print_tools(orch)
            continue

        if stripped_input == "/clear":
            cmd_clear(orch)
            continue

        if stripped_input == "/memory":
            print_memory(orch)
            continue

        if stripped_input == "/stats":
            print_stats(orch)
            continue

        if stripped_input == "/forget":
            cmd_forget(orch)
            continue

        # Clean the input text for LLM processing
        # This handles pasted text with special chars, excessive whitespace, etc.
        original_input = user_input
        user_input = clean_pasted_text(user_input)

        # Provide feedback if text was significantly cleaned
        if is_likely_paste(original_input) and len(original_input) > len(user_input) + 10:
            console.print(f"  [dim]→ Cleaned pasted text ({len(original_input)} → {len(user_input)} chars)[/dim]")

        if not user_input:
            console.print("  [yellow]Input is empty after cleaning. Please try again.[/yellow]")
            continue

        # Process chat with spinner
        try:
            # Start trace for this interaction
            trace_id = tracer.start_trace(user_input, turn_index=turn_counter)

            # Log STM stats before processing
            stats_before = orch.stm_stats()
            tracer.log_stm_snapshot({
                "total_tokens": stats_before.total_tokens,
                "utilisation_ratio": stats_before.utilisation_ratio,
                "message_count": stats_before.message_count,
                "pinned_count": stats_before.pinned_count,
            }, trigger="turn_start")

            with Progress(
                SpinnerColumn(),
                TextColumn("[dim]Thinking...[/dim]"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("thinking", total=None)
                response = orch.chat(user_input)

            # Log the RAW response BEFORE any processing (key acceptance criteria)
            # This captures the response exactly as received from the LLM
            tracer.log_raw_response(response, model=BASE_MODEL)

            # Log STM stats after processing
            stats_after = orch.stm_stats()
            tracer.log_stm_snapshot({
                "total_tokens": stats_after.total_tokens,
                "utilisation_ratio": stats_after.utilisation_ratio,
                "message_count": stats_after.message_count,
                "pinned_count": stats_after.pinned_count,
            }, trigger="turn_end")

            # Render response as markdown
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold cyan]Assistant[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            ))

            # End trace with final response
            tracer.end_trace(final_response=response)

            print_diagnostics(orch)

            turn_counter += 1

        except Exception as e:
            tracer.end_trace(error=str(e))
            console.print(f"\n[red bold]ERROR:[/red bold] {rich_markup.escape(str(e))}\n")


if __name__ == "__main__":
    main()