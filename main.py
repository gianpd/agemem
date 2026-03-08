"""
main.py
───────
AgeMem Interactive Chat — REPL entry point for the AgeMem-hybrid system.

Usage:
    python main.py

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
    LTM_PERSIST_PATH        default: agent_memory/ltm_store.json
"""

from __future__ import annotations


import os
import sys
import signal
import time
import warnings
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI

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
LTM_PERSIST_PATH = os.getenv("LTM_PERSIST_PATH", "agent_memory/ltm_store.json")
STM_TOKEN_LIMIT = int(os.getenv("STM_TOKEN_LIMIT", "6000"))


# ── Imports from AgeMem package ──────────────────────────────────────────────

from core.config import AgememConfig
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator
from memory.ltm_store import LTMStore


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
        LEARNING_SCORE_PROMPT_EVERY_N=3,
        TRIGGER_EVERY_N_TURNS=10,
        DEFAULT_MAX_TOKENS=BASE_MAX_TOKENS,
        DEFAULT_TEMPERATURE=BASE_TEMPERATURE,
        PERSIST_DIR="agemem_state",
    )

    llm = LLMClient(client, default_model=cfg.DEFAULT_MODEL)
    
    # Create LTM store with persistence
    ltm_path = Path(LTM_PERSIST_PATH)
    ltm_path.parent.mkdir(parents=True, exist_ok=True)
    ltm_store = LTMStore(cfg, persist_path=ltm_path)
    
    orch = Orchestrator(llm=llm, config=cfg, ltm_store=ltm_store)
    
    # Set up tools - combine all tool definitions
    all_tools = WEB_TOOL_DEFINITIONS + CORPUS_TOOL_DEFINITIONS
    orch.set_tools(all_tools)
    
    return orch


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
    
    # Progress bar
    bar = format_progress_bar(stats.utilisation_ratio)
    
    print(f"  [STM {bar} {stats.utilisation_ratio:.0%} ~{stats.total_tokens}tok | LTM {ltm_count} entries | turn {turn}]")
    
    # Memory ops
    for op in trace.ops_applied:
        if op.success:
            trigger_name = op.trigger.value if hasattr(op.trigger, 'value') else str(op.trigger)
            print(f"  [MEM] {op.op.value.upper()} triggered by {trigger_name} — {op.detail}")

    # Learning feedback
    if trace.feedback:
        print(f"  [LEARN] score={trace.feedback.score:.2f} — {trace.feedback.rationale[:50]}...")

    # Memory Agent rationale
    if trace.memory_agent_rationale:
        print(f"  [AGENT] {trace.memory_agent_rationale[:60]}...")


# ── REPL Commands ────────────────────────────────────────────────────────────

def print_help():
    """Print help message."""
    print("""
Available commands:
  /tools    List all available tools the agent can use.
  /clear    Reset STM context. LTM is retained.
  /memory   Print current LTM snapshot.
  /stats    Print current STM stats.
  /forget   Wipe LTM store from disk and memory.
  /help     Show this help message.
  Ctrl-C    Exit gracefully.
""")


def print_tools(orch: Orchestrator):
    """Print available tools."""
    tools = orch.get_available_tools()
    if not tools:
        print("  [No tools available]")
        return

    print(f"\n  Available Tools ({len(tools)}):")
    print("  " + "-" * 50)

    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "No description available.")
        # Get first line of description only
        desc_first_line = desc.split(".")[0] + "." if "." in desc else desc
        print(f"    /{name}")
        print(f"      {desc_first_line[:60]}{'...' if len(desc_first_line) > 60 else ''}")

    print("\n  Use these tools by mentioning them, e.g.:")
    print('    "Search the web for Python tutorials"  → uses web_search')
    print('    "List all documents"                   → uses list_documents')
    print('    "Write this to output.txt"             → uses write_file')
    print()


def print_memory(orch: Orchestrator):
    """Print LTM snapshot."""
    entries = orch.ltm_snapshot()
    if not entries:
        print("  [LTM is empty]")
        return
    
    print(f"\n  LTM Store: {len(entries)} entries")
    for entry in entries:
        score = entry.get('learning_score', 0)
        content = entry.get('content', '')[:80]
        entry_id = entry.get('entry_id', 'unknown')
        print(f"    [{entry_id}] score={score:.2f}: {content}{'...' if len(entry.get('content', '')) > 80 else ''}")
    print()


def print_stats(orch: Orchestrator):
    """Print STM stats."""
    stats = orch.stm_stats()
    print(f"\n  STM Stats:")
    print(f"    Tokens: {stats.total_tokens}")
    print(f"    Utilisation: {stats.utilisation_ratio:.1%}")
    print(f"    Messages: {stats.message_count}")
    print(f"    Pinned: {stats.pinned_count}")
    print()


def cmd_clear(orch: Orchestrator) -> bool:
    """Reset STM context."""
    orch.reset_stm()
    print("  [STM cleared. LTM retained.]")
    return True


def cmd_forget(orch: Orchestrator) -> bool:
    """Wipe LTM store."""
    confirm = input("  Are you sure you want to wipe all LTM entries? [y/N]: ").strip().lower()
    if confirm == 'y':
        orch.clear_ltm()
        print("  [LTM wiped.]")
    else:
        print("  [Cancelled.]")
    return True


# ── Main REPL ────────────────────────────────────────────────────────────────

def print_banner(orch: Orchestrator):
    """Print startup banner."""
    ltm_count = len(orch.ltm_snapshot())
    tools_count = len(orch.get_available_tools())

    print(f"""
AgeMem Chat
  Model   : {BASE_MODEL} @ {BASE_URL}
  STM     : {STM_TOKEN_LIMIT} token limit
  LTM     : {LTM_PERSIST_PATH} ({ltm_count} entries loaded)
  Memory  : STM resets on /clear — LTM persists across sessions
  Tools   : {tools_count} available (type /tools to list)
""")

    print()


def main():
    """Main entry point."""
    # Handle Ctrl-C gracefully
    def signal_handler(sig, frame):
        print("\n\nGoodbye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    # Build orchestrator
    try:
        orch = build_orchestrator()
    except Exception as e:
        print(f"ERROR: Failed to initialize orchestrator: {e}")
        sys.exit(1)
    
    # Print banner
    print_banner(orch)
    
    # REPL loop
    print("Type /help for commands or start chatting!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input == "/help":
            print_help()
            continue

        if user_input == "/tools":
            print_tools(orch)
            continue

        if user_input == "/clear":
            cmd_clear(orch)
            continue
        
        if user_input == "/memory":
            print_memory(orch)
            continue
        
        if user_input == "/stats":
            print_stats(orch)
            continue
        
        if user_input == "/forget":
            cmd_forget(orch)
            continue
        
        # Process chat
        try:
            response = orch.chat(user_input)
            print(f"\nAssistant: {response}\n")
            print_diagnostics(orch)
            print()
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()
