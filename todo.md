Here is a well-scoped internal dev ticket.

---

# Dev Ticket: AgeMem Interactive Chat — `main.py`

**Reference implementations:** `example_usage.py`, `ask.py` (tri-tier swarm)
**Target system:** AgeMem-hybrid (`agents/orchestrator.py` and supporting modules)

---

## Context

The AgeMem-hybrid system currently has no entry point for interactive use. `example_usage.py` demonstrates the `Orchestrator` API with a hardcoded turn list and exits. This ticket implements `main.py`: a persistent REPL where a user chats with a local Qwen model (via llama.cpp), the full STM/LTM memory cycle runs on every turn, and web search is available as a tool.

Study `example_usage.py` before starting — it shows the correct way to construct `AgememConfig`, wire `LLMClient`, and call `Orchestrator.chat()`. Do not bypass the Orchestrator to call the LLM directly.

---

## Scope

Single file: `main.py` at the project root, alongside `example_usage.py`.

No new modules. No changes to `core/`, `memory/`, `triggers/`, or `agents/` unless a genuine bug is found — in that case open a separate ticket.

---

## Functional Requirements

### F1 — LLM Backend: llama.cpp via OpenAI-compatible API

The local Qwen model is served by llama.cpp at a configurable host. The client must be an `openai.OpenAI` instance pointed at the llama.cpp `/v1` endpoint, passed into `LLMClient` exactly as shown in `example_usage.py`'s `build_orchestrator()`.

Environment variables (all optional, with defaults):

```
LLAMA_HOST        default: http://localhost:8080
LLAMA_MODEL       default: qwen3-4b
LLAMA_MAX_TOKENS  default: 2048
LLAMA_TEMPERATURE default: 0.2
```

On startup, verify the server is reachable before entering the REPL. If unreachable, print a clear error with the start command and exit. Do not silently proceed with a broken client.

### F2 — Web Search Tool

The agent must have access to a `web_search` tool. This is a new capability not present in the current AgeMem codebase.

The tool must be integrated into the Orchestrator's LLM call, not called outside it. Concretely: the messages list passed to `LLMClient.chat()` must include the tool schema, and tool call responses must be appended to the STM context before the next LLM call.

This requires a targeted extension to `LLMClient.chat()` and `Orchestrator.chat()` to support tool schemas and tool result injection. The extension must be backward-compatible — existing callers that pass no tools must continue to work unchanged.

Implementation of `web_search` itself: use `duckduckgo-search` (no API key required). Cap results at `TOOL_RESULT_MAX_CHARS` characters. Return title, URL, and snippet for each result.

```
WEB_SEARCH_MAX_RESULTS  default: 5
TOOL_RESULT_MAX_CHARS   default: 4000
```

### F3 — Memory: Full STM + LTM Cycle

The Orchestrator already manages STM and LTM. `main.py` must not re-implement any memory logic. The requirements here are about configuration and session behaviour.

**LTM persistence.** Pass a `persist_path` to `LTMStore` so memories survive process restarts. Default path: `agent_memory/ltm_store.json`. The Orchestrator constructor accepts an optional `ltm_store` argument — construct the `LTMStore` with the path before passing it in.

**STM reset on `/clear`.** When the user types `/clear`, reset the STM context to its initial state (system prompt only) without touching the LTM. The LTM must survive a `/clear` — it is session-persistent, not turn-persistent.

**No session cap.** There is no `MAX_STEPS` or turn limit. The session runs until the user exits. The STM overflow guards in `STMContext.force_fit()` are the only context management mechanism — do not add an outer turn counter.

### F4 — REPL Commands

```
/clear    Reset STM context. Print confirmation. LTM is retained.
/memory   Print current LTM snapshot (all entries, score, content).
/stats    Print current STM stats (tokens, utilisation %, message count).
/forget   Wipe LTM store from disk and memory. Prompt for confirmation.
/help     List available commands.
Ctrl-C    Exit gracefully.
```

### F5 — Session Startup Banner

On launch, print:

```
AgeMem Chat
  Model   : <LLAMA_MODEL> @ <LLAMA_HOST>
  STM     : <STM_TOKEN_LIMIT> token limit
  LTM     : <path> (<N> entries loaded)
  Memory  : STM resets on /clear — LTM persists across sessions
  Web     : web_search enabled
```

### F6 — Per-turn Diagnostics

After each assistant response, print a compact status line:

```
  [STM ██████░░░░ 61% ~3700tok | LTM 12 entries | turn 7]
```

Format: progress bar (10 chars, `█`/`░`), utilisation %, token estimate, LTM entry count, turn index. This mirrors the bar in `ask.py`'s REPL and gives the user visibility into memory pressure without reading logs.

If any memory ops fired during the turn (from `trace.ops_applied`), print them below the bar:

```
  [MEM] SUMMARY triggered by system_rule — Compressed 8 messages (~640 tokens saved)
  [MEM] ADD triggered by learning_score — Stored entry a3f9c1
```

Only print ops where `op.success == True`.

---

## Non-Functional Requirements

### N1 — No MAX_STEPS enforcement in main.py

The `LLAMA_MAX_STEPS` pattern from `ask.py` must not appear. AgeMem's overflow guards are the only context management. If the developer finds the Orchestrator needs a step budget for tool-calling loops, that belongs in the Orchestrator, not in `main.py`.

### N2 — Async not required

`ask.py` is fully async because the Archivist runs in the background. AgeMem's Orchestrator is synchronous. `main.py` must be synchronous. Do not introduce `asyncio` unless the web search backend strictly requires it — if so, use `asyncio.run()` at the call site only, not propagated upward.

### N3 — Config in one place

All tuneable values must be read from environment variables at the top of `main.py`, with explicit defaults. No magic numbers inline. Mirror the pattern from `ask.py`'s config block.

### N4 — Graceful degradation

If `gliner` or `duckduckgo-search` are not installed, warn at startup and disable the affected feature rather than crashing. Web search being unavailable must not prevent the REPL from starting.

### N5 — Import hygiene

`main.py` imports from the AgeMem package only via its public API:

```python
from core.config import AgememConfig
from core.types import MemoryOp, TriggerKind
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator
from memory.ltm_store import LTMStore
```

No direct imports from `memory/stm_context.py` internals or `triggers/system_rules.py`. If something needed by `main.py` is not reachable via these imports, the fix is to expose it through the existing public interface, not to reach into internals.

---

## Integration Points Requiring Orchestrator Extension

Two small extensions to existing files are in scope for this ticket because they are prerequisites. Both must be backward-compatible.

**E1 — Tool support in `LLMClient.chat()`.**

Add an optional `tools` parameter:

```python
def chat(
    self,
    messages: list[dict],
    tools: list[dict] | None = None,   # ← add this
    model: str | None = None,
    ...
) -> str:
```

When `tools` is not None, include it in the API call. When a tool call is returned instead of text content, return a structured sentinel or raise a typed exception — do not silently return empty string. The caller (`Orchestrator`) is responsible for detecting and handling tool calls.

**E2 — Tool call handling in `Orchestrator.chat()`.**

When the LLM returns a tool call rather than a final answer, the Orchestrator must:
1. Parse the tool name and arguments
2. Execute the tool (dispatch table, starting with `web_search`)
3. Append the tool result as a `tool` role message to STM
4. Loop back to the LLM call
5. Record all tool calls in `TurnTrace.ops_applied` with `TriggerKind.MAIN_AGENT`

This loop has no hard iteration cap (N1), but must be protected against the same duplicate-call pattern as `ask.py`'s `LoopGuard`. Implement a lightweight per-turn call tracker: if the same tool is called with identical arguments twice in one turn, inject a system message telling the agent to try a different approach rather than silently looping.

---

## Out of Scope

- Oracle / Archivist pattern from `ask.py` — AgeMem does not have a background writer
- Corpus ingestion (`ingest.py`) — not relevant to chat mode
- Multi-agent escalation — single model only
- Streaming responses — not supported by the current `LLMClient`
- GUI or web interface

---

## Acceptance Criteria

1. `python main.py` starts, prints the banner, and enters the REPL
2. A conversation of 30+ turns does not crash or exceed `STM_TOKEN_LIMIT` (verified by `/stats`)
3. After `/clear`, `/stats` shows message count reset to 1 (system prompt only); `/memory` still shows entries from before the clear
4. After process restart, `/memory` shows the same entries as before restart (LTM persisted to disk)
5. Typing a query that requires current information (e.g. "what happened in the news today") results in a `web_search` tool call visible in the diagnostics line
6. All existing tests in `tests/test_agemem.py` still pass — the 28-test suite must be green before the PR is opened