# AgeMem-Hybrid: Long-Term Memory (LTM) Architecture

This document provides a detailed technical explanation of how Long-Term Memory (LTM) works in the AgeMem-Hybrid system, including triggering mechanisms, autonomous operation, and tool execution capabilities.

---

## 1. What is LTM?

LTM (Long-Term Memory) is the persistent memory tier that survives across conversation sessions. Unlike STM (Short-Term Memory) which is the active context window limited to ~6000 tokens, LTM:

- **Persists to disk** (`{PERSIST_DIR}/ltm_store.json` where PERSIST_DIR defaults to `agent_memory`)
- **Has a capacity** of up to 500 entries (configurable via `LTM_MAX_ENTRIES`)
- **Uses token-overlap retrieval** instead of embeddings (inference-only constraint)
- **Stores facts, preferences, and knowledge** the agent deems worth remembering

### LTM Entry Structure

```python
@dataclass
class MemoryEntry:
    content: str                    # The actual memory content
    entry_id: str                   # Unique SHA1-based identifier
    created_at: float               # Unix timestamp
    updated_at: float               # Last modification time
    access_count: int               # How many times retrieved
    learning_score: float           # Aggregated novelty score (0-1)
    tags: list[str]                 # Categorical tags
    source_turn: int                # Which conversation turn created this
```

---

## 2. LTM Operations

The LTM supports four fundamental operations defined in `core/types.py`:

| Operation | Description | Trigger Sources |
|-----------|-------------|-----------------|
| **ADD** | Store a new memory entry | System Rule, Learning Score, Memory Agent |
| **UPDATE** | Modify an existing entry | Memory Agent only |
| **DELETE** | Remove an entry | Memory Agent only |
| **RETRIEVE** | Pull relevant entries into STM | Automatic per-turn |

---

## 3. What Triggers LTM Store Mechanisms?

LTM operations are triggered through a **three-layer hybrid control architecture**:

### Layer 1: System Rules (Deterministic)

Located in `triggers/system_rules.py`, these are threshold-based rules that fire without any LLM call:

| Rule ID | Condition | Action |
|---------|-----------|--------|
| **R1 (OVERFLOW_WARN)** | STM >= 75% capacity | Trigger SUMMARY only |
| **R2 (OVERFLOW_CRITICAL)** | STM >= 90% capacity | Force FILTER + SUMMARY |
| **R3 (PERIODIC_REVIEW)** | Every N turns (default: 10) | Invoke MemoryAgent for review |
| **R4 (LEARNING_SPIKE)** | Learning score >= threshold | Immediate LTM ADD candidate |
| **R5 (RELEVANCE_DECAY)** | Low relevance messages | Tag for FILTER priority |

### Layer 2: Learning Scorer (Self-Assessment)

Located in `agents/learning_scorer.py`, this collects agent self-ratings:

```python
# After every LEARNING_SCORE_PROMPT_EVERY_N turns (default: 3)
# The agent answers: "On a 0-1 scale, how much new information did you encounter?"
```

**Scoring Guide:**
- **1.0** — Highly novel, specific, reusable fact (user's name, project details)
- **0.7** — Useful context likely needed later in this session
- **0.4** — Potentially relevant but uncertain
- **0.1** — Routine exchange, no new persistent knowledge
- **0.0** — Pure procedural step, nothing to retain

**Critical Threshold:**
- If `score >= LTM_PROMOTE_THRESHOLD` (default: 0.65), the content is **immediately promoted to LTM** via `TriggerKind.LEARNING_SCORE`.

### Layer 3: Memory Agent (LLM-Based)

Located in `agents/memory_agent.py`, this is a dedicated sub-agent that performs qualitative memory decisions:

**When It Runs:**
1. On **periodic review** (every N turns via R3)
2. On **learning spike** (high learning score via R4)

**What It Decides:**
- Which entries to ADD/UPDATE/DELETE in LTM
- Context relevance scores for STM messages
- Whether a SUMMARY is needed

**Output Schema:**
```json
{
  "ltm_operations": [
    {
      "op": "add" | "update" | "delete",
      "entry_id": "<id or null>",
      "content": "<text>",
      "tags": ["..."],
      "confidence": 0.0-1.0
    }
  ],
  "context_relevance": [...],
  "summary_needed": true | false,
  "rationale": "<explanation>"
}
```

---

## 4. Can the Agent Autonomously Add to LTM?

**Yes.** The agent can autonomously add entries to LTM through two primary mechanisms:

### Mechanism A: Learning Score Threshold (Immediate)

In `agents/orchestrator.py` (lines 441-452):

```python
if (
    feedback
    and feedback.score >= self._config.LTM_PROMOTE_THRESHOLD  # >= 0.65
    and feedback.affected_content
):
    add_op = self._ltm.add(
        content=feedback.affected_content,
        learning_score=feedback.score,
        source_turn=turn_after,
        trigger=TriggerKind.LEARNING_SCORE,
    )
```

**This happens automatically** when the agent self-reports a high learning score. No human intervention required.

### Mechanism B: Memory Agent Review (Qualitative)

In `agents/orchestrator.py` (lines 454-462):

```python
if should_run_memory_agent:
    decision_obj = self._memory_agent.review(...)
    ops.extend(self._apply_memory_agent_decision(decision_obj, ...))
```

The Memory Agent may decide to ADD entries with `confidence >= 0.7` (per its system prompt).

### Confidence Thresholds

| Source | Minimum Confidence | Notes |
|--------|-------------------|-------|
| Learning Score | 0.65 | Automatic promotion |
| Memory Agent ADD | 0.7 | Per system prompt instruction |
| Orchestrator apply | 0.6 | Gate in `_apply_memory_agent_decision()` |

---

## 5. Complete Event Flow: When LTM is Triggered

Here is the complete per-turn lifecycle from `agents/orchestrator.py`:

### Pre-Turn Phase

```
1. STM.force_fit()          — Prevent overflow before LLM call
2. LTM.search(user_query)   — Retrieve top-k relevant memories into STM
```

### Main Turn Phase

```
3. Add user message to STM
4. Call LLM with tool support (may loop for multiple tool calls)
5. Add assistant response to STM
6. STM.force_fit()          — Post-response overflow guard
```

### Post-Turn Phase (Where LTM Operations Happen)

```
7. LearningScorer.collect() — Every N turns, get self-assessment
8. SystemRules.evaluate()   — Check all threshold rules

   If R2 (OVERFLOW_CRITICAL):  → FILTER + SUMMARY
   If R3 (PERIODIC_REVIEW):    → flag for MemoryAgent
   If R4 (LEARNING_SPIKE):     → flag for MemoryAgent + immediate LTM ADD

9. Immediate LTM promotion on learning spike
10. MemoryAgent.review() if flagged
11. Apply MemoryAgent decisions (ADD/UPDATE/DELETE)
12. Persist STM to disk
```

---

## 6. LTM Retrieval (Search)

LTM entries are automatically retrieved at the start of every turn:

```python
relevant = self._ltm.search(user_input, top_k=3)
if relevant:
    retrieve_op = self._stm.retrieve(relevant, trigger=TriggerKind.SYSTEM_RULE)
```

### Scoring Algorithm

In `memory/ltm_store.py` (lines 146-172), retrieval uses a composite score:

```python
score = 0.5 * overlap_score +           # Token overlap (Jaccard-like)
        0.3 * entry.learning_score +    # Aggregated novelty
        0.2 * recency                   # Exponential decay over 7 days
```

**No embeddings are used** — this is an intentional design constraint for inference-only deployments.

---

## 7. Multiple Tool Calls: LoopGuard Protection

**Yes, the agent can confidently run multiple tool calls** when chatting via `main.py`. The system is designed for multi-tool workflows:

### Tool Call Loop Architecture

In `agents/orchestrator.py` (lines 332-399):

```python
while True:
    messages = self._stm.openai_messages()
    try:
        assistant_text = self._llm.chat(..., tools=self._tools)
        break  # No tool call = we have a response
    except ToolCallResponse as e:
        # Check for duplicates (LoopGuard)
        call = ToolCall(name=tool_name, arguments=tool_args)
        if self._tool_tracker.record(call):
            # Duplicate detected — inject warning and continue
            self._stm.add_message(role="user", content="[SYSTEM] Duplicate tool call...")
            continue

        # Execute the tool
        tool_result = self._execute_tool(tool_name, tool_args)

        # Add tool result to context
        self._stm.add_message(role="tool", content=tool_result, ...)

        # Loop back to LLM with the tool result
        continue
```

### LoopGuard Deduplication

The `ToolCallTracker` prevents the agent from calling the same tool with identical arguments in a single turn:

```python
@dataclass
class ToolCall:
    name: str
    arguments: dict

    def key(self) -> str:
        args_str = str(sorted(self.arguments.items()))
        return f"{self.name}:{args_str}"
```

**Key Features:**
- **No hard iteration cap** — the agent can make as many tool calls as needed
- **Duplicate protection** — same tool+args won't execute twice per turn
- **Automatic continuation** — after each tool result, the LLM is called again
- **Context preservation** — all tool calls and results are stored in STM

### Example Multi-Tool Flow

```
User: "Search for Python tutorials and save the results to a file"

→ Agent calls web_search(query="Python tutorials")
→ Tool result added to STM
→ Agent called again with result
→ Agent calls write_file(path="results.txt", content=...)
→ Tool result added to STM
→ Agent called again
→ Agent provides final response: "I've searched and saved the results."
```

---

## 8. Configuration Parameters

All LTM-related thresholds are in `core/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LTM_MAX_ENTRIES` | 500 | Maximum LTM entries before pruning |
| `LTM_PROMOTE_THRESHOLD` | 0.65 | Learning score threshold for immediate LTM ADD |
| `LTM_UPDATE_THRESHOLD` | 0.7 | Score needed to update existing entry |
| `LTM_SIMILARITY_WORDS` | 10 | Word count for duplicate detection |
| `LEARNING_SCORE_PROMPT_EVERY_N` | 3 | Collect learning feedback every N turns |
| `LEARNING_SCORE_THRESHOLD_IMMEDIATE` | 0.8 | High spike threshold for immediate promotion |
| `TRIGGER_EVERY_N_TURNS` | 10 | MemoryAgent review cadence |

---

## 9. Persistence and Durability

### LTM Persistence

- **Path:** `{PERSIST_DIR}/ltm_store.json` where PERSIST_DIR is configurable via env var (default: `agent_memory`)
- **When:** Every write triggers `_maybe_persist()` — immediate durability
- **Format:** JSON array of `MemoryEntry` objects

### STM Persistence

- **Path:** `{PERSIST_DIR}/stm_context.json` where PERSIST_DIR is configurable via env var (default: `agent_memory`)
- **When:** After every turn in `chat()` method
- **Purpose:** Survive process restarts without losing conversation context

---

## 10. Key Design Invariants

1. **Context must be within bounds at the end of every turn** — Double overflow guard (pre-turn AND post-response)
2. **Pinned messages are never evicted** — System prompt and retrieved LTM are protected
3. **Tool calls are deduplicated per-turn** — LoopGuard prevents infinite loops
4. **LTM persists on every write** — No data loss on crash
5. **No RL training** — All behavior is prompted or rule-based
6. **No embeddings** — Token overlap is sufficient for ≤500 entries

---

## Summary

| Question | Answer |
|----------|--------|
| **What triggers LTM?** | Three-layer system: System Rules (thresholds), Learning Scorer (self-assessment), Memory Agent (periodic reviews) |
| **Can the agent autonomously add to LTM?** | **Yes** — via Learning Score >= 0.65 or Memory Agent decisions |
| **Which events trigger LTM storage?** | High learning scores, periodic reviews (every 10 turns), overflow conditions, and Memory Agent qualitative decisions |
| **Can the agent run multiple tool calls?** | **Yes** — via an unbounded tool loop with LoopGuard deduplication |

---

*Document version: 1.0*
*Generated for AgeMem-Hybrid codebase*
