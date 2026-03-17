# API Reference

## Core Classes

### Orchestrator

The central coordinator for all memory operations.

**Location**: [agents/orchestrator.py](../../agents/orchestrator.py)

```python
from agents.orchestrator import Orchestrator

orch = Orchestrator(
    llm: LLMClient,                    # Required: LLM client
    config: AgememConfig = DEFAULT_CONFIG,
    ltm_store: Optional[LTMStore] = None,
    stm_context: Optional[STMContext] = None
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `chat(message: str)` | `str` | Process a user message and return response |
| `last_trace()` | `TurnTrace` | Get audit record of last turn |
| `ltm_snapshot()` | `list[MemoryEntry]` | Get all LTM entries |
| `stm_stats()` | `ContextStats` | Get current context statistics |
| `clear_stm()` | `None` | Reset STM (keep LTM) |
| `wipe_ltm()` | `None` | Clear all LTM entries |

#### TurnTrace

```python
@dataclass
class TurnTrace:
    turn_index: int
    user_input: str
    assistant_response: str
    stm_stats_before: ContextStats
    stm_stats_after: ContextStats
    ops_applied: list[MemoryOpResult]
    feedback: Optional[LearningFeedback]
    memory_agent_rationale: str
    latency_ms: float
    prompt_versions: dict[str, str]
```

---

### LTMStore

Long-term memory storage with semantic retrieval.

**Location**: [memory/ltm_store.py](../../memory/ltm_store.py)

```python
from memory.ltm_store import LTMStore

ltm = LTMStore(
    config: AgememConfig = DEFAULT_CONFIG,
    persist_path: Optional[Path] = None,
    semantic_db_path: Optional[Path] = None,
    enable_semantic_search: bool = False,
    llm_client: Optional[Any] = None  # For query expansion
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add(entry: MemoryEntry)` | `MemoryOpResult` | Store new entry |
| `update(entry_id: str, content: str)` | `MemoryOpResult` | Update existing entry |
| `delete(entry_id: str)` | `MemoryOpResult` | Remove entry |
| `search(query: str, top_k: int = 5)` | `list[tuple[MemoryEntry, float]]` | Retrieve relevant entries |
| `entries()` | `dict[str, MemoryEntry]` | Get all entries |
| `prune(max_entries: int)` | `int` | Remove lowest-scored entries |

#### Search Result Format

```python
# Returns list of (entry, score) tuples
results = ltm.search("user preferences about email", top_k=5)
for entry, score in results:
    print(f"Score: {score:.3f} | {entry.content[:50]}...")
```

---

### STMContext

Short-term memory (active context window).

**Location**: [memory/stm_context.py](../../memory/stm_context.py)

```python
from memory.stm_context import STMContext

stm = STMContext(
    config: AgememConfig = DEFAULT_CONFIG,
    token_counter: Optional[TokenCounter] = None,
    summary_fn: Optional[Callable] = None,
    persist_path: Optional[Path] = None
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `append(message: ContextMessage)` | `None` | Add message to context |
| `messages()` | `list[ContextMessage]` | Get all messages |
| `openai_messages()` | `list[dict]` | Get OpenAI-compatible format |
| `stats()` | `ContextStats` | Get token statistics |
| `filter()` | `MemoryOpResult` | Drop low-relevance messages |
| `summary()` | `MemoryOpResult` | Compress message window |
| `force_fit()` | `list[MemoryOpResult]` | Ensure context within limits |
| `retrieve(entries: list[MemoryEntry])` | `MemoryOpResult` | Inject LTM entries |
| `persist()` | `None` | Save to disk |
| `load()` | `None` | Load from disk |

---

### SystemRules

Deterministic trigger engine.

**Location**: [triggers/system_rules.py](../../triggers/system_rules.py)

```python
from triggers.system_rules import SystemRules, RuleID

rules = SystemRules(config: AgememConfig = DEFAULT_CONFIG)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(stats, turn_index, feedback)` | `list[TriggerDecision]` | Check all rules |

#### TriggerDecision

```python
@dataclass
class TriggerDecision:
    rule_id: RuleID
    recommended_op: MemoryOp
    priority: int
    reason: str
    metadata: dict
```

#### Rule IDs

| ID | Name | Priority |
|----|------|----------|
| R1 | OVERFLOW_WARN | 70 |
| R2 | OVERFLOW_CRITICAL | 100 |
| R3 | PERIODIC_REVIEW | 40 |
| R4 | LEARNING_SPIKE | 90 |
| R5 | RELEVANCE_DECAY | 30 |

---

### MemoryAgent

LLM-driven memory decision agent.

**Location**: [agents/memory_agent.py](../../agents/memory_agent.py)

```python
from agents.memory_agent import MemoryAgent

agent = MemoryAgent(
    llm: LLMClient,
    config: AgememConfig = DEFAULT_CONFIG
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `review(context, ltm_entries, feedback)` | `MemoryAgentDecision` | Analyze and recommend operations |

#### MemoryAgentDecision

```python
@dataclass
class MemoryAgentDecision:
    ltm_operations: list[LTMOperation]
    context_relevance: list[dict]
    summary_needed: bool
    rationale: str
```

---

### LearningScorer

Self-assessment feedback collector.

**Location**: [agents/learning_scorer.py](../../agents/learning_scorer.py)

```python
from agents.learning_scorer import LearningScorer

scorer = LearningScorer(
    llm: LLMClient,
    config: AgememConfig = DEFAULT_CONFIG
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `collect(context_messages)` | `Optional[LearningFeedback]` | Get self-assessment score |
| `should_collect(turn_index)` | `bool` | Check if collection needed this turn |

---

### LLMClient

OpenAI-compatible API wrapper.

**Location**: [agents/llm_client.py](../../agents/llm_client.py)

```python
from agents.llm_client import LLMClient
from openai import OpenAI

client = OpenAI(api_key="...")
llm = LLMClient(
    client,
    default_model="gpt-4o-mini",
    default_max_tokens=2048,
    default_temperature=0.2
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `chat(messages, tools, tool_choice)` | `str` | Simple chat completion |
| `chat_with_tools(messages, tools)` | `ToolCallResponse` | Chat with tool support |
| `stream(messages)` | `Iterator[str]` | Streaming completion |

---

## Configuration

### AgememConfig

**Location**: [core/config.py](../../core/config.py)

```python
from core.config import AgememConfig, DEFAULT_CONFIG

# Use defaults
config = DEFAULT_CONFIG

# Custom configuration
config = AgememConfig(
    STM_TOKEN_LIMIT=4096,
    LTM_MAX_ENTRIES=1000,
    ENABLE_SEMANTIC_SEARCH=True,
    # ... see full list below
)
```

### Key Parameters

#### STM Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STM_TOKEN_LIMIT` | 9000 | Hard context ceiling |
| `STM_WARNING_THRESHOLD` | 0.75 | Trigger SUMMARY above this |
| `STM_CRITICAL_THRESHOLD` | 0.90 | Force FILTER above this |
| `STM_MIN_MESSAGES` | 4 | Minimum to keep under pressure |

#### LTM Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LTM_MAX_ENTRIES` | 5000 | Maximum stored entries |
| `LTM_PROMOTE_THRESHOLD` | 0.65 | Score for LTM ADD |
| `LTM_DEDUP_THRESHOLD` | 0.92 | Semantic duplicate threshold |
| `LTM_DEDUP_OVERLAP_THRESHOLD` | 0.70 | Jaccard duplicate threshold |

#### Semantic Search

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_SEMANTIC_SEARCH` | True | Use vector embeddings |
| `SEMANTIC_EMBEDDING_MODEL` | "Qwen/Qwen3-Embedding-0.6B" | Embedding model |
| `SEMANTIC_EMBEDDING_DIM` | 1024 | Embedding dimensions |
| `SEMANTIC_RETRIEVAL_MULTIPLIER` | 3 | Broad-pass multiplier |

#### Triggers

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRIGGER_EVERY_N_TURNS` | 10 | MemoryAgent review cadence |
| `LEARNING_SCORE_PROMPT_EVERY_N` | 3 | Self-assessment cadence |
| `LEARNING_SCORE_THRESHOLD_IMMEDIATE` | 0.8 | Bypass cadence threshold |

---

## Extension Points

### Custom Token Counter

```python
from tiktoken import encoding_for_model
from core.types import TokenCounter

encoder = encoding_for_model("gpt-4o")
counter = TokenCounter(encoder)

stm = STMContext(token_counter=counter)
```

### Custom Summary Function

```python
def my_summarizer(messages: list[ContextMessage]) -> str:
    # Custom summarization logic
    return "Summary of conversation..."

stm = STMContext(summary_fn=my_summarizer)
```

### Custom Vector Store

Replace `sqlite-vec` with your preferred vector database:

```python
class CustomVectorStore:
    def add(self, entry_id: str, embedding: np.ndarray) -> None: ...
    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def delete(self, entry_id: str) -> None: ...
```

### Custom Rules

Extend `SystemRules.evaluate()`:

```python
from triggers.system_rules import SystemRules, TriggerDecision

class CustomRules(SystemRules):
    def evaluate(self, stats, turn_index, feedback):
        decisions = super().evaluate(stats, turn_index, feedback)
        # Add custom rules
        if my_custom_condition:
            decisions.append(TriggerDecision(...))
        return decisions
```