# AgeMem Technical Specification

**Version:** 1.0  
**Date:** 2026-03-18  
**Status:** Draft  
**Repository:** https://github.com/gianpd/agemem

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Core Purpose](#2-system-core-purpose)
3. [Comparative Analysis: AgeMem vs Standard RAG](#3-comparative-analysis-agemem-vs-standard-rag)
4. [Memory Architecture](#4-memory-architecture)
5. [Document Ingestion Pipeline](#5-document-ingestion-pipeline)
6. [Standardized Usage Instructions](#6-standardized-usage-instructions)
7. [Formal Evaluation Framework](#7-formal-evaluation-framework)
8. [Appendices](#8-appendices)

---

## 1. Executive Summary

AgeMem is a hybrid memory management system that provides any LLM agent—running on any OpenAI-compatible endpoint, including fully local models via Ollama—a disciplined, auditable memory architecture with two tiers: Short-Term Memory (STM) for active context management and Long-Term Memory (LTM) for persistent knowledge storage. The system operates entirely at inference time, requiring no RL training, gradient updates, or fine-tuning.

**Core Thesis:** *500 perfectly curated memories on a 9B model will consistently outperform 10,000 uncurated RAG chunks on a 70B model.* [DocID: 01_overview_bbde6e]

---

## 2. System Core Purpose

### 2.1 Problem Statement

Every serious agent deployment hits the same wall: a 1-million-token context window and a 70B model will still hallucinate a user preference told three sessions ago, because that fact was silently evicted when the context overflowed—or buried so deep in irrelevant tokens that the model's attention never reached it. **The memory wall is not a model problem. It is a systems problem.** [DocID: 01_overview_bbde6e]

### 2.2 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Inference-Only** | No RL training, no gradient updates. All behavior from deterministic rules, prompted decisions, and self-assessed learning scores |
| **Privacy by Default** | Entire stack runs locally: llama.cpp inference, sqlite-vec vector index, GLiNER NER enrichment, JSON persistence |
| **Resource Efficiency** | Proven on 8GB RTX 4060 at 36 tokens/second |
| **Double-Boundary Overflow Guard** | Overflow invariant enforced at both message-append boundaries—before user message and after assistant response |

### 2.3 Target Use Cases

- Local LLM deployments (Ollama, llama.cpp, vLLM)
- Privacy-sensitive applications (healthcare, legal, finance)
- Multi-session agents requiring persistent memory
- Resource-constrained environments (consumer GPUs)

---

## 3. Comparative Analysis: AgeMem vs Standard RAG

### 3.1 Architectural Differences

| Dimension | Standard RAG Pipeline | AgeMem |
|-----------|----------------------|--------|
| **Memory Model** | Flat document chunks with vector similarity | Two-tier (STM/LTM) with hybrid scoring and promotion logic |
| **Retrieval Strategy** | Single-pass semantic search | Two-stage pipeline: semantic broad-pass → overlap refinement |
| **Context Management** | Fixed chunk injection, no eviction | Dynamic FILTER/SUMMARY with pinned message protection |
| **Persistence** | Stateless per session | Persistent LTM with learning scores and access tracking |
| **Self-Assessment** | None | Learning scores (0-1 novelty scale) drive memory promotion |
| **Deduplication** | Basic or none | Semantic (cosine ≥ 0.92) and overlap (Jaccard ≥ 0.70) dedup |
| **Query Expansion** | Rare or manual | Automatic paraphrase generation for coverage maximization |
| **Validation** | None | Retrieved memories validated for relevance, groundedness, faithfulness |

### 3.2 Retrieval Accuracy Improvements

| Metric | Standard RAG | AgeMem | Improvement Mechanism |
|--------|-------------|--------|----------------------|
| **Precision** | Moderate (noise from uncurated chunks) | High (curated memories with learning scores) | Learning score gating (threshold 0.65) prevents low-value storage |
| **Recall** | High (exhaustive search) | High (semantic + overlap fallback) | Two-stage retrieval with query expansion |
| **Relevance** | Variable (no quality control) | Validated (Tier 3 tools) | `validate_ltm_relevance` checks groundedness and faithfulness |
| **Freshness** | None (static corpus) | Dynamic (recency decay, 7-day half-life) | Hybrid scoring: `0.25 × recency_decay` |
| **Deduplication** | Basic | Semantic + overlap | Prevents redundant storage, updates existing entries |

### 3.3 Response Quality Improvements

| Aspect | Standard RAG | AgeMem |
|--------|-------------|--------|
| **Context Coherence** | Fragmented (unrelated chunks injected) | Coherent (STM manages message flow, summarizes when needed) |
| **Memory Persistence** | Session-only | Cross-session via LTM with JSON persistence |
| **Hallucination Reduction** | Moderate (context provided but unvalidated) | High (validated memories, pinned system prompt) |
| **Adaptive Behavior** | Static | Dynamic (MemoryAgent reviews, LearningScorer self-assesses) |
| **Overflow Handling** | Truncation or failure | Double-boundary force_fit with priority-based eviction |

### 3.4 Key Differentiators

1. **Hybrid Retrieval Scoring:** `score = 0.6 × cosine_similarity + 0.25 × recency_decay + 0.15 × learning_score` [DocID: 01_overview_bbde6e]
2. **Three-Layer Control System:** Deterministic rules → LLM-driven agent → self-assessed learning scores [DocID: 02_architecture_b3c135]
3. **LTM Self-Introspection Toolkit:** Agent-directed retrieval, validation, and persistence across 5 tool tiers [DocID: 02_architecture_b3c135]
4. **Double-Boundary Overflow Guard:** Context integrity enforced before user message and after assistant response [DocID: 01_overview_bbde6e]

---

## 4. Memory Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Orchestrator                            │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ System Rules │  │  Memory Agent   │  │ Learning Scorer  │   │
│  │ (pure logic) │  │  (LLM-driven)   │  │ (self-assessed)  │   │
│  └──────┬───────┘  └────────┬────────┘  └────────┬─────────┘   │
│         │                   │                     │             │
│  ┌──────▼───────────────────▼─────────────────────▼─────────┐  │
│  │                    STM Context Window                      │  │
│  │         FILTER · SUMMARY · RETRIEVE · force_fit           │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                               │ promote / retrieve               │
│  ┌────────────────────────────▼──────────────────────────────┐  │
│  │                      LTM Store                             │  │
│  │     ADD · UPDATE · DELETE · SEARCH (semantic + overlap)    │  │
│  │     sqlite-vec vector index  ·  JSON persistence           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                               │                                  │
│  ┌────────────────────────────▼──────────────────────────────┐  │
│  │              LTM Self-Introspection Toolkit                │  │
│  │    Agent-directed retrieval, validation, and persistence   │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

[DocID: 02_architecture_b3c135]

### 4.2 Data Structures

#### 4.2.1 MemoryEntry (LTM Core)

```python
@dataclass
class MemoryEntry:
    content: str              # The actual memory content
    entry_id: str             # SHA1 hash (auto-generated)
    created_at: float         # Unix timestamp
    updated_at: float         # Last modification time
    access_count: int         # Times retrieved
    learning_score: float     # Aggregated novelty signal
    tags: list[str]           # User/agent assigned tags
    source_turn: int          # Conversation turn origin
```

[DocID: 04_memory_system_9e022d]

#### 4.2.2 ContextMessage (STM Core)

```python
@dataclass
class ContextMessage:
    role: str                 # "system", "user", "assistant"
    content: str              # Message content
    is_pinned: bool           # Protected from eviction
    relevance_score: float    # Computed relevance (0-1)
    token_count: int          # Estimated token count
    metadata: dict            # Additional context
```

#### 4.2.3 LearningFeedback (Self-Assessment)

```python
@dataclass
class LearningFeedback:
    score: float              # 0-1 novelty scale
    rationale: str            # Agent's self-assessment reasoning
    turn_index: int           # When assessed
    promoted_to_ltm: bool     # Whether promoted
```

#### 4.2.4 TriggerDecision (Rule Engine Output)

```python
@dataclass
class TriggerDecision:
    rule_id: RuleID           # R1-R5 rule identifier
    recommended_op: MemoryOp  # FILTER, SUMMARY, REVIEW, etc.
    priority: int             # 0-100 execution priority
    reason: str               # Human-readable explanation
    metadata: dict            # Additional context
```

### 4.3 Persistence Layers

#### 4.3.1 LTM Persistence

| Component | Format | Location | Purpose |
|-----------|--------|----------|---------|
| **LTM Store** | JSON | `{PERSIST_DIR}/ltm_store.json` | MemoryEntry serialization |
| **Vector Index** | SQLite + vec | `{PERSIST_DIR}/ltm_semantic.db` | Semantic similarity search |
| **Embeddings** | 1024-dim vectors | sqlite-vec | Qwen/Qwen3-Embedding-0.6B |

#### 4.3.2 STM Persistence

| Component | Format | Location | Purpose |
|-----------|--------|----------|---------|
| **Context Window** | JSON | `{PERSIST_DIR}/stm_context.json` | Active conversation messages |
| **Token Counter** | Heuristic or tiktoken | In-memory | Token estimation |

#### 4.3.3 Persistence Flow

```
LTMStore loads from disk on initialization
STMContext loads from disk on initialization
  ↓
Turn processing (in-memory operations)
  ↓
persist() called after each turn
  ↓
JSON serialization to disk
```

[DocID: 04_memory_system_9e022d]

### 4.4 Three-Layer Control System

#### Layer 1: System Rules (Deterministic)

Zero LLM cost. Pure rule engine fires memory operations based on measurable thresholds.

| Rule | ID | Trigger | Action | Priority |
|------|-----|---------|--------|----------|
| Overflow Warning | R1 | utilisation >= 75% | Force SUMMARY | 70 |
| Overflow Critical | R2 | utilisation >= 90% | Force FILTER + SUMMARY | 100 |
| Periodic Review | R3 | Every N turns | Invoke MemoryAgent | 40 |
| Learning Spike | R4 | score >= 0.8 | Immediate LTM candidacy | 90 |
| Relevance Decay | R5 | Access count threshold | Re-evaluate entry | 30 |

[DocID: 02_architecture_b3c135]

#### Layer 2: Memory Agent (LLM-Driven)

A dedicated sub-agent handles qualitative decisions:
- What content is worth storing in LTM
- Which context messages are low-relevance
- Whether compression is warranted

Triggered by the rule engine, not on every turn, keeping inference cost bounded.

#### Layer 3: Learning Score (Self-Assessed)

After every N turns (configurable), the main agent rates its own output:

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 1.0 | Explicit user preferences, permanent facts | Immediate LTM candidacy |
| 0.7 | Temporary operational state | Periodic review |
| 0.4 | Inferred goals without concrete parameters | No action |
| 0.0 | Tool executions, procedural acknowledgments | No action |

Promotion threshold: 0.65  
Spike threshold: 0.8 (bypasses periodic cadence)

[DocID: 02_architecture_b3c135]

### 4.5 LTM Self-Introspection Toolkit

A self-directed introspection system enabling the agent to reason about, orchestrate, and validate its own memory operations.

#### Tool Organization (5 Tiers)

**Tier 1 — State Assessment (Introspection)**
- `assess_conversation_drift` — Detect topic drift from conversation anchor
- `self_assess_confidence` — Evaluate confidence in current context
- `are_you_ready_to_get_in_context_ltm` — Pre-flight check combining drift and confidence

**Tier 2 — Retrieval Orchestration (Action)**
- `paraphrase_for_coverage` — Generate semantic variants for better search coverage
- `trigger_contextual_ltm_retrieval` — Execute retrieval with mode selection (single_query, multi_paraphrase, anchored)

**Tier 3 — Validation & Refinement (Quality Control)**
- `validate_ltm_relevance` — Validate retrieved memories for relevance, groundedness, faithfulness
- `refine_retrieval_target` — Generate refined query strategy after validation failure
- `compress_conversation_for_ltm` — Compress context for LTM storage

**Tier 4 — Meta-Cognitive Tools (Learning)**
- `log_retrieval_decision` — Record decision chain for policy calibration
- `suggest_retrieval_strategy` — Recommend strategy based on conversation profile

**Tier 5 — Persistence Assurance (Memory Integrity)**
- `assess_persistence_need` — Detect explicit memory commands ("remember that...", "store this...")
- `force_memory_persistence` — Bypass gating for immediate persistence
- `validate_memory_commit` — Confirm persistence succeeded (prevents "agent lies about recording" bug)
- `log_persistence_failure` — Capture failure details for debugging

[DocID: 02_architecture_b3c135]

### 4.6 Retrieval Methods

#### Semantic Search (Default)

When `ENABLE_SEMANTIC_SEARCH=true`, uses vector embeddings:

```python
# Hybrid scoring formula
score = 0.60 × cosine_similarity    # Semantic relevance
      + 0.25 × recency_decay        # Time-based decay (exp, 7-day half-life)
      + 0.15 × learning_score       # Importance weight
```

**Model:** `Qwen/Qwen3-Embedding-0.6B` (1024 dimensions)  
**Vector Store:** `sqlite-vec` for efficient similarity search

#### Overlap Fallback

When semantic search is disabled, uses token-based Jaccard similarity:

```python
def jaccard_similarity(query_tokens: set, entry_tokens: set) -> float:
    intersection = len(query_tokens & entry_tokens)
    union = len(query_tokens | entry_tokens)
    return intersection / union if union > 0 else 0.0
```

Threshold: `LTM_DEDUP_OVERLAP_THRESHOLD=0.7`

#### Context-Aware Retrieval

When `CONTEXT_AWARE_RETRIEVAL=true`, queries are enriched with conversation context:

```python
CONTEXT_CURRENT_QUERY_WEIGHT = 0.50
CONTEXT_PREVIOUS_TURN_WEIGHT = 0.30
CONTEXT_TURN_BEFORE_WEIGHT = 0.15
CONTEXT_OLDEST_TURN_WEIGHT = 0.05
CONTEXT_WINDOW_SIZE = 3  # Turns to consider
```

[DocID: 04_memory_system_9e022d]

---

## 5. Document Ingestion Pipeline

### 5.1 Overview

AgeMem includes a document processing pipeline for converting external documents into searchable corpus entries with structured metadata and entity extraction.

### 5.2 Processing Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **PDF Conversion** | Docling | PDF-to-markdown conversion |
| **OCR Detection** | PyMuPDF | Automatic detection for scanned PDFs |
| **Table Recognition** | FAST/ACCURATE modes | Structure preservation |
| **Entity Extraction** | GLiNER | Zero-shot named entity recognition |

### 5.3 Entity Extraction

Built-in label sets for different domains:

| Label Set | Domain | Example Entities |
|-----------|--------|------------------|
| `legal` | Legal documents and contracts | Party, Clause, Date, Jurisdiction |
| `research` | Scientific papers and academic publications | Author, Institution, Method, Dataset |
| `edilizia` | Italian construction and public tenders | Person, Organization, Location, Date |

Custom label configurations via YAML files:

```yaml
my_domain:
  description: "Custom domain labels"
  labels:
    - person
    - organization
    - custom_entity
  label_map:
    person: people
    organization: orgs
    custom_entity: custom
```

### 5.4 Output Format

Each ingested document becomes a markdown file with YAML frontmatter containing:

```yaml
---
doc_id: unique_hash_identifier
doc_title: Extracted or provided title
doc_type: document | research | legal | contract
source_file: Original file path
source_hash: sha256:content_hash
doc_date: Extracted or provided date
ingested_at: ISO-8601 timestamp
ner_config: Label set used
entities:
  people: [list of person entities]
  orgs: [list of organization entities]
  locations: [list of location entities]
  dates: [list of date entities]
  fields: [list of domain-specific entities]
sections: [list of document sections]
page_count: Number of pages (PDF only)
has_tables: Boolean
has_figures: Boolean
has_code: Boolean
language: Detected language
---
```

### 5.5 Usage

```bash
# Single PDF
python -m ingest.ingest report.pdf --labels research

# Markdown file
python -m ingest.ingest notes.md --labels research

# Directory batch
python -m ingest.ingest documents/ --labels edilizia

# Custom label configuration
python -m ingest.ingest report.pdf --labels /path/to/config.yaml:my_domain
```

### 5.6 Installation

Document ingestion requires additional dependencies:

```bash
# Using uv
uv sync --extra ingest

# Using pip
pip install -e ".[ingest]"
```

[DocID: 02_architecture_b3c135], [DocID: 03_getting_started_26310d]

---

## 6. Standardized Usage Instructions

### 6.1 Prerequisites

- Python 3.10+
- Git
- (Optional) CUDA-capable GPU for local inference

### 6.2 Installation

```bash
# Clone the repository
git clone https://github.com/gianpd/agemem.git
cd agemem

# Install dependencies (using uv - recommended)
uv sync

# Or using pip
pip install -e .
```

### 6.3 Basic Usage

```python
import openai
from core.config import AgememConfig
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator

# Connect to any OpenAI-compatible endpoint
client = openai.OpenAI(api_key="sk-...")

# Configure the system
cfg = AgememConfig(DEFAULT_MODEL="gpt-4o-mini")

# Initialize components
llm = LLMClient(client, default_model=cfg.DEFAULT_MODEL)
orch = Orchestrator(llm=llm, config=cfg)

# Chat with memory
response = orch.chat("My name is Alice and I'm building a Kafka pipeline.")
print(response)

# Inspect memory state
trace = orch.last_trace()
print(f"STM: {trace.stm_stats_after.utilisation_ratio:.0%} full")
print(f"LTM: {len(orch.ltm_snapshot())} entries stored")
```

### 6.4 Local Models via Ollama

```python
import openai

# Connect to Ollama
client = openai.OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

cfg = AgememConfig(
    DEFAULT_MODEL="qwen3-4b",
    STM_TOKEN_LIMIT=4096  # Smaller for local models
)

llm = LLMClient(client, default_model=cfg.DEFAULT_MODEL)
orch = Orchestrator(llm=llm, config=cfg)
```

### 6.5 Interactive REPL

```bash
uv run main.py
```

**REPL Commands:**

| Command | Effect |
|---------|--------|
| `/clear` | Reset STM (LTM retained) |
| `/memory` | Show LTM snapshot |
| `/stats` | Show STM statistics |
| `/forget` | Wipe LTM (requires confirmation) |
| `/help` | Show help |

### 6.6 Environment Variables

```bash
# LLM API settings
BASE_URL=http://localhost:8080         # LLM API base URL
BASE_MODEL=qwen3-4b                    # Model name
BASE_MAX_TOKENS=2048                   # Max tokens per request
BASE_TEMPERATURE=0.2                   # Sampling temperature

# API keys (required for non-local endpoints)
API_KEY=your-api-key                   # Primary API key
OPENAI_API_KEY=your-key                # Fallback for OpenAI

# Memory persistence
PERSIST_DIR=agent_memory               # Directory for LTM + STM storage
STM_TOKEN_LIMIT=6000                   # Context window size

# Semantic search
ENABLE_SEMANTIC_SEARCH=true
SEMANTIC_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
```

### 6.7 Running Tests

All tests are offline—no LLM calls, no network required:

```bash
# Run all tests
python -m unittest tests.test_agemem -v

# Run specific test file
python -m unittest tests.test_query_expansion -v

# Run with coverage
python -m pytest tests/ --cov=.
```

**Key Test Cases:**

| Test | Validates |
|------|-----------|
| T20 | Double-boundary overflow invariant |
| T19 | LTM promotion via learning score |
| T13-T15 | System rule firing conditions |
| T07-T08 | FILTER respects pinned messages |

[DocID: 03_getting_started_26310d]

---

## 7. Formal Evaluation Framework

### 7.1 Evaluation Objectives

1. **Retrieval Quality:** Measure precision, recall, and relevance of memory retrieval
2. **Response Quality:** Assess coherence, accuracy, and hallucination reduction
3. **Memory Persistence:** Validate cross-session memory retention and recall
4. **Resource Efficiency:** Benchmark token usage, latency, and memory footprint
5. **Comparative Performance:** Benchmark against contemporary competitors

### 7.2 Required Datasets

#### 7.2.1 Conversational Memory Benchmarks

| Benchmark | Domain | Metrics | Priority |
|-----------|--------|---------|----------|
| **LongMemEval** | Long-context memory | Recall accuracy, temporal reasoning | High |
| **LoCoMo** | Long-context modeling | Context retention, coherence | High |
| **ConvoMem** | Conversational memory | Entity recall, preference tracking | High |

#### 7.2.2 Agent Task Benchmarks

| Benchmark | Domain | Metrics | Priority |
|-----------|--------|---------|----------|
| **ALFWorld** | Embodied agents | Task completion, memory utilization | High |
| **SciWorld** | Scientific reasoning | Knowledge retention, reasoning accuracy | Medium |
| **BabyAI** | Navigation | Spatial memory, goal persistence | Medium |

#### 7.2.3 Retrieval Quality Datasets

| Dataset | Purpose | Metrics |
|---------|---------|---------|
| **MS MARCO** | Passage retrieval | MRR@K, Recall@K |
| **Natural Questions** | Open-domain QA | Exact Match, F1 |
| **Custom LTM Corpus** | Domain-specific retrieval | Precision@K, Recall@K |

### 7.3 Evaluation Metrics

#### 7.3.1 Retrieval Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MRR@K** | `1/rank_of_first_relevant` | Mean Reciprocal Rank at K |
| **Precision@K** | `relevant_in_topK / K` | Fraction of relevant results in top K |
| **Recall@K** | `relevant_in_topK / total_relevant` | Coverage of relevant results |
| **NDCG@K** | DCG@K / IDCG@K | Normalized Discounted Cumulative Gain |

#### 7.3.2 Memory Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Retention Rate** | % of promoted memories retained after N turns | ≥ 95% |
| **Deduplication Accuracy** | % of true duplicates correctly merged | ≥ 90% |
| **Learning Score Correlation** | Pearson correlation between score and utility | ≥ 0.7 |
| **Context Utilization** | % of injected memories referenced in response | ≥ 60% |

#### 7.3.3 Response Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Hallucination Rate** | % of responses with unsupported claims | ≤ 5% |
| **Coherence Score** | Human-rated response coherence (1-5) | ≥ 4.0 |
| **Memory Grounding** | % of memory-dependent claims with valid citations | ≥ 90% |
| **Preference Accuracy** | % of user preferences correctly recalled | ≥ 95% |

### 7.4 Competitor Benchmarks

| System | Architecture | Key Difference |
|--------|--------------|----------------|
| **MemGPT** | Virtual context management | RL-trained memory policy |
| **Letta** | Agent memory systems | Cloud-hosted, API-dependent |
| **LangChain RAG** | Standard retrieval-augmented generation | Flat chunk retrieval, no memory tiers |
| **LlamaIndex** | Document indexing and retrieval | Query-focused, no persistent memory |
| **Base LLM (no memory)** | No memory system | Baseline comparison |

### 7.5 Evaluation Protocol

#### Phase 1: Retrieval Quality (Automated)

1. Populate LTM with 500 curated memories from benchmark datasets
2. Execute 1000 queries across benchmark test sets
3. Measure MRR@K, Precision@K, Recall@K at K=1,5,10
4. Compare semantic vs overlap retrieval modes
5. Log variant hit-rate for query expansion

#### Phase 2: Memory Persistence (Multi-Session)

1. Conduct 50-turn conversations across 10 sessions
2. Track promoted memories and their retrieval across sessions
3. Measure retention rate and deduplication accuracy
4. Validate learning score correlation with actual utility

#### Phase 3: Response Quality (Human Evaluation)

1. Conduct 100 conversations with human evaluators
2. Rate responses on coherence, accuracy, memory grounding
3. Measure hallucination rate via fact verification
4. Track preference accuracy across sessions

#### Phase 4: Comparative Benchmarking

1. Run identical tasks across all competitor systems
2. Normalize metrics for fair comparison
3. Measure resource usage (tokens, latency, memory)
4. Generate comparative performance report

### 7.6 Evaluation Infrastructure

#### SearchTrace Instrumentation

```python
@dataclass
class SearchTrace:
    query: str
    query_embedding: list[float]
    results: list[tuple[str, float]]  # (entry_id, score)
    latency_ms: float
    mode: str  # "semantic", "overlap", "expanded"
    variant_used: Optional[str]
```

#### SQLite Logging

```sql
CREATE TABLE search_traces (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    query TEXT,
    query_embedding BLOB,
    results_json TEXT,
    latency_ms REAL,
    mode TEXT,
    variant_used TEXT
);
```

#### MRR Harness

```python
def evaluate_mrr(queries: list[Query], ltm: LTMStore, k: int = 10) -> float:
    """Calculate Mean Reciprocal Rank at K"""
    reciprocal_ranks = []
    for query in queries:
        results = ltm.search(query.text, top_k=k)
        for rank, (entry, score) in enumerate(results, 1):
            if entry.entry_id in query.relevant_ids:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

### 7.7 Reporting

Evaluation results should be reported in the following format:

```markdown
## Evaluation Report: AgeMem v[VERSION]

### Retrieval Quality
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MRR@5 | 0.XX | ≥ 0.70 | ✅/❌ |
| Precision@5 | 0.XX | ≥ 0.60 | ✅/❌ |
| Recall@10 | 0.XX | ≥ 0.80 | ✅/❌ |

### Memory Persistence
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Retention Rate | XX% | ≥ 95% | ✅/❌ |
| Dedup Accuracy | XX% | ≥ 90% | ✅/❌ |

### Response Quality
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Hallucination Rate | XX% | ≤ 5% | ✅/❌ |
| Coherence Score | X.X | ≥ 4.0 | ✅/❌ |

### Comparative Performance
| System | MRR@5 | Halluc. Rate | Tokens/Query |
|--------|-------|--------------|--------------|
| AgeMem | 0.XX | XX% | XXX |
| MemGPT | 0.XX | XX% | XXX |
| LangChain RAG | 0.XX | XX% | XXX |
```

[DocID: 07_roadmap_9f6c7e]

---

## 8. Appendices

### 8.1 Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `STM_TOKEN_LIMIT` | 6000 | Context window size in tokens |
| `LTM_MAX_ENTRIES` | 1000 | Maximum LTM entries before pruning |
| `LTM_DEDUP_THRESHOLD` | 0.92 | Cosine similarity dedup threshold |
| `LTM_DEDUP_OVERLAP_THRESHOLD` | 0.70 | Jaccard similarity dedup threshold |
| `OVERFLOW_WARN` | 0.75 | Warning threshold (force SUMMARY) |
| `OVERFLOW_CRITICAL` | 0.90 | Critical threshold (force FILTER) |
| `LEARNING_SPIKE` | 0.80 | Immediate LTM candidacy threshold |
| `SEMANTIC_EMBEDDING_MODEL` | Qwen/Qwen3-Embedding-0.6B | Embedding model |
| `QUERY_EXPANSION_N_VARIANTS` | 3 | Number of query variants |

### 8.2 Directory Structure

```
agemem/
├── core/           # Data types, configuration, utilities
├── memory/         # LTM/STM storage implementations
├── triggers/       # Rule engine
├── agents/         # LLM-facing components
├── tools/          # Utility tools (web search, etc.)
├── skills/         # Skill detection and loading
├── prompts/        # Prompt registry
├── ingest/         # Document ingestion pipeline
└── tests/          # Unit tests
```

### 8.3 Design Patterns

| Pattern | Usage |
|---------|-------|
| **Orchestrator** | Central coordinator, only place that writes to LTM/STM |
| **Strategy** | Retrieval modes (semantic, overlap, expanded) |
| **Observer** | SystemRules fire based on state changes |
| **Template Method** | Introspection tools follow tiered structure |
| **Factory** | MemoryEntry creation with auto-generated IDs |

### 8.4 Critical Invariants

1. **Overflow Invariant:** Context must be within STM_TOKEN_LIMIT at both pre-turn and post-turn boundaries
2. **Pinned Protection:** Messages with `is_pinned=True` are never evicted by FILTER
3. **Single Writer:** Only Orchestrator writes to LTM/STM; other components are read-only
4. **Offline Testability:** All tests run without LLM calls or network access

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-18 | AgeMem Team | Initial specification |

---

*This specification is derived from the AgeMem corpus documentation. All citations reference source documents by DocID for traceability.*
