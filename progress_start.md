# AgeMem: Progress Start

**An Open-Source Memory Data Layer for the Agentic World**

*Research & Positioning Document — March 2026*

---

## What AgeMem Actually Is

AgeMem is an inference-only, hybrid memory management system for LLM agents. It implements a principled three-layer architecture that compensates for the absence of RL training (as found in the AgeMem academic paper) through deterministic safety rules, a dedicated Memory Agent sub-component, and agent self-assessment feedback loops.

At its core, AgeMem solves one problem: **how do you give an LLM agent persistent memory without fine-tuning the model?** The answer is a control system that decides what to remember, when to compress, and how to retrieve — all running alongside the main agent without modifying its weights.

---

## Core Architecture: The Three-Layer Hybrid

### Layer 1: System Rules (Deterministic Safety)
A rule engine that fires memory operations based on measurable thresholds:
- **R1 (Overflow Warning)**: At 75% context utilisation → trigger SUMMARY
- **R2 (Critical Overflow)**: At 90% utilisation → force FILTER + SUMMARY
- **R3 (Periodic Review)**: Every N turns → invoke MemoryAgent
- **R4 (Learning Spike)**: High novelty score → immediate LTM promotion

These rules are unconditional, auditable, and testable without any LLM calls. They form the safety floor.

### Layer 2: Memory Agent (Qualitative Judgment)
A dedicated LLM sub-agent that makes nuanced decisions:
- What content deserves long-term storage (ADD/UPDATE/DELETE)
- Which messages are low-relevance for eviction
- When context compression is warranted

The MemoryAgent runs only when triggered by Layer 1, controlling inference costs while providing the qualitative judgment a rules engine cannot.

### Layer 3: Learning Scorer (Signal Source)
The main agent periodically rates its own turns on a 0–1 novelty scale. This self-assessment:
- Drives LTM promotion decisions
- Feeds into MemoryAgent context
- Bypasses periodic cadence for high-novelty exchanges

This is an inference-time approximation of the reward signal that RL training would bake into weights.

---

## Originality: What Makes AgeMem Different

### 1. The Double-Boundary Overflow Invariant
A concrete engineering contribution: AgeMem enforces context bounds **both** before the user message is appended **and** after the assistant response. This is necessary because a long assistant reply can push context from 70% to 105% utilisation in one step. Most inference-only systems guard only at turn start and fail on long generations.

### 2. Inference-Only Design Philosophy
AgeMem accepts a constraint most systems ignore: the LLM weights are frozen. It does not require:
- Fine-tuning or RL training
- Embedding model dependencies (token-overlap fallback available)
- External vector databases (SQLite-based semantic search optional)

This makes AgeMem deployable on commodity hardware with any OpenAI-compatible endpoint.

### 3. The Ingest Pipeline: NER + Light RAG
AgeMem includes a document ingestion system that differentiates it from pure memory frameworks:

**4-Stage Pipeline:**
1. **Parse**: Uses Docling (IBM) to convert PDFs to markdown with table structure recognition
2. **Extract**: Uses GLiNER (zero-shot NER) to extract named entities without training
3. **Write**: Creates markdown with YAML frontmatter containing entities, document identity, and structure metadata
4. **Index**: Updates corpus index for quick lookups

**Domain-Specific Label Sets:**
- `edilizia`: Italian construction/tenders (CIG, CUP, RUP codes)
- `research`: Scientific papers (datasets, algorithms, citations)
- `legal`: Contracts (parties, clauses, jurisdictions)

This is a **local, privacy-preserving document intelligence system** — no API calls during ingestion.

### 4. Skills System (Dynamic Capability Injection)
Skills are documents with trigger keywords that get injected into context when relevant. Unlike static system prompts, skills:
- Load dynamically from a corpus directory
- Activate based on keyword matching
- Provide contextual hints without code changes

Example: A "profittability" skill triggers on construction tender keywords and guides the agent through profitability calculations.

### 5. Semantic Search (Optional, Zero-Additional-Infra)
Recent addition using sqlite-vec for vector similarity search:
- Embedding model: Qwen3-Embedding-0.6B (600M params, Apache 2.0)
- Storage: SQLite virtual tables (no new infrastructure)
- Algorithm: Brute-force cosine similarity (exact, not HNSW)
- Fallback: Token-overlap scoring when embeddings unavailable

This scales from hundreds to tens of thousands of entries without architectural changes.

---

## Competitive Landscape: Honest Comparison

### MemGPT (Letta)
**Similarity**: Both implement unified LTM/STM management
**Difference**: MemGPT requires wrapping the LLM in a control loop with explicit memory tools. AgeMem is transparent — it runs alongside any OpenAI-compatible API without requiring the agent to call memory tools. MemGPT's approach needs model cooperation; AgeMem's works with frozen weights.

**Trade-off**: MemGPT can be more powerful with a cooperative model; AgeMem works with any model but makes conservative decisions.

### LangChain Memory
**Similarity**: Both provide conversation memory abstractions
**Difference**: LangChain offers simple buffer windows, summary chains, and entity extraction. AgeMem provides:
- Automatic context compression with forced bounds
- Learning-score-driven memory promotion
- Dedicated MemoryAgent for qualitative decisions
- Persistent LTM with retrieval scoring

**Trade-off**: LangChain is simpler and more widely adopted; AgeMem is more opinionated and handles edge cases (overflow, learning spikes) explicitly.

### OpenAI Assistants API
**Similarity**: Both provide retrieval and thread management
**Difference**: OpenAI's API is closed-source, hosted, and embedding-based. AgeMem is:
- Fully open-source and auditable
- Runs locally with no external dependencies
- Token-overlap fallback requires no embedding models
- Configurable thresholds and rules

**Trade-off**: OpenAI's solution is zero-config and scales infinitely; AgeMem is private, offline-capable, and tunable.

### Zep AI
**Similarity**: Both provide long-term memory for LLM apps
**Difference**: Zep is a hosted service with automatic fact extraction and user memory. AgeMem is:
- Self-hosted with no external API calls
- Explicit about when and what to remember
- Includes document ingestion pipeline

**Trade-off**: Zep abstracts complexity; AgeMem exposes it for control.

### Honest Assessment of Gaps

| Capability | AgeMem Status | Notes |
|------------|---------------|-------|
| RL-trained memory policy | ❌ Not available | Paper shows 4-8% gains from RL; AgeMem compensates with rules |
| Embedding-based retrieval | ⚠️ Optional | sqlite-vec added; not as battle-tested as Chroma/Qdrant |
| Multi-user isolation | ⚠️ Partial | Persistence directories configurable per user |
| Distributed memory | ❌ Not implemented | Single-node only |
| Web UI | ❌ Not included | CLI-only REPL |
| Model fine-tuning | ❌ Not applicable | Inference-only by design |

---

## Long-Term Vision: The Founder Perspective

### The Problem Worth Solving
Current LLM agents are amnesiac. They forget the conversation when the window closes. They cannot accumulate knowledge across sessions. They cannot reference documents without expensive re-ingestion. They treat every user the same, regardless of history.

The agentic world needs a **memory data layer** — not a feature, but infrastructure. Like databases revolutionised application development by separating persistence from logic, AgeMem aims to separate memory from agent implementation.

### The Open-Source Thesis
Memory is too important to be proprietary. User data, conversation history, and learned preferences should not be locked in closed APIs. AgeMem is committed to:

1. **Full transparency**: Every memory decision is auditable (TurnTrace)
2. **Local-first**: Runs entirely on user's hardware
3. **Standard compatibility**: OpenAI-compatible, works with any model
4. **No vendor lock-in**: JSON persistence, SQLite storage, plain-text corpus

### Technical Roadmap

**Phase 1: Core Stability (Current)**
- Hybrid memory architecture complete
- Ingest pipeline with NER + RAG functional
- Semantic search with sqlite-vec operational
- 28 unit tests, no LLM required for validation

**Phase 2: Scale & Quality**
- [ ] Hybrid search (semantic + keyword fusion)
- [ ] Memory quality metrics (precision@k, recall)
- [ ] Multi-session user identity
- [ ] Web UI for memory inspection

**Phase 3: Integration Ecosystem**
- [ ] MCP (Model Context Protocol) server
- [ ] LangChain/LlamaIndex adapters
- [ ] REST API for memory operations
- [ ] Sync protocol for cross-device memory

**Phase 4: Advanced Memory**
- [ ] Hierarchical memory (episodic → semantic → procedural)
- [ ] Memory consolidation (sleep-like compression)
- [ ] Cross-document knowledge graph
- [ ] Federated learning for shared knowledge (opt-in)

### The Differentiator
Most memory systems treat memory as a **cache** — something to be managed automatically and forgotten when convenient. AgeMem treats memory as a **database** — something to be intentionally designed, queried, and preserved.

The ingest pipeline with domain-specific NER is the embodiment of this: documents are not just embedded; they are **understood** at ingestion time, with entities extracted into searchable buckets. This is the difference between "find similar text" and "find all documents mentioning CIG code X in the last 6 months."

### Business Model (Hypothetical)
AgeMem remains MIT-licensed and free. Sustainability through:
- **Managed hosting**: Optional cloud service for teams
- **Enterprise features**: SSO, audit logs, compliance tools
- **Consulting**: Implementation and custom label set development
- **Support**: SLA-backed support for production deployments

The core remains open. The value is in the ecosystem and the guarantee that user data stays user-owned.

---

## Current State: What Works Today

✅ **Robust memory management**: STM/LTM with overflow guards
✅ **Document ingestion**: PDF → Markdown with NER extraction
✅ **Semantic search**: sqlite-vec with Qwen3 embeddings
✅ **Skill system**: Dynamic capability injection
✅ **Tool integration**: Web search, corpus search, file I/O
✅ **Persistence**: JSON + SQLite across sessions
✅ **Configurability**: All thresholds tunable via config
✅ **Observability**: Turn-by-turn traces with operation logs

⚠️ **Needs validation**: Semantic search at scale (>10K entries)
⚠️ **Needs development**: Web UI, multi-user, distributed sync
⚠️ **Needs research**: Memory quality metrics, consolidation

---

## Conclusion

AgeMem is not the most sophisticated memory system. It does not use RL training. It does not require a specific model. It does not have the largest feature set.

What it offers is **principled simplicity**: a memory layer that works with any OpenAI-compatible endpoint, runs locally, respects privacy, and makes its decisions auditable. The ingest pipeline with NER + light RAG provides immediate utility beyond chat history — it turns documents into structured, searchable knowledge.

The long-term bet is that the agentic world will need memory infrastructure as foundational as databases were for the web. AgeMem aims to be that foundation — open, local, and built to last.

---

*Document Version: 1.0*
*Last Updated: 2026-03-11*
