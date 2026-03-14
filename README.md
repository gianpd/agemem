# AgeMem — Memory Data Layer for the Agentic AI Era

**Inference-only, privacy-first, long-term + short-term memory management for LLM agents on any hardware.**

[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen)](tests/test_agemem.py)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

---

## The Problem We're Solving

Every serious agent deployment hits the same wall.

You can buy a 1-million-token context window. You can throw a 70B model at it. You'll still watch your agent hallucinate a user preference it was told three sessions ago, because that fact was silently evicted when the context overflowed — or it was buried so deep in a sea of irrelevant tokens that the model's attention never reached it.

**The memory wall is not a model problem. It is a systems problem.**

The industry's current answer — bigger contexts, more VRAM, cloud-hosted memory APIs — is the software equivalent of solving a bad search engine by making the database bigger. It is compute-inefficient, privacy-hostile, and fundamentally non-local.

AgeMem is built on a different thesis:

> **500 perfectly curated memories on a 9B model will consistently outperform 10,000 uncurated RAG chunks on a 70B model.**

We prove this not with benchmarks on a leased datacenter cluster, but on an **8GB RTX 4060 at 36 tokens/second**.

---

## What AgeMem Is

AgeMem is a hybrid memory management system that gives any LLM agent — running on any OpenAI-compatible endpoint, including fully local models via Ollama — a disciplined, auditable memory architecture with two tiers:

- **Short-Term Memory (STM)** — the active context window, managed with surgical precision. Messages are filtered by relevance score, summarised when the window fills, and hard-dropped only when no other option remains. Pinned content (system prompt, injected LTM) is never evicted under any pressure level.

- **Long-Term Memory (LTM)** — a persistent store of high-value facts, promoted from STM based on a learning signal and retrieved via semantic search. Backed by `sqlite-vec` for vector similarity, with a Jaccard overlap fallback that works on CPU with zero dependencies.

The system decides when to move information between tiers, when to compress, and when to discard — without requiring fine-tuned weights. It runs at inference time, on your hardware, with your data never leaving your machine.

---

## Why Now

Bryan Catanzaro (VP of Applied Deep Learning Research, NVIDIA) recently articulated what the open AI ecosystem is converging toward: *AI as open infrastructure*, not walled-garden products. The Nemotron Nano releases are one data point. Llama, Qwen, Mistral are others. Highly capable open models are rapidly approaching the quality of frontier closed APIs — and they run locally.

When that transition completes, the final bottleneck will not be the model's reasoning. It will be **state management**. An agent that can reason beautifully but forgets everything is not an agent. It is an expensive autocomplete.

AgeMem is building that missing layer — the memory infrastructure that makes local open models genuinely useful across sessions, users, and tasks.

---

## Architecture

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
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Three-layer control

**Layer 1 — System Rules (deterministic, zero LLM cost).** A pure rule engine fires memory operations based on measurable thresholds: context token utilisation, turn count, learning score magnitude. These rules form the correctness floor. They cannot be overridden by the agent.

**Layer 2 — Memory Agent (LLM-driven, dedicated sub-agent).** A separate LLM call handles qualitative decisions: what content is worth storing in LTM, which context messages are low-relevance, whether compression is warranted. Triggered by the rule engine, not on every turn, keeping inference cost bounded.

**Layer 3 — Learning Score (self-assessed signal).** After every N turns, the main agent rates its own output on a 0–1 novelty scale. Scores above the promotion threshold trigger LTM candidacy. Scores above the spike threshold bypass the periodic cadence entirely. This is the inference-time proxy for the reward signal that RL training would otherwise require.

---

## Key Technical Properties

### Dual overflow guard (the double-boundary invariant)

A single `force_fit()` call at turn start is insufficient. An assistant response can itself be longer than the remaining token budget, pushing a 70% context to 105% in a single step. AgeMem enforces the overflow invariant at **both** message-append boundaries — before the user message and after the assistant response. Test T20 was written specifically to catch this failure mode.

### Semantic deduplication

`_find_similar()` uses full-content Jaccard similarity in the overlap-only path (not leading-word prefix matching, which collapses distinct facts that happen to share an opening phrase). When semantic search is enabled, cosine similarity on unit-normalised embeddings replaces the heuristic entirely, with a configurable threshold (`LTM_DEDUP_THRESHOLD=0.92`).

### Hybrid retrieval scoring

```
score = 0.6 × cosine_similarity
      + 0.25 × recency_decay (exp, 7-day half-life)
      + 0.15 × learning_score
```

Semantic relevance dominates, but recent and high-salience entries get a measurable boost. Query expansion generates paraphrase variants and merges results, with per-variant attribution so you can measure which variant actually retrieved the winning entry.

### Privacy by default

The entire stack — `llama.cpp` inference, `sqlite-vec` vector index, GLiNER NER enrichment, JSON persistence — runs locally. No telemetry, no cloud round-trips, no API keys required beyond your local model server.

---

## Quickstart

```python
import openai
from core.config import AgememConfig
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator

client = openai.OpenAI(api_key="sk-...")          # or ollama / any compatible endpoint
cfg = AgememConfig(DEFAULT_MODEL="gpt-4o-mini")
llm = LLMClient(client, default_model=cfg.DEFAULT_MODEL)
orch = Orchestrator(llm=llm, config=cfg)

response = orch.chat("My name is Alice and I'm building a Kafka pipeline.")
print(response)

trace = orch.last_trace()
print(f"STM: {trace.stm_stats_after.utilisation_ratio:.0%} full")
print(f"LTM: {len(orch.ltm_snapshot())} entries stored")
```

**For local models via Ollama:**
```python
client = openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
```

**Run the test suite (no API key, no network):**
```bash
python -m unittest tests.test_agemem -v
# 35 tests, 0 failures
```

---

## Configuration

All thresholds live in a single `AgememConfig` dataclass in `core/config.py`. Nothing is hardcoded.

| Parameter | Default | Effect |
|---|---|---|
| `STM_TOKEN_LIMIT` | 6000 | Hard context ceiling |
| `STM_WARNING_THRESHOLD` | 0.75 | SUMMARY fires above this |
| `STM_CRITICAL_THRESHOLD` | 0.90 | FILTER + hard-drop above this |
| `LTM_PROMOTE_THRESHOLD` | 0.65 | Learning score → LTM ADD |
| `LTM_DEDUP_THRESHOLD` | 0.92 | Cosine sim threshold for dedup (semantic) |
| `LTM_DEDUP_OVERLAP_THRESHOLD` | 0.70 | Jaccard threshold for dedup (overlap fallback) |
| `LEARNING_SCORE_PROMPT_EVERY_N` | 3 | Feedback collection cadence |
| `TRIGGER_EVERY_N_TURNS` | 10 | Memory Agent full review cadence |
| `MEMORY_AGENT_MODEL` | gpt-4o-mini | Can differ from main agent |
| `ENABLE_QUERY_EXPANSION` | False | Multi-variant retrieval |

---

## Project Structure

```
agemem/
├── core/
│   ├── types.py            # All data contracts
│   └── config.py           # All thresholds — one place
├── memory/
│   ├── ltm_store.py        # LTM: ADD/UPDATE/DELETE/SEARCH/PRUNE
│   ├── stm_context.py      # STM: FILTER/SUMMARY/RETRIEVE/force_fit
│   ├── embedding.py        # Embedding model (lazy-loaded)
│   └── vector_index.py     # sqlite-vec wrapper
├── triggers/
│   └── system_rules.py     # Deterministic rule engine (R1–R4)
├── agents/
│   ├── llm_client.py       # OpenAI-compatible wrapper
│   ├── memory_agent.py     # Qualitative memory decisions
│   ├── learning_scorer.py  # Self-assessment feedback
│   └── orchestrator.py     # Turn coordinator
├── tools/
│   └── query_expansion.py  # Paraphrase variant generation
└── tests/
    └── test_agemem.py      # 35 offline unit tests — no LLM required
```

---

## Roadmap

The following items are actively being worked on or designed. Contributions in any of these areas are especially welcome.

**Near-term**
- [ ] MRR@K evaluation harness with `SearchTrace` instrumentation and SQLite logging
- [ ] Variant hit-rate metric — measures query expansion ROI against latency cost
- [ ] `_find_similar` semantic path for overlap-only stores (full Jaccard → embedding upgrade path)
- [ ] Entity-retention checks via GLiNER NER to prevent important named entities being pruned

**Medium-term**
- [ ] Multi-agent memory sharing — shared LTM store across agent instances with conflict resolution
- [ ] Memory compaction — periodic background consolidation of related LTM entries
- [ ] Streaming token counting for models with non-whitespace tokenisers
- [ ] Benchmarks on AgeMem paper tasks: ALFWorld, SciWorld, BabyAI

**Long-term vision**
- [ ] On-device fine-tuning of the memory promotion policy using AgeMem's own `learning_score` signal as reward — closing the loop between inference-only and the RL training the original paper required
- [ ] Cross-session memory graphs — structured entity relationships, not just flat key-value facts
- [ ] Memory federation — privacy-preserving sync across devices with local encryption

---

## Contributing

AgeMem is at an inflection point. The core architecture is proven and tested. The retrieval layer is getting measurably better. The gap between this system and the RL-trained AgeMem paper baseline is understood and documented — and there is a clear engineering path to close it without requiring fine-tuning compute.

**This is the moment to get involved.**

If you work on any of the following, your contribution will have immediate, measurable impact:

- **Retrieval quality** — embedding models, reranking, hybrid BM25+semantic, query expansion tuning
- **Evaluation** — MRR@K harness, shadow-mode A/B testing, benchmark integration (ALFWorld, BabyAI)
- **Local inference** — llama.cpp integration, quantisation testing, edge hardware profiling
- **NLP** — NER-guided memory promotion, entity extraction, coreference resolution
- **Infrastructure** — async orchestration, multi-agent coordination, persistence backends

To contribute:

1. Fork and clone the repo
2. Run `python -m unittest tests.test_agemem -v` — all 35 should pass
3. Open an issue describing what you want to work on, or pick one from the roadmap
4. Submit a PR with tests — the test suite is the contract

Please read the inline documentation in `core/config.py` and `memory/ltm_store.py` before opening a PR. The codebase is intentionally small and readable. A new contributor should be able to understand the full system in an afternoon.

---

## Why AgeMem Matters

NVIDIA's Catanzaro framed it precisely: *"Every fast computer is also a slow computer."* The purpose of accelerated compute is not to do everything — it is to prioritise and focus on the workloads that matter.

The agent memory problem is the same problem. A 1-million-token context is not intelligence. It is compute spent on attention over irrelevant tokens. AgeMem applies the same logic that makes accelerated computing efficient to the software layer: filter ruthlessly, retain purposefully, retrieve precisely.

As open-weight models get smaller and smarter — Llama, Qwen, Nemotron, and whatever comes next — the model ceases to be the bottleneck. The agent that wins in a world of capable local models is the one with the best memory, not the biggest context.

AgeMem is building that memory layer. It is open, auditable, runs on your hardware, and is designed to be extended by the community.

**Come build it with us.**

---

## Reference

Yu, Y., Yao, L., Xie, Y., Tan, Q., Feng, J., Li, Y., & Wu, L. (2026). *Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents*. arXiv:2601.01885.

---

*AgeMem is an independent open-source project. It is not affiliated with or endorsed by NVIDIA, Anthropic, or the authors of the AgeMem paper.*