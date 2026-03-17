# Roadmap & Future Direction

## Current State

AgeMem is at an inflection point. The core architecture is proven and tested with 35+ offline unit tests. The retrieval layer is getting measurably better. The gap between this system and the RL-trained AgeMem paper baseline is understood and documented.

### What's Working

- Double-boundary overflow protection
- Semantic search with sqlite-vec
- Learning score-driven promotion
- Context-aware retrieval
- Query expansion with fallback transforms
- Full offline testability

### Known Limitations

| Area | Current State | Target |
|------|---------------|--------|
| Retrieval Quality | Good for ≤500 entries | Scale to 5000+ |
| Token Counting | Heuristic approximation | tiktoken accuracy |
| Multi-Agent | Single agent only | Shared memory |
| Evaluation | Manual testing | MRR@K benchmarks |

---

## Near-Term (Q1-Q2 2026)

### MRR@K Evaluation Harness

**Goal**: Measurable retrieval quality

- Implement `SearchTrace` instrumentation
- SQLite logging for query/result pairs
- Mean Reciprolical Rank at K evaluation
- Compare semantic vs overlap retrieval

**Contribution Opportunity**: Build the evaluation infrastructure

### Variant Hit-Rate Metric

**Goal**: Measure query expansion ROI

- Track which variant retrieved winning entry
- Calculate latency vs quality tradeoff
- Auto-tune `QUERY_EXPANSION_N_VARIANTS`

**Contribution Opportunity**: Metrics and dashboards

### Entity-Retention Checks

**Goal**: Prevent important named entities from being pruned

- GLiNER NER integration
- Entity-aware FILTER decisions
- Entity importance scoring

**Contribution Opportunity**: NLP pipeline integration

---

## Medium-Term (Q2-Q4 2026)

### Multi-Agent Memory Sharing

**Goal**: Shared LTM across agent instances

```
┌──────────────┐     ┌──────────────┐
│   Agent A    │     │   Agent B    │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
        ┌───────▼───────┐
        │  Shared LTM   │
        │  (conflict    │
        │   resolution) │
        └───────────────┘
```

**Challenges**:
- Conflict resolution (same key, different values)
- Access control and permissions
- Consistency guarantees

**Contribution Opportunity**: Architecture and implementation

### Memory Compaction

**Goal**: Background consolidation of related LTM entries

- Cluster similar entries
- Generate summary entries
- Prune redundant information
- Maintain source attribution

**Contribution Opportunity**: Clustering algorithms

### Streaming Token Counting

**Goal**: Accurate counts for non-whitespace tokenizers

- Character-level streaming
- Model-specific tokenizer support
- Fallback for unknown models

**Contribution Opportunity**: Tokenizer integration

### Benchmark Integration

**Goal**: Validate on standard agent benchmarks

| Benchmark | Domain | Priority |
|-----------|--------|----------|
| ALFWorld | Embodied agents | High |
| SciWorld | Scientific reasoning | Medium |
| BabyAI | Navigation | Medium |

**Contribution Opportunity**: Benchmark runner implementation

---

## Long-Term Vision (2026+)

### On-Device Fine-Tuning

**Goal**: Close the RL loop using AgeMem's own signals

The original AgeMem paper required RL training to learn the memory policy. Our inference-only approach compensates through prompting and rules. The next step: use the `learning_score` signal as reward for on-device fine-tuning.

```
User Interaction
       │
       ▼
Learning Score (reward signal)
       │
       ▼
LoRA/QLoRA fine-tuning
       │
       ▼
Improved memory policy
```

**Challenges**:
- Compute requirements for fine-tuning
- Data quality and quantity
- Catastrophic forgetting prevention

**Contribution Opportunity**: Training pipeline design

### Cross-Session Memory Graphs

**Goal**: Structured entity relationships, not flat facts

```
User ─── works_at ─── Company
  │                    │
  └── uses ─── Tool ───┘
```

Features:
- Entity extraction and linking
- Relationship inference
- Graph-based retrieval
- Temporal evolution tracking

**Contribution Opportunity**: Graph database integration

### Memory Federation

**Goal**: Privacy-preserving sync across devices

```
┌─────────────┐     ┌─────────────┐
│   Device A  │     │   Device B  │
│  (encrypted)│◄───►│  (encrypted)│
└─────────────┘     └─────────────┘
        │                 │
        └────────┬────────┘
                 │
         ┌───────▼───────┐
         │  Sync Server  │
         │  (encrypted)  │
         └───────────────┘
```

Features:
- End-to-end encryption
- Conflict-free replicated data types (CRDTs)
- Selective sharing

**Contribution Opportunity**: Cryptography and sync protocols

---

## How to Contribute

This is the moment to get involved. Your contribution will have immediate, measurable impact.

### High-Impact Areas

| Area | Skills Needed | Impact |
|------|---------------|--------|
| Retrieval Quality | Embeddings, reranking, BM25 | Direct user experience improvement |
| Evaluation | MRR harness, benchmark integration | Measurable quality metrics |
| Local Inference | llama.cpp, quantization | Edge deployment enablement |
| NLP | NER, entity extraction, coreference | Smarter memory promotion |
| Infrastructure | Async, multi-agent, persistence | Scale and reliability |

### Getting Started

1. Read [01-Overview.md](01-Overview.md) and [02-Architecture.md](02-Architecture.md)
2. Run `python -m unittest tests.test_agemem -v`
3. Pick an issue from the roadmap
4. Open a PR with tests

### Research References

- [AgeMem Paper](https://arxiv.org/abs/2601.01885) - Original research
- [MemGPT](https://memgpt.ai/) - Virtual context management
- [Letta](https://letta.com/) - Agent memory systems

---

## Project Philosophy

### Why This Matters

NVIDIA's Bryan Catanzaro: *"Every fast computer is also a slow computer."* The purpose of accelerated compute is not to do everything — it is to prioritise and focus on the workloads that matter.

The agent memory problem is the same problem. A 1-million-token context is not intelligence. It is compute spent on attention over irrelevant tokens. AgeMem applies the same logic that makes accelerated computing efficient to the software layer: **filter ruthlessly, retain purposefully, retrieve precisely.**

### Open by Design

As open-weight models get smaller and smarter — Llama, Qwen, Nemotron — the model ceases to be the bottleneck. The agent that wins in a world of capable local models is the one with the best memory, not the biggest context.

AgeMem is building that memory layer. It is open, auditable, runs on your hardware, and is designed to be extended by the community.

**Come build it with us.**