# AgeMem — Project Overview

## The Problem We're Solving

Every serious agent deployment hits the same wall. You can buy a 1-million-token context window. You can throw a 70B model at it. You'll still watch your agent hallucinate a user preference it was told three sessions ago, because that fact was silently evicted when the context overflowed — or it was buried so deep in a sea of irrelevant tokens that the model's attention never reached it.

**The memory wall is not a model problem. It is a systems problem.**

The industry's current answer — bigger contexts, more VRAM, cloud-hosted memory APIs — is the software equivalent of solving a bad search engine by making the database bigger. It is compute-inefficient, privacy-hostile, and fundamentally non-local.

## Core Thesis

> **500 perfectly curated memories on a 9B model will consistently outperform 10,000 uncurated RAG chunks on a 70B model.**

AgeMem proves this not with benchmarks on a leased datacenter cluster, but on an **8GB RTX 4060 at 36 tokens/second**.

## What AgeMem Is

AgeMem is a hybrid memory management system that gives any LLM agent — running on any OpenAI-compatible endpoint, including fully local models via Ollama — a disciplined, auditable memory architecture with two tiers:

| Tier | Purpose | Behavior |
|------|---------|----------|
| **Short-Term Memory (STM)** | Active context window | Messages filtered by relevance, summarised when full, hard-dropped only as last resort |
| **Long-Term Memory (LTM)** | Persistent knowledge store | High-value facts promoted from STM, retrieved via semantic search |

The system decides when to move information between tiers, when to compress, and when to discard — **without requiring fine-tuned weights**. It runs at inference time, on your hardware, with your data never leaving your machine.

## Key Differentiators

### 1. Inference-Only Architecture

AgeMem requires no RL training, no gradient updates, no fine-tuning. All behavior is derived from:
- Deterministic system rules (threshold-based triggers)
- Prompted agent decisions (structured JSON output)
- Self-assessed learning scores (novelty rating 0-1)

### 2. Privacy by Default

The entire stack runs locally:
- `llama.cpp` inference
- `sqlite-vec` vector index
- GLiNER NER enrichment
- JSON persistence

No telemetry, no cloud round-trips, no API keys required beyond your local model server.

### 3. Double-Boundary Overflow Guard

A single `force_fit()` call at turn start is insufficient. An assistant response can itself be longer than the remaining token budget, pushing a 70% context to 105% in a single step. AgeMem enforces the overflow invariant at **both** message-append boundaries — before the user message and after the assistant response.

### 4. Hybrid Retrieval Scoring

```
score = 0.6 × cosine_similarity
      + 0.25 × recency_decay (exp, 7-day half-life)
      + 0.15 × learning_score
```

Semantic relevance dominates, but recent and high-salience entries get measurable boosts.

## When to Use AgeMem

**Ideal for:**
- Local LLM deployments (Ollama, llama.cpp, vLLM)
- Privacy-sensitive applications (healthcare, legal, finance)
- Multi-session agents that need persistent memory
- Resource-constrained environments (consumer GPUs)

**Not ideal for:**
- Cloud-only deployments with unlimited context budgets
- Single-turn stateless interactions
- Applications requiring RL-based memory optimization

## Reference Implementation

AgeMem is a principled adaptation of the research paper:

> Yu, Y., Yao, L., Xie, Y., Tan, Q., Feng, J., Li, Y., & Wu, L. (2026). *Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents*. arXiv:2601.01885.

The key difference: AgeMem compensates for the lack of RL training through a three-layer hybrid control architecture that achieves comparable behavior through inference-time mechanisms.