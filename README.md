# AgeMem-Hybrid

**Inference-Only Long-Term and Short-Term Memory Management for LLM Agents**

---

## What This Is

This repository implements a hybrid memory management system for LLM agents, directly inspired by the architecture described in *Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents* (Yu et al., 2026, arXiv:2601.01885). It is written in Python and targets any OpenAI-compatible API endpoint.

The system is **not** a reproduction of AgeMem. It is a principled inference-only adaptation — a deliberate architectural response to the paper's own finding that the RL training is the source of its performance gains, not the tool interface alone. That finding is the starting point for this design.

---

## Background: What the Paper Actually Claims

The paper proposes AgeMem, a framework where an LLM agent autonomously manages both long-term memory (LTM) and short-term memory (STM) through a unified tool-based interface. The core contribution is a three-stage progressive reinforcement learning strategy with step-wise GRPO that teaches the model *when* to invoke memory tools by back-propagating final task rewards through intermediate memory decisions.

The paper includes an explicit ablation called **AgeMem-noRL**: the same tool interface, the same architecture, but without RL fine-tuning. This is the critical data point. AgeMem-noRL performs worse than most baselines on several benchmarks — it scores 8.87% on PDDL versus 13–18% for baselines, and 46.34% on BabyAI versus 58–60% for baselines. The RL training contributes 8.53 and 8.72 percentage point improvements over AgeMem-noRL.

**The tool design is necessary but not sufficient.** Any inference-only deployment of AgeMem is, by the paper's own definition, running AgeMem-noRL at best. This system acknowledges that constraint and compensates for it with a hybrid control architecture.

---

## Design Philosophy: The Hybrid Approach

Because a frozen, non-fine-tuned LLM cannot learn *when* to invoke memory tools from reward signals, this system distributes that responsibility across three layers with different guarantees:

**Layer 1 — System Rules (deterministic, no LLM).** A rule engine fires memory operations based on measurable thresholds: context token utilisation, turn count, and learning score magnitude. These rules are unconditional and cannot be overridden by the agent. They are the correctness floor of the system.

**Layer 2 — Memory Agent (LLM-based, dedicated sub-agent).** A separate LLM call is responsible for qualitative memory decisions: what content is worth storing in LTM, which context messages are low-relevance, whether a compression summary is warranted. This is the "agent-based LTM" pattern from Figure 1b of the paper, but applied to both memory tiers and triggered by the rule engine rather than on every turn.

**Layer 3 — Learning Score Feedback (agent self-assessment).** The main agent periodically rates its own turns on a 0–1 novelty scale. Scores above the promotion threshold trigger immediate LTM candidacy. Scores above the spike threshold bypass the periodic cadence entirely. This is an inference-only approximation of the reward signal that RL training would otherwise bake into the weights.

The three layers together cover the space that the paper's RL training covers in a single fine-tuned model: deterministic safety, qualitative judgment, and signal-driven prioritisation.

---

## Project Structure

```
agemem/
├── core/
│   ├── types.py          # All data contracts: MemoryEntry, ContextMessage,
│   │                     #   LearningFeedback, MemoryOpResult, ContextStats
│   └── config.py         # All tunable thresholds in one place
├── memory/
│   ├── ltm_store.py      # Persistent LTM store with overlap-based retrieval
│   └── stm_context.py    # Active context window with FILTER, SUMMARY, RETRIEVE
├── triggers/
│   └── system_rules.py   # Deterministic rule engine (R1–R4)
├── agents/
│   ├── llm_client.py     # Thin OpenAI-compatible wrapper
│   ├── memory_agent.py   # Dedicated sub-agent for qualitative memory decisions
│   ├── learning_scorer.py# Agent self-assessment feedback collector
│   └── orchestrator.py   # Central turn coordinator
├── tests/
│   └── test_agemem.py    # 28 offline unit tests, no LLM required
└── example_usage.py      # Wiring example for OpenAI / local models
```

---

## Features and Their AgeMem Correspondence

### Unified Tool Interface

The paper exposes six memory operations as agent-callable tools: ADD, UPDATE, DELETE (LTM), and RETRIEVE, SUMMARY, FILTER (STM). This system implements all six as first-class operations with typed return values (`MemoryOpResult`), a trigger provenance field (`TriggerKind`), and full audit logging per turn. The difference from the paper is that these operations are invoked by the control system rather than selected by a trained policy. The interface itself is faithful.

### Long-Term Memory Store (`ltm_store.py`)

Corresponds to the LTM management component in AgeMem. Implements ADD with near-duplicate detection (avoiding redundant storage, which the paper's penalty term `Ppenalty` discourages), UPDATE via exponential moving average of learning scores (approximating the paper's storage quality reward), DELETE, and keyword-overlap search for retrieval. The retrieval scoring combines token-overlap, recency decay, and accumulated learning score — a hand-crafted approximation of the semantic relevance component the paper trains via RL.

The paper measures Memory Quality (MQ) as LLM-judged relevance between stored entries and ground-truth facts. This system collects the same signal indirectly through `LearningFeedback.score`, which drives both what gets stored and how highly it is ranked for future retrieval.

### Short-Term Memory Context (`stm_context.py`)

Corresponds to the STM management component in AgeMem. The critical design decision is the **double overflow guard** in the orchestrator: `force_fit()` runs both before the user message is appended (pre-turn) and after the assistant response is appended (post-turn). This is necessary because a long assistant response can push a context from 70% utilisation to 105% in a single step. The paper's RL training prevents this implicitly by learning proactive summarisation. In the inference-only setting, the guard must be explicit and run at both boundaries.

The `STM_WARNING_THRESHOLD` (75%) and `STM_CRITICAL_THRESHOLD` (90%) map to the paper's concept of preventive actions and overflow penalties respectively. SUMMARY is triggered at warning; FILTER plus hard-drop is triggered at critical. Pinned messages (system prompt, injected LTM entries) are never evicted under any pressure level.

### System Rules (`system_rules.py`)

This layer has no direct equivalent in the paper. It is the structural contribution of the hybrid approach. The paper replaces heuristics with learned policy; this system acknowledges that the policy cannot be learned at inference time and instead formalises the heuristics as an explicit, auditable rule engine.

Rules:
- **R1** — SUMMARY recommendation at warning threshold
- **R2** — forced FILTER + SUMMARY at critical threshold  
- **R3** — periodic MemoryAgent review cycle every N turns
- **R4** — immediate LTM candidacy on learning score spike

Each rule carries a priority, a `RuleID`, and is independently testable without any LLM. The rule engine is the component that prevents the system from degrading to the naïve AgeMem-noRL baseline.

### Memory Agent (`memory_agent.py`)

Corresponds most closely to the "agent-based LTM" pattern (Figure 1b of the paper) extended to cover STM as well. The MemoryAgent is a dedicated LLM call with a structured output contract (JSON schema) that produces ADD/UPDATE/DELETE recommendations for LTM, per-message relevance scores for STM, and a summary trigger signal. It is invoked only when the system rules determine a review is warranted — not on every turn — to control inference cost.

The key limitation relative to the paper: the MemoryAgent relies on a crafted prompt to encode what AgeMem's RL training encodes in weights. It is calibrated, not learned. Its decisions are correct on the schema level (validated before application) but not on the quality level in the way a trained model would be.

### Learning Score Feedback (`learning_scorer.py`)

This is the acceptance-criterion mechanism that has no precise parallel in the paper. The paper uses task completion reward `Rtask` propagated backwards through trajectory steps. In inference, there is no trajectory and no reward signal. The learning score is an agent self-assessment: after every N turns, the main agent answers "how much new, reusable information did you just encounter?" on a 0–1 scale.

This score drives: promotion of content to LTM (score ≥ `LTM_PROMOTE_THRESHOLD`), update rather than add for near-duplicates (score ≥ `LTM_UPDATE_THRESHOLD`), and bypass of the periodic review cadence for high-novelty turns (score ≥ `LEARNING_SCORE_THRESHOLD_IMMEDIATE`). It also feeds into the MemoryAgent's review context, giving it a pre-labelled salience signal. The score is the closest inference-time proxy available for the paper's reward-weighted memory quality signal.

---

## What This Is Not

This system does not reproduce the paper's performance numbers and does not claim to. The paper's gains — 4.82 to 8.57 percentage points over the best baselines on five benchmarks — come from the three-stage RL training and step-wise GRPO. Those are training-time interventions. They require labelled trajectories, rollout groups, and fine-tuning compute. None of that is available at inference time.

The system also does not implement embedding-based retrieval. LTM search uses token-overlap scoring, which is sufficient for small stores (≤500 entries) and requires no external dependencies. For larger deployments, the `LTMStore.search()` method is the correct extension point for a vector database integration.

---

## Research Contribution

This implementation makes one novel engineering contribution that is not present in the paper and is not implied by it.

**The double-boundary overflow invariant.** The paper's RL training learns proactive compression behaviours that prevent the context from approaching overflow. In the inference-only setting, no such proactive behaviour can be assumed. A single `force_fit()` call at turn start is insufficient: an assistant response can itself be longer than the remaining token budget. The correct invariant is that the context must be within bounds at the *end* of every turn, not just the start. This requires `force_fit()` to be called after the assistant message is appended, not before the user message is appended. The test `T20` was written to catch this specific failure, and it did — catching a real bug in the first implementation where a 6-turn session produced a context at 106% utilisation despite the pre-turn guard being present.

This double-boundary pattern is a concrete and transferable finding: any inference-only memory system that guards context overflow only at turn ingress will fail on long assistant responses. The fix is a one-line addition, but the architectural insight — that the overflow invariant must be enforced at message-append boundaries, not at turn boundaries — has implications for all similar systems.

A secondary contribution is the formalisation of the rule engine as a pure, LLM-free, independently testable layer. In the AgeMem paper and in most prior work, the equivalent logic is embedded inside prompt text or hardcoded in the agent loop. Separating it into a typed rule engine with priority ordering and `TriggerKind` provenance tracking makes the system's behaviour auditable and its failure modes diagnosable without inspecting LLM outputs.

---

## Quickstart

```python
import openai
from core.config import AgememConfig
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator

client = openai.OpenAI(api_key="sk-...")
cfg = AgememConfig(DEFAULT_MODEL="gpt-4o-mini")
llm = LLMClient(client, default_model=cfg.DEFAULT_MODEL)
orch = Orchestrator(llm=llm, config=cfg)

response = orch.chat("My name is Alice and I'm building a Kafka pipeline.")
print(response)

# Inspect what happened
trace = orch.last_trace()
print(f"STM: {trace.stm_stats_after.utilisation_ratio:.0%} full")
print(f"Ops: {[op.op.value for op in trace.ops_applied if op.success]}")
print(f"LTM: {len(orch.ltm_snapshot())} entries stored")
```

For local models via Ollama:

```python
client = openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
```

---

## Running Tests

No network or API key required:

```bash
cd agemem
python -m unittest tests.test_agemem -v
```

All 28 tests cover: token counting, LTM add/update/delete/search/prune, STM filter/summary/retrieve/force_fit, system rule firing conditions, MemoryAgent JSON schema parsing, and full orchestrator turn lifecycle including overflow behaviour and LTM promotion via learning score.

---

## Configuration Reference

All thresholds are in `core/config.py` as a single `AgememConfig` dataclass. Key parameters:

| Parameter | Default | Effect |
|---|---|---|
| `STM_TOKEN_LIMIT` | 6000 | Hard context ceiling |
| `STM_WARNING_THRESHOLD` | 0.75 | SUMMARY fires above this |
| `STM_CRITICAL_THRESHOLD` | 0.90 | FILTER + SUMMARY forces above this |
| `LTM_PROMOTE_THRESHOLD` | 0.65 | Learning score above this → LTM ADD |
| `LEARNING_SCORE_PROMPT_EVERY_N` | 3 | Ask agent for feedback every N turns |
| `TRIGGER_EVERY_N_TURNS` | 10 | MemoryAgent full review cadence |
| `MEMORY_AGENT_MODEL` | gpt-4o-mini | Can differ from main agent model |

---

## Reference

Yu, Y., Yao, L., Xie, Y., Tan, Q., Feng, J., Li, Y., & Wu, L. (2026). *Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents*. arXiv:2601.01885.
