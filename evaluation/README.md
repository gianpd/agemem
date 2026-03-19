# AgeMem Evaluation Pipeline

A comprehensive evaluation framework for testing the AgeMem memory system against the LongMemEval benchmark. The pipeline validates both retrieval quality and end-to-end memory lifecycle behavior through the production `orchestrator.chat()` codepath.

## What It Evaluates

### Phase 1: Retrieval Quality

Tests how effectively the system retrieves relevant memories from the Long-Term Memory (LTM) store.

| Metric | Description | Target |
|--------|-------------|--------|
| **MRR@K** | Mean Reciprocal Rank at K - measures rank of first relevant result | >= 0.85 @10 |
| **Recall@K** | Fraction of relevant items found in top K | >= 0.90 @5 |
| **Precision@K** | Fraction of top K results that are relevant | - |
| **NDCG@K** | Normalized Discounted Cumulative Gain - ranking quality | - |

**Implementation:** [`evaluation/pipeline/metrics_pipeline.py:207-350`](evaluation/pipeline/metrics_pipeline.py#L207-L350)

### Phase 2: Memory Lifecycle

Tests the complete memory behaviors users experience in production:

| Test Area | What It Validates |
|-----------|-------------------|
| **Memory Operations** | ADD/UPDATE/DELETE trigger accuracy |
| **Learning Scores** | Score evolution and LTM promotion |
| **Context-Aware Retrieval** | Retrieval with conversation context vs baseline |
| **Query Expansion** | Recall improvement from query variants |

**Implementation:** [`evaluation/pipeline/phase2_pipeline.py:175-418`](evaluation/pipeline/phase2_pipeline.py#L175-L418)

### Behavior Testing (LongMemEval Alignment)

Tests all 5 LongMemEval memory behavior categories:

| Behavior | Code | Description |
|----------|------|-------------|
| **Information Extraction** | IE | Single-session detail retrieval |
| **Multi-Session Reasoning** | MR | Synthesis across 30-40 sessions |
| **Knowledge Updates** | KU | Most recent value tracking |
| **Temporal Reasoning** | TR | Time-aware queries |
| **Abstention** | ABS | Correct "I don't know" when info missing |

**Implementation:** [`evaluation/pipeline/metrics_pipeline.py:378-455`](evaluation/pipeline/metrics_pipeline.py#L378-L455)

---

## Codebase Components Tested

### Core Production Codepath

The evaluation tests the **exact production codepath** users experience through `orchestrator.chat()`:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  agents/orchestrator.py:Orchestrator.chat()                 │  ← Primary entry point
│                                                             │
│  ├── memory/stm_context.py:STMContext                       │  ← STM overflow guard
│  ├── memory/context_retrieval.py:ContextAwareRetriever      │  ← Context-aware retrieval
│  ├── memory/ltm_store.py:LTMStore.search()                  │  ← Semantic search
│  ├── triggers/memory_trigger_engine.py:MemoryTriggerEngine  │  ← Post-turn triggers
│  └── agents/learning_scorer.py:LearningScorer              │  ← Learning feedback
└─────────────────────────────────────────────────────────────┘
```

**Implementation:** [`evaluation/orchestrator_test_harness.py:455-562`](evaluation/orchestrator_test_harness.py#L455-L562)

### Component Coverage Matrix

| Component | File | Phase 1 | Phase 2 | Orchestrator |
|-----------|------|---------|---------|--------------|
| LTMStore.search() | `memory/ltm_store.py` | ✅ | ✅ | ✅ |
| STMContext overflow guard | `memory/stm_context.py` | - | - | ✅ |
| ContextAwareRetriever | `memory/context_retrieval.py` | - | ✅ | ✅ |
| QueryExpander | `tools/query_expansion.py` | - | ✅ | ✅ |
| MemoryTriggerEngine | `triggers/memory_trigger_engine.py` | - | ✅ | ✅ |
| LearningScorer | `agents/learning_scorer.py` | - | ✅ | ✅ |
| Orchestrator.chat() | `agents/orchestrator.py` | - | - | ✅ |
| Corpus fallback | `agents/orchestrator.py` | - | - | ✅ |
| Skill injection | `agents/orchestrator.py` | - | - | ✅ |

---

## Metrics Collected

### Retrieval Metrics

Defined in [`evaluation/pipeline/metrics_pipeline.py:32-54`](evaluation/pipeline/metrics_pipeline.py#L32-L54):

```python
@dataclass
class RetrievalMetrics:
    mrr_at_1: float      # Mean Reciprocal Rank @1
    mrr_at_5: float      # Mean Reciprocal Rank @5
    mrr_at_10: float     # Mean Reciprocal Rank @10
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_5: float   # Target: >= 0.90
    recall_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    avg_latency_ms: float
```

### Behavior-Segmented Metrics

Defined in [`evaluation/pipeline/metrics_pipeline.py:148-169`](evaluation/pipeline/metrics_pipeline.py#L148-L169):

```python
@dataclass
class BehaviorMetrics:
    behavior_name: str      # IE, MR, KU, TR, ABS
    query_count: int
    mrr_at_10: float
    recall_at_5: float
    precision_at_5: float
    ndcg_at_10: float
    avg_latency_ms: float
```

### Phase 2 Lifecycle Metrics

Defined in [`evaluation/pipeline/phase2_pipeline.py:47-150`](evaluation/pipeline/phase2_pipeline.py#L47-L150):

| Metric Class | Key Fields |
|--------------|------------|
| `MemoryOperationMetrics` | add_operations_total, add_precision, add_recall, update_*, delete_* |
| `LearningScoreMetrics` | scores_measured, avg_score, promotion_recall |
| `ContextAwareRetrievalMetrics` | baseline_mrr, context_aware_mrr, mrr_improvement |
| `QueryExpansionMetrics` | baseline_recall, expanded_recall, variant_hit_rate |

### Orchestrator Test Metrics

Defined in [`evaluation/pipeline/phase2_pipeline.py:782-834`](evaluation/pipeline/phase2_pipeline.py#L782-L834):

```python
@dataclass
class OrchestratorTestMetrics:
    # Retrieval through orchestrator
    retrieval_via_orchestrator_count: int
    stm_overflow_guard_triggered: int
    ltm_retrieval_triggered: int
    corpus_fallback_used: int

    # Memory lifecycle
    post_turn_triggers_fired: int
    learning_feedback_collected: int

    # Behavior-specific results
    ie_correct: int  # Information Extraction
    mr_correct: int  # Multi-Session Reasoning
    ku_correct: int  # Knowledge Updates
    tr_correct: int  # Temporal Reasoning
    abs_correct: int # Abstention
```

---

## How to Run

### Prerequisites

1. Dataset: LongMemEval S format (`evaluation/data/longmemeval_s_cleaned.json`)
2. Python 3.10+
3. Dependencies: `pip install -r requirements.txt`

### Quick Start

Run a quick evaluation with 10 queries:

```bash
python evaluation/run.py \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --mode full \
    --queries 10
```

### Evaluation Modes

| Mode | Description | Command |
|------|-------------|---------|
| `retrieval` | Phase 1: Retrieval quality only | `--mode retrieval` |
| `lifecycle` | Phase 2: Memory lifecycle testing | `--mode lifecycle` |
| `full` | Complete Phase 1 + Phase 2 | `--mode full` (default) |

### Reproducible Evaluation

For reproducible runs with full manifest logging:

```bash
python evaluation/reproducible_runner.py \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --queries 50 \
    --seed 42 \
    --output-dir evaluation/results
```

Outputs:
- `{run_id}_manifest.json` - Full reproducibility manifest
- `{run_id}_metrics.json` - All metrics
- `{run_id}_report.md` - Human-readable report

**Implementation:** [`evaluation/reproducible_runner.py:141-313`](evaluation/reproducible_runner.py#L141-L313)

### End-to-End Evaluation

Run comprehensive Phase 1 + Phase 2 with orchestrator-based tests:

```bash
python -m evaluation.pipeline.end_to_end_runner \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --queries 20 \
    --output-dir evaluation/results \
    --use-orchestrator
```

**Implementation:** [`evaluation/pipeline/end_to_end_runner.py:496-605`](evaluation/pipeline/end_to_end_runner.py#L496-L605)

### Full Dataset Evaluation

Evaluate against all queries:

```bash
python evaluation/run.py \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --mode full \
    --output-dir evaluation/results
```

### Command Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Path to benchmark dataset | Required |
| `--mode` | Evaluation mode (retrieval/lifecycle/full) | `full` |
| `--queries` | Number of queries (0 = all) | `0` |
| `--output-dir` | Output directory | `evaluation/results` |
| `--seed` | Random seed for reproducibility | `42` |
| `--persist-session` | Keep session data for debugging | `false` |
| `--verbose` / `-v` | Enable debug logging | `false` |

### Orchestrator-Specific Options

| Flag | Description |
|------|-------------|
| `--use-orchestrator` | Enable orchestrator-based Phase 2 tests (default) |
| `--no-orchestrator` | Use traditional component-level tests only |
| `--disable-query-expansion` | Skip query expansion testing |
| `--disable-context-aware` | Skip context-aware retrieval testing |

---

## Output Files

After running, the output directory contains:

| File | Description |
|------|-------------|
| `eval_*_report.md` | Human-readable evaluation report |
| `eval_*_metrics.json` | Machine-readable metrics |
| `phase2_*.md` | Phase 2 lifecycle report (if applicable) |
| `end_to_end_report.md` | Combined Phase 1 + Phase 2 report |
| `traces.db` | SQLite database of retrieval traces |

---

## Architecture

```
evaluation/
├── run.py                          # Main CLI entry point
├── reproducible_runner.py          # Reproducible evaluation with manifest
├── question_evaluator.py           # Question evaluation through orchestrator
├── session_replay.py               # Session replay through orchestrator
├── orchestrator_test_harness.py    # Orchestrator-based test infrastructure
├── mock_llm.py                     # Mock LLM for deterministic testing
├── factory.py                      # Orchestrator factory for evaluation
├── trace_capture.py                # Turn trace capture
├── pipeline/
│   ├── dataset_pipeline.py         # Dataset loading (LongMemEval format)
│   ├── inference_pipeline.py       # Query execution
│   ├── metrics_pipeline.py         # Metric calculation
│   ├── phase2_pipeline.py          # Phase 2 lifecycle testing
│   ├── end_to_end_runner.py        # Combined Phase 1 + Phase 2 runner
│   └── report_generator.py         # Report generation
└── docs/
    ├── longmemeval_guide.md        # LongMemEval benchmark guide
    └── eval_status.md              # Current evaluation status
```

---

## Reproducibility

The evaluation pipeline implements full reproducibility per the audit recommendations:

1. **Seed Control**: All random seeds (Python, NumPy, PyTorch) are set
2. **Dataset Hashing**: SHA256 hash of dataset recorded
3. **Environment Capture**: Python version, platform, git commit logged
4. **Package Versions**: Key dependency versions recorded
5. **Deterministic LLM**: Mock LLM for reproducible responses

**Implementation:** [`evaluation/reproducible_runner.py:37-81`](evaluation/reproducible_runner.py#L37-L81)

---

## Target Metrics

Per the technical specification:

| Metric | Target | Status |
|--------|--------|--------|
| MRR@10 | >= 0.85 | Validated |
| Recall@5 | >= 0.90 | Validated |
| Avg Latency | < 500ms | Validated |

Check current results in [`evaluation/docs/eval_status.md`](evaluation/docs/eval_status.md).