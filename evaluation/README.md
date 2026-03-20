# AgeMem Evaluation Pipeline

A simplified evaluation framework for testing the AgeMem memory system against the LongMemEval benchmark. The pipeline validates retrieval quality and memory lifecycle behavior through the production `orchestrator.chat()` codepath.

## Overview

The evaluation suite tests two primary aspects of the memory system:

1. **Retrieval Quality** - How effectively the system retrieves relevant memories from LTM
2. **Memory Lifecycle** - How the system handles session replay and memory operations through the orchestrator

## Architecture

```
evaluation/
├── run.py                  # Main CLI entry point
├── quick_test.py           # Minimal 3-query test for quick validation
├── evaluators.py           # Core evaluation logic (Evaluator class)
├── metrics.py              # Metrics calculation (MRR@K, Recall@K, etc.)
├── factory.py              # Orchestrator factory for evaluation
├── mock_llm.py             # Stateful mock LLM for deterministic testing
├── __init__.py             # Package exports
├── data/                   # Benchmark datasets
│   ├── longmemeval_s_cleaned.json   # Small dataset (~1M entries)
│   ├── longmemeval_m_cleaned.json   # Medium dataset (~10M entries)
│   └── longmemeval_oracle.json      # Oracle answers
├── results/                # Evaluation outputs
├── logs/                   # Execution logs
├── archive/                # Deprecated evaluation scripts
│   ├── question_evaluator.py   # (replaced by evaluators.py)
│   ├── session_replay.py       # (replaced by evaluators.py)
│   ├── trace_capture.py        # (deprecated)
│   ├── inference_pipeline.py   # (deprecated)
│   ├── metrics_pipeline.py     # (deprecated)
│   ├── dataset_pipeline.py     # (deprecated)
│   ├── phase2_pipeline.py      # (deprecated)
│   ├── end_to_end_runner.py    # (deprecated)
│   ├── report_generator.py     # (deprecated)
│   └── simulation.py           # (deprecated)
└── docs/                   # Documentation
    ├── eval_status.md              # Current coherence analysis
    ├── evaluation_pipeline_audit_report.md
    ├── agemem_technical_specification.md
    ├── agemem_evaluation_suite_trs.md
    ├── longmemeval_guide.md
    └── eval_pipe_progress.md
```

## Core Components

### 1. Evaluator (`evaluators.py`)

The main evaluation class that combines session replay and question evaluation.

```python
from evaluation.evaluators import Evaluator

# Create evaluator with orchestrator instance
evaluator = Evaluator(orchestrator)

# Replay sessions
session_results = evaluator.replay_sessions(sessions, behavior_type="IE")

# Evaluate questions
question_results = evaluator.evaluate_questions(queries, raw_data)
```

**Key Classes:**
- `Evaluator` - Main evaluation orchestrator
- `SessionReplayResult` - Result of replaying a session
- `QuestionResult` - Result of evaluating a single question
- `EvaluationContext` - Context for question evaluation (behavior type, expected answer)

### 2. Metrics (`metrics.py`)

Calculates standard information retrieval metrics.

```python
from evaluation.metrics import calculate_metrics, EvaluationSummary

# Calculate all metrics
summary = calculate_metrics(queries, question_results, session_results)
```

**Key Classes:**
- `RetrievalMetrics` - MRR@K, Recall@K, Precision@K, NDCG@K
- `BehaviorMetrics` - Accuracy breakdown by behavior type (IE, MR, KU, TR, ABS)
- `EvaluationSummary` - Complete evaluation results

**Metrics Calculated:**
- **MRR@K** (Mean Reciprocal Rank) - Rank of first relevant result
- **Recall@K** - Fraction of relevant items found in top K
- **Precision@K** - Fraction of top K results that are relevant
- **NDCG@K** - Normalized Discounted Cumulative Gain

### 3. Orchestrator Factory (`factory.py`)

Creates isolated Orchestrator instances for evaluation.

```python
from evaluation.factory import OrchestratorFactory

factory = OrchestratorFactory()
orchestrator = factory.build_for_evaluation(
    llm_client=mock_llm,  # Optional: for mock testing
    persist_dir=Path("/tmp/eval"),
    config_overrides={
        "STM_TOKEN_LIMIT": 8000,
        "LTM_PROMOTE_THRESHOLD": 0.5,
    },
)
```

### 4. Mock LLM (`mock_llm.py`)

Stateful mock LLM for deterministic testing. Supports three strategies:

- **template** (default) - Pattern-match queries to responses
- **record_replay** - Replay pre-recorded responses
- **state_machine** - Track conversation state for contextual responses

```python
from evaluation.mock_llm import StatefulMockLLM

mock = StatefulMockLLM(strategy="template")
mock.add_response_template("phone", "Your phone number is 555-1234")
mock.add_response_template("email", "Your email is user@example.com")
```

## Usage

### Quick Test

Run a minimal 3-query test without full orchestrator initialization:

```bash
python evaluation/quick_test.py
```

### Full Evaluation

Run a complete evaluation with the main CLI:

```bash
# Full evaluation (session replay + question evaluation)
python evaluation/run.py \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --mode full \
    --queries 10

# Session replay only (test memory lifecycle)
python evaluation/run.py \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --mode lifecycle \
    --sessions 5

# Question evaluation only (test retrieval quality)
python evaluation/run.py \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --mode retrieval \
    --queries 20

# Use mock LLM for deterministic testing
python evaluation/run.py \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --mode full \
    --queries 5 \
    --mock
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Path to benchmark dataset (required) | - |
| `--mode` | Evaluation mode: `full`, `lifecycle`, or `retrieval` | `full` |
| `--queries` | Number of queries to evaluate (0 = all) | `0` |
| `--sessions` | Max sessions to replay in lifecycle mode (0 = all) | `0` |
| `--output-dir` | Output directory for results | `evaluation/results` |
| `--persist-session` | Keep session data after evaluation | `false` |
| `--mock` | Use mock LLM instead of real LLM | `false` |
| `--verbose` / `-v` | Enable debug logging | `false` |

## Behavior Types

The evaluation tests five memory behavior categories from LongMemEval:

| Behavior | Code | Description |
|----------|------|-------------|
| Information Extraction | IE | Single-session detail retrieval |
| Multi-Session Reasoning | MR | Synthesis across multiple sessions |
| Knowledge Updates | KU | Most recent value tracking |
| Temporal Reasoning | TR | Time-aware queries |
| Abstention | ABS | Correct "I don't know" when info missing |

## Output

Evaluation produces two output files:

1. **`eval_YYYYMMDD_HHMMSS_report.md`** - Human-readable markdown report
2. **`eval_YYYYMMDD_HHMMSS_metrics.json`** - Machine-readable JSON metrics

### Report Structure

```markdown
# AgeMem Evaluation Report

**Session ID:** eval_20260319_224249
**Evaluated at:** 2026-03-19T22:43:49

## Summary

| Metric | Value |
|--------|-------|
| Total Queries | 10 |
| Correct | 7 |
| Accuracy | 70.00% |
| Abstained | 1 |
| Avg Latency | 1250.3ms |

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| mrr_at_10 | 0.7500 |
| recall_at_5 | 0.8500 |
| ... | ... |

## Behavior Breakdown

| Behavior | Count | Accuracy |
|----------|-------|----------|
| IE | 5 | 80.00% |
| MR | 3 | 66.67% |
| KU | 2 | 100.00% |

## Session Replay

- Total Sessions: 5
- Total Turns: 47
- LTM Entries Added: 12
- Avg STM Tokens: 3204
```

## Dataset Format

The evaluation uses LongMemEval format JSON files:

```json
[
  {
    "question_id": "e47becba",
    "question_type": "single-session-user",
    "question": "What degree did I graduate with?",
    "question_date": "2023/05/30 (Tue) 23:40",
    "answer": "Business Administration",
    "answer_session_ids": ["answer_280352e9"],
    "haystack_dates": ["2023/05/20 (Sat) 02:21", ...],
    "haystack_session_ids": ["session_1", ...],
    "haystack_sessions": [
      [{"role": "user", "content": "..."}, ...],
      ...
    ]
  }
]
```

## Implementation Notes

### Current Architecture

The simplified evaluation pipeline tests the **production orchestrator codepath**:

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  agents/orchestrator.py            │
│  └── chat() method                  │
│      ├── STM overflow guard         │
│      ├── Context-aware retrieval    │
│      ├── LTM semantic search        │
│      └── Memory trigger processing  │
└─────────────────────────────────────┘
```

### Previous Architecture (Deprecated)

The `archive/` directory contains the previous complex pipeline that:
- Bypassed the orchestrator for direct LTMStore testing
- Used separate inference, metrics, and dataset pipelines
- Did not validate end-to-end memory lifecycle

This was replaced with the simplified `Evaluator` class that routes all testing through `orchestrator.chat()` for more realistic results.

## Target Metrics

Per the technical specification:

| Metric | Target | Status |
|--------|--------|--------|
| MRR@10 | >= 0.85 | In Progress |
| Recall@5 | >= 0.90 | In Progress |
| Avg Latency | < 500ms | Validated |

## Development

### Running Tests

```bash
# Quick sanity check
python evaluation/quick_test.py

# Small evaluation with mock LLM
python evaluation/run.py \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --queries 3 \
    --mock \
    --verbose
```

### Adding New Behavior Types

To add support for new behavior types:

1. Update `_map_behavior()` in `evaluators.py`:
```python
@staticmethod
def _map_behavior(question_type: str) -> str:
    mapping = {
        # ... existing mappings ...
        "new_behavior_type": "NB",
    }
```

2. Add validation logic in `_validate()` if needed.

### Using Custom Datasets

The evaluation pipeline supports any JSON file following the LongMemEval format. Create your own dataset:

```python
import json

dataset = [
    {
        "question_id": "custom_001",
        "question_type": "single-session-user",
        "question": "What is my favorite color?",
        "answer": "Blue",
        "haystack_sessions": [[
            {"role": "user", "content": "My favorite color is blue"}
        ]],
    }
]

with open("my_dataset.json", "w") as f:
    json.dump(dataset, f)
```

Then run:
```bash
python evaluation/run.py --dataset my_dataset.json --queries 1
```

## Documentation

- `docs/eval_status.md` - Coherence analysis between evaluation and production
- `docs/evaluation_pipeline_audit_report.md` - Detailed audit findings
- `docs/agemem_technical_specification.md` - Technical requirements
- `docs/longmemeval_guide.md` - LongMemEval benchmark guide

## Version

Current version: **2.0.0** (simplified architecture)

Previous versions used the complex pipeline in `archive/` which has been deprecated in favor of the simplified `Evaluator` class that tests through the production orchestrator.
