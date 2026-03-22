# AgeMem Evaluation Pipeline

A simplified evaluation framework for testing the AgeMem memory system against the LongMemEval benchmark. The pipeline validates retrieval quality and memory lifecycle behavior through the production `orchestrator.chat()` codepath.

## Overview

The evaluation suite tests two primary aspects of the memory system:

1. **Retrieval Quality** - How effectively the system retrieves relevant memories from LTM
2. **Memory Lifecycle** - How the system handles session replay and memory operations through the orchestrator

## Architecture

```
evaluation/
├── cli.py                  # Single CLI entry point (--batch-size for checkpointing)
├── runner.py               # BatchRunner orchestration class
├── loader.py               # DatasetLoader for unified dataset loading
├── evaluator.py            # Evaluator class (query/session evaluation)
├── checkpoint.py           # CheckpointManager for resumable evaluations
├── report.py               # ReportGenerator for markdown/JSON output
├── metrics.py              # Metrics calculation (MRR@K, Recall@K, etc.)
├── factory.py              # Orchestrator factory for evaluation
├── llm_judge.py            # LLM-as-Judge client for answer validation
├── mock_llm.py             # Stateful mock LLM for deterministic testing
├── quick_test.py           # Minimal 3-query test for quick validation
├── __init__.py             # Package exports
│
├── batch_runner.py         # [DEPRECATED] Use runner.py
├── run.py                  # [DEPRECATED] Use cli.py
├── run_batch.py            # [DEPRECATED] Use cli.py
├── evaluators.py           # [DEPRECATED] Use evaluator.py
│
├── data/                   # Benchmark datasets
│   ├── longmemeval_s_cleaned.json   # Small dataset (~1M entries)
│   ├── longmemeval_m_cleaned.json   # Medium dataset (~10M entries)
│   └── longmemeval_oracle.json      # Oracle answers
├── results/                # Evaluation outputs
├── logs/                   # Execution logs
└── archive/                # Deprecated evaluation scripts
```

## Core Components

### 1. CLI (`cli.py`)

Single entry point for all evaluation operations.

```bash
# Run new evaluation (no checkpointing)
python -m evaluation.cli --dataset evaluation/data/longmemeval_s_cleaned.json

# Run with batch checkpointing (resumable)
python -m evaluation.cli --dataset evaluation/data/longmemeval_s_cleaned.json --batch-size 10

# Resume from checkpoint
python -m evaluation.cli --resume <session_id>

# Generate partial report
python -m evaluation.cli --report <session_id>

# List checkpoints
python -m evaluation.cli --list-checkpoints
```

### 2. BatchRunner (`runner.py`)

Orchestration class that coordinates evaluation runs.

```python
from evaluation.runner import BatchRunner, BatchConfig
from evaluation.factory import OrchestratorFactory

factory = OrchestratorFactory()
config = BatchConfig(batch_size=10, checkpoint_interval=5)

runner = BatchRunner(config, factory)
summary = runner.run(
    dataset_path=Path("evaluation/data/longmemeval_s_cleaned.json"),
    mode="full",
    max_interactions=100
)
```

### 3. DatasetLoader (`loader.py`)

Unified dataset loading with support for queries and sessions.

```python
from evaluation.loader import DatasetLoader

loader = DatasetLoader()
entries, queries, raw_data = loader.load(
    path="evaluation/data/longmemeval_s_cleaned.json",
    subset="all",      # or "queries_only"
    limit=10
)

# Drop-in replacement for old run.load_dataset()
from evaluation.loader import load_dataset
entries, queries, raw_data = load_dataset(dataset_path, query_limit=10)
```

### 4. Evaluator (`evaluator.py`)

Main evaluation class for query and session evaluation.

```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator(orchestrator, llm_judge=None, use_llm_judge=True)

# Evaluate a single query
result = evaluator.evaluate_query(query, instance)

# Evaluate multiple questions
results = evaluator.evaluate_questions(queries, raw_data)

# Replay sessions
session_results = evaluator.replay_sessions(sessions, behavior_type="IE")
```

**Key Classes:**
- `Evaluator` - Main evaluation orchestrator
- `SessionReplayResult` - Result of replaying a session
- `QuestionResult` - Result of evaluating a single question
- `EvaluationContext` - Context for question evaluation

### 5. CheckpointManager (`checkpoint.py`)

Manages evaluation state for resumable runs.

```python
from evaluation.checkpoint import CheckpointManager, CheckpointState

manager = CheckpointManager(checkpoint_dir=Path("evaluation/checkpoints"))

# Save checkpoint
manager.save(state)

# Load checkpoint
state = manager.load(session_id)

# List available checkpoints
checkpoints = manager.list_checkpoints()
```

### 6. ReportGenerator (`report.py`)

Generates markdown and JSON reports for evaluations.

```python
from evaluation.report import ReportGenerator

reporter = ReportGenerator(output_dir=Path("evaluation/results"))

# Full report
md_path = reporter.generate_markdown(summary, session_id)
json_path = reporter.generate_json(summary, session_id)

# Partial report (for in-progress evaluations)
md_path = reporter.generate_partial_markdown(summary, session_id, checkpoint)
```

### 7. Metrics (`metrics.py`)

Calculates standard information retrieval metrics.

```python
from evaluation.metrics import calculate_metrics, EvaluationSummary

summary = calculate_metrics(queries, question_results, session_results)
```

**Metrics Calculated:**
- **MRR@K** (Mean Reciprocal Rank) - Rank of first relevant result
- **Recall@K** - Fraction of relevant items found in top K
- **Precision@K** - Fraction of top K results that are relevant
- **NDCG@K** - Normalized Discounted Cumulative Gain

### 8. LLM-as-Judge (`llm_judge.py`)

Uses an LLM to validate answers when exact matching fails.

```python
from evaluation.llm_judge import LLMJudge

judge = LLMJudge(model="claude-3-5-sonnet")
is_correct, confidence = judge.evaluate(
    question="What is my phone number?",
    predicted="555-1234",
    expected="555-1234"
)
```

### 9. Orchestrator Factory (`factory.py`)

Creates isolated Orchestrator instances for evaluation.

```python
from evaluation.factory import OrchestratorFactory

factory = OrchestratorFactory()
orchestrator = factory.build_for_evaluation(
    llm_client=mock_llm,
    persist_dir=Path("/tmp/eval"),
    config_overrides={"STM_TOKEN_LIMIT": 8000}
)
```

## Usage

### Quick Test

Run a minimal 3-query test without full orchestrator initialization:

```bash
python evaluation/quick_test.py
```

### Full Evaluation

```bash
# Simple evaluation (no checkpointing)
python -m evaluation.cli --dataset evaluation/data/longmemeval_s_cleaned.json

# With batch checkpointing (resumable)
python -m evaluation.cli \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --batch-size 10 \
    --max-interactions 100

# Resume interrupted evaluation
python -m evaluation.cli --resume eval_20260322_143052

# Generate partial report for running evaluation
python -m evaluation.cli --report eval_20260322_143052
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Path to benchmark dataset | - |
| `--mode` | Evaluation mode: `full`, `lifecycle`, `retrieval` | `full` |
| `--batch-size` | Interactions per batch (0 = no checkpointing) | `0` |
| `--max-interactions` | Max queries to evaluate (0 = all) | `0` |
| `--max-batches` | Max batches to process (0 = unlimited) | `0` |
| `--session-id` | Custom session ID | auto-generated |
| `--output-dir` | Output directory for results | `evaluation/results` |
| `--checkpoint-interval` | Save checkpoint every N batches | `5` |
| `--resume` | Resume from checkpoint by session ID | - |
| `--report` | Generate partial report for session | - |
| `--list-checkpoints` | List available checkpoints | - |
| `--cleanup` | Remove checkpoint files for session | - |
| `--no-resume` | Start fresh (ignore existing checkpoint) | `false` |
| `-v, --verbose` | Enable debug logging | `false` |

### Batch Mode vs Simple Mode

**Simple mode** (`--batch-size 0`):
- No checkpointing
- Faster for small evaluations
- Cannot resume if interrupted

**Batch mode** (`--batch-size N`):
- Checkpoints saved every N interactions
- Can resume if interrupted
- Creates fresh orchestrator per batch (prevents STM accumulation)
- Generates partial reports during execution

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

1. **`{session_id}_report.md`** - Human-readable markdown report
2. **`{session_id}_metrics.json`** - Machine-readable JSON metrics

For partial evaluations: **`{session_id}_partial_report.md`**

### Report Structure

```markdown
# AgeMem Evaluation Report

**Session ID:** eval_20260322_143052
**Evaluated at:** 2026-03-22T14:30:52

## Summary

| Metric | Value |
|--------|-------|
| Total Queries | 10 |
| Correct | 7 |
| Accuracy | 70.00% |
| Abstained | 1 |
| Avg Latency | 1250.3ms |

## LLM-as-Judge Statistics

| Metric | Value |
|--------|-------|
| Judge Calls | 5 |
| Judge Correct | 4 |
| Heuristic Fallback | 1 |

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| mrr_at_10 | 0.7500 |
| recall_at_5 | 0.8500 |

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

## Migration from Deprecated Modules

If you have code importing from the old modules:

```python
# OLD (deprecated)
from evaluation.batch_runner import BatchEvaluationRunner
from evaluation.run import load_dataset, generate_report
from evaluation.evaluators import Evaluator

# NEW
from evaluation.runner import BatchRunner
from evaluation.loader import load_dataset, DatasetLoader
from evaluation.report import ReportGenerator
from evaluation.evaluator import Evaluator
```

The deprecated modules emit `DeprecationWarning` when imported but continue to work through re-exports.

## Development

### Running Tests

```bash
# Quick sanity check
python evaluation/quick_test.py

# Small evaluation
python -m evaluation.cli \
    --dataset evaluation/data/longmemeval_s_cleaned.json \
    --max-interactions 3 \
    -v
```

### Adding New Behavior Types

Update `BEHAVIOR_MAP` in `evaluator.py`:

```python
class Evaluator:
    BEHAVIOR_MAP = {
        "single-session-user": "IE",
        "multi-session": "MR",
        "knowledge-update": "KU",
        "time-weighted": "TR",
        "abstention": "ABS",
        "new_behavior_type": "NB",
    }
```

## Target Metrics

| Metric | Target | Status |
|--------|--------|--------|
| MRR@10 | >= 0.85 | In Progress |
| Recall@5 | >= 0.90 | In Progress |
| Avg Latency | < 500ms | Validated |

## Version

Current version: **3.0.0** (refactored architecture)

**Changelog:**
- v3.0.0 - Refactored into modular architecture (loader, runner, evaluator, checkpoint, report, cli)
- v2.0.0 - Simplified to test through production orchestrator
- v1.0.0 - Original complex pipeline (archived)