# Getting Started

## Prerequisites

- Python 3.10+
- Git
- (Optional) CUDA-capable GPU for local inference

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/gianpd/agemem.git
cd agemem

# Install dependencies
uv sync

# Or with document ingestion support
uv sync --extra ingest
```

### Using pip

```bash
pip install -e .

# With document ingestion support
pip install -e ".[ingest]"
```

## Quick Start

### Basic Usage

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

### Local Models via Ollama

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

## Interactive REPL

Run the interactive chat interface:

```bash
uv run main.py
```

### REPL Commands

| Command | Effect |
|---------|--------|
| `/clear` | Reset STM (LTM retained) |
| `/memory` | Show LTM snapshot |
| `/stats` | Show STM statistics |
| `/forget` | Wipe LTM (requires confirmation) |
| `/help` | Show help |

### Keybindings

| Key | Action |
|-----|--------|
| Enter | Send message |
| Alt+Enter | Insert newline |
| Escape+Enter | Insert newline (alternative) |
| Ctrl+C | Cancel input or exit |

## Environment Variables

### Core Configuration

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
```

### Remote Providers

```bash
# OpenRouter example
BASE_URL=https://openrouter.ai/api/v1
BASE_MODEL=anthropic/claude-sonnet-4
API_KEY=sk-or-...

# OpenAI example
BASE_URL=https://api.openai.com
BASE_MODEL=gpt-4o-mini
API_KEY=sk-...
```

### Optional Features

```bash
# Web search
WEB_SEARCH_MAX_RESULTS=5
TOOL_RESULT_MAX_CHARS=4000

# Semantic search
ENABLE_SEMANTIC_SEARCH=true
SEMANTIC_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

# Debug/tracing
DEBUG_MODE=false
TRACE_LOG_DIR=logs
TRACE_RETENTION_DAYS=30
```

### Backward Compatibility

Old `LLAMA_*` variable names are supported but deprecated:

| Deprecated | Use Instead |
|------------|-------------|
| `LLAMA_HOST` | `BASE_URL` |
| `LLAMA_MODEL` | `BASE_MODEL` |
| `LLAMA_MAX_TOKENS` | `BASE_MAX_TOKENS` |
| `LLAMA_TEMPERATURE` | `BASE_TEMPERATURE` |

## Running Tests

All tests are offline — no LLM calls, no network required:

```bash
# Run all tests
python -m unittest tests.test_agemem -v

# Run specific test file
python -m unittest tests.test_query_expansion -v

# Run with coverage
python -m pytest tests/ --cov=.
```

### Key Test Cases

| Test | Validates |
|------|-----------|
| T20 | Double-boundary overflow invariant |
| T19 | LTM promotion via learning score |
| T13-T15 | System rule firing conditions |
| T07-T08 | FILTER respects pinned messages |

## Document Ingestion

The `ingest/` module provides PDF-to-markdown conversion with configurable NER extraction:

```bash
# Requires: uv pip install -e ".[ingest]"
uv run ingest/ingest.py report.pdf [doc_type]

# Built-in label sets
uv run ingest/ingest.py paper.pdf research --labels research
uv run ingest/ingest.py contract.pdf legal --labels legal
uv run ingest/ingest.py gara.pdf bando --labels edilizia
```

### Custom Label Configuration

Create a YAML file (see `ingest/gliner_config.yaml`):

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

Then reference it: `--labels /path/to/config.yaml:my_domain`

## Next Steps

- Read [02-Architecture.md](02-Architecture.md) to understand the system design
- Read [04-Memory-System.md](04-Memory-System.md) for LTM/STM mechanics
- Check [core/config.py](../../core/config.py) for all configuration options