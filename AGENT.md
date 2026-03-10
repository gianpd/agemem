# CLAUDE.md — AgeMem-Hybrid

## Project Overview

AgeMem-Hybrid is an inference-only memory management system for LLM agents, implementing unified long-term memory (LTM) and short-term memory (STM) management. It is a principled adaptation of the AgeMem paper (Yu et al., 2026) that compensates for the lack of RL training through a three-layer hybrid control architecture.

## Architecture

```
agemem/
├── core/
│   ├── types.py          # Data contracts: MemoryEntry, ContextMessage, etc.
│   └── config.py         # All tunable thresholds (AgememConfig)
├── memory/
│   ├── ltm_store.py      # Persistent LTM with overlap-based retrieval
│   └── stm_context.py    # Active context window management
├── triggers/
│   └── system_rules.py   # Deterministic rule engine (R1-R4)
├── agents/
│   ├── llm_client.py     # OpenAI-compatible wrapper
│   ├── memory_agent.py   # Dedicated sub-agent for qualitative decisions
│   ├── learning_scorer.py# Agent self-assessment feedback
│   └── orchestrator.py   # Central turn coordinator
├── tools/
│   ├── tool_registry.py  # Tool registration and discovery
│   └── web_tools.py      # Web search implementation
├── ingest/
│   ├── ingest.py         # Document ingestion pipeline
│   ├── gliner_labels.py  # NER label definitions for different domains
│   └── gliner_config.yaml# YAML label configurations
├── test_agemem.py        # 28 offline unit tests
└── main.py               # REPL entry point
```

## Three-Layer Hybrid Control

1. **System Rules (deterministic)** — Threshold-based triggers for overflow protection
2. **Memory Agent (LLM-based)** — Qualitative decisions on memory operations
3. **Learning Score (self-assessment)** — Novelty scoring drives LTM promotion

## Quick Commands

```bash
# Run tests (no LLM required)
python -m unittest test_agemem -v

# Start interactive chat
uv run main.py

# Install dependencies (basic)
uv pip install -e .

# Install with document ingestion support (includes docling + gliner)
uv pip install -e ".[ingest]"
```

## Document Ingestion

The `ingest/` module provides PDF-to-markdown conversion with configurable NER extraction:

```bash
# Requires: uv pip install -e ".[ingest]"
uv run ingest/ingest.py report.pdf [doc_type]

# Use built-in label sets for different domains
uv run ingest/ingest.py paper.pdf research --labels research
uv run ingest/ingest.py contract.pdf legal --labels legal
uv run ingest/ingest.py gara.pdf bando --labels edilizia

# Use custom labels from YAML config
uv run ingest/ingest.py doc.pdf custom --labels /path/to/config.yaml:medical
```

### Built-in Label Sets

| Set | Description | Use Case |
|-----|-------------|----------|
| `edilizia` | Italian construction/tenders (CIG, CUP, appalti) | Public procurement documents |
| `research` | Scientific papers (datasets, algorithms, citations) | Academic publications |
| `legal` | Contracts and legal documents (parties, clauses) | Legal text analysis |

### Custom Label Configuration

Create a YAML file (see `ingest/gliner_config.yaml` for examples):

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
  buckets:
    - people
    - orgs
    - custom
```

Then reference it: `--labels /path/to/config.yaml:my_domain`

| Feature | Package | Purpose |
|---------|---------|---------|
| PDF parsing | `docling` | Convert PDFs to markdown |
| Entity extraction | `gliner` | Zero-shot NER with domain-specific labels |
| Frontmatter | `PyYAML` | Metadata in YAML format |

## Key Configuration

All thresholds in `core/config.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `STM_TOKEN_LIMIT` | 6000 | Hard context ceiling |
| `STM_WARNING_THRESHOLD` | 0.75 | Trigger SUMMARY |
| `STM_CRITICAL_THRESHOLD` | 0.90 | Force FILTER + SUMMARY |
| `LTM_PROMOTE_THRESHOLD` | 0.65 | Learning score → LTM ADD |
| `TRIGGER_EVERY_N_TURNS` | 10 | MemoryAgent review cadence |

## Environment Variables

### Primary Variables (Recommended)

```bash
# Core LLM settings
BASE_URL=http://localhost:8080         # LLM API base URL
BASE_MODEL=qwen3-4b                    # Model name
BASE_MAX_TOKENS=2048                   # Max tokens per request
BASE_TEMPERATURE=0.2                   # Sampling temperature

# API key (required for non-local endpoints)
API_KEY=your-api-key                   # Primary API key
OPENAI_API_KEY=your-key                # Fallback for OpenAI-compatible services

# Memory and persistence
# Both LTM and STM are stored in PERSIST_DIR:
#   - {PERSIST_DIR}/ltm_store.json  (LTM entries)
#   - {PERSIST_DIR}/stm_context.json (STM context)
PERSIST_DIR=agent_memory
STM_TOKEN_LIMIT=6000
```

### Using Remote Providers (OpenRouter, OpenAI, etc.)

For non-localhost endpoints, an API key is required:

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

### Backward Compatibility

Old `LLAMA_*` variable names are still supported but deprecated:

| Deprecated | Use Instead |
|------------|-------------|
| `LLAMA_HOST` | `BASE_URL` |
| `LLAMA_MODEL` | `BASE_MODEL` |
| `LLAMA_MAX_TOKENS` | `BASE_MAX_TOKENS` |
| `LLAMA_TEMPERATURE` | `BASE_TEMPERATURE` |
| `LLAMA_CTX_SIZE` | `BASE_CTX_SIZE` |

If both old and new names are set, the new name takes precedence.

## Testing Philosophy

All tests are offline — no LLM calls, no network. Mock LLMClient for orchestrator tests. Key tests:

- **T20**: Double-boundary overflow invariant (critical architectural check)
- **T19**: LTM promotion via learning score
- **T13-T15**: System rule firing conditions

## Design Constraints

- **No RL training**: This is inference-only; all behavior is prompted or rule-based
- **No embeddings**: LTM retrieval uses token overlap (sufficient for ≤500 entries)
- **Tool LoopGuard**: Prevents duplicate tool calls with same arguments per turn
- **Double overflow guard**: `force_fit()` runs both pre-turn AND post-response

## Extension Points

- **Vector DB**: Replace `LTMStore.search()` for large-scale deployments
- **New tools**: Add to `tools/` and register in `tool_registry.py`
- **Custom rules**: Extend `SystemRules.evaluate()` in `triggers/system_rules.py`

## Critical Invariants

1. Context must be within bounds at the **end** of every turn (not just start)
2. Pinned messages (system prompt, retrieved LTM) are never evicted
3. Tool calls are deduplicated per-turn via LoopGuard
4. LTM persists to disk on every write; STM persists after every turn

## REPL Commands

| Command | Effect |
|---------|--------|
| `/clear` | Reset STM (LTM retained) |
| `/memory` | Show LTM snapshot |
| `/stats` | Show STM statistics |
| `/forget` | Wipe LTM (with confirmation) |
| `/help` | Show help |
