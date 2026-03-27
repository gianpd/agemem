# Corpus Mode Evaluation Experiment

## Overview

The corpus mode evaluation experiment tests AgeMem's ability to answer complex multi-hop questions by retrieving information from an external document corpus using tool-based interactions. Unlike the LTM (Long-Term Memory) mode where context is pre-loaded into memory, corpus mode simulates a real-world scenario where the agent must actively search for and retrieve relevant documents to answer questions.

## What the Experiment Tests

The corpus mode evaluation tests the following capabilities:

1. **Multi-hop Retrieval**: The agent's ability to identify and retrieve multiple documents that collectively contain the answer, where each document provides a piece of the puzzle.

2. **Tool Use Proficiency**: How effectively the agent uses corpus search tools (`grep_corpus`, `search_metadata`, `list_documents`, `read_document`) to locate relevant information.

3. **Reasoning Over Retrieved Context**: The ability to synthesize information from multiple retrieved documents to construct a correct answer.

4. **Information Navigation**: Skill in navigating between documents, reading specific sections, and extracting the right information without human guidance.

5. **Query Reformulation**: When initial searches don't yield results, the ability to reformulate search patterns (e.g., using pipe-separated alternatives in `grep_corpus`).

## Tools Evaluated (Corpus Tool Suite)

The evaluation tests the following corpus interaction tools defined in `tools/corpus.py`:

| Tool | Purpose | Evaluation Focus |
|------|---------|------------------|
| `list_documents` | List all available documents in the corpus | Document discovery and inventory awareness |
| `search_metadata` | Search document metadata (titles, types, tags) | Finding documents by known attributes |
| `grep_corpus` | Full-text search using regex patterns across document bodies | Primary retrieval mechanism; tests pattern construction and result interpretation |
| `read_document` | Retrieve full content of a specific document | Reading and comprehending retrieved documents |
| `read_lines` | Read specific line ranges from documents | Efficient navigation within large documents |
| `ingest_document` | Add new documents to the corpus | Corpus management (used during setup, not evaluation) |

The `grep_corpus` tool is particularly critical as it supports pipe-separated OR patterns (e.g., `"Siemens|partnership|industrial"`), which the agent must learn to use effectively for multi-concept queries.

## What Good Results Mean

A high J-score in corpus mode indicates:

- **Effective Search Strategy**: The agent successfully identifies which documents contain the answer without being told which ones to read.
- **Low False Positive Rate**: The agent can distinguish relevant from irrelevant documents among search results.
- **Robust Information Synthesis**: The agent correctly combines facts from multiple sources, even when each source only provides partial information.
- **Tool Mastery**: The agent understands the tool semantics—knowing when to use metadata search vs. full-text grep, and how to read documents efficiently.
- **Generalization**: The system works without task-specific tuning—it's not hardcoded for HotpotQA but demonstrates general retrieval competence.

**J-score Interpretation:**
- **1.0**: Perfect answer match
- **0.7-0.9**: Substantially correct with minor errors
- **0.4-0.6**: Partially correct, missing key information
- **0.0-0.3**: Incorrect or completely missed

## Why HotpotQA is Ideal for Testing Multi-hop Corpus Retrieval

HotpotQA is specifically designed to require multi-hop reasoning across documents:

### Dataset Characteristics

1. **Explicit Multi-hop Structure**: Each question requires reasoning over **2-10 supporting facts** from **2+ different documents** (on average). A question cannot be answered from a single document.

2. **Two Question Types**:
   - **Bridge Questions**: Require chaining facts where the answer to a sub-question enables finding the next fact (e.g., "What university did the author of 'X' attend?" requires first identifying the author, then finding their university).
   - **Comparison Questions**: Require comparing attributes across two entities (e.g., "Which is larger, X or Y?" requires retrieving facts about both).

3. **Difficulty Levels**: Questions are categorized as easy, medium, or hard, allowing evaluation of retrieval complexity scaling.

4. **Gold Supporting Facts**: Each question is annotated with the exact documents and sentences required to answer it, enabling precise evaluation of whether the agent retrieved the necessary information.

5. **Distractor Setting**: The "distractor" configuration includes both gold paragraphs and irrelevant ones, testing the agent's ability to filter noise—exactly what corpus mode must handle.

### Why It Maps to Corpus Mode

| HotpotQA Feature | Corpus Mode Challenge |
|-----------------|----------------------|
| Multiple required documents | Must use search tools to find relevant docs |
| Bridging entities across documents | Must track entities and follow references |
| Distractor paragraphs | Must filter search results for relevance |
| Varied question types | Must adapt search strategy (entity lookup vs. comparison) |
| Annotated supporting facts | Can measure retrieval accuracy against gold standard |

## Expected LLM-as-Judge Scores

Based on the AgeMem paper (arXiv:2601.01885v1) and empirical evaluation runs:

### Target Benchmarks

| Configuration | Expected J-Score | Notes |
|--------------|------------------|-------|
| AgeMem-noRL (paper) | 54.49 | Baseline without reinforcement learning |
| AgeMem-RL (paper) | 55.49 | With reinforcement learning optimization |
| **Corpus Mode Target** | **45-55** | Achievable with effective tool use |
| **LTM Mode Baseline** | **50-60** | Oracle access to gold paragraphs |

### Score Interpretation for Corpus Mode

| Range | Assessment |
|-------|------------|
| 55+ | Excellent: Agent effectively uses corpus tools, retrieves correct documents, synthesizes answers accurately |
| 45-55 | Good: Solid retrieval with occasional misses or synthesis errors |
| 35-45 | Fair: Basic retrieval works but struggles with complex multi-hop chains |
| <35 | Poor: Significant issues with tool use or retrieval strategy |

### Coverage Rate

In addition to the mean J-score, the **coverage rate** (% of successful judge calls) should be monitored:

- **Target**: >95% coverage (few parse failures or exceptions)
- **Acceptable**: 90-95%
- **Concern**: <90% (indicates system instability or tool failures)

## Running the Evaluation

```bash
# Corpus mode evaluation (dynamically retrieves from corpus)
python evaluation/run_hotpotqa.py --mode corpus --limit 100 --output evaluation/results/corpus_results.json

# LTM mode baseline (oracle access to gold paragraphs)
python evaluation/run_hotpotqa.py --mode ltm --limit 100 --output evaluation/results/ltm_results.json

# Generate LLM-as-Judge report
python evaluation/llm_judge_eval.py --log evaluation/logs/hotpotqa_*.log --output evaluation/reports/judge_eval.json --csv evaluation/reports/judge_eval.csv
```

## Output Schema

The evaluation produces results with the following structure:

```json
{
  "sample_id": "5a8bb29b55429924...",
  "index": 33,
  "query": "Which American film actor...",
  "gold": "George Raft",
  "gold_titles": ["Some Like It Hot", "George Raft"],
  "prediction": "George Raft",
  "llm_judge_score": 1.0,
  "judge_status": "ok",
  "latency_ms": 45230
}
```

## Key Differences from LTM Mode

| Aspect | LTM Mode | Corpus Mode |
|--------|----------|-------------|
| Context source | Pre-loaded into LTM | Retrieved via tools during inference |
| Information access | Oracle (guaranteed relevant) | Must search and filter |
| Tool use | None required | Active tool calling required |
| Realism | Synthetic upper bound | Realistic deployment scenario |
| Expected score | Higher (50-60) | Lower but more meaningful (45-55) |

The corpus mode evaluation represents the true test of AgeMem's retrieval capabilities in a production-like setting where information must be discovered rather than provided.