# Technical Requirements Specification: AgeMem Automated Testing Suite

**Document ID:** TRS-AGEMEM-EVAL-001
**Version:** 1.1
**Date:** 2026-03-18
**Status:** Draft
**Author:** AgeMem Evaluation Team
**References:** [AgeMem Technical Specification v1.0](./agemem_technical_specification.md)

---

## 1. Introduction

### 1.1 Purpose

This Technical Requirements Specification (TRS) defines the architectural requirements for a Python-based automated testing suite designed to evaluate the AgeMem system. The specification ensures 100% adherence to the protocols established in [Section 7: Formal Evaluation Framework](./agemem_technical_specification.md#7-formal-evaluation-framework) of the AgeMem Technical Specification.

### 1.2 Scope

The testing suite shall implement four core components:

1. [**Dataset Pipeline**](#31-dataset-pipeline) — Ingests and validates benchmark datasets per [Section 7.2](./agemem_technical_specification.md#72-required-datasets)
2. [**Inference Test Pipeline**](#32-inference-test-pipeline) — Executes AgeMem with telemetry capture per [Section 7.6](./agemem_technical_specification.md#76-evaluation-infrastructure)
3. [**KPI Metrics Collection Pipeline**](#33-kpi-metrics-collection-pipeline) — Calculates metrics defined in [Section 7.3](./agemem_technical_specification.md#73-evaluation-metrics)
4. [**Evaluation Metrics Report**](#34-evaluation-metrics-report) — Generates comparative analysis per [Section 7.4](./agemem_technical_specification.md#74-competitor-benchmarks) and [Section 7.7](./agemem_technical_specification.md#77-reporting)

### 1.3 Definitions

| Term | Definition | Source |
|------|------------|--------|
| **AgeMem** | Hybrid memory management system with STM/LTM architecture | [Section 2](./agemem_technical_specification.md#2-system-core-purpose) |
| **STM** | Short-Term Memory — active context window management | [Section 4.1](./agemem_technical_specification.md#41-high-level-architecture) |
| **LTM** | Long-Term Memory — persistent knowledge storage | [Section 4.1](./agemem_technical_specification.md#41-high-level-architecture) |
| **SearchTrace** | Instrumentation data structure capturing query execution details | [Section 7.6](./agemem_technical_specification.md#76-evaluation-infrastructure) |
| **MRR@K** | [Mean Reciprocal Rank at K](#mrr-k-formula) — primary retrieval quality metric | [Section 7.3.1](./agemem_technical_specification.md#731-retrieval-metrics) |
| **Learning Score** | Self-assessed novelty signal (0-1 scale) driving LTM promotion | [Section 4.4](./agemem_technical_specification.md#44-three-layer-control-system) |
| **LongMemEval** | Long-context memory benchmark dataset | [Section 7.2.1](./agemem_technical_specification.md#721-conversational-memory-benchmarks) |
| **LoCoMo** | Long-context modeling benchmark dataset | [Section 7.2.1](./agemem_technical_specification.md#721-conversational-memory-benchmarks) |
| **ConvoMem** | Conversational memory benchmark dataset | [Section 7.2.1](./agemem_technical_specification.md#721-conversational-memory-benchmarks) |

---

## 2. References

### 2.1 Primary Reference

- **[AgeMem Technical Specification v1.0](./agemem_technical_specification.md)**
  - [Section 7.1: Evaluation Objectives](./agemem_technical_specification.md#71-evaluation-objectives)
  - [Section 7.2: Dataset Requirements](./agemem_technical_specification.md#72-required-datasets)
  - [Section 7.3: KPI Metrics](./agemem_technical_specification.md#73-evaluation-metrics)
  - [Section 7.4: Competitor Benchmarks](./agemem_technical_specification.md#74-competitor-benchmarks)
  - [Section 7.5: Evaluation Protocol](./agemem_technical_specification.md#75-evaluation-protocol)
  - [Section 7.6: Evaluation Infrastructure](./agemem_technical_specification.md#76-evaluation-infrastructure)
  - [Section 7.7: Reporting](./agemem_technical_specification.md#77-reporting)

### 2.2 Supporting Documents

| Document | Location | Description |
|----------|----------|-------------|
| AgeMem Architecture Overview | [`docs/ARCHITECTURE_LAYERS.md`](/home/jaco/develops/WORKS/agemem/docs/ARCHITECTURE_LAYERS.md) | Component architecture and data flow |
| Memory System Implementation | [`docs/memory_enhanced.md`](/home/jaco/develops/WORKS/agemem/docs/memory_enhanced.md) | Enhanced memory system documentation |
| Core Data Structures | [Section 4.2](./agemem_technical_specification.md#42-data-structures) | MemoryEntry, ContextMessage definitions |
| Configuration Reference | [Appendix 8.1](./agemem_technical_specification.md#81-configuration-reference) | Environment variables and defaults |

### 2.3 External References

| Resource | URL | Purpose |
|----------|-----|---------|
| LongMemEval Dataset | <https://github.com/declare-lab/LongMemEval> | Primary long-context memory benchmark |
| LoCoMo Dataset | <https://github.com/nyu-mll/LoCoMo> | Long-context modeling evaluation |
| MS MARCO Dataset | <https://microsoft.github.io/msmarco/> | Passage retrieval benchmarking |
| Natural Questions | <https://ai.google.com/research/NaturalQuestions> | Open-domain QA evaluation |
| sqlite-vec Documentation | <https://github.com/asg017/sqlite-vec> | Vector search implementation |
| GLiNER NER | <https://github.com/urchade/GLiNER> | Named entity recognition for ingestion |

---

## 3. Architectural Requirements

### 3.1 Dataset Pipeline

#### 3.1.1 Functional Requirements

The Dataset Pipeline shall ingest and validate the three primary benchmark datasets as specified in [Section 7.2.1](./agemem_technical_specification.md#721-conversational-memory-benchmarks) of the AgeMem Technical Specification:

| Dataset | Domain | Primary Metrics | Technical Definition | Priority |
|---------|--------|-----------------|---------------------|----------|
| [**LongMemEval**](https://github.com/declare-lab/LongMemEval) | Long-context memory | Recall accuracy, temporal reasoning | [Recall@K](#recallk-formula) per [Section 7.3.1](./agemem_technical_specification.md#731-retrieval-metrics) | High |
| [**LoCoMo**](https://github.com/nyu-mll/LoCoMo) | Long-context modeling | Context retention, coherence | [Coherence Score](#coherence-formula) per [Section 7.3.3](./agemem_technical_specification.md#733-response-quality-metrics) | High |
| [**ConvoMem**](https://huggingface.co/datasets/MultiOn/ConvoMem) | Conversational memory | Entity recall, preference tracking | [Preference Accuracy](#preference-formula) per [Section 7.3.3](./agemem_technical_specification.md#733-response-quality-metrics) | High |

#### 3.1.2 Data Ingestion Requirements

1. **Format Support:** The pipeline shall support JSON, CSV, and Parquet formats for dataset ingestion per [Section 5](./agemem_technical_specification.md#5-document-ingestion-pipeline)

2. **Schema Validation:** Each dataset shall be validated against its respective schema before processing. Schema definitions must comply with [MemoryEntry structure](./agemem_technical_specification.md#421-memoryentry-ltm-core)

3. **Data Partitioning:** Datasets shall be partitioned into training, validation, and test sets with configurable ratios (default: 70/15/15) per [Phase 1 protocol](./agemem_technical_specification.md#phase-1-retrieval-quality-automated)

4. **Entity Extraction:** The pipeline shall extract and normalize entities using [GLiNER NER](https://github.com/urchade/GLiNER) as specified in [Section 5.3](./agemem_technical_specification.md#53-entity-extraction)

5. **Temporal Annotation:** All dataset entries shall include temporal annotations for [temporal reasoning evaluation](./agemem_technical_specification.md#recency-decay-formula) using the formula:
   ```
   recency_decay = exp(-ln(2) × days_elapsed / 7)
   ```

#### 3.1.3 Validation Requirements

1. **Completeness Check:** Verify all required fields are present per [MemoryEntry dataclass](./agemem_technical_specification.md#421-memoryentry-ltm-core):
   - `content` (str): The actual memory content
   - `entry_id` (str): SHA1 hash (auto-generated)
   - `created_at` (float): Unix timestamp
   - `learning_score` (float): Aggregated novelty signal (0-1)

2. **Consistency Check:** Ensure cross-reference integrity between related entries

3. **Quality Metrics:** Calculate and report data quality metrics:
   - Missing field percentage
   - Entity extraction accuracy
   - Temporal annotation coverage

4. **Validation Report:** Generate a validation report per [Section 7.7 reporting format](./agemem_technical_specification.md#77-reporting)

#### 3.1.4 Output Specifications

The Dataset Pipeline shall produce:

- Validated dataset files in standardized JSON format
- Dataset metadata including entry counts, entity distributions, and temporal ranges
- Validation reports with quality metrics per [reporting specification](./agemem_technical_specification.md#77-reporting)

### 3.2 Inference Test Pipeline

#### 3.2.1 Functional Requirements

The Inference Test Pipeline shall programmatically trigger AgeMem system processes and capture output telemetry as specified in [Section 7.6: Evaluation Infrastructure](./agemem_technical_specification.md#76-evaluation-infrastructure).

#### 3.2.2 System Integration Requirements

1. **AgeMem Interface:** The pipeline shall interface with AgeMem through its [Python API](./agemem_technical_specification.md#63-basic-usage) or command-line interface

2. **Configuration Management:** Support configurable AgeMem parameters per [Appendix 8.1](./agemem_technical_specification.md#81-configuration-reference):

   | Parameter | Default | Description | Source |
   |-----------|---------|-------------|--------|
   | `LTM_DEDUP_OVERLAP_THRESHOLD` | 0.70 | Jaccard similarity threshold for overlap dedup | [Section 4.6](./agemem_technical_specification.md#46-retrieval-methods) |
   | `LTM_DEDUP_THRESHOLD` | 0.92 | Cosine similarity dedup threshold | [Appendix 8.1](./agemem_technical_specification.md#81-configuration-reference) |
   | `LEARNING_SPIKE` | 0.80 | Immediate LTM candidacy threshold | [Section 4.4](./agemem_technical_specification.md#layer-3-learning-score-self-assessed) |
   | `STM_TOKEN_LIMIT` | 6000 | Context window size in tokens | [Appendix 8.1](./agemem_technical_specification.md#81-configuration-reference) |

3. **Session Management:** Manage multiple concurrent evaluation sessions with isolated memory stores per [Phase 2 protocol](./agemem_technical_specification.md#phase-2-memory-persistence-multi-session)

#### 3.2.3 Telemetry Capture Requirements

1. **SearchTrace Instrumentation:** Capture SearchTrace data as defined in [Section 7.6](./agemem_technical_specification.md#76-evaluation-infrastructure):

   ```python
   @dataclass
   class SearchTrace:
       query: str                           # Original query text
       query_embedding: list[float]         # 1024-dim vector from Qwen/Qwen3-Embedding-0.6B
       results: list[tuple[str, float]]     # (entry_id, score) per [hybrid scoring formula](#hybrid-scoring)
       latency_ms: float                    # End-to-end retrieval latency
       mode: str                            # "semantic", "overlap", or "expanded" per [Section 4.6](./agemem_technical_specification.md#46-retrieval-methods)
       variant_used: Optional[str]          # Query expansion variant if applicable
   ```

2. **SQLite Logging:** Implement SQLite logging for query/result pairs as specified in [Section 7.6](./agemem_technical_specification.md#76-evaluation-infrastructure):

   ```sql
   CREATE TABLE search_traces (
       id INTEGER PRIMARY KEY,
       timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
       query TEXT NOT NULL,
       query_embedding BLOB,              -- Serialized 1024-dim float array
       results_json TEXT NOT NULL,        -- JSON array of (entry_id, score) tuples
       latency_ms REAL NOT NULL,
       mode TEXT NOT NULL CHECK(mode IN ('semantic', 'overlap', 'expanded')),
       variant_used TEXT,
       session_id TEXT NOT NULL,
       FOREIGN KEY (session_id) REFERENCES evaluation_sessions(id)
   );
   ```

3. **Additional Telemetry:** Capture:
   - Memory operation logs (ADD, UPDATE, DELETE, SEARCH) per [LTM Store operations](./agemem_technical_specification.md#41-high-level-architecture)
   - Context window utilization metrics per [Overflow Invariant](./agemem_technical_specification.md#84-critical-invariants)
   - Learning score assessments per [LearningFeedback structure](./agemem_technical_specification.md#423-learningfeedback-self-assessment)
   - Tool execution traces (T1-T5 tiers) per [Section 4.5](./agemem_technical_specification.md#45-ltm-self-introspection-toolkit)

#### 3.2.4 Execution Requirements

1. **Batch Processing:** Support batch execution of [1000 queries](./agemem_technical_specification.md#phase-1-retrieval-quality-automated) from benchmark datasets

2. **Parallel Execution:** Enable configurable parallel test execution with resource limits per [Section 5.1: Scalability](#51-scalability)

3. **Checkpointing:** Implement checkpointing for long-running evaluations per [Phase 2: 50-turn conversations](./agemem_technical_specification.md#phase-2-memory-persistence-multi-session)

4. **Error Handling:** Graceful handling of system failures with detailed error logging

#### 3.2.5 Output Specifications

The Inference Test Pipeline shall produce:

- Raw telemetry data in SQLite database per [schema definition](#sqlite-logging)
- SearchTrace logs in JSON format per [SearchTrace dataclass](#searchtrace-instrumentation)
- Execution reports with success/failure statistics
- Performance metrics (latency, throughput, resource usage)

### 3.3 KPI Metrics Collection Pipeline

#### 3.3.1 Functional Requirements

The KPI Metrics Collection Pipeline shall calculate MRR@K and other accuracy benchmarks defined in [Section 7.3: Evaluation Metrics](./agemem_technical_specification.md#73-evaluation-metrics).

#### 3.3.2 Metric Calculation Requirements

The pipeline shall calculate the following metrics as specified in [Section 7.3](./agemem_technical_specification.md#73-evaluation-metrics):

##### Retrieval Metrics

<a name="mrrk-formula"></a>
**MRR@K (Mean Reciprocal Rank at K)**

```python
def calculate_mrr(queries: list[Query], k: int = 10) -> float:
    """
    Mean Reciprocal Rank at K per Section 7.3.1
    Formula: 1 / rank_of_first_relevant
    """
    reciprocal_ranks = []
    for query in queries:
        results = ltm.search(query.text, top_k=k)
        for rank, (entry, score) in enumerate(results, 1):
            if entry.entry_id in query.relevant_ids:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

<a name="precisionk-formula"></a>
**Precision@K**

```python
def calculate_precision(results: list, relevant_ids: set, k: int) -> float:
    """
    Fraction of relevant results in top K per Section 7.3.1
    Formula: relevant_in_topK / K
    """
    top_k = results[:k]
    relevant_in_topk = sum(1 for entry, _ in top_k if entry.entry_id in relevant_ids)
    return relevant_in_topk / k
```

<a name="recallk-formula"></a>
**Recall@K**

```python
def calculate_recall(results: list, relevant_ids: set, k: int) -> float:
    """
    Coverage of relevant results per Section 7.3.1
    Formula: relevant_in_topK / total_relevant
    """
    top_k = results[:k]
    relevant_in_topk = sum(1 for entry, _ in top_k if entry.entry_id in relevant_ids)
    return relevant_in_topk / len(relevant_ids) if relevant_ids else 0.0
```

<a name="ndcgk-formula"></a>
**NDCG@K (Normalized Discounted Cumulative Gain)**

```python
def calculate_dcg(scores: list, k: int) -> float:
    """DCG = sum((2^relevance - 1) / log2(rank + 1))"""
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(scores[:k]))

def calculate_ndcg(results: list, relevance_scores: dict, k: int) -> float:
    """
    NDCG = DCG@K / IDCG@K per Section 7.3.1
    """
    actual_scores = [relevance_scores.get(entry.entry_id, 0) for entry, _ in results[:k]]
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    dcg = calculate_dcg(actual_scores, k)
    idcg = calculate_dcg(ideal_scores, k)
    return dcg / idcg if idcg > 0 else 0.0
```

##### Memory Quality Metrics

| Metric | Technical Definition | Target | Source |
|--------|---------------------|--------|--------|
| **Retention Rate** | `% of promoted memories retained after N turns` | ≥ 95% | [Section 7.3.2](./agemem_technical_specification.md#732-memory-quality-metrics) |
| **Deduplication Accuracy** | `% of true duplicates correctly merged` | ≥ 90% | [Section 7.3.2](./agemem_technical_specification.md#732-memory-quality-metrics) |
| **Learning Score Correlation** | `Pearson correlation between score and utility` | ≥ 0.7 | [Section 7.3.2](./agemem_technical_specification.md#732-memory-quality-metrics) |
| **Context Utilization** | `% of injected memories referenced in response` | ≥ 60% | [Section 7.3.2](./agemem_technical_specification.md#732-memory-quality-metrics) |

##### Response Quality Metrics

<a name="coherence-formula"></a>
| Metric | Technical Definition | Target | Source |
|--------|---------------------|--------|--------|
| **Hallucination Rate** | `% of responses with unsupported claims` | ≤ 5% | [Section 7.3.3](./agemem_technical_specification.md#733-response-quality-metrics) |
| **Coherence Score** | `Human-rated response coherence (1-5)` | ≥ 4.0 | [Section 7.3.3](./agemem_technical_specification.md#733-response-quality-metrics) |
| **Memory Grounding** | `% of memory-dependent claims with valid citations` | ≥ 90% | [Section 7.3.3](./agemem_technical_specification.md#733-response-quality-metrics) |
| **Preference Accuracy** | `% of user preferences correctly recalled` | ≥ 95% | [Section 7.3.3](./agemem_technical_specification.md#733-response-quality-metrics) |

##### Hybrid Scoring Formula

<a name="hybrid-scoring"></a>
Per [Section 4.6: Semantic Search](./agemem_technical_specification.md#46-retrieval-methods):

```python
def calculate_hybrid_score(
    cosine_similarity: float,
    days_elapsed: float,
    learning_score: float
) -> float:
    """
    Hybrid scoring formula for memory retrieval ranking.
    Source: Section 4.6, Semantic Search
    """
    recency_decay = math.exp(-math.log(2) * days_elapsed / 7)  # 7-day half-life
    score = (
        0.60 * cosine_similarity +      # Semantic relevance
        0.25 * recency_decay +           # Time-based decay
        0.15 * learning_score            # Importance weight
    )
    return score
```

#### 3.3.3 Performance Benchmarks

The pipeline shall evaluate against the performance targets from [Section 7.3](./agemem_technical_specification.md#73-evaluation-metrics):

| Metric | Target | Category | Technical Definition |
|--------|--------|----------|---------------------|
| **MRR@10** | ≥ 0.85 | Retrieval | [Mean Reciprocal Rank](#mrrk-formula) at K=10 |
| **Recall@5** | ≥ 0.90 | Retrieval | [Recall@K](#recallk-formula) at K=5 |
| **Hallucination Rate** | ≤ 5% | Quality | [% unsupported claims](#response-quality-metrics) |
| **Coherence Score** | ≥ 4.0 | Quality | [1-5 human rating](#response-quality-metrics) |
| **Memory Grounding** | ≥ 90% | Quality | [% valid citations](#response-quality-metrics) |
| **Preference Accuracy** | ≥ 95% | Quality | [% correct recall](#response-quality-metrics) |

#### 3.3.4 Aggregation Requirements

1. **Cross-Dataset Aggregation:** Aggregate metrics across [LongMemEval](https://github.com/declare-lab/LongMemEval), [LoCoMo](https://github.com/nyu-mll/LoCoMo), and [ConvoMem](https://huggingface.co/datasets/MultiOn/ConvoMem) datasets per [Section 7.2.1](./agemem_technical_specification.md#721-conversational-memory-benchmarks)

2. **Retrieval Mode Comparison:** Compare [semantic vs overlap retrieval modes](./agemem_technical_specification.md#46-retrieval-methods) per Phase 1 requirement

3. **Temporal Analysis:** Analyze performance across different temporal ranges using [recency decay formula](#hybrid-scoring)

4. **Statistical Significance:** Calculate confidence intervals and statistical significance of results

#### 3.3.5 Output Specifications

The KPI Metrics Collection Pipeline shall produce:

- Metric calculation results in JSON format per [Section 7.7 reporting format](./agemem_technical_specification.md#77-reporting)
- Comparative analysis reports
- Statistical significance tests
- Performance trend visualizations

### 3.4 Evaluation Metrics Report

#### 3.4.1 Functional Requirements

The Evaluation Metrics Report shall generate a final performance summary against the competitor benchmarks defined in [Section 7.4](./agemem_technical_specification.md#74-competitor-benchmarks) of the AgeMem Technical Specification.

#### 3.4.2 Competitor Benchmarks

The report shall compare AgeMem against the following systems as specified in [Section 7.4](./agemem_technical_specification.md#74-competitor-benchmarks):

| System | Architecture | Key Difference | Reference |
|--------|--------------|----------------|-----------|
| **MemGPT** | Virtual context management | RL-trained memory policy | <https://github.com/cpacker/MemGPT> |
| **Letta** | Agent memory systems | Cloud-hosted, API-dependent | <https://github.com/letta-ai/letta> |
| **LangChain RAG** | Standard retrieval-augmented generation | Flat chunk retrieval, no memory tiers | <https://github.com/langchain-ai/langchain> |
| **LlamaIndex** | Document indexing and retrieval | Query-focused, no persistent memory | <https://github.com/run-llama/llama_index> |
| **Base LLM (no memory)** | No memory system | Baseline comparison | N/A |

#### 3.4.3 Report Generation Requirements

1. **Executive Summary:** High-level performance comparison with key findings per [Section 7.7](./agemem_technical_specification.md#77-reporting)

2. **Detailed Metrics:** Comprehensive breakdown of all KPIs by dataset and retrieval mode per [metric definitions](#332-metric-calculation-requirements)

3. **Competitive Analysis:** Side-by-side comparison with competitor systems per [Section 7.4](./agemem_technical_specification.md#74-competitor-benchmarks)

4. **Resource Efficiency:** Token usage, latency, and memory footprint analysis per [Resource Efficiency objective](./agemem_technical_specification.md#71-evaluation-objectives)

5. **Recommendations:** Actionable recommendations based on evaluation results

#### 3.4.4 Output Formats

The report shall be generated in multiple formats per [Section 7.7](./agemem_technical_specification.md#77-reporting):

- **Markdown:** For documentation and version control
- **PDF:** For formal reporting and distribution
- **JSON:** For programmatic consumption
- **HTML:** For interactive web-based viewing

#### 3.4.5 Automation Requirements

1. **Scheduled Generation:** Support scheduled report generation (daily, weekly, on-demand)
2. **Versioning:** Maintain report version history with diff comparisons
3. **Distribution:** Automated distribution via email, webhook, or file storage
4. **Customization:** Configurable report sections and metrics inclusion

---

## 4. Data Interfaces

### 4.1 Input Interfaces

1. **Benchmark Datasets:** [LongMemEval](https://github.com/declare-lab/LongMemEval), [LoCoMo](https://github.com/nyu-mll/LoCoMo), [ConvoMem](https://huggingface.co/datasets/MultiOn/ConvoMem) in JSON/CSV/Parquet format per [Section 7.2.1](./agemem_technical_specification.md#721-conversational-memory-benchmarks)

2. **AgeMem Configuration:** YAML/JSON configuration files per [Appendix 8.1](./agemem_technical_specification.md#81-configuration-reference)

3. **Test Queries:** JSON files containing test queries with ground truth relevance judgments per [Phase 1 protocol](./agemem_technical_specification.md#phase-1-retrieval-quality-automated)

### 4.2 Output Interfaces

1. **SQLite Database:** SearchTrace logs and telemetry data per [Section 7.6 SQLite Logging](./agemem_technical_specification.md#76-evaluation-infrastructure)

2. **JSON Files:** Metric results, validation reports, performance data per [Section 7.7](./agemem_technical_specification.md#77-reporting)

3. **Markdown/PDF Reports:** Final evaluation reports per [reporting specification](./agemem_technical_specification.md#77-reporting)

4. **Visualization Files:** Charts and graphs in PNG/SVG format

### 4.3 API Interfaces

1. **AgeMem Python API:** For system interaction and telemetry capture per [Section 6.3](./agemem_technical_specification.md#63-basic-usage)

2. **Database APIs:** SQLite and potential PostgreSQL support per [Section 4.3.1](./agemem_technical_specification.md#431-ltm-persistence)

3. **Reporting APIs:** Integration with external reporting tools

---

## 5. Performance Requirements

### 5.1 Scalability

1. **Dataset Size:** Support evaluation of datasets with up to 100,000 entries per [LTM_MAX_ENTRIES limit](./agemem_technical_specification.md#81-configuration-reference)

2. **Concurrent Sessions:** Handle up to 50 concurrent evaluation sessions per [Phase 2: 10 sessions](./agemem_technical_specification.md#phase-2-memory-persistence-multi-session)

3. **Query Throughput:** Process at least 100 queries per minute per session

### 5.2 Reliability

1. **Uptime:** 99.5% availability during evaluation runs

2. **Data Integrity:** Zero data loss for telemetry and metric results per [Offline Testability invariant](./agemem_technical_specification.md#84-critical-invariants)

3. **Error Recovery:** Automatic recovery from transient failures

### 5.3 Performance Metrics

1. **Latency:** End-to-end evaluation completion within 24 hours for full benchmark suite per [Phase 1-4 protocols](./agemem_technical_specification.md#75-evaluation-protocol)

2. **Resource Usage:** Memory usage < 16GB, CPU utilization < 80% on reference hardware per [proven on 8GB RTX 4060](./agemem_technical_specification.md#22-design-principles)

3. **Storage:** Efficient storage of telemetry data with configurable retention policies

---

## 6. Validation Requirements

### 6.1 Unit Testing

1. **Component Testing:** 90% code coverage for all pipeline components per [Offline Testability invariant](./agemem_technical_specification.md#84-critical-invariants)

2. **Integration Testing:** End-to-end testing of complete evaluation workflow

3. **Regression Testing:** Automated regression test suite for continuous integration

### 6.2 Validation Against Specification

1. **Protocol Adherence:** 100% compliance with [Section 7 evaluation protocols](./agemem_technical_specification.md#75-evaluation-protocol)

2. **Metric Accuracy:** Validation of metric calculations against known benchmarks per [MRR Harness](./agemem_technical_specification.md#76-evaluation-infrastructure)

3. **Report Completeness:** Verification that all required report sections per [Section 7.7](./agemem_technical_specification.md#77-reporting) are generated

### 6.3 Acceptance Criteria

1. **Functional Completeness:** All specified requirements implemented and tested
2. **Performance Compliance:** Meets all [performance requirements](#5-performance-requirements)
3. **Documentation:** Complete user and developer documentation

---

## 7. Appendices

### Appendix A: Evaluation Protocol Checklist

Based on [Section 7.5](./agemem_technical_specification.md#75-evaluation-protocol) of the AgeMem Technical Specification:

#### Phase 1: Retrieval Quality (Automated)

Per [Phase 1 specification](./agemem_technical_specification.md#phase-1-retrieval-quality-automated):

1. ✅ Populate LTM with 500 curated memories from benchmark datasets
2. ✅ Execute 1000 queries across benchmark test sets
3. ✅ Measure [MRR@K](#mrrk-formula), [Precision@K](#precisionk-formula), [Recall@K](#recallk-formula) at K=1,5,10
4. ✅ Compare [semantic vs overlap retrieval modes](./agemem_technical_specification.md#46-retrieval-methods)
5. ✅ Log variant hit-rate for query expansion per [Tier 2 tools](./agemem_technical_specification.md#tier-2--retrieval-orchestration-action)

#### Phase 2: Memory Persistence (Multi-Session)

Per [Phase 2 specification](./agemem_technical_specification.md#phase-2-memory-persistence-multi-session):

1. ✅ Conduct 50-turn conversations across 10 sessions
2. ✅ Track promoted memories and their retrieval across sessions
3. ✅ Measure [retention rate](#memory-quality-metrics) and [deduplication accuracy](#memory-quality-metrics)
4. ✅ Validate [learning score correlation](#memory-quality-metrics) with actual utility

#### Phase 3: Response Quality (Human Evaluation)

Per [Phase 3 specification](./agemem_technical_specification.md#phase-3-response-quality-human-evaluation):

1. ✅ Conduct 100 conversations with human evaluators
2. ✅ Rate responses on [coherence](#response-quality-metrics), accuracy, [memory grounding](#response-quality-metrics)
3. ✅ Measure [hallucination rate](#response-quality-metrics) via fact verification
4. ✅ Track [preference accuracy](#response-quality-metrics) across sessions

#### Phase 4: Comparative Benchmarking

Per [Phase 4 specification](./agemem_technical_specification.md#phase-4-comparative-benchmarking):

1. ✅ Run identical tasks across all [competitor systems](#342-competitor-benchmarks)
2. ✅ Normalize metrics for fair comparison
3. ✅ Measure resource usage (tokens, latency, memory)
4. ✅ Generate comparative performance report

### Appendix B: SearchTrace Schema

Complete SQLite schema per [Section 7.6](./agemem_technical_specification.md#76-evaluation-infrastructure):

```sql
CREATE TABLE search_traces (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    query TEXT NOT NULL,
    query_embedding BLOB,              -- 1024-dim float array per [Section 4.3.1](./agemem_technical_specification.md#431-ltm-persistence)
    results_json TEXT NOT NULL,        -- [(entry_id, score), ...] per [hybrid scoring](#hybrid-scoring)
    latency_ms REAL NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('semantic', 'overlap', 'expanded')),
    variant_used TEXT,                 -- Query expansion variant per [Tier 2 tools](./agemem_technical_specification.md#tier-2--retrieval-orchestration-action)
    session_id TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES evaluation_sessions(id)
);
```

### Appendix C: Configuration Parameters

Key configuration parameters from [AgeMem Technical Specification Appendix 8.1](./agemem_technical_specification.md#81-configuration-reference):

| Parameter | Default | Description | Impact on Evaluation |
|-----------|---------|-------------|---------------------|
| `LTM_DEDUP_OVERLAP_THRESHOLD` | 0.70 | Jaccard similarity threshold for overlap dedup | Affects [deduplication accuracy](#memory-quality-metrics) |
| `LTM_DEDUP_THRESHOLD` | 0.92 | Cosine similarity dedup threshold | Affects [deduplication accuracy](#memory-quality-metrics) |
| `CONTEXT_CURRENT_QUERY_WEIGHT` | 0.50 | Weight for current query in context-aware retrieval | Affects [MRR@K](#mrrk-formula) |
| `CONTEXT_PREVIOUS_TURN_WEIGHT` | 0.30 | Weight for previous turn in context-aware retrieval | Affects [MRR@K](#mrrk-formula) |
| `CONTEXT_TURN_BEFORE_WEIGHT` | 0.15 | Weight for turn before previous in context-aware retrieval | Affects [MRR@K](#mrrk-formula) |
| `LEARNING_SPIKE` | 0.80 | Immediate LTM candidacy threshold | Affects [retention rate](#memory-quality-metrics) |
| `OVERFLOW_WARN` | 0.75 | Warning threshold (force SUMMARY) | Affects [context utilization](#memory-quality-metrics) |
| `OVERFLOW_CRITICAL` | 0.90 | Critical threshold (force FILTER) | Affects [context utilization](#memory-quality-metrics) |

### Appendix D: Evaluation Objectives Mapping

Based on [Section 7.1](./agemem_technical_specification.md#71-evaluation-objectives) of the AgeMem Technical Specification:

| Objective | Pipeline Component | Metrics | Technical Definition |
|-----------|-------------------|---------|---------------------|
| **Retrieval Quality** | [KPI Metrics Collection](#33-kpi-metrics-collection-pipeline) | MRR@K, Precision@K, Recall@K, NDCG@K | [Section 7.3.1](./agemem_technical_specification.md#731-retrieval-metrics) |
| **Response Quality** | [Evaluation Metrics Report](#34-evaluation-metrics-report) | Coherence Score, Hallucination Rate | [Section 7.3.3](./agemem_technical_specification.md#733-response-quality-metrics) |
| **Memory Persistence** | [Inference Test Pipeline](#32-inference-test-pipeline) | Retention rate, Deduplication accuracy | [Section 7.3.2](./agemem_technical_specification.md#732-memory-quality-metrics) |
| **Resource Efficiency** | [Inference Test Pipeline](#32-inference-test-pipeline) | Token usage, Latency, Memory footprint | Telemetry capture per [Section 7.6](./agemem_technical_specification.md#76-evaluation-infrastructure) |
| **Comparative Performance** | [Evaluation Metrics Report](#34-evaluation-metrics-report) | Cross-system MRR@K comparison | [Section 7.4](./agemem_technical_specification.md#74-competitor-benchmarks) |

### Appendix E: Data Structure Reference

Core data structures per [Section 4.2](./agemem_technical_specification.md#42-data-structures):

**MemoryEntry (LTM Core)**
```python
@dataclass
class MemoryEntry:
    content: str              # The actual memory content
    entry_id: str             # SHA1 hash (auto-generated)
    created_at: float         # Unix timestamp
    updated_at: float         # Last modification time
    access_count: int         # Times retrieved
    learning_score: float     # Aggregated novelty signal (0-1)
    tags: list[str]           # User/agent assigned tags
    source_turn: int          # Conversation turn origin
```

**ContextMessage (STM Core)**
```python
@dataclass
class ContextMessage:
    role: str                 # "system", "user", "assistant"
    content: str              # Message content
    is_pinned: bool           # Protected from eviction per [Pinned Protection invariant](./agemem_technical_specification.md#84-critical-invariants)
    relevance_score: float    # Computed relevance (0-1)
    token_count: int          # Estimated token count
    metadata: dict            # Additional context
```

**LearningFeedback (Self-Assessment)**
```python
@dataclass
class LearningFeedback:
    score: float              # 0-1 novelty scale per [Layer 3](./agemem_technical_specification.md#layer-3-learning-score-self-assessed)
    rationale: str            # Agent's self-assessment reasoning
    turn_index: int           # When assessed
    promoted_to_ltm: bool    # Whether promoted
```

---

## 8. Approval

**Prepared by:** AgeMem Evaluation Team
**Reviewed by:** [Reviewer Name]
**Approved by:** [Approver Name]
**Date:** [Approval Date]

---

*This Technical Requirements Specification serves as the authoritative reference for developers to build the AgeMem automated testing suite, ensuring all [Evaluation Objectives (Section 7.1)](./agemem_technical_specification.md#71-evaluation-objectives) and [Evaluation Protocols (Section 7.5)](./agemem_technical_specification.md#75-evaluation-protocol) are programmatically satisfied.*

**Related Documents:**
- [AgeMem Technical Specification](./agemem_technical_specification.md)
- [Architecture Layers](/home/jaco/develops/WORKS/agemem/docs/ARCHITECTURE_LAYERS.md)
- [Memory Enhanced Documentation](/home/jaco/develops/WORKS/agemem/docs/memory_enhanced.md)
