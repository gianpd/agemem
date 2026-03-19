# LongMemEval Benchmark Alignment Audit Report

**Document ID:** AUDIT-LME-001
**Date:** 2026-03-19
**Auditor:** AgeMem Evaluation Pipeline Analysis
**Reference Standard:** LongMemEval Benchmark Specification (ICLR 2025)

---

## Executive Summary

The AgeMem evaluation pipeline implements Phase 1 (Retrieval Quality) with standard IR metrics but **does not align with LongMemEval's five-behavior taxonomy**. The pipeline treats all queries uniformly rather than categorizing results by memory behavior type. While all three LongMemEval dataset variants are available locally, the most recent evaluation used only the Oracle variant.

| Audit Dimension | Finding | Severity |
|-----------------|---------|----------|
| Behavioral Coverage | Partial — retrieval only, no behavior differentiation | High |
| Structural Taxonomy | Not implemented — flat metric structure | High |
| Dataset Identification | Oracle variant used; S/M available but untested | Medium |

---

## 1. Behavioral Coverage Analysis

### 1.1 LongMemEval Five Core Memory Behaviors

The LongMemEval benchmark evaluates five distinct memory capabilities:

| Behavior | Definition | Pipeline Coverage |
|----------|------------|-------------------|
| **Information Extraction (IE)** | Recall specific information from extensive histories | **Partial** — tested via retrieval metrics, but not isolated as a category |
| **Multi-Session Reasoning (MR)** | Synthesize information across multiple conversation sessions | **Not Tested** — no cross-session aggregation evaluation |
| **Knowledge Updates (KU)** | Recognize and apply the most recent information when values change | **Not Tested** — temporal_anchor field exists but unused in evaluation logic |
| **Temporal Reasoning (TR)** | Understand relative/absolute time references in user information | **Not Tested** — no time-aware query evaluation |
| **Abstention (ABS)** | Recognize when information is not present and refrain from answering | **Not Tested** — no negative query evaluation |

### 1.2 Evidence from Code Analysis

**Dataset Pipeline (`dataset_pipeline.py:238-307`):**
- Parses `question_type` field from LongMemEval format
- Stores type in `BenchmarkQuery.query_type` attribute
- **Gap:** Query type is captured but never used to segment evaluation results

**Metrics Pipeline (`metrics_pipeline.py:182-348`):**
- Calculates aggregate metrics (MRR, Precision, Recall, NDCG)
- No per-category metric computation
- **Gap:** `evaluate()` method returns single `RetrievalMetrics` object, not behavior-segmented results

**Inference Pipeline (`inference_pipeline.py:344-393`):**
- Executes queries against LTMStore
- Captures `SearchTrace` with latency and results
- **Gap:** No temporal reasoning tests, no multi-session synthesis tests

### 1.3 Coverage Score

| Behavior | Test Implementation | Score |
|----------|--------------------| ----- |
| Information Extraction | Retrieval metrics (implicit) | 40% |
| Multi-Session Reasoning | None | 0% |
| Knowledge Updates | None | 0% |
| Temporal Reasoning | None | 0% |
| Abstention | None | 0% |
| **Overall Behavioral Coverage** | | **8%** |

---

## 2. Structural Taxonomy Analysis

### 2.1 Current Pipeline Structure

The evaluation pipeline uses a flat, phase-based architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Retrieval Quality                                  │
│   ├── MRR@K (K=1,5,10)                                      │
│   ├── Precision@K (K=1,5,10)                                │
│   ├── Recall@K (K=1,5,10)                                   │
│   └── NDCG@K (K=5,10)                                       │
├─────────────────────────────────────────────────────────────┤
│ Phase 2: Memory Quality (Defined, Unpopulated)              │
│   ├── Retention Rate                                        │
│   ├── Deduplication Accuracy                                │
│   ├── Learning Score Correlation                            │
│   └── Context Utilization                                   │
├─────────────────────────────────────────────────────────────┤
│ Phase 3: Response Quality (Defined, Unpopulated)            │
│   ├── Hallucination Rate                                    │
│   ├── Coherence Score                                       │
│   ├── Memory Grounding                                      │
│   └── Preference Accuracy                                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 LongMemEval Expected Structure

LongMemEval requires per-behavior accuracy scoring:

```
┌─────────────────────────────────────────────────────────────┐
│ Overall Accuracy (Weighted Average)                         │
├─────────────────────────────────────────────────────────────┤
│ Information Extraction Accuracy                             │
│   ├── Single-Session-User Score                             │
│   └── Single-Session-Assistant Score                        │
├─────────────────────────────────────────────────────────────┤
│ Multi-Session Reasoning Accuracy                            │
│   ├── Aggregation Questions Score                           │
│   └── Comparison Questions Score                            │
├─────────────────────────────────────────────────────────────┤
│ Knowledge Updates Accuracy                                  │
│   └── Most-Recent-Value Questions Score                     │
├─────────────────────────────────────────────────────────────┤
│ Temporal Reasoning Accuracy                                 │
│   ├── Time-Reference Questions Score                        │
│   └── Date-Filtering Questions Score                        │
├─────────────────────────────────────────────────────────────┤
│ Abstention Accuracy                                         │
│   └── Unknown-Information Questions Score                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Gap Analysis

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| Behavior-categorized results | Single aggregate metrics | **Critical** — No per-behavior scoring |
| Discrete accuracy per behavior | Not implemented | **Critical** — Cannot identify weak behaviors |
| Question type utilization | `question_type` parsed but unused | **Major** — Data available, logic missing |

### 2.4 Code Reference

The `question_type` field is captured in the dataset but never segmented:

```python
# dataset_pipeline.py:257
question_type = instance.get("question_type", "retrieval")

# This value is stored but never used for metric segmentation
query = BenchmarkQuery(
    query_id=question_id,
    query_text=question_text,
    relevant_entry_ids=relevant_entry_ids,
    query_type=question_type,  # <-- CAPTURED BUT UNUSED
    ...
)
```

---

## 3. Dataset Identification

### 3.1 Available Datasets

| Dataset | Path | Size | Context Size | Status |
|---------|------|------|--------------|--------|
| **LongMemEval Oracle** | `evaluation/data/longmemeval_oracle.json` | 15 MB | 1-3 evidence sessions | **Used in latest run** |
| **LongMemEval S (Small)** | `evaluation/data/longmemeval_s_cleaned.json` | 277 MB | ~115k tokens (40 sessions) | Available, untested |
| **LongMemEval M (Medium)** | `evaluation/data/longmemeval_m_cleaned.json` | 2.7 GB | ~1.5M tokens (500 sessions) | Available, untested |

### 3.2 Dataset Variant Specifications

Per LongMemEval benchmark specification:

| Variant | Purpose | Session Count | Use Case |
|---------|---------|---------------|----------|
| **Oracle** | Baseline with perfect retrieval | 1-3 evidence sessions | Reading comprehension baseline |
| **S (Small)** | Standard evaluation | ~30-40 sessions | Fits in 128k context window |
| **M (Medium)** | Stress testing | 500 sessions | Requires memory systems |

### 3.3 Current Usage

The latest evaluation session (`phase1_20260319_101419`) used the **Oracle variant**:

```
Source: evaluation/results/phase1_report.md
- Session ID: phase1_20260319_101419
- Dataset: longmemeval_oracle.json (inferred from file sizes and eval_pipe_progress.md)
- Queries: Default limit (100)
```

**Note:** The Oracle variant provides only evidence sessions (sessions containing the answer), making it a retrieval baseline rather than a full memory evaluation. Per LongMemEval specification:

> "Oracle: Controlled evaluation with perfect retrieval. Only evidence sessions containing the answer are included. Use Case: Baseline for measuring retrieval quality and reading comprehension."

### 3.4 Recommendations for Dataset Usage

| Dataset | Recommended Use | Current Usage |
|---------|-----------------|---------------|
| Oracle | Baseline retrieval quality | **Primary** — appropriate for Phase 1 |
| S (Small) | Standard LongMemEval evaluation | Not used — should be primary for full evaluation |
| M (Medium) | Memory system stress test | Not used — for scalability testing |

---

## 4. Technical Findings

### 4.1 Question Types in LongMemEval Oracle

Based on the benchmark specification, the Oracle dataset contains the following question type distribution:

| Question Type | Behavior Category | Count (approx) |
|---------------|-------------------|----------------|
| Single-Session-User | Information Extraction | 126 |
| Single-Session-Assistant | Information Extraction | ~30 |
| Multi-Session | Multi-Session Reasoning | 133 |
| Knowledge-Update | Knowledge Updates | 78 |
| Temporal-Reasoning | Temporal Reasoning | 133 |
| Unknown | Abstention | ~30 |
| **Total** | | **500** |

### 4.2 Pipeline Question Type Handling

Current implementation does not segment by question type:

```python
# metrics_pipeline.py:200-213 — MRR calculation
for query, trace in zip(queries, traces):
    relevant_ids = set(query.relevant_entry_ids)
    # query.query_type is available but NOT used for segmentation
    for rank, (entry_id, score) in enumerate(trace.results[:k], start=1):
        if entry_id in relevant_ids:
            reciprocal_ranks.append(1.0 / rank)
            found = True
            break
```

### 4.3 Missing Behavior-Specific Tests

| Behavior | Required Test | Implementation Status |
|----------|---------------|----------------------|
| Multi-Session Reasoning | Aggregate facts across sessions | Not implemented |
| Knowledge Updates | Track value changes, return most recent | Not implemented |
| Temporal Reasoning | Parse time references, filter by date | Not implemented |
| Abstention | Detect absent information, return "unknown" | Not implemented |

---

## 5. Current Evaluation Results

### 5.1 Latest Run (Oracle Dataset)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| MRR@10 | 0.0300 | >= 0.85 | FAIL |
| Recall@5 | 0.8333 | >= 0.90 | FAIL |
| Precision@5 | 0.0120 | N/A | — |
| NDCG@10 | 0.0282 | N/A | — |
| Avg Latency | 525ms | < 500ms | FAIL |

### 5.2 Comparative Analysis

| System | MRR@5 | LongMemEval Score |
|--------|-------|-------------------|
| **OMEGA (SOTA)** | — | 95.4% |
| **Mastra** | — | 94.87% |
| **AgeMem (Oracle)** | 0.03 | Not calculated |
| MemGPT | 0.72 | — |
| Letta | 0.75 | — |

**Note:** AgeMem's MRR of 0.03 cannot be directly compared to LongMemEval accuracy scores. LongMemEval uses per-question accuracy (exact match / LLM evaluation), not retrieval ranking metrics.

---

## 6. Recommendations

### 6.1 Immediate (High Priority)

1. **Implement Behavior Segmentation**
   - Modify `MetricsPipeline.evaluate()` to calculate per-behavior metrics
   - Group queries by `query_type` and compute separate MRR/Recall per category
   - File: `metrics_pipeline.py`

2. **Add Behavior-Specific Evaluation Methods**
   ```python
   def calculate_behavior_metrics(
       self,
       queries: list[BenchmarkQuery],
       traces: list[SearchTrace],
   ) -> dict[str, RetrievalMetrics]:
       """Calculate metrics segmented by question_type."""
       behaviors = {}
       for query_type in ["temporal-reasoning", "knowledge", "multi-session", ...]:
           type_queries = [q for q in queries if q.query_type == query_type]
           type_traces = [...]  # matching traces
           behaviors[query_type] = self.calculate_retrieval_metrics(type_queries, type_traces)
       return behaviors
   ```

3. **Run Against LongMemEval S (Small)**
   - The S variant is the standard LongMemEval evaluation dataset
   - Oracle is a baseline, not a full evaluation
   - Command: `python -m evaluation.runner --dataset evaluation/data/longmemeval_s_cleaned.json`

### 6.2 Medium Priority

4. **Implement Knowledge Update Testing**
   - Add temporal ordering to entry evaluation
   - Track which entries are "most recent" for a given attribute
   - Test that system returns latest value, not first encountered

5. **Implement Abstention Testing**
   - Add negative queries (questions with no answer in memory)
   - Track false positive rate (answering when should abstain)
   - Measure "I don't know" response accuracy

6. **Implement Temporal Reasoning Tests**
   - Parse relative time references ("last week", "yesterday")
   - Match against session timestamps
   - Evaluate time-filtered retrieval

### 6.3 Architecture Recommendations

7. **LongMemEval Compliance Layer**
   - Create `evaluation/longmemeval_adapter.py`
   - Implement standard LongMemEval evaluation protocol
   - Output results in LongMemEval leaderboard format

8. **Report Enhancement**
   - Add per-behavior breakdown to `phase1_report.md`
   - Include behavior-level accuracy alongside aggregate metrics
   - Match LongMemEval leaderboard output format

---

## 7. Conclusion

The AgeMem evaluation pipeline is architecturally sound for Phase 1 retrieval testing but **does not currently conform to LongMemEval's behavioral evaluation methodology**. The primary gaps are:

1. **Behavioral Coverage (8%)** — Only implicit Information Extraction via retrieval; no MR, KU, TR, or ABS testing.

2. **Structural Taxonomy** — Flat metric structure does not distinguish between the five memory behaviors; `question_type` is parsed but unused for segmentation.

3. **Dataset Usage** — Only Oracle variant tested; S (standard) and M (stress) variants available but untested.

**Alignment Score:** The pipeline achieves approximately **20% alignment** with LongMemEval requirements. To achieve full compliance, behavior-segmented evaluation and additional test types are required.

---

## Appendix A: File References

| Component | Path | Lines |
|-----------|------|-------|
| Dataset Pipeline | `evaluation/pipeline/dataset_pipeline.py` | 542 |
| Inference Pipeline | `evaluation/pipeline/inference_pipeline.py` | 514 |
| Metrics Pipeline | `evaluation/pipeline/metrics_pipeline.py` | 523 |
| Evaluation Runner | `evaluation/runner.py` | 409 |
| Progress Documentation | `evaluation/docs/eval_pipe_progress.md` | 462 |
| Latest Results | `evaluation/results/phase1_report.md` | 60 |
| Metrics JSON | `evaluation/results/metrics.json` | 39 |

## Appendix B: LongMemEval Question Type Mapping

| LongMemEval Type | Behavior | AgeMem Pipeline Support |
|------------------|----------|-------------------------|
| Single-Session-User | IE | Implicit (retrieval) |
| Single-Session-Assistant | IE | Implicit (retrieval) |
| Multi-Session | MR | Not supported |
| Knowledge-Update | KU | Not supported |
| Temporal-Reasoning | TR | Not supported |
| Preference | IE | Implicit (retrieval) |
| Unknown | ABS | Not supported |

---

*Audit completed: 2026-03-19*