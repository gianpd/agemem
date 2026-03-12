# AgeMem — Query Expansion Tool: Software Specification

**Feature:** `tools/query_expansion.py`  
**Status:** ✅ SHIPPED  
**Date:** 2026-03-12  
**Target branch:** `main`

---

## 1. Problem Statement

The current retrieval pipeline in `memory/ltm_store.py` uses two paths:

1. **Semantic path** (Phase 2, when `ENABLE_SEMANTIC_SEARCH=True`): `sqlite-vec` KNN over Qwen3-Embedding-0.6B vectors
2. **Lexical fallback** (always available): token-overlap scoring with recency and learning-score weighting

Both paths share the same entry point: the raw query string passed by the orchestrator.

**The gap:** Neither path handles paraphrase distance — cases where the query and stored entry share no surface tokens and the cosine distance is high because the semantic models diverge in embedding space for domain-specific synonyms. This is not a model quality problem; it is a coverage problem. The solution is query expansion: generate N paraphrase variants of the query before retrieval, run all variants, and merge results.

**Why not full vector search replacement?** The existing GLiNER NER enrichment at ingestion time has already resolved most of the classic "grep misses synonyms" gap. Regex over NER-normalized entity types handles morphological variance. The remaining gap is narrow and specific: distributional synonyms that neither the model's embedding space nor any entity normalizer bridges. Query expansion targets exactly that gap without increasing index complexity.

---

## 2. Architecture

### 2a. Position in the stack

```
orchestrator.py
    └── ltm_store.search(query)
            └── [NEW] query_expansion.expand(query)  →  [q0, q1, q2, ..., qN]
                        └── for each qi: _semantic_search(qi) OR _token_overlap_search(qi)
                                └── merge_results([results_q0, results_q1, ...])
                                        └── deduplicate by entry_id
                                                └── re-rank by best_score
```

The expansion is **transparent to the orchestrator** — `ltm_store.search()` signature does not change.

### 2b. New file

**`tools/query_expansion.py`**

Sits alongside existing `tools/web_tools.py`. Self-contained, no circular imports. Can be called independently by any agent tool or the retrieval pipeline.

### 2c. Integration point

`memory/ltm_store.py` → `search()` method.  
One import, one call, one merge. No schema changes. No new dependencies beyond what Phase 2 already added.

---

## 3. Specification: `tools/query_expansion.py`

### 3a. Public interface

```python
class QueryExpander:
    """
    Generates paraphrase variants of a query string using a local LLM
    via the existing LLMClient. Falls back to regex-based expansion
    if the LLM call fails or returns malformed output.
    
    All expansion is done at query time, not at index time.
    Zero schema changes required.
    """

    def __init__(
        self,
        llm_client: LLMClient,                  # existing agents/llm_client.py
        model: str,                              # inherits from AgememConfig.MEMORY_AGENT_MODEL
        n_variants: int = 3,                     # total expansions including original
        use_ner_hints: bool = True,              # inject NER entity types into prompt
        fallback_on_error: bool = True,          # return [original] on LLM failure
        timeout_ms: int = 2000,                  # abort expansion if LLM takes > 2s
    ): ...

    def expand(
        self,
        query: str,
        ner_entities: list[dict] | None = None,  # optional: GLiNER output for the query
    ) -> list[str]:
        """
        Returns [original_query, variant_1, ..., variant_N].
        First element is always the unmodified original.
        Never raises — returns [query] on any failure.
        """
        ...
```

### 3b. LLM prompt (system)

```
You are a query expansion assistant for a document retrieval system.
Given a search query, generate {n} alternative phrasings that express
the same intent using different vocabulary. Focus on:
- Domain synonyms and technical aliases
- Nominalization variants (e.g. "throttle requests" → "request throttling")
- Abbreviation expansion and contraction
- Passive/active voice alternates for action queries

Return ONLY a JSON array of strings. No explanation. No markdown.
Example output: ["variant one", "variant two", "variant three"]
```

### 3c. LLM prompt (user)

```
Query: {query}
{ner_block}
Generate {n_variants - 1} alternative phrasings.
```

Where `ner_block` (injected only when `use_ner_hints=True` and entities present):
```
Known entities in query: {entity_type}: "{entity_text}", ...
```

This gives the model grounding to expand around the right semantic axis rather than hallucinating unrelated synonyms.

### 3d. Fallback expansion (no LLM / timeout)

When LLM is unavailable or `timeout_ms` exceeded, apply deterministic regex transformations:

| Transform | Example |
|---|---|
| Verb → nominal | `"authenticate user"` → `"user authentication"` |
| Nominal → verb phrase | `"rate limiting"` → `"limit the rate"`, `"apply rate limit"` |
| Add "how to" prefix | `"deploy container"` → `"how to deploy container"` |
| Acronym expansion (config-supplied dict) | `"LTM"` → `"long term memory"` |

These are cheap, deterministic, and cover the most common English nominalisation patterns that trip up token-overlap search.

Fallback is controlled by `AgememConfig.QUERY_EXPANSION_FALLBACK_TRANSFORMS` — a list of enabled transform names.

---

## 4. Specification: `memory/ltm_store.py` changes

### 4a. `search()` method update

```python
def search(
    self,
    query: str,
    top_k: int | None = None,
    *,
    expand_query: bool | None = None,   # None = respect config default
) -> list[MemoryEntry]:

    effective_top_k = top_k or self.config.LTM_SEARCH_TOP_K
    should_expand = (
        expand_query
        if expand_query is not None
        else self.config.ENABLE_QUERY_EXPANSION
    )

    if should_expand and self._expander is not None:
        queries = self._expander.expand(query)
    else:
        queries = [query]

    all_results: dict[str, tuple[MemoryEntry, float]] = {}

    for q in queries:
        if self._semantic_enabled:
            results = self._semantic_search(q, top_k=effective_top_k)
        else:
            results = self._token_overlap_search(q, top_k=effective_top_k)

        for entry, score in results:
            if entry.entry_id not in all_results or score < all_results[entry.entry_id][1]:
                all_results[entry.entry_id] = (entry, score)  # keep best (lowest) distance

    merged = sorted(all_results.values(), key=lambda x: x[1])
    return [entry for entry, _ in merged[:effective_top_k]]
```

**Key invariant:** deduplication by `entry_id`, best score wins. The original query always runs first — if expansion fails silently, behaviour is identical to today.

### 4b. Constructor update

```python
# Existing:
def __init__(self, config: AgememConfig, llm_client: LLMClient | None = None):

# Add after semantic backend init:
self._expander: QueryExpander | None = None
if config.ENABLE_QUERY_EXPANSION and llm_client is not None:
    self._expander = QueryExpander(
        llm_client=llm_client,
        model=config.MEMORY_AGENT_MODEL,
        n_variants=config.QUERY_EXPANSION_N_VARIANTS,
        use_ner_hints=config.QUERY_EXPANSION_USE_NER_HINTS,
        timeout_ms=config.QUERY_EXPANSION_TIMEOUT_MS,
    )
```

---

## 5. Specification: `core/config.py` additions

```python
# --- Query Expansion ---
ENABLE_QUERY_EXPANSION: bool = False          # opt-in, safe default
QUERY_EXPANSION_N_VARIANTS: int = 3           # total queries including original
QUERY_EXPANSION_USE_NER_HINTS: bool = True    # inject GLiNER entities into prompt
QUERY_EXPANSION_TIMEOUT_MS: int = 2000        # LLM timeout before fallback
QUERY_EXPANSION_FALLBACK_TRANSFORMS: list[str] = field(
    default_factory=lambda: ["nominalize", "add_how_to"]
)
QUERY_EXPANSION_ACRONYM_DICT: dict[str, str] = field(
    default_factory=dict                      # user-supplied, e.g. {"LTM": "long term memory"}
)
```

---

## 6. Specification: `agents/orchestrator.py` changes

### 6a. Pass `llm_client` to `LTMStore`

The orchestrator already holds `self.llm`. It currently initializes LTMStore without passing the client:

```python
# Current (approximate):
self.ltm = LTMStore(config=self.config)

# Updated:
self.ltm = LTMStore(config=self.config, llm_client=self.llm)
```

This is the only orchestrator change required.

### 6b. NER hint passthrough (optional, Phase 2 of this feature)

If the ingest pipeline already runs GLiNER and stores entity metadata alongside LTM entries, the search call can pass NER hints:

```python
# Optional enhancement — not required for v1
entities = self._extract_query_entities(query)   # call GLiNER on the query itself
results = self.ltm.search(query, ner_entities=entities)
```

Leave this wired as `None` for v1. The `QueryExpander` handles `None` gracefully.

---

## 7. Specification: `tests/test_query_expansion.py`

### Required test cases

| ID | Name | What it tests |
|---|---|---|
| T1 | `test_expand_returns_original_first` | `result[0] == query` always |
| T2 | `test_expand_returns_n_variants` | `len(result) == n_variants` when LLM succeeds |
| T3 | `test_expand_never_raises` | wrap in try/except, assert no exception on bad LLM response |
| T4 | `test_expand_fallback_on_timeout` | mock LLM to sleep > timeout_ms, assert returns [query] |
| T5 | `test_expand_fallback_on_malformed_json` | mock LLM returns `"not json"`, assert returns [query] |
| T6 | `test_nominalize_transform` | `"authenticate user"` → includes `"user authentication"` |
| T7 | `test_acronym_expansion` | config dict `{"API": "application programming interface"}`, query `"API key"` → expanded |
| T8 | `test_search_deduplication` | two variants return same entry_id, merged result has it once |
| T9 | `test_search_best_score_wins` | same entry returned by two variants with different scores, lower score kept |
| T10 | `test_search_expand_false_bypasses_expander` | `search(q, expand_query=False)` calls expander zero times |
| T11 | `test_search_disabled_config` | `ENABLE_QUERY_EXPANSION=False` → expander never called |
| T12 | `test_ner_hints_injected_in_prompt` | mock LLM, assert prompt contains entity text when `use_ner_hints=True` |

All tests must run without a real LLM — mock `LLMClient.chat()` via `unittest.mock.patch`.

---

## 8. File change summary

| File | Change type | Description |
|---|---|---|
| `tools/query_expansion.py` | **NEW** | `QueryExpander` class, LLM + regex fallback |
| `memory/ltm_store.py` | MODIFIED | `search()` multi-query loop + merge; `__init__` wires expander |
| `core/config.py` | MODIFIED | 6 new config fields under `# --- Query Expansion ---` |
| `agents/orchestrator.py` | MODIFIED | Pass `llm_client` to `LTMStore.__init__` |
| `tests/test_query_expansion.py` | **NEW** | 12 unit tests, all offline |
| `memory/__init__.py` | MODIFIED | Export `QueryExpander` if needed externally |
| `pyproject.toml` | NO CHANGE | No new dependencies — uses existing `sentence-transformers`, `sqlite-vec`, `openai` |

---

## 9. Execution order for the agent

Ship in this exact order to avoid broken intermediate states:

```
1. core/config.py          — add fields (no logic, safe first)
2. tools/query_expansion.py — new file, no imports from ltm_store
3. tests/test_query_expansion.py — write tests before wiring
4. memory/ltm_store.py     — wire expander into search()
5. agents/orchestrator.py  — pass llm_client to LTMStore
6. memory/__init__.py       — add exports
7. Run: pytest tests/test_query_expansion.py -v
8. Run: pytest tests/test_agemem.py -v  (regression — must still pass 28/28)
```

---

## 10. Acceptance criteria

- [ ] All 12 new tests pass
- [ ] All 28 existing tests still pass
- [ ] `ENABLE_QUERY_EXPANSION=False` (default) produces zero behavioural change vs. current codebase
- [ ] `ENABLE_QUERY_EXPANSION=True` with mocked LLM returning 2 variants: search calls `_semantic_search` (or `_token_overlap_search`) exactly 3 times (original + 2 variants)
- [ ] LLM timeout or JSON parse failure: `search()` returns results identical to today (single-query path)
- [ ] No new required dependencies in `pyproject.toml`

---

## 11. Design decisions and rationale

**Why not expand at index time?** Expanding at index time would require re-embedding on every LTM add/update, multiplying index size by N. It would also require choosing expansion variants without knowing future query vocabulary. Query-time expansion is cheaper, more flexible, and doesn't touch the schema.

**Why N=3 default?** Empirically, the marginal recall gain of variant 4+ drops fast relative to LLM inference cost. Three total queries (original + 2 variants) hits the knee of the recall/cost curve for short domain queries.

**Why `ENABLE_QUERY_EXPANSION=False` default?** The feature adds an LLM call per retrieval. For agents running against small LTM stores (< 100 entries), the token-overlap search is fast and precise enough that expansion is not worth the latency. The default keeps existing behaviour unchanged until the user opts in.

**Why keep the original query as `result[0]`?** Determinism. If all LLM expansion fails, the caller always gets at least what it would have gotten without expansion. The merge step cannot make results worse than the baseline.

**Why timeout at 2000ms?** Agent turns are latency-sensitive. The memory retrieval step should not become the bottleneck. At 2s, even a slow local Ollama call has time to respond; anything slower is treated as degraded and falls back.

Given your specific stack, here's the precise metric to use:

---

## The Right Metric: **MRR@K with Query Variant Attribution**

Not generic NDCG or Recall@K. Here's why and how.

### The metric

**Mean Reciprocal Rank at K (MRR@K)** measured separately for:

1. **Original query only** — baseline, what you have today
2. **Expanded query set** — after merging variants
3. **Per-variant contribution** — which variant was the one that actually retrieved the winning entry

```python
MRR@K = (1/|Q|) * Σ (1 / rank_of_first_relevant_result)
```

Where "relevant" is defined by `learning_score >= LTM_PROMOTE_THRESHOLD` on the retrieved entry — you already have this signal in your schema, no human labelling needed.

---

### Why MRR and not Recall@K

Recall@K tells you *if* the right entry appeared in the top K. MRR tells you *where* it ranked. For an agent memory system, position matters: the entry at rank 1 goes into the LLM context with full weight; rank 5 may get truncated. MRR penalises correct-but-late retrievals, which Recall doesn't.

---

### The attribution dimension (your unique advantage)

Because you run N queries and merge, you can log *which query index* produced each result:

```python
{
  "entry_id": "abc123",
  "best_score": 0.21,
  "source_query_index": 2,   # 0 = original, 1+ = variants
  "source_query_text": "request throttling mechanism",
  "original_query": "rate limit enforcement"
}
```

This gives you a **variant hit rate** metric — the fraction of top-1 results that came from a variant rather than the original. If this is consistently 0, expansion is adding latency with no recall benefit and you should tune the prompt or reduce N. If it's above ~15%, expansion is earning its cost.

---

### What to instrument in `ltm_store.search()`

```python
@dataclass
class SearchTrace:
    query_original: str
    queries_expanded: list[str]
    results_per_query: list[list[tuple[str, float]]]   # entry_id, score per query
    merged_results: list[tuple[str, float]]
    top1_source_query_index: int                        # 0 = original won
    latency_expansion_ms: float
    latency_retrieval_ms: float
    expansion_succeeded: bool                           # False = fell back
```

Store this trace in a `search_traces` SQLite table alongside your existing `ltm_entries`. No external observability stack needed — it's already SQLite.

---

### The comparison you can run today without labels

Since `learning_score` is already on every entry, define a proxy ground truth:

> **A retrieval is "correct"** if the top-1 returned entry has `learning_score >= config.LTM_PROMOTE_THRESHOLD` AND `access_count >= 2`

The `access_count` guard filters out entries that were stored but never confirmed useful by the agent's own retrieval behaviour. This gives you a self-supervised signal from data you already have.

Run two modes in shadow mode (both paths execute, only one is returned to the agent) for 50–100 agent turns, then compare:

```
MRR@3 (original only)  vs  MRR@3 (expanded)
Variant hit rate
Mean latency_expansion_ms
Expansion fallback rate (LLM timeout %)
```

If `MRR@3(expanded) - MRR@3(original) < 0.03` after 100 turns, the gap is too narrow to justify the latency. That's your kill switch criterion — put it in the spec as an explicit rollback condition.