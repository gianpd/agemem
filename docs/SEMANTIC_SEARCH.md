# Semantic Search Implementation Guide

Research date: 2026-03-11

## Executive Summary

For AgeMem's LTM system, the recommended stack is:

- **Embedding Model**: `Qwen/Qwen3-Embedding-0.6B` (primary) or `BAAI/bge-m3` (fallback)
- **Vector Storage**: `sqlite-vec` extension (zero new dependencies, embedded)
- **Model Caching**: HuggingFace local cache with configurable path
- **Similarity**: Cosine similarity via native SQL functions

This combination provides semantic retrieval with minimal complexity overhead, scaling from hundreds to tens of thousands of entries without architectural changes.

---

## Embedding Model Selection

### Top Contenders for Local Deployment

| Model | MTEB Retrieval | Parameters | Context | Dimensions | Best For |
|-------|---------------|------------|---------|------------|----------|
| **Qwen3-Embedding-0.6B** | 61.41 nDCG@10 | 0.6B | 32K | 32-1024 | Best quality/size ratio |
| **BAAI/bge-m3** | 54.0 avg | 568M | 8192 | 1024 | Multilingual, proven |
| **Qwen3-Embedding-8B** | 70.58 | 8B | 32K | 32-1024 | #1 MTEB multilingual |
| **all-MiniLM-L6-v2** | 42.0 | 46M | 512 | 384 | Lightweight, prototyping |

### Recommendation: Qwen3-Embedding-0.6B

**Verified Specifications (from HuggingFace model card):**

- **Parameters**: 0.6B (600M)
- **Context Length**: 32,768 tokens
- **Embedding Dimensions**: Flexible 32-1024 (Matryoshka support)
- **Languages**: 100+ natural and programming languages
- **License**: Apache 2.0
- **Release**: June 2025

**MTEB Benchmarks:**

| Category | Score |
|----------|-------|
| Multilingual Mean | 64.33 |
| Retrieval (nDCG@10) | 61.41 |
| Classification | 66.83 |
| Reranking (MAP) | 64.64 |
| STS (Spearman) | 76.17 |

**Why Qwen3-Embedding-0.6B:**

1. **Best quality/size ratio**: Outperforms bge-m3 with same parameter count
2. **Long context**: 32K tokens handles full documents, not just snippets
3. **Matryoshka embeddings**: Flexible dimensions (32-1024) for speed/accuracy tradeoff
4. **Instruction-aware**: 1-5% boost with task-specific instructions
5. **Modern architecture**: Released June 2025, state-of-the-art techniques

**Fallback: BAAI/bge-m3**

For maximum stability with proven production track record. Slightly lower benchmarks but battle-tested.

**Lightweight: all-MiniLM-L6-v2**

For constrained environments (edge, low memory), this 46MB model is viable for prototyping. Trade-off: ~30% lower retrieval quality.

### Model Loading with Caching

```python
import os
from sentence_transformers import SentenceTransformer

# Configure cache location (do this BEFORE loading model)
CACHE_DIR = os.path.expanduser("~/.cache/agemem/models")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# Load model (downloads to cache on first run, uses cache thereafter)
model = SentenceTransformer(
    'Qwen/Qwen3-Embedding-0.6B',
    cache_folder=CACHE_DIR,  # Explicit cache folder
    trust_remote_code=True   # Required for Qwen models
)

# Generate embeddings
embeddings = model.encode(
    ["text1", "text2"],
    normalize_embeddings=True,  # For cosine similarity
    show_progress_bar=False
)
```

### Cache Directory Structure

```
~/.cache/agemem/models/
├── models--Qwen--Qwen3-Embedding-0.6B/
│   ├── snapshots/
│   │   └── <commit-hash>/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer.json
│   │       └── ...
│   ├── refs/
│   │   └── main  -> points to current snapshot
│   └── blobs/
│       └── <sha256-hashed-files>
```

### Fast Model Initialization

For production use, pre-download models to avoid startup latency:

```python
# Pre-download script (run once during setup)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'Qwen/Qwen3-Embedding-0.6B',
    cache_folder="~/.cache/agemem/models",
    trust_remote_code=True
)
print(f"Model cached at: ~/.cache/agemem/models")

# Subsequent loads are instant (<1 second)
# First load: ~30 seconds (download + load)
# Cached load: ~2 seconds (just load into memory)
```

### Environment Configuration

```python
# In config.py or .env
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_CACHE_DIR = os.path.expanduser("~/.cache/agemem/models")
EMBEDDING_DIMENSION = 1024  # Can reduce to 512 or 256 for speed

# Optional: Set globally for all HuggingFace operations
os.environ["HF_HOME"] = EMBEDDING_CACHE_DIR
```

---

## Vector Database Selection

### Comparison Matrix

| Database | Type | Algorithm | Query Speed | Scale | Complexity |
|----------|------|-----------|-------------|-------|------------|
| **sqlite-vec** | Embedded | Brute-force | Good | <100K | Very Low |
| **vectorlite** | Embedded | HNSW | Excellent | <1M | Low |
| **ChromaDB** | Embedded/Server | HNSW | Excellent | <10M | Low |
| **LanceDB** | Embedded | IVF-PQ | Excellent | <10M | Low |
| **Qdrant** | Server | HNSW | Excellent | >10M | Medium |

### Decision: sqlite-vec

**Why for AgeMem:**

1. **Zero new infrastructure**: AgeMem already uses SQLite for LTM persistence
2. **Transactional**: Embedding updates happen in same transaction as content updates
3. **Portable**: Same `.db` file works everywhere SQLite runs
4. **Simple API**: Just SQL functions, no new query language
5. **No background processes**: Unlike ChromaDB server mode

**Algorithm Note:**

sqlite-vec uses **brute-force (exact) search**, not HNSW. This means:

- **Pros**: Exact results, no accuracy loss, simple implementation
- **Cons**: O(n) query time, slower for >100K vectors

For AgeMem's expected scale (<10K LTM entries initially), brute-force is sufficient and simpler.

**When to upgrade to HNSW:**

If LTM exceeds 100K entries or query latency becomes problematic, migrate to **vectorlite**:

```bash
pip install vectorlite
```

```python
# vectorlite with HNSW (2x-40x faster queries than sqlite-vec)
import vectorlite
db.enable_load_extension(True)
vectorlite.load(db)

# Create HNSW index
db.execute("""
    CREATE VIRTUAL TABLE ltm_hnsw
    USING vectorlite(
        embedding FLOAT[1024],
        hnsw(max_elements=100000, M=16, ef_construction=200)
    )
""")
```

### sqlite-vec Implementation

```python
import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32
import numpy as np

# Setup
db = sqlite3.connect("ltm.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Check version
version = db.execute("SELECT vec_version()").fetchone()[0]
print(f"sqlite-vec version: {version}")

# Create vector table
db.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS ltm_embeddings
    USING vec0(
        entry_id TEXT PRIMARY KEY,
        embedding FLOAT[1024]
    )
""")

# Insert embedding
embedding = model.encode("memory content", normalize_embeddings=True)
db.execute(
    "INSERT INTO ltm_embeddings VALUES (?, ?)",
    [entry_id, serialize_float32(embedding)]
)

# Query similar (brute-force cosine distance)
query_emb = model.encode("search query", normalize_embeddings=True)
results = db.execute("""
    SELECT entry_id, vec_distance_cosine(embedding, ?) as distance
    FROM ltm_embeddings
    ORDER BY distance
    LIMIT 10
""", [serialize_float32(query_emb)]).fetchall()
```

### Schema Integration

```sql
-- Augment existing ltm_entries table
ALTER TABLE ltm_entries ADD COLUMN embedding BLOB;
ALTER TABLE ltm_entries ADD COLUMN embedding_model TEXT DEFAULT 'Qwen3-Embedding-0.6B';
ALTER TABLE ltm_entries ADD COLUMN embedding_dim INTEGER DEFAULT 1024;

-- Create virtual table for fast similarity
CREATE VIRTUAL TABLE ltm_vec_index
USING vec0(
    entry_id TEXT PRIMARY KEY,
    embedding FLOAT[1024]
);
```

---

## Best Practices for Agent Memory

### 1. Embedding Generation Strategy

**When to embed:**

- On `add_entry()`: Generate embedding immediately (adds ~50ms latency)
- Batch embedding for existing entries: Use background migration

**What to embed:**

```python
# Embed the core content, not metadata
embedding_text = f"{entry['content']}"

# For Qwen3, can add instruction prefix for better retrieval
# embedding_text = "Represent this memory for retrieval: " + entry['content']
```

**Normalization:**

Always normalize embeddings for cosine similarity:

```python
embeddings = model.encode(texts, normalize_embeddings=True)
# Now dot product == cosine similarity
```

### 2. Retrieval Pipeline

```python
def retrieve_relevant_ltm(query: str, top_k: int = 10) -> list[dict]:
    """Two-stage retrieval for best results."""

    # Stage 1: Semantic retrieval (broad)
    query_emb = model.encode(query, normalize_embeddings=True)
    candidates = db.execute("""
        SELECT e.*, v.distance
        FROM ltm_entries e
        JOIN ltm_vec_index v ON e.id = v.entry_id
        ORDER BY v.distance
        LIMIT ?
    """, [serialize_float32(query_emb), top_k * 3]).fetchall()

    # Stage 2: Recency decay + tag filtering (narrow)
    scored = []
    for entry in candidates:
        score = entry['distance']
        # Apply recency decay
        age_hours = (now - entry['created_at']).total_seconds() / 3600
        recency_factor = 1.0 / (1.0 + 0.01 * age_hours)
        score *= recency_factor
        scored.append((score, entry))

    return sorted(scored, key=lambda x: x[0])[:top_k]
```

### 3. Memory Update Strategy

When LTM entries are updated:

```python
def update_entry(entry_id: str, new_content: str):
    """Update content and re-embed."""
    new_embedding = model.encode(new_content, normalize_embeddings=True)

    with db:
        db.execute("UPDATE ltm_entries SET content = ? WHERE id = ?",
                   [new_content, entry_id])
        db.execute("UPDATE ltm_vec_index SET embedding = ? WHERE entry_id = ?",
                   [serialize_float32(new_embedding), entry_id])
```

### 4. Hybrid Search (Future Enhancement)

Combine semantic + keyword matching:

```python
def hybrid_search(query: str, top_k: int = 10) -> list:
    """Combine BM25-style keyword with semantic."""

    # Semantic results
    sem_results = semantic_search(query, top_k * 2)

    # Keyword results (existing token overlap)
    kw_results = keyword_search(query, top_k * 2)

    # Reciprocal rank fusion
    return rrf_fusion(sem_results, kw_results, top_k)
```

---

## Implementation Phases

### Phase 1: Core Integration (Week 1)

- [ ] Add `sqlite-vec` to dependencies
- [ ] Add embedding columns to LTM schema
- [ ] Configure model cache directory
- [ ] Implement `embed_entry()` function
- [ ] Replace `token_overlap_search()` with `semantic_search()`

### Phase 2: Migration (Week 2)

- [ ] Batch-embed existing LTM entries
- [ ] Add embedding cache for repeated queries
- [ ] Benchmark retrieval quality vs. old system

### Phase 3: Optimization (Week 3)

- [ ] Add hybrid search (semantic + keyword)
- [ ] Implement query embedding caching
- [ ] Add metrics: retrieval latency, precision@k
- [ ] Evaluate vectorlite for HNSW if scale warrants

---

## Metrics to Track

| Metric | Target | Measurement |
|--------|--------|-------------|
| Retrieval Precision@10 | ≥0.8 | Manual eval with known facts |
| Retrieval Recall | ≥0.9 | Can LTM find injected facts? |
| Query Latency P95 | <100ms | Time from query to results |
| Embedding Generation | <50ms | Time to embed single entry |
| Model Load Time (cached) | <2s | Time to load model from cache |
| Memory Utilization | Track trend | % of entries ever retrieved |

---

## Dependencies

```txt
# requirements.txt additions
sentence-transformers>=3.0.0
sqlite-vec>=0.1.0
numpy>=1.24.0

# Optional: for HNSW when scaling
# vectorlite>=0.2.0
```

---

## References

- [Qwen3-Embedding-0.6B on HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) - Model card
- [Qwen3-Embedding GitHub](https://github.com/QwenLM/Qwen3-Embedding) - Official repo
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding model benchmarks
- [sqlite-vec Documentation](https://alexgarcia.xyz/sqlite-vec/python.html) - Python integration
- [vectorlite GitHub](https://github.com/1yefuwang1/vectorlite) - HNSW for SQLite
- [HuggingFace Cache Documentation](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) - Cache management
- [Sentence-Transformers Documentation](https://www.sbert.net/) - Usage guide