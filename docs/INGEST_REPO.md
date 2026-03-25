# Ingest Repository & Storage Framework

This document describes the AgeMem ingest system as a resource repository and storage framework, mapping operations to the ADD/UPDATE/GET/DELETE paradigm.

## Overview

AgeMem implements two complementary storage systems:

1. **Corpus Store** — File-based document repository with entity extraction
2. **Memory Store** — JSON/SQLite-based long-term memory with semantic search

Both systems expose CRUD-style operations, though with different storage backends and access patterns.

---

## 1. Corpus Store (Document Repository)

### Storage Backend

- **Location**: `corpus/` directory
- **Format**: Markdown files with YAML frontmatter
- **Index**: `corpus/_index.yaml` (document registry)

### Document Schema

Each document is stored as `corpus/{doc_id}.md`:

```yaml
---
doc_id: report_abc123
doc_title: "Document Title"
doc_type: contract
source_file: /path/to/original.pdf
source_hash: sha256:abc123...
doc_date: 2024-01-15
ingested_at: 2024-01-15T10:30:00
ner_config: edilizia
entities:
  people: [John Smith, Jane Doe]
  orgs: [Acme Corp]
  dates: [2024-01-15]
  locations: [Milan, Rome]
sections: [Introduction, Methods, Results]
page_count: 15
has_tables: true
has_figures: false
---

# Document content in markdown...
```

### CRUD Operations

| Operation | Method | File | Description |
|-----------|--------|------|-------------|
| **ADD** | `ingest()` | `ingest/ingest.py:913` | Ingest PDF into corpus |
| **ADD** | `ingest_markdown()` | `ingest/ingest.py:858` | Ingest markdown file |
| **ADD** | `write_document()` | `ingest/ingest.py:777` | Write document to corpus |
| **ADD** | `ingest_document()` | `tools/corpus.py` | Agent-callable wrapper |
| **GET** | `read_document()` | `tools/corpus.py:296` | Read full document by ID |
| **GET** | `read_lines()` | `tools/corpus.py:315` | Read specific line range |
| **LIST** | `list_documents()` | `tools/corpus.py:148` | List all corpus documents |
| **SEARCH** | `search_metadata()` | `tools/corpus.py:163` | Search document metadata |
| **SEARCH** | `grep_corpus()` | `tools/corpus.py:186` | Search document body text |
| **UPDATE** | Re-ingest | — | Overwrite by re-running ingest |
| **DELETE** | Manual | — | Delete `corpus/{doc_id}.md` |

### ADD Operation Details

#### `ingest(pdf_path, doc_type, labels_arg, ...)` — Main Entry Point

```python
def ingest(
    pdf_path: str,
    doc_type: str = "document",
    labels_arg: Optional[str] = None,
    auto_detect_ocr: bool = True,
    force_ocr: bool = False,
    fast_mode: bool = True,
    disable_tables: bool = False,
    post_process: bool = True,
    post_process_config: str = "default",
    enable_multiscale: Optional[bool] = None,
) -> str:
```

**Pipeline**:
1. Load label configuration (`load_labels()`)
2. Parse PDF via Docling (`parse_pdf()`) → markdown + sections
3. Extract entities via GLiNER (`extract_entities()`)
4. Write to corpus (`write_document()`)
5. Update index (`update_index()`)

**Returns**: `doc_id` string

#### `write_document(pdf, markdown, sections, entities, doc_type, label_config)` — Storage

```python
def write_document(
    pdf: Path,
    markdown: str,
    sections: List[str],
    entities: Dict[str, List[str]],
    doc_type: str,
    label_config: Dict[str, Any],
) -> Path:
```

Generates:
- Unique `doc_id` from filename + content hash
- YAML frontmatter with metadata
- Markdown body

### GET Operation Details

#### `read_document(doc_id)` — Full Document Read

```python
def read_document(doc_id: str) -> str:
```

Returns full document content (truncated at 8000 chars for large documents).

#### `read_lines(doc_id, start_line, end_line)` — Partial Read

```python
def read_lines(doc_id: str, start_line: int, end_line: int) -> str:
```

Returns specific line range (max 75 lines per call). Line numbers are 1-indexed.

### SEARCH Operations

#### `search_metadata(keyword)` — Frontmatter Search

```python
def search_metadata(keyword: str) -> str:
```

Searches YAML frontmatter fields (titles, types, entities). Returns JSON with matching documents.

#### `grep_corpus(pattern, context_lines)` — Body Text Search

```python
def grep_corpus(pattern: str, context_lines: int = 3) -> str:
```

Regex search over document body text. Automatically:
- Skips YAML frontmatter
- Converts space-separated queries to pipe-separated patterns
- Groups results by document
- Caps output at ~4000 characters

### Agent Tools (Tool Definitions)

Defined in `tools/corpus.py`:

```python
tool_definitions = [
    {"name": "list_documents", ...},
    {"name": "search_metadata", ...},
    {"name": "grep_corpus", ...},
    {"name": "read_document", ...},
    {"name": "read_lines", ...},
    {"name": "ingest_document", ...},
]
```

---

## 2. Memory Store (Long-Term Memory)

### Storage Backend

- **Primary**: JSON file (`{PERSIST_DIR}/ltm_store.json`)
- **Semantic**: SQLite with sqlite-vec (`{PERSIST_DIR}/ltm_semantic.db`)

### Entry Schema

```python
@dataclass
class MemoryEntry:
    content: str
    entry_id: str           # SHA1 hash of content
    created_at: float       # Unix timestamp
    updated_at: float       # Unix timestamp
    access_count: int       # Times retrieved
    learning_score: float   # Importance (0.0 - 1.0)
    tags: list[str]
    source_turn: int
```

### CRUD Operations

| Operation | Method | File | Description |
|-----------|--------|------|-------------|
| **ADD** | `add()` | `memory/ltm_store.py:157` | Create new memory entry |
| **GET** | `get()` | `memory/ltm_store.py:508` | Retrieve entry by ID |
| **GET** | `search()` | `memory/ltm_store.py:265` | Semantic/token search |
| **GET** | `search_by_vector()` | `memory/ltm_store.py:329` | Vector-based retrieval |
| **LIST** | `all_entries()` | `memory/ltm_store.py:504` | Get all entries |
| **UPDATE** | `update()` | `memory/ltm_store.py:201` | Update existing entry |
| **DELETE** | `delete()` | `memory/ltm_store.py:235` | Remove entry |
| **COUNT** | `size()` | `memory/ltm_store.py:513` | Entry count |

### ADD Operation Details

```python
def add(
    self,
    content: str,
    learning_score: float = 0.0,
    tags: list[str] | None = None,
    source_turn: int = 0,
    trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
) -> MemoryOpResult:
```

**Behavior**:
- If similar content exists and `learning_score >= LTM_UPDATE_THRESHOLD`, calls `update()` instead
- Creates `MemoryEntry` with auto-generated `entry_id` (SHA1 of content)
- Inserts embedding into vector index (if semantic search enabled)
- Prunes if entry count exceeds `LTM_MAX_ENTRIES`

**Returns**: `MemoryOpResult` with success status and affected entries

### UPDATE Operation Details

```python
def update(
    self,
    entry_id: str,
    content: str,
    learning_score: float = 0.0,
    trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
) -> MemoryOpResult:
```

**Behavior**:
- Updates `content` and `updated_at` timestamp
- Applies exponential moving average to `learning_score`: `0.6 * old + 0.4 * new`
- Updates vector embedding (if semantic search enabled)

### DELETE Operation Details

```python
def delete(
    self,
    entry_id: str,
    trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
) -> MemoryOpResult:
```

**Behavior**:
- Removes entry from in-memory dict
- Deletes embedding from vector index
- Persists changes to JSON

### SEARCH Operations

#### `search(query, top_k, ...)` — Query-Based Retrieval

```python
def search(
    self,
    query: str,
    top_k: int = 5,
    *,
    expand_query: bool | None = None,
    ner_entities: list[dict] | None = None,
) -> list[MemoryEntry]:
```

**Scoring (semantic mode)**:
- Cosine similarity + recency decay + learning_score

**Scoring (fallback mode)**:
- Token overlap (Jaccard) + recency decay + learning_score

**Query Expansion**: When enabled, generates paraphrase variants and merges results.

#### `search_by_vector(query_vector, top_k, min_similarity)` — Vector Retrieval

```python
def search_by_vector(
    self,
    query_vector: np.ndarray,
    top_k: int = 5,
    min_similarity: Optional[float] = None,
) -> list[MemoryEntry]:
```

Used for context-aware retrieval with pre-computed embeddings.

---

## 3. Vector Index (Embedding Store)

### Storage Backend

- **Format**: SQLite virtual table with sqlite-vec extension
- **Table**: `ltm_vec_index`
- **Dimension**: 1024 (Qwen3-Embedding)

### CRUD Operations

| Operation | Function | File | Description |
|-----------|----------|------|-------------|
| **ADD** | `insert_embedding()` | `memory/vector_index.py:72` | Insert new embedding |
| **UPDATE** | `update_embedding()` | `memory/vector_index.py:97` | Update existing embedding |
| **DELETE** | `delete_embedding()` | `memory/vector_index.py:124` | Remove embedding |
| **SEARCH** | `query_similar()` | `memory/vector_index.py:145` | Cosine similarity search |
| **COUNT** | `get_embedding_count()` | `memory/vector_index.py:180` | Count embeddings |
| **EXISTS** | `entry_exists()` | `memory/vector_index.py:199` | Check existence |
| **INIT** | `ensure_table_exists()` | `memory/vector_index.py:48` | Create table |

### ADD Operation Details

```python
def insert_embedding(db: Any, entry_id: str, embedding: np.ndarray) -> None:
```

Inserts serialized float32 embedding. Fails silently if entry_id exists.

### SEARCH Operation Details

```python
def query_similar(
    db: Any,
    query_embedding: np.ndarray,
    limit: int = 10
) -> list[tuple[str, float]]:
```

**Returns**: List of `(entry_id, distance)` tuples, sorted by ascending distance.

**Distance Scale**: Cosine distance
- `0.0` = identical vectors
- `2.0` = opposite vectors

---

## 4. Operation Summary Matrix

### Corpus Store (File-Based)

| CRUD | Method | Tool Name | Agent Callable |
|------|--------|-----------|----------------|
| CREATE | `ingest()` | `ingest_document` | Yes |
| CREATE | `ingest_markdown()` | `ingest_document` | Yes |
| CREATE | `write_document()` | — | Internal |
| READ | `read_document()` | `read_document` | Yes |
| READ | `read_lines()` | `read_lines` | Yes |
| LIST | `list_documents()` | `list_documents` | Yes |
| SEARCH | `search_metadata()` | `search_metadata` | Yes |
| SEARCH | `grep_corpus()` | `grep_corpus` | Yes |
| UPDATE | Re-ingest | — | Manual |
| DELETE | File deletion | — | Manual |

### Memory Store (LTM)

| CRUD | Method | Agent Callable |
|------|--------|----------------|
| CREATE | `add()` | Via agent internals |
| READ | `get()` | Via agent internals |
| READ | `search()` | Via agent internals |
| READ | `search_by_vector()` | Via agent internals |
| LIST | `all_entries()` | Via agent internals |
| UPDATE | `update()` | Via agent internals |
| DELETE | `delete()` | Via agent internals |
| COUNT | `size()` | Via agent internals |

### Vector Index

| CRUD | Function | Called By |
|------|----------|-----------|
| CREATE | `insert_embedding()` | `LTMStore._insert_embedding_for_entry()` |
| UPDATE | `update_embedding()` | `LTMStore._update_embedding_for_entry()` |
| DELETE | `delete_embedding()` | `LTMStore.delete()` |
| SEARCH | `query_similar()` | `LTMStore.search_by_vector()` |
| EXISTS | `entry_exists()` | Validation checks |
| INIT | `ensure_table_exists()` | `LTMStore._init_semantic_backend()` |

---

## 5. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOCUMENT INGESTION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PDF File ──► Docling ──► Markdown ──► GLiNER ──► Entities         │
│                                  │                                  │
│  Markdown File ──────────────────┘                                  │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ corpus/{doc_id}.md                                            │ │
│  │ ---                                                           │ │
│  │ doc_id: abc123                                                │ │
│  │ doc_title: "Document Title"                                   │ │
│  │ entities: {people: [...], orgs: [...]}                        │ │
│  │ ---                                                           │ │
│  │ # Document content...                                         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ corpus/_index.yaml                                            │ │
│  │ abc123: {title: "...", type: "...", date: "..."}              │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        MEMORY STORAGE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐│
│  │ LTMStore                │    │ Vector Index (sqlite-vec)       ││
│  │ (JSON + SQLite)         │    │                                 ││
│  │                         │    │ insert_embedding()              ││
│  │ add()    ──────────────►│    │ update_embedding()              ││
│  │ update() ──────────────►│    │ delete_embedding()              ││
│  │ delete() ──────────────►│    │ query_similar()                 ││
│  │ get()                   │    │                                 ││
│  │ search()                │    │                                 ││
│  └─────────────────────────┘    └─────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Configuration

Key settings in `core/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CORPUS` | `Path("corpus")` | Document corpus directory |
| `PERSIST_DIR` | `"agent_memory"` | Memory persistence directory |
| `LTM_MAX_ENTRIES` | 500000 | Max LTM entries before pruning |
| `LTM_UPDATE_THRESHOLD` | 0.50 | Score threshold for update vs add |
| `LTM_DEDUP_THRESHOLD` | 0.85 | Semantic similarity for dedup |
| `ENABLE_SEMANTIC_SEARCH` | True | Enable vector-based retrieval |
| `MAX_READ_LINES` | 75 | Max lines for read_lines() |

---

## 7. File Reference

### Ingest Module

| File | Purpose |
|------|---------|
| `ingest/ingest.py` | Main ingestion orchestration |
| `ingest/__init__.py` | Package exports |
| `ingest/entity_post_processor.py` | Entity filtering/validation |
| `ingest/gliner_labels/__init__.py` | Label definitions |
| `ingest/gliner_labels/gliner_labels.py` | NER label sets |

### Corpus Tools

| File | Purpose |
|------|---------|
| `tools/corpus.py` | Corpus access and ingestion tools |

### Memory Module

| File | Purpose |
|------|---------|
| `memory/ltm_store.py` | Long-term memory store |
| `memory/vector_index.py` | Vector index operations |
| `memory/embedding.py` | Embedding generation |
| `memory/retrieval.py` | LTM retrieval utilities |
| `memory/stm_context.py` | Short-term memory context |

### Types

| File | Purpose |
|------|---------|
| `core/types.py` | `MemoryEntry`, `MemoryOp`, `MemoryOpResult` |
| `core/config.py` | Configuration constants |

---

## 8. Usage Examples

### Ingest a PDF

```bash
python ingest/ingest.py document.pdf contract --labels legal
```

### Ingest a Markdown File

```bash
python ingest/ingest.py notes.md document
```

### Ingest a Directory

```bash
python ingest/ingest.py documents/ --labels generic
```

### Programmatic Usage

```python
from ingest.ingest import ingest, ingest_markdown, ingest_directory

# Single PDF
doc_id = ingest("report.pdf", "report", labels_arg="research")

# Markdown file
doc_id = ingest_markdown("notes.md", "document")

# Directory
doc_ids = ingest_directory("papers/", labels_arg="research")
```

### Corpus Tools (Agent Context)

```python
from tools.corpus import list_documents, read_document, grep_corpus

# List all documents
docs = list_documents()

# Read a specific document
content = read_document("report_abc123")

# Search body text
matches = grep_corpus("contract|agreement")
```

### Memory Store

```python
from memory.ltm_store import LTMStore

store = LTMStore(
    persist_path=Path("agent_memory/ltm_store.json"),
    semantic_db_path=Path("agent_memory/ltm_semantic.db"),
    enable_semantic_search=True,
)

# Add
result = store.add("Important fact about X", learning_score=0.8)

# Search
entries = store.search("X", top_k=5)

# Get
entry = store.get(entries[0].entry_id)

# Update
store.update(entry.entry_id, content="Updated fact", learning_score=0.9)

# Delete
store.delete(entry.entry_id)
```