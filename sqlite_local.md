# SQLite Local Database Guide

Database location: `agent_memory/ltm_semantic.db`

## Overview

This database stores long-term memory (LTM) entries with semantic search capabilities using the sqlite-vec extension. It contains two main tables:

- `ltm_entries` - Main memory entries with content and metadata
- `ltm_vec_index` - Virtual table for vector similarity search (1024-dimensional embeddings)

## Quick Start

```bash
# Open the database
sqlite3 agent_memory/ltm_semantic.db

# Or use the full path
sqlite3 /home/jaco/develops/WORKS/agemem/agent_memory/ltm_semantic.db
```

## Schema Reference

### `ltm_entries` Table

| Column | Type | Description |
|--------|------|-------------|
| `entry_id` | TEXT PK | Unique 12-character hex identifier |
| `content` | TEXT | The memory content |
| `created_at` | REAL | Unix timestamp when created |
| `updated_at` | REAL | Unix timestamp when last updated |
| `access_count` | INTEGER | Number of times retrieved |
| `learning_score` | REAL | Importance score (0.0-1.0) |
| `tags` | TEXT | JSON array of tags |
| `source_turn` | INTEGER | Conversation turn when created |
| `embedding` | BLOB | Serialized 1024-dim vector |
| `embedding_model` | TEXT | Model name (default: Qwen3-Embedding-0.6B) |
| `embedding_dim` | INTEGER | Vector dimension (default: 1024) |

### `ltm_vec_index` Virtual Table

SQLite-vec virtual table for cosine similarity search. Access via the rowids table:

```sql
SELECT id FROM ltm_vec_index_rowids;
```

---

## Useful Queries for Testing & Checking

### Basic Inspection

```sql
-- Count total entries
SELECT COUNT(*) as total_entries FROM ltm_entries;

-- View all entries with key metrics
SELECT
    entry_id,
    substr(content, 1, 60) || '...' as content_preview,
    learning_score,
    access_count,
    datetime(created_at, 'unixepoch', 'localtime') as created
FROM ltm_entries
ORDER BY created_at DESC;

-- Check schema version and embedding model
SELECT DISTINCT embedding_model, embedding_dim FROM ltm_entries;
```

### Health Checks

```sql
-- Verify vector index is populated (should match ltm_entries count)
SELECT
    (SELECT COUNT(*) FROM ltm_entries) as entries,
    (SELECT COUNT(*) FROM ltm_vec_index_rowids) as vectors,
    CASE
        WHEN (SELECT COUNT(*) FROM ltm_entries) = (SELECT COUNT(*) FROM ltm_vec_index_rowids)
        THEN 'OK'
        ELSE 'MISMATCH'
    END as status;

-- Find entries without embeddings (should return 0)
SELECT entry_id, content
FROM ltm_entries
WHERE embedding IS NULL;

-- Find orphaned vectors (in vec_index but not in ltm_entries)
SELECT v.id
FROM ltm_vec_index_rowids v
LEFT JOIN ltm_entries e ON v.id = e.entry_id
WHERE e.entry_id IS NULL;

-- Check for stale embeddings (updated_at newer than embedding creation)
-- Note: Embeddings are regenerated on update, so this shouldn't happen
SELECT entry_id, updated_at
FROM ltm_entries
WHERE embedding IS NOT NULL
ORDER BY updated_at DESC;
```

### Data Quality Queries

```sql
-- Top entries by learning score
SELECT entry_id, learning_score, substr(content, 1, 50) as preview
FROM ltm_entries
ORDER BY learning_score DESC
LIMIT 10;

-- Most accessed memories
SELECT entry_id, access_count, substr(content, 1, 50) as preview
FROM ltm_entries
ORDER BY access_count DESC
LIMIT 10;

-- Entries with high learning_score but low access (potentially underutilized)
SELECT entry_id, learning_score, access_count, substr(content, 1, 50) as preview
FROM ltm_entries
WHERE learning_score > 0.5 AND access_count < 2
ORDER BY learning_score DESC;

-- Distribution of learning scores
SELECT
    CASE
        WHEN learning_score >= 0.8 THEN '0.8-1.0'
        WHEN learning_score >= 0.5 THEN '0.5-0.8'
        WHEN learning_score >= 0.2 THEN '0.2-0.5'
        ELSE '0.0-0.2'
    END as score_range,
    COUNT(*) as count
FROM ltm_entries
GROUP BY score_range
ORDER BY score_range DESC;

-- Recent entries (last 7 days)
SELECT entry_id, substr(content, 1, 60) as preview, learning_score
FROM ltm_entries
WHERE created_at > strftime('%s', 'now', '-7 days')
ORDER BY created_at DESC;
```

### Tag Analysis

```sql
-- Find entries with specific tags
SELECT entry_id, substr(content, 1, 60) as preview, tags
FROM ltm_entries
WHERE tags IS NOT NULL AND tags != '[]'
LIMIT 10;

-- Extract and count tags (requires JSON1 extension)
SELECT value as tag, COUNT(*) as usage_count
FROM ltm_entries, json_each(tags)
WHERE tags IS NOT NULL AND tags != '[]'
GROUP BY value
ORDER BY usage_count DESC;
```

### Debugging Queries

```sql
-- Check a specific entry by ID
SELECT * FROM ltm_entries WHERE entry_id = '4cbac66f372e';

-- Find entries containing specific text
SELECT entry_id, content, learning_score
FROM ltm_entries
WHERE content LIKE '%bridge%';

-- Check timestamp consistency
SELECT
    entry_id,
    datetime(created_at, 'unixepoch', 'localtime') as created,
    datetime(updated_at, 'unixepoch', 'localtime') as updated,
    updated_at - created_at as seconds_since_creation
FROM ltm_entries
ORDER BY created_at DESC;

-- Find duplicate content (potential issues)
SELECT content, COUNT(*) as duplicates
FROM ltm_entries
GROUP BY content
HAVING COUNT(*) > 1;
```

### Vector Index Inspection

```sql
-- Count vectors in index
SELECT COUNT(*) as vector_count FROM ltm_vec_index_rowids;

-- List all vector IDs
SELECT id FROM ltm_vec_index_rowids ORDER BY rowid;

-- Check vector chunk storage
SELECT chunk_id, size FROM ltm_vec_index_chunks;
```

---

## Python Access Example

```python
import sqlite3

# Connect with extension support
conn = sqlite3.connect('agent_memory/ltm_semantic.db')
conn.enable_load_extension(True)

# Load sqlite-vec
import sqlite_vec
sqlite_vec.load(conn)

# Run queries
cursor = conn.execute("SELECT COUNT(*) FROM ltm_entries")
print(f"Total entries: {cursor.fetchone()[0]}")

conn.close()
```

---

## Maintenance Operations

### Clean Up Stale Vectors

```sql
-- Remove vectors that have no corresponding entry
DELETE FROM ltm_vec_index_rowids
WHERE id NOT IN (SELECT entry_id FROM ltm_entries);
```

### Reset Access Counts

```sql
-- Reset all access counts (use sparingly)
UPDATE ltm_entries SET access_count = 0;
```

### Export to JSON

```bash
sqlite3 agent_memory/ltm_semantic.db \
    "SELECT json_group_array(json_object(
        'entry_id', entry_id,
        'content', content,
        'learning_score', learning_score,
        'tags', tags
    )) FROM ltm_entries;" > ltm_export.json
```

---

## Common Issues

1. **Vector count mismatch**: Run the health check query to detect; may need to resync embeddings
2. **Missing embeddings**: Check if embedding model is available and sqlite-vec loaded correctly
3. **Stale entries**: Updated content should regenerate embeddings automatically via `_update_embedding_for_entry`

---

## Related Files

- `memory/ltm_store.py` - Main store implementation
- `memory/vector_index.py` - Vector index operations
- `memory/retrieval.py` - Two-stage retrieval pipeline
- `core/db_migrations.py` - Schema migrations
- `memory/embedding.py` - Embedding generation