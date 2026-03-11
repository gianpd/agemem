"""
memory/retrieval.py
────────────────────
Two-stage retrieval pipeline for LTM semantic search.

Responsibilities
────────────────
* STAGE 1 (Semantic broad-pass) - Query vector index for candidates
* STAGE 2 (Recency-decay re-rank) - Re-score based on recency and learning_score

Design decisions
────────────────
* Two-stage approach balances semantic quality with temporal relevance.
* Recency decay uses exponential formula: exp(-age_days * decay_rate)
* Final score weights: semantic 50%, learning_score 30%, recency 20%
* Returns top_k results after re-ranking from top_k * 3 candidates.
* learning_score is clamped to [0, 1] on read; storage layer should normalize.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from memory.embedding import embed_text
from memory.vector_index import query_similar

logger = logging.getLogger(__name__)


def _fetch_entry_metadata(db: Any, entry_ids: list[str]) -> dict[str, dict]:
    """
    Fetch metadata for entries from ltm_entries table.

    Args:
        db: SQLite connection.
        entry_ids: List of entry IDs to fetch.

    Returns:
        Dict mapping entry_id to metadata dict with content, learning_score, created_at.
    """
    if not entry_ids:
        return {}

    placeholders = ",".join("?" * len(entry_ids))
    query = f"""
        SELECT entry_id, content, learning_score, created_at
        FROM ltm_entries
        WHERE entry_id IN ({placeholders})
    """

    try:
        cursor = db.execute(query, entry_ids)
        return {
            row[0]: {
                "content": row[1] or "",
                "learning_score": max(0.0, min(1.0, (row[2] or 0.0))),  # clamp to [0, 1]
                "created_at": row[3] or time.time(),
            }
            for row in cursor.fetchall()
        }
    except Exception as e:
        logger.warning("_fetch_entry_metadata failed for %d entries: %s", len(entry_ids), e)
        return {}


def retrieve_relevant_ltm(
    db: Any,  # SQLite connection with vec extension
    model: Any,  # Embedding model (EmbeddingModule instance or similar)
    query: str,
    top_k: int = 10,
    recency_decay_rate: float = 0.01,
) -> list[dict]:
    """
    Retrieve the top_k most relevant LTM entries for a query.

    Two-stage pipeline:
      1. Semantic broad-pass: Query vector index for top_k * 3 candidates
      2. Recency-decay re-rank: Combine semantic, learning, and recency scores

    Re-ranking formula:
        final_score = semantic_score * 0.5 + learning_score * 0.3 + recency_factor * 0.2
        recency_factor = exp(-age_days * recency_decay_rate)

    Args:
        db: SQLite connection with sqlite-vec extension loaded.
            Expected to have ltm_entries and ltm_vec_index tables.
        model: Embedding model with embed_text() method (e.g., EmbeddingModule).
        query: The search query string.
        top_k: Number of results to return after re-ranking. Default 10.
        recency_decay_rate: Rate of recency decay. Higher = faster decay.
                           Default 0.01 means ~100 day half-life.

    Returns:
        List of dicts with keys: entry_id, content, score, learning_score, age_days.
        Sorted by final_score descending.
    """
    # Guard against empty queries
    if not query or not query.strip():
        return []

    # Stage 1: Semantic broad-pass
    # Get more candidates than needed for re-ranking
    candidate_count = top_k * 3

    # Generate query embedding
    # Handle both module-level function and model instance
    if hasattr(model, "embed_text"):
        query_embedding = model.embed_text(query)
    else:
        query_embedding = embed_text(query)

    if query_embedding is None:
        return []

    # Ensure numpy array for vector_index
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding, dtype=np.float32)

    # Query vector index for similar entries
    # Returns list of (entry_id, distance) tuples
    similar_results = query_similar(db, query_embedding, limit=candidate_count)

    if not similar_results:
        return []

    # Extract entry IDs and distances
    entry_ids = [entry_id for entry_id, _ in similar_results]
    distances = {entry_id: dist for entry_id, dist in similar_results}

    # Fetch metadata for all candidates
    metadata = _fetch_entry_metadata(db, entry_ids)

    # Stage 2: Recency-decay re-rank
    now = time.time()
    scored_entries: list[tuple[float, dict]] = []

    for entry_id in entry_ids:
        # Skip stale vector index entries (deleted from ltm_entries)
        if entry_id not in metadata:
            logger.debug("Skipping stale vector entry: %s", entry_id)
            continue

        entry_meta = metadata[entry_id]

        content = entry_meta["content"]
        semantic_distance = distances.get(entry_id, 1.0)
        learning_score = entry_meta["learning_score"]
        created_at = entry_meta["created_at"]

        # Convert distance to similarity score (lower distance = higher similarity)
        # sqlite-vec returns cosine distance, so similarity = 1 - distance
        similarity = 1.0 - semantic_distance

        # Calculate age in days
        age_days = (now - created_at) / 86_400  # seconds per day

        # Recency factor with exponential decay
        recency_factor = math.exp(-age_days * recency_decay_rate)

        # Final score: weighted combination
        # semantic 50%, learning 30%, recency 20%
        final_score = (
            similarity * 0.5 +
            learning_score * 0.3 +
            recency_factor * 0.2
        )

        result = {
            "entry_id": entry_id,
            "content": content,
            "score": final_score,
            "learning_score": learning_score,
            "age_days": age_days,
        }
        scored_entries.append((final_score, result))

    # Sort by final score descending
    scored_entries.sort(key=lambda x: x[0], reverse=True)

    # Return top_k results
    return [entry for _, entry in scored_entries[:top_k]]


def retrieve_by_tags(
    db: Any,
    tags: list[str],
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieve LTM entries matching specific tags.

    This is a lightweight filter for tag-based retrieval without semantic search.
    Useful for quick lookups when the exact category is known.

    Args:
        db: SQLite connection with ltm_entries table.
        tags: List of tags to match (OR logic).
        top_k: Maximum number of results.

    Returns:
        List of dicts with entry metadata, sorted by learning_score.
    """
    if not tags:
        return []

    # Build parameterized query for tag matching
    placeholders = ",".join("?" * len(tags))
    query = f"""
        SELECT entry_id, content, learning_score, created_at, updated_at
        FROM ltm_entries
        WHERE tags IS NOT NULL
        AND (
            SELECT COUNT(*) FROM json_each(tags)
            WHERE value IN ({placeholders})
        ) > 0
        ORDER BY learning_score DESC
        LIMIT ?
    """

    try:
        cursor = db.execute(query, tags + [top_k])
        results = []
        now = time.time()

        for row in cursor.fetchall():
            entry_id, content, learning_score, created_at, updated_at = row
            age_days = (now - created_at) / 86_400

            results.append({
                "entry_id": entry_id,
                "content": content,
                "score": learning_score,  # Use learning_score as score for tag search
                "learning_score": max(0.0, min(1.0, learning_score or 0.0)),  # clamp to [0, 1]
                "age_days": age_days,
            })

        return results
    except Exception as e:
        logger.warning("retrieve_by_tags failed for tags=%s: %s", tags, e)
        return []


def retrieve_recent(
    db: Any,
    days: int = 7,
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieve LTM entries from the last N days, sorted by recency.

    Useful for context loading when recency matters more than semantic relevance.

    Args:
        db: SQLite connection with ltm_entries table.
        days: Number of days to look back. Default 7.
        top_k: Maximum number of results.

    Returns:
        List of dicts with entry metadata, sorted by updated_at descending.
    """
    cutoff = time.time() - (days * 86_400)

    query = """
        SELECT entry_id, content, learning_score, created_at, updated_at
        FROM ltm_entries
        WHERE updated_at >= ?
        ORDER BY updated_at DESC
        LIMIT ?
    """

    try:
        cursor = db.execute(query, [cutoff, top_k])
        results = []
        now = time.time()

        for row in cursor.fetchall():
            entry_id, content, learning_score, created_at, updated_at = row
            age_days = (now - created_at) / 86_400

            # Score based on recency: 1.0 for most recent, approaching 0.0 at cutoff
            recency_score = (updated_at - cutoff) / (days * 86_400)

            results.append({
                "entry_id": entry_id,
                "content": content,
                "score": recency_score,
                "learning_score": max(0.0, min(1.0, learning_score or 0.0)),  # clamp to [0, 1]
                "age_days": age_days,
            })

        return results
    except Exception as e:
        logger.warning("retrieve_recent failed for days=%d: %s", days, e)
        return []