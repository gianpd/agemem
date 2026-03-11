"""
memory/vector_index.py
----------------------
Vector index module for semantic search over LTM entries.

Responsibilities
----------------
* INSERT  - store a new embedding vector
* UPDATE  - overwrite an existing embedding
* DELETE  - remove an embedding
* QUERY   - find similar entries by cosine distance

Design decisions
----------------
* Uses sqlite-vec extension for embedded vector storage
* Brute-force (exact) search - acceptable for <100K vectors
* Table name: ltm_vec_index (virtual table using vec0)
* Cosine distance via vec_distance_cosine SQL function
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import sqlite_vec

logger = logging.getLogger(__name__)

TABLE_NAME = "ltm_vec_index"
DEFAULT_DIMENSION = 1024


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    """
    Convert numpy array to sqlite-vec blob format.

    Args:
        embedding: 1D numpy array of floats

    Returns:
        Serialized bytes suitable for sqlite-vec storage
    """
    return sqlite_vec.serialize_float32(embedding)


def ensure_table_exists(db: Any, dimension: int = DEFAULT_DIMENSION) -> None:
    """
    Create the vector index table if it does not exist.

    Should be called once after opening the database connection.

    Args:
        db: SQLite connection with sqlite-vec extension loaded
        dimension: Embedding dimension (default 1024 for Qwen3-Embedding)
    """
    try:
        db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE_NAME}
            USING vec0(
                entry_id TEXT PRIMARY KEY,
                embedding FLOAT[{dimension}]
            )
        """)
        logger.debug(f"Ensured vector index table '{TABLE_NAME}' exists (dim={dimension})")
    except Exception as e:
        logger.error(f"Failed to create vector index table: {e}")
        raise


def insert_embedding(db: Any, entry_id: str, embedding: np.ndarray) -> None:
    """
    Insert a new embedding into the vector index.

    Args:
        db: SQLite connection with sqlite-vec extension loaded
        entry_id: Unique identifier for the memory entry
        embedding: 1D numpy array of floats (normalized for cosine similarity)

    Note:
        If entry_id already exists, the insert will fail.
        Use update_embedding to modify existing entries.
    """
    try:
        serialized = _serialize_embedding(embedding)
        db.execute(
            f"INSERT INTO {TABLE_NAME} (entry_id, embedding) VALUES (?, ?)",
            (entry_id, serialized)
        )
        logger.debug(f"Inserted embedding for entry_id={entry_id}")
    except Exception as e:
        # Log but don't raise - allow caller to continue
        logger.error(f"Failed to insert embedding for {entry_id}: {e}")


def update_embedding(db: Any, entry_id: str, embedding: np.ndarray) -> None:
    """
    Update an existing embedding in the vector index.

    Args:
        db: SQLite connection with sqlite-vec extension loaded
        entry_id: Unique identifier for the memory entry
        embedding: 1D numpy array of floats (normalized for cosine similarity)

    Note:
        If entry_id does not exist, this will not insert a new row.
        Check affected rows if you need to verify the update succeeded.
    """
    try:
        serialized = _serialize_embedding(embedding)
        cursor = db.execute(
            f"UPDATE {TABLE_NAME} SET embedding = ? WHERE entry_id = ?",
            (serialized, entry_id)
        )
        if cursor.rowcount == 0:
            logger.warning(f"No embedding found to update for entry_id={entry_id}")
        else:
            logger.debug(f"Updated embedding for entry_id={entry_id}")
    except Exception as e:
        logger.error(f"Failed to update embedding for {entry_id}: {e}")


def delete_embedding(db: Any, entry_id: str) -> None:
    """
    Delete an embedding from the vector index.

    Args:
        db: SQLite connection with sqlite-vec extension loaded
        entry_id: Unique identifier for the memory entry
    """
    try:
        cursor = db.execute(
            f"DELETE FROM {TABLE_NAME} WHERE entry_id = ?",
            (entry_id,)
        )
        if cursor.rowcount == 0:
            logger.debug(f"No embedding to delete for entry_id={entry_id}")
        else:
            logger.debug(f"Deleted embedding for entry_id={entry_id}")
    except Exception as e:
        logger.error(f"Failed to delete embedding for {entry_id}: {e}")


def query_similar(
    db: Any,
    query_embedding: np.ndarray,
    limit: int = 10
) -> list[tuple[str, float]]:
    """
    Find entries with similar embeddings using cosine distance.

    Args:
        db: SQLite connection with sqlite-vec extension loaded
        query_embedding: 1D numpy array of floats (should be normalized)
        limit: Maximum number of results to return (default 10)

    Returns:
        List of (entry_id, distance) tuples, sorted by ascending distance.
        Lower distance = more similar (cosine distance: 0 = identical, 2 = opposite)
    """
    try:
        serialized = _serialize_embedding(query_embedding)
        cursor = db.execute(
            f"""
            SELECT entry_id, vec_distance_cosine(embedding, ?) as distance
            FROM {TABLE_NAME}
            ORDER BY distance
            LIMIT ?
            """,
            (serialized, limit)
        )
        results = cursor.fetchall()
        return [(row[0], row[1]) for row in results]
    except Exception as e:
        logger.error(f"Failed to query similar embeddings: {e}")
        return []


def get_embedding_count(db: Any) -> int:
    """
    Get the total number of embeddings in the index.

    Args:
        db: SQLite connection with sqlite-vec extension loaded

    Returns:
        Number of embeddings stored, or 0 on error.
    """
    try:
        cursor = db.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        logger.error(f"Failed to count embeddings: {e}")
        return 0


def entry_exists(db: Any, entry_id: str) -> bool:
    """
    Check if an embedding exists for the given entry_id.

    Args:
        db: SQLite connection with sqlite-vec extension loaded
        entry_id: Unique identifier for the memory entry

    Returns:
        True if embedding exists, False otherwise.
    """
    try:
        cursor = db.execute(
            f"SELECT 1 FROM {TABLE_NAME} WHERE entry_id = ? LIMIT 1",
            (entry_id,)
        )
        return cursor.fetchone() is not None
    except Exception as e:
        logger.error(f"Failed to check embedding existence for {entry_id}: {e}")
        return False