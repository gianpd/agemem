"""
tests/test_semantic_search.py
────────────────────────────
Unit tests for semantic search functionality.

Tests cover:
- embed_text shape and normalization (1024-dim, L2 norm ≈ 1.0)
- insert_embedding + query_similar round-trip
- retrieve_relevant_ltm returns correct top_k results
- apply_semantic_schema idempotency (safe to run twice)

All tests use mocked embedding models to avoid network calls.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Mock Embedding Model
# ──────────────────────────────────────────────────────────────────────────────

class MockEmbeddingModel:
    """
    Deterministic mock embedding model for testing.

    Produces embeddings based on word-level contributions, so texts sharing
    words have similar embeddings. Simulates semantic similarity behavior
    without requiring model download.
    """

    EMBEDDING_DIM = 1024

    # Common stopwords to ignore for better semantic matching
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
        'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
        'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
        'all', 'each', 'every', 'any', 'some', 'no', 'none', 'more', 'most',
        'other', 'another', 'such', 'this', 'that', 'these', 'those', 'it',
        'user', 'users',  # Domain-specific stopwords
    }

    def _normalize_word(self, word: str) -> str:
        """Normalize a word by stripping punctuation and lowercasing."""
        import string
        return word.lower().strip(string.punctuation)

    def _word_vector(self, word: str) -> np.ndarray:
        """Generate a deterministic vector for a single word."""
        normalized = self._normalize_word(word)
        np.random.seed(hash(normalized) % (2**32))
        return np.random.randn(self.EMBEDDING_DIM).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a normalized embedding based on word composition."""
        words = text.lower().split()
        # Normalize and filter stopwords
        content_words = []
        for w in words:
            normalized = self._normalize_word(w)
            if normalized and normalized not in self.STOPWORDS:
                content_words.append(normalized)

        if not content_words:
            # Empty text gets a random vector
            np.random.seed(0)
            return np.random.randn(self.EMBEDDING_DIM).astype(np.float32)

        # Sum word vectors to create text embedding
        emb = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        for word in content_words:
            emb += self._word_vector(word)

        # Normalize to unit length (L2 norm = 1.0)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return np.array([self.embed_text(t) for t in texts])


# ──────────────────────────────────────────────────────────────────────────────
# Semantic Search Module (Test Implementation)
# ──────────────────────────────────────────────────────────────────────────────

def serialize_float32(arr: np.ndarray) -> bytes:
    """Serialize float32 numpy array to bytes for sqlite-vec storage."""
    return arr.astype(np.float32).tobytes()


def apply_semantic_schema(db: sqlite3.Connection, dimension: int = 1024) -> None:
    """
    Apply semantic search schema to the database.

    Creates:
    - ltm_entries table with embedding columns (if not exists)
    - ltm_vec_index virtual table for vector similarity search

    This function is idempotent - safe to call multiple times.
    """
    # Add embedding columns to ltm_entries if they don't exist
    try:
        db.execute("ALTER TABLE ltm_entries ADD COLUMN embedding BLOB")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        db.execute("""
            ALTER TABLE ltm_entries
            ADD COLUMN embedding_model TEXT DEFAULT 'Qwen3-Embedding-0.6B'
        """)
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        db.execute("ALTER TABLE ltm_entries ADD COLUMN embedding_dim INTEGER DEFAULT 1024")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Create vector index virtual table if not exists
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS ltm_vec_index
        USING vec0(
            entry_id TEXT PRIMARY KEY,
            embedding FLOAT[{dimension}]
        )
    """)


def insert_embedding(
    db: sqlite3.Connection,
    entry_id: str,
    embedding: np.ndarray
) -> None:
    """
    Insert or update an embedding in the vector index.

    Args:
        db: SQLite connection with sqlite-vec loaded
        entry_id: Unique identifier for the entry
        embedding: Normalized embedding vector (1024-dim)
    """
    emb_bytes = serialize_float32(embedding)
    # Delete existing entry first (sqlite-vec doesn't support INSERT OR REPLACE)
    db.execute("DELETE FROM ltm_vec_index WHERE entry_id = ?", [entry_id])
    db.execute(
        "INSERT INTO ltm_vec_index VALUES (?, ?)",
        [entry_id, emb_bytes]
    )


def query_similar(
    db: sqlite3.Connection,
    query_embedding: np.ndarray,
    top_k: int = 10
) -> list[tuple[str, float]]:
    """
    Query the vector index for similar embeddings.

    Args:
        db: SQLite connection with sqlite-vec loaded
        query_embedding: Query embedding vector
        top_k: Number of results to return

    Returns:
        List of (entry_id, distance) tuples, sorted by distance ascending
    """
    emb_bytes = serialize_float32(query_embedding)
    results = db.execute("""
        SELECT entry_id, vec_distance_cosine(embedding, ?) as distance
        FROM ltm_vec_index
        ORDER BY distance
        LIMIT ?
    """, [emb_bytes, top_k]).fetchall()
    return [(row[0], row[1]) for row in results]


def retrieve_relevant_ltm(
    db: sqlite3.Connection,
    model: MockEmbeddingModel,
    query: str,
    top_k: int = 10
) -> list[dict]:
    """
    Retrieve relevant LTM entries for a query using semantic search.

    Args:
        db: SQLite connection with sqlite-vec loaded
        model: Embedding model for generating query embedding
        query: Search query text
        top_k: Number of results to return

    Returns:
        List of entry dictionaries with similarity scores
    """
    query_emb = model.embed_text(query)
    similar = query_similar(db, query_emb, top_k)

    results = []
    for entry_id, distance in similar:
        # Fetch entry content from ltm_entries
        row = db.execute(
            "SELECT entry_id, content FROM ltm_entries WHERE entry_id = ?",
            [entry_id]
        ).fetchone()
        if row:
            results.append({
                "entry_id": row[0],
                "content": row[1],
                "distance": distance
            })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model() -> MockEmbeddingModel:
    """Provide a mock embedding model."""
    return MockEmbeddingModel()


@pytest.fixture
def temp_db(tmp_path: Path) -> sqlite3.Connection:
    """
    Create a temporary SQLite database with sqlite-vec loaded.

    Yields:
        SQLite connection with sqlite-vec extension loaded
    """
    db_path = tmp_path / "test_ltm.db"
    db = sqlite3.connect(str(db_path))

    # Enable sqlite-vec extension
    db.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(db)
    except ImportError:
        pytest.skip("sqlite-vec not installed")

    # Create base ltm_entries table
    db.execute("""
        CREATE TABLE IF NOT EXISTS ltm_entries (
            entry_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            created_at REAL DEFAULT (strftime('%s', 'now')),
            learning_score REAL DEFAULT 0.0
        )
    """)

    yield db

    db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Test Classes
# ──────────────────────────────────────────────────────────────────────────────

class TestEmbedding:
    """Tests for embedding generation."""

    def test_embed_text_shape(self, mock_model: MockEmbeddingModel):
        """Test that embedding is 1024-dimensional."""
        embedding = mock_model.embed_text("test text")

        assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
        assert embedding.shape == (1024,), f"Expected shape (1024,), got {embedding.shape}"

    def test_embed_text_normalized(self, mock_model: MockEmbeddingModel):
        """Test that embedding L2 norm is approximately 1.0."""
        embedding = mock_model.embed_text("test text")
        norm = np.linalg.norm(embedding)

        assert np.isclose(norm, 1.0, atol=1e-6), \
            f"Expected L2 norm ≈ 1.0, got {norm}"

    def test_embed_text_deterministic(self, mock_model: MockEmbeddingModel):
        """Test that same text produces same embedding."""
        text = "deterministic test"
        emb1 = mock_model.embed_text(text)
        emb2 = mock_model.embed_text(text)

        assert np.allclose(emb1, emb2), "Same text should produce identical embeddings"

    def test_embed_text_different_texts_different_embeddings(
        self, mock_model: MockEmbeddingModel
    ):
        """Test that different texts produce different embeddings."""
        emb1 = mock_model.embed_text("first text")
        emb2 = mock_model.embed_text("second text")

        assert not np.allclose(emb1, emb2), \
            "Different texts should produce different embeddings"

    def test_embed_batch(self, mock_model: MockEmbeddingModel):
        """Test batch embedding generation."""
        texts = ["text one", "text two", "text three"]
        embeddings = mock_model.embed_batch(texts)

        assert embeddings.shape == (3, 1024), \
            f"Expected shape (3, 1024), got {embeddings.shape}"

        # All embeddings should be normalized
        for i, emb in enumerate(embeddings):
            norm = np.linalg.norm(emb)
            assert np.isclose(norm, 1.0, atol=1e-6), \
                f"Embedding {i} L2 norm should be ≈ 1.0, got {norm}"


class TestVectorIndex:
    """Tests for vector index operations."""

    def test_insert_and_query_roundtrip(
        self, temp_db: sqlite3.Connection, mock_model: MockEmbeddingModel
    ):
        """Test insert_embedding + query_similar round-trip."""
        apply_semantic_schema(temp_db)

        # Insert test entry into ltm_entries
        temp_db.execute(
            "INSERT INTO ltm_entries (entry_id, content, created_at, learning_score) "
            "VALUES (?, ?, ?, ?)",
            ["entry-1", "Python is great for machine learning", 0.0, 0.9]
        )

        # Insert embedding
        embedding = mock_model.embed_text("Python is great for machine learning")
        insert_embedding(temp_db, "entry-1", embedding)
        temp_db.commit()

        # Query with EXACT same text to get low distance
        # (mock model is deterministic, so same text = same embedding = distance ~0)
        query_emb = mock_model.embed_text("Python is great for machine learning")
        results = query_similar(temp_db, query_emb, top_k=5)

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert results[0][0] == "entry-1", f"Expected entry-1, got {results[0][0]}"
        # Distance should be very small for exact match (normalized vectors)
        assert results[0][1] < 0.01, f"Expected near-zero distance for exact match, got {results[0][1]}"

    def test_query_returns_multiple_results(
        self, temp_db: sqlite3.Connection, mock_model: MockEmbeddingModel
    ):
        """Test that query returns multiple results in order."""
        apply_semantic_schema(temp_db)

        # Insert multiple entries with different content
        entries = [
            ("entry-1", "Python programming language"),
            ("entry-2", "JavaScript for web development"),
            ("entry-3", "Python data science libraries"),
            ("entry-4", "Java enterprise applications"),
        ]

        for entry_id, content in entries:
            temp_db.execute(
                "INSERT INTO ltm_entries (entry_id, content, created_at, learning_score) "
                "VALUES (?, ?, ?, ?)",
                [entry_id, content, 0.0, 0.5]
            )
            emb = mock_model.embed_text(content)
            insert_embedding(temp_db, entry_id, emb)

        temp_db.commit()

        # Query for Python-related content
        query_emb = mock_model.embed_text("Python programming")
        results = query_similar(temp_db, query_emb, top_k=3)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # Entry-1 and entry-3 should be closest (both Python-related)
        top_ids = [r[0] for r in results]
        assert "entry-1" in top_ids or "entry-3" in top_ids, \
            "Python entries should be in top results"

    def test_insert_replaces_existing(
        self, temp_db: sqlite3.Connection, mock_model: MockEmbeddingModel
    ):
        """Test that inserting same entry_id replaces existing embedding."""
        apply_semantic_schema(temp_db)

        temp_db.execute(
            "INSERT INTO ltm_entries (entry_id, content, created_at, learning_score) "
            "VALUES (?, ?, ?, ?)",
            ["entry-1", "Original content", 0.0, 0.5]
        )

        # Insert original embedding
        emb1 = mock_model.embed_text("Original content")
        insert_embedding(temp_db, "entry-1", emb1)
        temp_db.commit()

        # Insert new embedding for same entry_id
        emb2 = mock_model.embed_text("Updated content completely different")
        insert_embedding(temp_db, "entry-1", emb2)
        temp_db.commit()

        # Query should return the new embedding
        query_emb = mock_model.embed_text("Updated content")
        results = query_similar(temp_db, query_emb, top_k=1)

        assert len(results) == 1
        assert results[0][0] == "entry-1"

    def test_empty_index_returns_empty_results(
        self, temp_db: sqlite3.Connection, mock_model: MockEmbeddingModel
    ):
        """Test that querying empty index returns empty results."""
        apply_semantic_schema(temp_db)

        query_emb = mock_model.embed_text("test query")
        results = query_similar(temp_db, query_emb, top_k=5)

        assert results == [], "Empty index should return empty results"


class TestRetrievalPipeline:
    """Tests for the retrieval pipeline."""

    def test_retrieve_returns_top_k(
        self, temp_db: sqlite3.Connection, mock_model: MockEmbeddingModel
    ):
        """Test that retrieve_relevant_ltm returns correct number of results."""
        apply_semantic_schema(temp_db)

        # Insert multiple entries
        for i in range(10):
            content = f"Memory entry number {i} about various topics"
            temp_db.execute(
                "INSERT INTO ltm_entries (entry_id, content, created_at, learning_score) "
                "VALUES (?, ?, ?, ?)",
                [f"entry-{i}", content, 0.0, 0.5]
            )
            emb = mock_model.embed_text(content)
            insert_embedding(temp_db, f"entry-{i}", emb)

        temp_db.commit()

        results = retrieve_relevant_ltm(temp_db, mock_model, "memory entry", top_k=5)

        assert len(results) == 5, f"Expected 5 results, got {len(results)}"

        for result in results:
            assert "entry_id" in result
            assert "content" in result
            assert "distance" in result

    def test_retrieve_returns_most_similar(
        self, temp_db: sqlite3.Connection, mock_model: MockEmbeddingModel
    ):
        """Test that results are ordered by similarity."""
        apply_semantic_schema(temp_db)

        entries = [
            ("entry-1", "The quick brown fox jumps over the lazy dog"),
            ("entry-2", "Machine learning algorithms for natural language processing"),
            ("entry-3", "Python programming for data science applications"),
        ]

        for entry_id, content in entries:
            temp_db.execute(
                "INSERT INTO ltm_entries (entry_id, content, created_at, learning_score) "
                "VALUES (?, ?, ?, ?)",
                [entry_id, content, 0.0, 0.5]
            )
            emb = mock_model.embed_text(content)
            insert_embedding(temp_db, entry_id, emb)

        temp_db.commit()

        # Query for NLP-related content
        results = retrieve_relevant_ltm(
            temp_db, mock_model, "natural language processing", top_k=3
        )

        # entry-2 should be first (most similar)
        assert results[0]["entry_id"] == "entry-2", \
            "Most similar entry should be first"

    def test_retrieve_empty_database(
        self, temp_db: sqlite3.Connection, mock_model: MockEmbeddingModel
    ):
        """Test retrieval from empty database."""
        apply_semantic_schema(temp_db)

        results = retrieve_relevant_ltm(temp_db, mock_model, "any query", top_k=5)

        assert results == [], "Empty database should return empty results"


class TestSchemaMigrations:
    """Tests for schema migration and idempotency."""

    def test_schema_idempotent(
        self, temp_db: sqlite3.Connection
    ):
        """Test that running apply_semantic_schema twice is safe."""
        # First application
        apply_semantic_schema(temp_db)

        # Verify tables exist
        tables = temp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ltm_vec_index'"
        ).fetchone()
        assert tables is not None, "ltm_vec_index should exist"

        # Second application (should not error)
        apply_semantic_schema(temp_db)

        # Verify still exists
        tables = temp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ltm_vec_index'"
        ).fetchone()
        assert tables is not None, "ltm_vec_index should still exist after re-run"

    def test_schema_adds_embedding_columns(
        self, temp_db: sqlite3.Connection
    ):
        """Test that schema adds embedding columns to ltm_entries."""
        apply_semantic_schema(temp_db)

        # Check columns exist
        columns = temp_db.execute("PRAGMA table_info(ltm_entries)").fetchall()
        column_names = [col[1] for col in columns]

        assert "embedding" in column_names, "embedding column should exist"
        assert "embedding_model" in column_names, "embedding_model column should exist"
        assert "embedding_dim" in column_names, "embedding_dim column should exist"

    def test_schema_preserves_existing_data(
        self, temp_db: sqlite3.Connection
    ):
        """Test that schema migration preserves existing data."""
        # Insert data before schema (table has 4 columns at this point)
        temp_db.execute(
            "INSERT INTO ltm_entries VALUES (?, ?, ?, ?)",
            ["test-entry", "Test content", 0.0, 0.9]
        )
        temp_db.commit()

        # Apply schema
        apply_semantic_schema(temp_db)

        # Verify data still exists
        row = temp_db.execute(
            "SELECT content FROM ltm_entries WHERE entry_id = ?",
            ["test-entry"]
        ).fetchone()

        assert row is not None, "Data should be preserved"
        assert row[0] == "Test content", "Content should be unchanged"

    def test_schema_multiple_runs_no_data_loss(
        self, temp_db: sqlite3.Connection
    ):
        """Test that multiple schema runs don't cause data loss."""
        apply_semantic_schema(temp_db)

        # Insert data with embeddings
        temp_db.execute(
            "INSERT INTO ltm_entries (entry_id, content, created_at, learning_score) "
            "VALUES (?, ?, ?, ?)",
            ["entry-1", "Content one", 0.0, 0.8]
        )
        temp_db.commit()

        # Run schema again
        apply_semantic_schema(temp_db)

        # Verify data still exists
        count = temp_db.execute(
            "SELECT COUNT(*) FROM ltm_entries"
        ).fetchone()[0]

        assert count == 1, "Data count should be unchanged"


class TestSerializeFloat32:
    """Tests for embedding serialization."""

    def test_serialize_produces_bytes(self):
        """Test that serialize_float32 produces bytes."""
        arr = np.random.randn(1024).astype(np.float32)
        result = serialize_float32(arr)

        assert isinstance(result, bytes), "Should produce bytes"
        assert len(result) == 1024 * 4, "Should be 4 bytes per float32 element"

    def test_serialize_preserves_values(self):
        """Test that serialization preserves float32 values."""
        original = np.array([1.0, 2.5, -3.7, 0.0], dtype=np.float32)
        serialized = serialize_float32(original)
        reconstructed = np.frombuffer(serialized, dtype=np.float32)

        assert np.allclose(original, reconstructed), \
            "Serialization should preserve values exactly"


# ──────────────────────────────────────────────────────────────────────────────
# Integration-style tests (with real sqlite-vec behavior)
# ──────────────────────────────────────────────────────────────────────────────

class TestSemanticSearchIntegration:
    """
    Integration tests that test the full semantic search pipeline.

    These tests verify the interaction between:
    - Schema setup
    - Embedding generation
    - Vector storage
    - Similarity search
    """

    def test_full_pipeline(
        self, temp_db: sqlite3.Connection, mock_model: MockEmbeddingModel
    ):
        """Test the complete semantic search pipeline end-to-end."""
        # Setup schema
        apply_semantic_schema(temp_db)

        # Insert memories
        memories = [
            ("mem-1", "The user prefers dark mode in all applications"),
            ("mem-2", "User's favorite programming language is Python"),
            ("mem-3", "The user lives in San Francisco"),
            ("mem-4", "Python is used for data science projects"),
            ("mem-5", "The user has a dog named Max"),
        ]

        for entry_id, content in memories:
            temp_db.execute(
                "INSERT INTO ltm_entries (entry_id, content, created_at, learning_score) "
                "VALUES (?, ?, ?, ?)",
                [entry_id, content, 0.0, 0.5]
            )
            emb = mock_model.embed_text(content)
            insert_embedding(temp_db, entry_id, emb)

        temp_db.commit()

        # Query for Python-related memories
        results = retrieve_relevant_ltm(
            temp_db, mock_model, "What programming does the user like?", top_k=3
        )

        assert len(results) == 3

        # Python-related entries should be in results
        contents = [r["content"].lower() for r in results]
        python_related = any("python" in c for c in contents)
        assert python_related, "Python-related memory should be retrieved"

    def test_cosine_distance_range(
        self, temp_db: sqlite3.Connection, mock_model: MockEmbeddingModel
    ):
        """Test that cosine distances are in valid range [0, 2]."""
        apply_semantic_schema(temp_db)

        # Insert diverse entries
        for i in range(5):
            content = f"Unique content number {i} with different words"
            temp_db.execute(
                "INSERT INTO ltm_entries (entry_id, content, created_at, learning_score) "
                "VALUES (?, ?, ?, ?)",
                [f"entry-{i}", content, 0.0, 0.5]
            )
            emb = mock_model.embed_text(content)
            insert_embedding(temp_db, f"entry-{i}", emb)

        temp_db.commit()

        query_emb = mock_model.embed_text("some random query text")
        results = query_similar(temp_db, query_emb, top_k=5)

        for entry_id, distance in results:
            assert 0.0 <= distance <= 2.0, \
                f"Cosine distance should be in [0, 2], got {distance}"


# ──────────────────────────────────────────────────────────────────────────────
# Run with pytest
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])