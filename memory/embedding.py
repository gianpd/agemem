"""
memory/embedding.py
───────────────────
Embedding module for semantic search using Qwen3-Embedding-0.6B.

Responsibilities
────────────────
* Load and cache the Qwen3-Embedding-0.6B model
* Provide single-text and batch embedding functions
* Return normalized embeddings for cosine similarity

Design decisions
────────────────
* Model cache is configured to ~/.cache/agemem/models to keep
  AgeMem's models separate from other HuggingFace downloads.
* Both HF environment variables and the cache_folder kwarg are set
  to ensure models land in the right place regardless of how
  sentence-transformers internally resolves paths.
* trust_remote_code=True is required for Qwen models with custom code.
* Imports are lazy to avoid requiring sentence_transformers at import time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Cache configuration
# ──────────────────────────────────────────────────────────────────────────────

def _get_cache_path() -> Path:
    """Get the model cache directory, creating it if needed."""
    cache_path = Path.home() / ".cache" / "agemem" / "models"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _configure_hf_cache(cache_path: Path) -> None:
    """
    Configure HuggingFace cache via environment variables.

    Sets HF_HOME, HF_HUB_CACHE, and TRANSFORMERS_CACHE to ensure
    all model downloads go to the specified cache directory.
    """
    cache_str = str(cache_path)
    os.environ["HF_HOME"] = cache_str
    os.environ["HF_HUB_CACHE"] = cache_str
    os.environ["TRANSFORMERS_CACHE"] = cache_str


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingModule class
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingModule:
    """
    Embedding module using Qwen3-Embedding-0.6B for semantic search.

    Provides normalized 1024-dimensional embeddings suitable for
    cosine similarity search.

    Usage:
        embedder = EmbeddingModule()
        embedding = embedder.embed_text("Hello world")
        # embedding.shape == (1024,)

        batch = embedder.embed_batch(["Hello", "World"])
        # batch.shape == (2, 1024)
    """

    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIM = 1024

    _instance: Optional["EmbeddingModule"] = None

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        """
        Initialize the embedding module.

        Args:
            cache_path: Optional custom cache path. Defaults to
                        ~/.cache/agemem/models
        """
        self._cache_path = cache_path or _get_cache_path()
        _configure_hf_cache(self._cache_path)

        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy-load the model on first access."""
        if self._model is None:
            # Lazy import to avoid requiring sentence_transformers at import time
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.MODEL_NAME,
                cache_folder=str(self._cache_path),
                trust_remote_code=True,
            )
        return self._model

    @property
    def cache_path(self) -> Path:
        """Return the configured cache path."""
        return self._cache_path

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A normalized 1024-dimensional embedding vector.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of text strings.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of normalized embeddings, shape (len(texts), 1024).
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings

    @classmethod
    def get_instance(cls) -> "EmbeddingModule":
        """Get or create a singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Since embeddings are already normalized, this is just the dot product.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in range [-1, 1].
    """
    return float(np.dot(a, b))


# ──────────────────────────────────────────────────────────────────────────────
# Convenience functions (module-level)
# ──────────────────────────────────────────────────────────────────────────────

_default_module: Optional[EmbeddingModule] = None


def get_embedding_module() -> EmbeddingModule:
    """Get the default EmbeddingModule instance (singleton)."""
    global _default_module
    if _default_module is None:
        _default_module = EmbeddingModule()
    return _default_module


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text string using the default module.

    Args:
        text: The text to embed.

    Returns:
        A normalized 1024-dimensional embedding vector.
    """
    return get_embedding_module().embed_text(text)


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of text strings using the default module.

    Args:
        texts: List of texts to embed.

    Returns:
        Array of normalized embeddings, shape (len(texts), 1024).
    """
    return get_embedding_module().embed_batch(texts)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("AgeMem Embedding Module - Smoke Test")
    print("=" * 60)

    # Initialize module
    print("\nInitializing EmbeddingModule...")
    module = EmbeddingModule()
    print(f"Model: {module.MODEL_NAME}")
    print(f"Embedding dimension: {module.EMBEDDING_DIM}")
    print(f"Cache path: {module.cache_path}")

    # Test strings
    text_a = "The quick brown fox jumps over the lazy dog."
    text_b = "A fast auburn fox leaps above a sleepy canine."
    text_c = "Machine learning models require large datasets."

    print(f"\nTest strings:")
    print(f"  A: {text_a}")
    print(f"  B: {text_b}")
    print(f"  C: {text_c}")

    # Embed
    print("\nGenerating embeddings...")
    emb_a = module.embed_text(text_a)
    emb_b = module.embed_text(text_b)
    emb_c = module.embed_text(text_c)

    print(f"  Embedding A shape: {emb_a.shape}")
    print(f"  Embedding B shape: {emb_b.shape}")
    print(f"  Embedding C shape: {emb_c.shape}")

    # Cosine similarities
    sim_ab = cosine_similarity(emb_a, emb_b)
    sim_ac = cosine_similarity(emb_a, emb_c)
    sim_bc = cosine_similarity(emb_b, emb_c)

    print("\nCosine similarities:")
    print(f"  A <-> B (similar meaning): {sim_ab:.4f}")
    print(f"  A <-> C (different meaning): {sim_ac:.4f}")
    print(f"  B <-> C (different meaning): {sim_bc:.4f}")

    # Batch embedding test
    print("\nTesting batch embedding...")
    batch = module.embed_batch([text_a, text_b, text_c])
    print(f"  Batch shape: {batch.shape}")

    # Verify batch matches individual embeddings
    match_a = np.allclose(batch[0], emb_a)
    match_b = np.allclose(batch[1], emb_b)
    match_c = np.allclose(batch[2], emb_c)
    print(f"  Batch matches individual: A={match_a}, B={match_b}, C={match_c}")

    print("\n" + "=" * 60)
    print("Smoke test complete!")
    print("=" * 60)