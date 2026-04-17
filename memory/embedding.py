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
* OOM protection: pre-validates input size (~80k chars / ~30k tokens), chunks large batches,
  and truncates long texts intelligently.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Guardrail constants (derived from Qwen3-Embedding-0.6B specs + safety margin)
# ──────────────────────────────────────────────────────────────────────────────

MAX_BATCH_SIZE = 32
MAX_TEXT_CHARS = 80000
MAX_TEXT_TOKENS_ESTIMATE = 30000
TARGET_BATCH_BYTES = 512 * 1024 * 1024


# ──────────────────────────────────────────────────────────────────────────────
# Cache configuration
# ──────────────────────────────────────────────────────────────────────────────

def _get_cache_path() -> Path:
    """Get the model cache directory, creating it if needed."""
    cache_path = Path.home() / ".cache" / "agemem" / "models" / "hub"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _configure_hf_cache(cache_path: Path) -> None:
    cache_str = str(cache_path)
    os.environ["HF_HOME"] = cache_str
    os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ──────────────────────────────────────────────────────────────────────────────
# Input validation and preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_tokens(char_count: int) -> int:
    """Rough token estimate: ~4 chars per token for most text."""
    return char_count // 3 + 1


def _validate_text_input(text: str, label: str = "text") -> None:
    """Validate single text input. Raises ValueError on clearly invalid input."""
    if not isinstance(text, str):
        raise TypeError(f"Expected string for {label}, got {type(text).__name__}")

    if not text:
        return

    char_count = len(text)
    est_tokens = _estimate_tokens(char_count)

    if est_tokens > MAX_TEXT_TOKENS_ESTIMATE:
        raise ValueError(
            f"{label} too long: ~{est_tokens} tokens estimated "
            f"(max supported: {MAX_TEXT_TOKENS_ESTIMATE}). "
            f"Pre-chunk your text before embedding."
        )


def _truncate_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """
    Truncate text intelligently for embedding quality.

    Strategy: Keep beginning (most semantically important) and tail
    (often contains conclusions/references). For embeddings, the
    middle is typically least informative.
    """
    if len(text) <= max_chars:
        return text

    head_len = int(max_chars * 0.75)
    tail_len = max_chars - head_len - 4

    return text[:head_len] + " ... " + text[-tail_len:] if tail_len > 100 else text[:max_chars]


def _chunk_batch(texts: list[str], max_batch: int = MAX_BATCH_SIZE) -> list[list[str]]:
    """Split large batches into safe-sized chunks."""
    if len(texts) <= max_batch:
        return [texts]
    return [texts[i : i + max_batch] for i in range(0, len(texts), max_batch)]


def _validate_batch_input(texts: list[str]) -> None:
    """Validate batch input. Raises on clearly oversized data."""
    if not isinstance(texts, list):
        raise TypeError(f"Expected list of strings, got {type(texts).__name__}")

    if len(texts) > MAX_BATCH_SIZE * 16:
        raise ValueError(
            f"Batch too large: {len(texts)} items (max: {MAX_BATCH_SIZE * 16}). "
            f"Process in chunks or use embed_batch_streaming."
        )

    total_chars = sum(len(t) for t in texts if isinstance(t, str))
    est_bytes = total_chars * 4

    if est_bytes > TARGET_BATCH_BYTES * 8:
        logger.warning(
            "Large batch detected: ~%.1f MB estimated. "
            "Processing in chunks to avoid OOM.",
            est_bytes / (1024 * 1024),
        )


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingModule class
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingModule:
    """
    Embedding module using Qwen3-Embedding-0.6B for semantic search.

    Provides normalized 1024-dimensional embeddings suitable for
    cosine similarity search.

    OOM Protection:
    - Pre-validates input size before GPU allocation
    - Automatically truncates long texts (head + tail preservation)
    - Chunks large batches automatically
    - Clear error messages for oversized inputs

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
        self._cache_path = cache_path or _get_cache_path()
        _configure_hf_cache(self._cache_path)
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy-load the model on first access (local only, no network)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(
                    self.MODEL_NAME,
                    cache_folder=str(self._cache_path),
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Embedding model '{self.MODEL_NAME}' not found locally. "
                    f"Run: python scripts/preload_model.py  (with internet)"
                    f"Original error: {e}"
                ) from e
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

        Raises:
            TypeError: If text is not a string.
            ValueError: If text is unreasonably large (indicates upstream bug).
        """
        _validate_text_input(text)

        if not text:
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

        processed = _truncate_text(text)

        try:
            embedding = self.model.encode(
                processed,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embedding
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("OOM embedding %d chars. Text: %.100s...", len(text), text)
                raise RuntimeError(
                    f"GPU OOM embedding {len(text)} chars. "
                    "Pre-chunk long texts before calling embed_text."
                ) from e
            raise

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of text strings.

        Automatically chunks large batches to avoid OOM.
        Automatically truncates long texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of normalized embeddings, shape (len(texts), 1024).

        Raises:
            TypeError: If texts is not a list.
            ValueError: If batch is unreasonably large.
        """
        _validate_batch_input(texts)

        if not texts:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)

        if len(texts) == 1:
            single = self.embed_text(texts[0])
            return single.reshape(1, -1)

        processed = [_truncate_text(t) if t else "" for t in texts]
        chunks = _chunk_batch(processed)

        all_embeddings = []
        for chunk in chunks:
            try:
                embeddings = self.model.encode(
                    chunk,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                all_embeddings.append(embeddings)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("OOM in batch chunk (size=%d)", len(chunk))
                    raise RuntimeError(
                        f"GPU OOM in batch chunk (size={len(chunk)}). "
                        f"Reduce batch size or text length."
                    ) from e
                raise

        return np.vstack(all_embeddings)

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
    """Embed a single text string using the default module."""
    return get_embedding_module().embed_text(text)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of text strings using the default module."""
    return get_embedding_module().embed_batch(texts)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("AgeMem Embedding Module - Smoke Test")
    print("=" * 60)

    print("\nInitializing EmbeddingModule...")
    module = EmbeddingModule()
    print(f"Model: {module.MODEL_NAME}")
    print(f"Embedding dimension: {module.EMBEDDING_DIM}")
    print(f"Cache path: {module.cache_path}")

    text_a = "The quick brown fox jumps over the lazy dog."
    text_b = "A fast auburn fox leaps above a sleepy canine."
    text_c = "Machine learning models require large datasets."

    print(f"\nTest strings:")
    print(f"  A: {text_a}")
    print(f"  B: {text_b}")
    print(f"  C: {text_c}")

    print("\nGenerating embeddings...")
    emb_a = module.embed_text(text_a)
    emb_b = module.embed_text(text_b)
    emb_c = module.embed_text(text_c)

    print(f"  Embedding A shape: {emb_a.shape}")
    print(f"  Embedding B shape: {emb_b.shape}")
    print(f"  Embedding C shape: {emb_c.shape}")

    sim_ab = cosine_similarity(emb_a, emb_b)
    sim_ac = cosine_similarity(emb_a, emb_c)
    sim_bc = cosine_similarity(emb_b, emb_c)

    print("\nCosine similarities:")
    print(f"  A <-> B (similar): {sim_ab:.4f}")
    print(f"  A <-> C (different): {sim_ac:.4f}")
    print(f"  B <-> C (different): {sim_bc:.4f}")

    print("\nTesting batch embedding...")
    batch = module.embed_batch([text_a, text_b, text_c])
    print(f"  Batch shape: {batch.shape}")

    match_a = np.allclose(batch[0], emb_a)
    match_b = np.allclose(batch[1], emb_b)
    match_c = np.allclose(batch[2], emb_c)
    print(f"  Batch matches individual: A={match_a}, B={match_b}, C={match_c}")

    print("\nTesting long text truncation...")
    long_text = "word " * 20000  # 100k chars, will be truncated to 80k
    emb_long = module.embed_text(long_text)
    print(f"  Long text ({len(long_text)} chars -> ~{MAX_TEXT_CHARS}) embedded: {emb_long.shape}")

    print("\n" + "=" * 60)
    print("Smoke test complete!")
    print("=" * 60)
