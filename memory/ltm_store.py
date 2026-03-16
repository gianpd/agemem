"""
memory/ltm_store.py
────────────────────
Long-Term Memory store.

Responsibilities
────────────────
* ADD     – store a new MemoryEntry
* UPDATE  – overwrite an existing entry (by entry_id or content similarity)
* DELETE  – remove an entry
* SEARCH  – semantic retrieval using embeddings (or fallback to token overlap)
* PRUNE   – enforce LTM_MAX_ENTRIES by dropping lowest-scored entries

Design decisions
────────────────
* SEMANTIC_SEARCH: When enabled, uses sqlite-vec for vector similarity search.
* Thread safety is not a concern for a single-agent inference loop but a
  reentrant lock is included for completeness.
* Persistence: entries are serialised to JSON. Callers can pass a path to
  enable persistence across sessions.
* SEMANTIC_SEARCH: Optional SQLite backend for vector-based semantic search.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from core.types import MemoryEntry, MemoryOp, MemoryOpResult, TriggerKind
from core.config import AgememConfig, DEFAULT_CONFIG

# SEMANTIC_SEARCH: Lazy imports for embedding and vector modules
if TYPE_CHECKING:
    pass  # numpy already imported above for runtime use

# QUERY_EXPANSION: Import QueryExpander
from tools.query_expansion import QueryExpander

logger = logging.getLogger(__name__)


class LTMStore:

    def __init__(
        self,
        config: AgememConfig = DEFAULT_CONFIG,
        persist_path: Optional[Path] = None,
        # SEMANTIC_SEARCH: New optional parameters for semantic search
        semantic_db_path: Optional[Path] = None,
        enable_semantic_search: bool = False,
        # QUERY_EXPANSION: New optional parameter for query expansion
        llm_client: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._entries: dict[str, MemoryEntry] = {}
        self._lock = threading.RLock()
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load(persist_path)

        # SEMANTIC_SEARCH: Initialize SQLite backend if enabled
        self._semantic_enabled = enable_semantic_search
        self._semantic_db_path = semantic_db_path
        self._db: Optional[sqlite3.Connection] = None
        self._embedding_model = None

        if enable_semantic_search and semantic_db_path:
            self._init_semantic_backend()

        # QUERY_EXPANSION: Initialize QueryExpander if enabled
        self._expander: Optional[QueryExpander] = None
        if config.ENABLE_QUERY_EXPANSION and llm_client is not None:
            self._expander = QueryExpander(
                llm_client=llm_client,
                model=config.MEMORY_AGENT_MODEL,
                n_variants=config.QUERY_EXPANSION_N_VARIANTS,
                use_ner_hints=config.QUERY_EXPANSION_USE_NER_HINTS,
                timeout_ms=config.QUERY_EXPANSION_TIMEOUT_MS,
                fallback_transforms=config.QUERY_EXPANSION_FALLBACK_TRANSFORMS,
                acronym_dict=config.QUERY_EXPANSION_ACRONYM_DICT,
            )

    # SEMANTIC_SEARCH: Initialize SQLite and embedding model
    def _init_semantic_backend(self) -> None:
        """Initialize SQLite database and embedding model for semantic search."""
        try:
            # Apply schema migrations
            from memory.migrations import apply_semantic_schema
            apply_semantic_schema(str(self._semantic_db_path))

            # Open connection
            self._db = sqlite3.connect(str(self._semantic_db_path))

            # Load sqlite-vec extension
            self._db.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(self._db)
            except ImportError:
                logger.warning("sqlite-vec not installed. Semantic search disabled.")
                self._semantic_enabled = False
                return

            # Initialize embedding model (lazy)
            self._embedding_model = None  # Will be loaded on first use

            # Ensure vector index table exists
            from memory.vector_index import ensure_table_exists
            ensure_table_exists(self._db)

            logger.info(f"Semantic search enabled: {self._semantic_db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}")
            self._semantic_enabled = False

    # SEMANTIC_SEARCH: Get or load embedding model
    def _get_embedding_model(self):
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            try:
                from memory.embedding import EmbeddingModule
                self._embedding_model = EmbeddingModule.get_instance()
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                return None
        return self._embedding_model

    # SEMANTIC_SEARCH: Generate embedding for text
    def _generate_embedding(self, text: str) -> Optional["np.ndarray"]:
        """Generate embedding for text, returns None on failure."""
        model = self._get_embedding_model()
        if model is None:
            return None
        try:
            vec = model.embed_text(text)
            # Ensure unit norm for cosine similarity via dot product
            import numpy as np
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        learning_score: float = 0.0,
        tags: list[str] | None = None,
        source_turn: int = 0,
        trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
    ) -> MemoryOpResult:
        """
        Add a new entry.  If a near-duplicate exists and the learning_score is
        above UPDATE_THRESHOLD, updates it instead.
        """
        with self._lock:
            existing = self._find_similar(content)
            if existing and learning_score >= self._config.LTM_UPDATE_THRESHOLD:
                return self.update(
                    existing.entry_id,
                    content=content,
                    learning_score=learning_score,
                    trigger=trigger,
                )

            entry = MemoryEntry(
                content=content,
                learning_score=learning_score,
                tags=tags or [],
                source_turn=source_turn,
            )
            self._entries[entry.entry_id] = entry
            self._maybe_prune()
            self._maybe_persist()

            # SEMANTIC_SEARCH: Insert embedding into vector index
            if self._semantic_enabled and self._db:
                self._insert_embedding_for_entry(entry)

            return MemoryOpResult(
                op=MemoryOp.ADD,
                success=True,
                trigger=trigger,
                detail=f"Stored entry {entry.entry_id}",
                entries_affected=[entry.entry_id],
            )

    def update(
        self,
        entry_id: str,
        content: str,
        learning_score: float = 0.0,
        trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
    ) -> MemoryOpResult:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return MemoryOpResult(
                    op=MemoryOp.UPDATE,
                    success=False,
                    trigger=trigger,
                    detail=f"Entry {entry_id} not found",
                )
            entry.content = content
            entry.updated_at = time.time()
            # Exponential moving average of learning scores
            entry.learning_score = 0.6 * entry.learning_score + 0.4 * learning_score
            self._maybe_persist()

            # SEMANTIC_SEARCH: Update embedding in vector index
            if self._semantic_enabled and self._db:
                self._update_embedding_for_entry(entry)

            return MemoryOpResult(
                op=MemoryOp.UPDATE,
                success=True,
                trigger=trigger,
                detail=f"Updated entry {entry_id}",
                entries_affected=[entry_id],
            )

    def delete(
        self,
        entry_id: str,
        trigger: TriggerKind = TriggerKind.SYSTEM_RULE,
    ) -> MemoryOpResult:
        with self._lock:
            if entry_id not in self._entries:
                return MemoryOpResult(
                    op=MemoryOp.DELETE,
                    success=False,
                    trigger=trigger,
                    detail=f"Entry {entry_id} not found",
                )
            del self._entries[entry_id]
            self._maybe_persist()

            # SEMANTIC_SEARCH: Delete embedding from vector index
            if self._semantic_enabled and self._db:
                from memory.vector_index import delete_embedding
                delete_embedding(self._db, entry_id)
                self._db.commit()

            return MemoryOpResult(
                op=MemoryOp.DELETE,
                success=True,
                trigger=trigger,
                detail=f"Deleted entry {entry_id}",
                entries_affected=[entry_id],
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        expand_query: bool | None = None,
        ner_entities: list[dict] | None = None,
    ) -> list[MemoryEntry]:
        """
        Retrieve the top_k most relevant entries for a query.

        SEMANTIC_SEARCH: When semantic search is enabled, uses vector similarity.
        Falls back to token-overlap scoring when semantic search is unavailable.

        QUERY_EXPANSION: When enabled, generates paraphrase variants and merges results.

        Scoring (semantic): cosine similarity + recency decay + learning_score.
        Scoring (fallback): TF-IDF-inspired token overlap + recency decay + learning_score.
        """
        with self._lock:
            if not self._entries:
                return []

            # Determine effective top_k
            effective_top_k = top_k or self._config.LTM_SEARCH_TOP_K if hasattr(self._config, 'LTM_SEARCH_TOP_K') else top_k

            # QUERY_EXPANSION: Determine if we should expand the query
            should_expand = (
                expand_query
                if expand_query is not None
                else self._config.ENABLE_QUERY_EXPANSION
            )

            # Get query variants
            if should_expand and self._expander is not None:
                queries = self._expander.expand(query, ner_entities=ner_entities)
            else:
                queries = [query]

            # Run search for each query variant and merge results
            all_results: dict[str, tuple[MemoryEntry, float]] = {}

            for q in queries:
                if self._semantic_enabled and self._db:
                    results = self._semantic_search_with_scores(q, top_k=effective_top_k)
                    # Semantic search returns distance (lower is better)
                    # Normalize to similarity [0, 1] where 1 is best for consistent merging
                    for entry, distance in results:
                        # Cosine distance is in [0, 2], convert to similarity
                        similarity = 1.0 - (distance / 2.0)
                        if entry.entry_id not in all_results or similarity > all_results[entry.entry_id][1]:
                            all_results[entry.entry_id] = (entry, similarity)
                else:
                    results = self._token_overlap_search_with_scores(q, top_k=effective_top_k)
                    # Token overlap returns similarity score (higher is better)
                    for entry, score in results:
                        if entry.entry_id not in all_results or score > all_results[entry.entry_id][1]:
                            all_results[entry.entry_id] = (entry, score)

            # Sort by similarity score (descending, higher is better)
            # All scores are now normalized to [0, 1] similarity scale
            merged = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
            return [entry for entry, _ in merged[:effective_top_k]]

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_similarity: Optional[float] = None,
    ) -> list[MemoryEntry]:
        """
        Retrieve LTM entries using a pre-computed query vector.

        This is the core method for context-aware retrieval, allowing
        retrieval based on a weighted context embedding rather than
        just the current query string.

        Args:
            query_vector: Pre-computed embedding vector (must be normalized).
            top_k: Number of results to return.
            min_similarity: Minimum similarity score (0-1) to include a result.
                           Entries below this threshold are filtered out.

        Returns:
            List of MemoryEntry objects sorted by similarity.
        """
        with self._lock:
            if not self._entries:
                return []

            if not self._semantic_enabled or not self._db:
                # Semantic search not available - can't use vector search
                logger.warning("Semantic search not available for vector search")
                return []

            try:
                from memory.vector_index import query_similar

                # Query vector index for similar entries
                # Returns list of (entry_id, cosine_distance) tuples
                candidate_count = top_k * 3  # Retrieve more for filtering
                similar_results = query_similar(self._db, query_vector, limit=candidate_count)

                if not similar_results:
                    return []

                # Convert distances to similarities and filter
                min_sim = min_similarity or 0.0
                scored_entries: list[tuple[float, MemoryEntry]] = []

                for entry_id, distance in similar_results:
                    entry = self._entries.get(entry_id)
                    if not entry:
                        continue

                    # Convert cosine distance to similarity (0-1 scale)
                    # Cosine distance: 0 = identical, 2 = opposite
                    similarity = 1.0 - (distance / 2.0)

                    if similarity < min_sim:
                        continue

                    # Increment access count
                    entry.access_count += 1
                    scored_entries.append((similarity, entry))

                # Sort by similarity descending
                scored_entries.sort(key=lambda x: x[0], reverse=True)

                # Return top_k entries
                return [entry for _, entry in scored_entries[:top_k]]

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

    # SEMANTIC_SEARCH: Semantic search implementation
    def _semantic_search(self, query: str, top_k: int) -> list[MemoryEntry]:
        """Perform semantic search using vector similarity."""
        try:
            from memory.retrieval import retrieve_relevant_ltm

            results = retrieve_relevant_ltm(
                db=self._db,
                model=self._get_embedding_model(),
                query=query,
                top_k=top_k,
            )

            # Convert results to MemoryEntry objects
            entries = []
            for result in results:
                entry = self._entries.get(result["entry_id"])
                if entry:
                    entry.access_count += 1
                    entries.append(entry)

            return entries

        except Exception as e:
            logger.error(f"Semantic search failed, falling back to token overlap: {e}")
            return self._token_overlap_search(query, top_k)

    # QUERY_EXPANSION: Semantic search with scores for merging
    def _semantic_search_with_scores(self, query: str, top_k: int) -> list[tuple[MemoryEntry, float]]:
        """Perform semantic search and return (entry, score) tuples."""
        try:
            from memory.retrieval import retrieve_relevant_ltm

            results = retrieve_relevant_ltm(
                db=self._db,
                model=self._get_embedding_model(),
                query=query,
                top_k=top_k,
            )

            # Convert results to (MemoryEntry, score) tuples
            entries_with_scores = []
            for result in results:
                entry = self._entries.get(result["entry_id"])
                if entry:
                    entry.access_count += 1
                    # Use distance as score (lower is better)
                    score = result.get("distance", 0.0)
                    entries_with_scores.append((entry, score))

            return entries_with_scores

        except Exception as e:
            logger.error(f"Semantic search failed, falling back to token overlap: {e}")
            return self._token_overlap_search_with_scores(query, top_k)

    # SEMANTIC_SEARCH: Original token overlap search (renamed)
    def _token_overlap_search(self, query: str, top_k: int) -> list[MemoryEntry]:
        """
        Fallback retrieval using token overlap scoring.
        All O(n) — adequate for ≤500 entries without any embedding model.
        """
        query_tokens = self._tokenise(query)
        scored: list[tuple[float, MemoryEntry]] = []
        now = time.time()
        for entry in self._entries.values():
            overlap = self._overlap_score(query_tokens, entry.content)
            # Recency: exponential decay over 7 days
            age_days = (now - entry.updated_at) / 86_400
            recency = math.exp(-age_days / 7)
            score = 0.5 * overlap + 0.3 * entry.learning_score + 0.2 * recency
            scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        results = [e for _, e in scored[:top_k]]
        # Increment access count for retrieved entries
        for e in results:
            e.access_count += 1
        return results

    # QUERY_EXPANSION: Token overlap search with scores for merging
    def _token_overlap_search_with_scores(self, query: str, top_k: int) -> list[tuple[MemoryEntry, float]]:
        """
        Fallback retrieval using token overlap scoring, returns (entry, score) tuples.
        """
        query_tokens = self._tokenise(query)
        scored: list[tuple[float, MemoryEntry]] = []
        now = time.time()
        for entry in self._entries.values():
            overlap = self._overlap_score(query_tokens, entry.content)
            # Recency: exponential decay over 7 days
            age_days = (now - entry.updated_at) / 86_400
            recency = math.exp(-age_days / 7)
            score = 0.5 * overlap + 0.3 * entry.learning_score + 0.2 * recency
            scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        results = [(entry, score) for score, entry in scored[:top_k]]
        # Increment access count for retrieved entries
        for entry, _ in results:
            entry.access_count += 1
        return results

    def all_entries(self) -> list[MemoryEntry]:
        with self._lock:
            return list(self._entries.values())

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific entry by ID."""
        with self._lock:
            return self._entries.get(entry_id)

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    # SEMANTIC_SEARCH: Check if semantic search is enabled
    def is_semantic_enabled(self) -> bool:
        """Return True if semantic search is enabled and operational."""
        return self._semantic_enabled and self._db is not None

    # SEMANTIC_SEARCH: Get count of embeddings in vector index
    def embedding_count(self) -> int:
        """Return the number of entries with embeddings."""
        if not self._semantic_enabled or not self._db:
            return 0
        try:
            from memory.vector_index import get_embedding_count
            return get_embedding_count(self._db)
        except Exception:
            return 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> set[str]:
        """Lowercase word tokens, stopwords excluded."""
        STOPWORDS = {
            "the", "a", "an", "is", "in", "on", "at", "to", "of",
            "and", "or", "but", "for", "with", "was", "be", "are",
        }
        return {
            w.lower().strip(".,!?;:\"'()")
            for w in text.split()
            if w.lower() not in STOPWORDS and len(w) > 2
        }

    def _overlap_score(self, query_tokens: set[str], content: str) -> float:
        """Jaccard-like overlap between query and content token sets."""
        content_tokens = self._tokenise(content)
        if not query_tokens or not content_tokens:
            return 0.0
        intersection = query_tokens & content_tokens
        union = query_tokens | content_tokens
        return len(intersection) / len(union)

    def _find_similar(self, content: str) -> Optional[MemoryEntry]:
        """
        Duplicate detection: uses embedding similarity when semantic search is enabled,
        otherwise uses full-content Jaccard overlap.

        SEMANTIC PATH: Uses cosine similarity on embeddings (requires sqlite-vec).
        OVERLAP PATH: Uses Jaccard similarity on tokenised content.

        Known limitations:
        - Overlap path cannot detect paraphrases (BUG2a) — requires semantic search.
        - Overlap path uses token overlap threshold, not semantic understanding.
        """
        # SEMANTIC_DEDUP: Use embedding similarity when semantic search is enabled
        if self._semantic_enabled and self._db is not None:
            vec = self._generate_embedding(content)
            if vec is not None:
                best_id: Optional[str] = None
                best_sim = 0.0
                for eid, entry in self._entries.items():
                    stored_vec = self._get_cached_embedding(eid)
                    if stored_vec is not None:
                        import numpy as np
                        sim = float(np.dot(vec, stored_vec))
                        if sim > best_sim:
                            best_sim, best_id = sim, eid
                if best_sim >= self._config.LTM_DEDUP_THRESHOLD:
                    return self._entries[best_id]
                return None
            # embedding generation failed — fall through to Jaccard
            logger.warning("_find_similar: embedding failed, falling back to Jaccard dedup")

        # OVERLAP_FALLBACK: Use full-content Jaccard similarity.
        # This fixes BUG2b (false-positive prefix collapse) by comparing entire
        # content rather than just leading words. BUG2a (paraphrase detection)
        # remains a known limitation of the overlap path.
        query_tokens = self._tokenise(content)
        best_entry: Optional[MemoryEntry] = None
        best_score = 0.0
        for entry in self._entries.values():
            score = self._overlap_score(query_tokens, entry.content)
            if score > best_score:
                best_score, best_entry = score, entry

        threshold = getattr(self._config, 'LTM_DEDUP_OVERLAP_THRESHOLD', 0.7)
        return best_entry if best_score >= threshold else None

    # SEMANTIC_DEDUP: Helper to get cached embedding for an entry
    def _get_cached_embedding(self, entry_id: str) -> Optional["np.ndarray"]:
        """Retrieve embedding from vector index for an entry."""
        if not self._db:
            return None
        try:
            from memory.vector_index import get_embedding
            return get_embedding(self._db, entry_id)
        except Exception:
            return None

    def _maybe_prune(self) -> None:
        """Drop entries beyond LTM_MAX_ENTRIES, lowest learning_score first."""
        if len(self._entries) <= self._config.LTM_MAX_ENTRIES:
            return
        sorted_ids = sorted(
            self._entries,
            key=lambda eid: self._entries[eid].learning_score,
        )
        excess = len(self._entries) - self._config.LTM_MAX_ENTRIES
        for eid in sorted_ids[:excess]:
            del self._entries[eid]
            # SEMANTIC_SEARCH: Also delete from vector index
            if self._semantic_enabled and self._db:
                from memory.vector_index import delete_embedding
                delete_embedding(self._db, eid)

        # Commit any deletes to SQLite
        if self._semantic_enabled and self._db and excess > 0:
            self._db.commit()

    def _maybe_persist(self) -> None:
        if self._persist_path is None:
            return
        data = [e.to_dict() for e in self._entries.values()]
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
            for item in data:
                entry = MemoryEntry(**item)
                self._entries[entry.entry_id] = entry
        except Exception:
            pass  # corrupt file → start fresh

    # SEMANTIC_SEARCH: Insert embedding for an entry
    def _insert_embedding_for_entry(self, entry: MemoryEntry) -> None:
        """Generate and insert embedding for an entry."""
        if not self._db:
            return
        try:
            embedding = self._generate_embedding(entry.content)
            if embedding is not None:
                from memory.vector_index import insert_embedding
                insert_embedding(self._db, entry.entry_id, embedding)

                # Also update SQLite ltm_entries table with embedding BLOB
                self._upsert_entry_to_sqlite(entry, embedding=embedding)

                # Commit the transaction to persist changes
                self._db.commit()

        except Exception as e:
            logger.error(f"Failed to insert embedding for {entry.entry_id}: {e}")

    # SEMANTIC_SEARCH: Update embedding for an entry
    def _update_embedding_for_entry(self, entry: MemoryEntry) -> None:
        """Generate and update embedding for an entry."""
        if not self._db:
            return
        try:
            embedding = self._generate_embedding(entry.content)
            if embedding is not None:
                from memory.vector_index import update_embedding
                update_embedding(self._db, entry.entry_id, embedding)

                # Also update SQLite ltm_entries table with embedding BLOB
                self._upsert_entry_to_sqlite(entry, embedding=embedding)

                # Commit the transaction to persist changes
                self._db.commit()

        except Exception as e:
            logger.error(f"Failed to update embedding for {entry.entry_id}: {e}")

    # SEMANTIC_SEARCH: Upsert entry to SQLite ltm_entries table
    def _upsert_entry_to_sqlite(
        self,
        entry: MemoryEntry,
        embedding: Optional["np.ndarray"] = None,
    ) -> None:
        """Insert or update an entry in the SQLite ltm_entries table."""
        if not self._db:
            return
        try:
            import json as json_mod

            # Serialize embedding to bytes if provided
            embedding_bytes: Optional[bytes] = None
            if embedding is not None:
                import numpy as np
                embedding_bytes = embedding.astype(np.float32).tobytes()

            self._db.execute("""
                INSERT OR REPLACE INTO ltm_entries
                (entry_id, content, created_at, updated_at, access_count, learning_score, tags, source_turn, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id,
                entry.content,
                entry.created_at,
                entry.updated_at,
                entry.access_count,
                entry.learning_score,
                json_mod.dumps(entry.tags),
                entry.source_turn,
                embedding_bytes,
            ))
        except Exception as e:
            logger.error(f"Failed to upsert entry {entry.entry_id} to SQLite: {e}")

    # SEMANTIC_SEARCH: Sync all entries to SQLite (for migration)
    def sync_to_sqlite(self) -> int:
        """
        Sync all in-memory entries to SQLite database.
        Returns the number of entries synced.
        """
        if not self._db:
            return 0
        count = 0
        for entry in self._entries.values():
            # Generate and insert embedding - this also upserts to ltm_entries with BLOB
            self._insert_embedding_for_entry(entry)
            # Fallback: if embedding generation failed, _insert_embedding_for_entry
            # will have logged the error but NOT written the row. Write it now without BLOB.
            # Check first to avoid overwriting a successfully written embedding.
            row = self._db.execute(
                "SELECT entry_id FROM ltm_entries WHERE entry_id = ?",
                (entry.entry_id,)
            ).fetchone()
            if row is None:
                self._upsert_entry_to_sqlite(entry)  # embedding=None, row didn't exist
            count += 1
        # Commit all changes
        if count > 0:
            self._db.commit()
        return count

    # SEMANTIC_SEARCH: Close database connection
    def close(self) -> None:
        """Close the SQLite database connection if open."""
        if self._db:
            self._db.close()
            self._db = None