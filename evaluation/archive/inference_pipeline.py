"""
Inference Test Pipeline Module
------------------------------

Executes AgeMem with telemetry capture per Section 3.2 of TRS-AGEMEM-EVAL-001.

Captures:
- SearchTrace instrumentation
- Memory operation logs
- Context window utilization metrics
- Learning score assessments
- Tool execution traces
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from core.types import MemoryEntry, ContextStats
from evaluation.pipeline.dataset_pipeline import BenchmarkEntry, BenchmarkQuery

logger = logging.getLogger(__name__)


@dataclass
class SearchTrace:
    """
    SearchTrace instrumentation per Section 7.6 of the technical specification.

    Captures query execution details for retrieval quality evaluation.
    """
    query: str                                    # Original query text
    query_embedding: list[float] = field(default_factory=list)  # 1024-dim vector
    results: list[tuple[str, float]] = field(default_factory=list)  # (entry_id, score)
    latency_ms: float = 0.0                       # End-to-end retrieval latency
    mode: str = "semantic"                        # "semantic", "overlap", or "expanded"
    variant_used: Optional[str] = None            # Query expansion variant if applicable
    session_id: str = ""                          # Evaluation session ID
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_embedding": self.query_embedding[:10] if self.query_embedding else [],  # Truncate for JSON
            "results": self.results,
            "latency_ms": self.latency_ms,
            "mode": self.mode,
            "variant_used": self.variant_used,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }


@dataclass
class MemoryOpTrace:
    """Trace record for memory operations."""
    op: str                                       # ADD, UPDATE, DELETE, RETRIEVE
    entry_id: str
    content_preview: str = ""                     # First 100 chars of content
    learning_score: float = 0.0
    trigger: str = ""                             # system_rule, memory_agent, learning_score
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionStats:
    """Statistics for an evaluation session."""
    session_id: str
    total_queries: int = 0
    total_turns: int = 0
    memory_ops: int = 0
    avg_latency_ms: float = 0.0
    context_utilization_avg: float = 0.0
    entries_promoted: int = 0
    entries_retrieved: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class InferencePipeline:
    """
    Inference Test Pipeline per Section 3.2 of TRS-AGEMEM-EVAL-001.

    Executes AgeMem with telemetry capture for evaluation.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._db_path = db_path or Path("evaluation/results/traces.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._session_id = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self._db: Optional[sqlite3.Connection] = None
        self._traces: list[SearchTrace] = []
        self._memory_traces: list[MemoryOpTrace] = []
        self._session_stats = SessionStats(session_id=self._session_id)

        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for trace logging per Section 7.6."""
        self._db = sqlite3.connect(str(self._db_path))
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables per Appendix B of TRS-AGEMEM-EVAL-001."""
        # Search traces table
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS search_traces (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                query_embedding BLOB,
                results_json TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                mode TEXT NOT NULL CHECK(mode IN ('semantic', 'overlap', 'expanded')),
                variant_used TEXT,
                session_id TEXT NOT NULL
            )
        """)

        # Memory operation traces table
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS memory_op_traces (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                op TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                content_preview TEXT,
                learning_score REAL,
                trigger TEXT,
                session_id TEXT NOT NULL
            )
        """)

        # Evaluation sessions table
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_sessions (
                session_id TEXT PRIMARY KEY,
                started_at DATETIME,
                ended_at DATETIME,
                total_queries INTEGER DEFAULT 0,
                total_turns INTEGER DEFAULT 0,
                memory_ops INTEGER DEFAULT 0,
                avg_latency_ms REAL DEFAULT 0,
                context_utilization_avg REAL DEFAULT 0,
                entries_promoted INTEGER DEFAULT 0,
                entries_retrieved INTEGER DEFAULT 0
            )
        """)

        # Context utilization table
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS context_utilization (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                total_tokens INTEGER,
                message_count INTEGER,
                utilisation_ratio REAL,
                overflow_risk INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._db.commit()

    # ── Search Tracing ───────────────────────────────────────────────────────

    def trace_search(
        self,
        query: str,
        results: list[tuple[str, float]],
        latency_ms: float,
        mode: str = "semantic",
        query_embedding: Optional[list[float]] = None,
        variant_used: Optional[str] = None,
    ) -> SearchTrace:
        """
        Record a search trace for retrieval quality evaluation.

        Args:
            query: The search query text
            results: List of (entry_id, score) tuples
            latency_ms: Search latency in milliseconds
            mode: Retrieval mode (semantic, overlap, expanded)
            query_embedding: Optional query embedding vector
            variant_used: Query expansion variant if applicable

        Returns:
            SearchTrace object
        """
        trace = SearchTrace(
            query=query,
            query_embedding=query_embedding or [],
            results=results,
            latency_ms=latency_ms,
            mode=mode,
            variant_used=variant_used,
            session_id=self._session_id,
        )

        self._traces.append(trace)
        self._store_search_trace(trace)
        self._session_stats.total_queries += 1

        return trace

    def _store_search_trace(self, trace: SearchTrace) -> None:
        """Store search trace in SQLite database."""
        import struct

        # Serialize embedding to bytes
        embedding_bytes = b""
        if trace.query_embedding:
            embedding_bytes = struct.pack(f"{len(trace.query_embedding)}f", *trace.query_embedding)

        self._db.execute("""
            INSERT INTO search_traces
            (query, query_embedding, results_json, latency_ms, mode, variant_used, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            trace.query,
            embedding_bytes,
            json.dumps(trace.results),
            trace.latency_ms,
            trace.mode,
            trace.variant_used,
            trace.session_id,
        ))
        self._db.commit()

    # ── Memory Operation Tracing ─────────────────────────────────────────────

    def trace_memory_op(
        self,
        op: str,
        entry_id: str,
        content_preview: str = "",
        learning_score: float = 0.0,
        trigger: str = "",
    ) -> MemoryOpTrace:
        """
        Record a memory operation trace.

        Args:
            op: Operation type (ADD, UPDATE, DELETE, RETRIEVE)
            entry_id: The affected entry ID
            content_preview: First 100 chars of content
            learning_score: Learning score for the operation
            trigger: What triggered this operation

        Returns:
            MemoryOpTrace object
        """
        trace = MemoryOpTrace(
            op=op,
            entry_id=entry_id,
            content_preview=content_preview[:100],
            learning_score=learning_score,
            trigger=trigger,
            session_id=self._session_id,
        )

        self._memory_traces.append(trace)
        self._store_memory_trace(trace)
        self._session_stats.memory_ops += 1

        return trace

    def _store_memory_trace(self, trace: MemoryOpTrace) -> None:
        """Store memory operation trace in SQLite database."""
        self._db.execute("""
            INSERT INTO memory_op_traces
            (op, entry_id, content_preview, learning_score, trigger, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            trace.op,
            trace.entry_id,
            trace.content_preview,
            trace.learning_score,
            trace.trigger,
            trace.session_id,
        ))
        self._db.commit()

    # ── Context Utilization Tracking ─────────────────────────────────────────

    def trace_context_utilization(
        self,
        turn_index: int,
        stats: ContextStats,
    ) -> None:
        """
        Record context utilization for a turn.

        Args:
            turn_index: Current turn number
            stats: ContextStats from STM
        """
        self._db.execute("""
            INSERT INTO context_utilization
            (session_id, turn_index, total_tokens, message_count, utilisation_ratio, overflow_risk)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            self._session_id,
            turn_index,
            stats.total_tokens,
            stats.message_count,
            stats.utilisation_ratio,
            1 if stats.overflow_risk else 0,
        ))
        self._db.commit()

        # Update running average
        prev_avg = self._session_stats.context_utilization_avg
        prev_count = self._session_stats.total_turns
        new_count = prev_count + 1
        self._session_stats.context_utilization_avg = (
            (prev_avg * prev_count + stats.utilisation_ratio) / new_count
        )
        self._session_stats.total_turns = new_count

    # ── Batch Execution ───────────────────────────────────────────────────────

    def execute_queries(
        self,
        queries: list[BenchmarkQuery],
        ltm_store: Any,  # LTMStore
        top_k: int = 10,
        mode: str = "semantic",
    ) -> list[SearchTrace]:
        """
        Execute a batch of queries against LTM and capture traces.

        Args:
            queries: List of BenchmarkQuery objects
            ltm_store: LTMStore instance to query
            top_k: Number of results to retrieve
            mode: Retrieval mode (semantic, overlap, expanded)

        Returns:
            List of SearchTrace objects
        """
        traces = []

        for query in queries:
            start_time = time.time()

            # Execute search
            results = ltm_store.search(query.query_text, top_k=top_k)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Convert results to (entry_id, score) format
            result_tuples = []
            for entry in results:
                # Calculate hybrid score for trace
                score = self._calculate_hybrid_score(entry)
                result_tuples.append((entry.entry_id, score))

            # Create trace
            trace = self.trace_search(
                query=query.query_text,
                results=result_tuples,
                latency_ms=latency_ms,
                mode=mode,
            )
            traces.append(trace)

            # Update session stats
            self._session_stats.entries_retrieved += len(results)

        return traces

    def _calculate_hybrid_score(
        self,
        entry: MemoryEntry,
        cosine_similarity: float = 0.8,  # Default for overlap mode
    ) -> float:
        """
        Calculate hybrid score per Section 4.6 of technical specification.

        Formula: 0.60 * cosine_similarity + 0.25 * recency_decay + 0.15 * learning_score
        """
        now = time.time()
        days_elapsed = (now - entry.created_at) / 86400  # Days

        # Recency decay with 7-day half-life
        recency_decay = math.exp(-math.log(2) * days_elapsed / 7)

        score = (
            0.60 * cosine_similarity +
            0.25 * recency_decay +
            0.15 * entry.learning_score
        )
        return score

    # ── Population ───────────────────────────────────────────────────────────

    def populate_ltm(
        self,
        entries: list[BenchmarkEntry],
        ltm_store: Any,  # LTMStore
    ) -> tuple[int, dict[str, str]]:
        """
        Populate LTM with benchmark entries per Phase 1 protocol.

        Args:
            entries: List of BenchmarkEntry objects to add
            ltm_store: LTMStore instance to populate

        Returns:
            Tuple of (count of entries added, mapping from benchmark entry_id to actual entry_id)
        """
        count = 0
        id_mapping: dict[str, str] = {}  # Maps benchmark entry_id -> actual LTM entry_id

        for entry in entries:
            # Store original entry_id for mapping
            benchmark_entry_id = entry.entry_id

            result = ltm_store.add(
                content=entry.content,
                learning_score=entry.learning_score,
                tags=entry.tags,
                source_turn=entry.source_turn,
            )
            if result.success:
                # Map the benchmark entry_id to the actual LTM entry_id
                actual_entry_id = result.entries_affected[0] if result.entries_affected else ""
                id_mapping[benchmark_entry_id] = actual_entry_id

                self.trace_memory_op(
                    op="ADD",
                    entry_id=actual_entry_id,
                    content_preview=entry.content[:100],
                    learning_score=entry.learning_score,
                    trigger="evaluation_setup",
                )
                count += 1
                self._session_stats.entries_promoted += 1

        return count, id_mapping

    # ── Session Management ───────────────────────────────────────────────────

    def start_session(self) -> None:
        """Start a new evaluation session."""
        self._db.execute("""
            INSERT INTO evaluation_sessions (session_id, started_at, total_queries, total_turns)
            VALUES (?, ?, 0, 0)
        """, (self._session_id, self._session_stats.started_at))
        self._db.commit()

    def end_session(self) -> SessionStats:
        """End the current session and return stats."""
        self._session_stats.ended_at = datetime.now().isoformat()

        self._db.execute("""
            UPDATE evaluation_sessions
            SET ended_at = ?, total_queries = ?, total_turns = ?, memory_ops = ?,
                avg_latency_ms = ?, context_utilization_avg = ?,
                entries_promoted = ?, entries_retrieved = ?
            WHERE session_id = ?
        """, (
            self._session_stats.ended_at,
            self._session_stats.total_queries,
            self._session_stats.total_turns,
            self._session_stats.memory_ops,
            self._session_stats.avg_latency_ms,
            self._session_stats.context_utilization_avg,
            self._session_stats.entries_promoted,
            self._session_stats.entries_retrieved,
            self._session_id,
        ))
        self._db.commit()

        return self._session_stats

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_traces(self) -> list[SearchTrace]:
        return self._traces

    def get_memory_traces(self) -> list[MemoryOpTrace]:
        return self._memory_traces

    def get_session_stats(self) -> SessionStats:
        return self._session_stats

    def close(self) -> None:
        """Close the database connection."""
        if self._db:
            self._db.close()