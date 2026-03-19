"""
Dataset Pipeline Module
-----------------------

Ingests and validates benchmark datasets per Section 3.1 of TRS-AGEMEM-EVAL-001.

Supports:
- LongMemEval: Long-context memory benchmark
- LoCoMo: Long-context modeling benchmark
- ConvoMem: Conversational memory benchmark
- Custom datasets in JSON, CSV, Parquet formats
"""

from __future__ import annotations

import json
import hashlib
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    """
    Standardized benchmark entry format.

    Maps to MemoryEntry structure per Section 4.2.1 of the technical specification.
    """
    content: str                                    # The memory content
    entry_id: str = ""                              # Auto-generated SHA1 hash
    created_at: float = field(default_factory=time.time)
    learning_score: float = 0.0                     # 0-1 novelty signal
    tags: list[str] = field(default_factory=list)
    source_turn: int = 0

    # Benchmark-specific fields
    query: Optional[str] = None                     # Query that should retrieve this entry
    relevant_ids: list[str] = field(default_factory=list)  # IDs of relevant entries for this query
    temporal_anchor: Optional[float] = None         # Timestamp for temporal reasoning
    entities: dict[str, list[str]] = field(default_factory=dict)  # Extracted entities
    preferences: dict[str, str] = field(default_factory=dict)     # User preferences for ConvoMem

    def __post_init__(self) -> None:
        if not self.entry_id:
            self.entry_id = hashlib.sha1(
                f"{self.content}{self.created_at}".encode()
            ).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkEntry":
        return cls(
            content=data.get("content", ""),
            entry_id=data.get("entry_id", ""),
            created_at=data.get("created_at", time.time()),
            learning_score=data.get("learning_score", 0.0),
            tags=data.get("tags", []),
            source_turn=data.get("source_turn", 0),
            query=data.get("query"),
            relevant_ids=data.get("relevant_ids", []),
            temporal_anchor=data.get("temporal_anchor"),
            entities=data.get("entities", {}),
            preferences=data.get("preferences", {}),
        )


@dataclass
class BenchmarkQuery:
    """
    Query with ground truth relevance judgments.

    Used for retrieval quality evaluation per Phase 1 protocol.
    """
    query_id: str
    query_text: str
    relevant_entry_ids: list[str]              # Ground truth relevant entry IDs
    relevance_scores: dict[str, float] = field(default_factory=dict)  # Optional: graded relevance
    query_type: str = "retrieval"              # retrieval, temporal, preference, etc.
    metadata: dict = field(default_factory=dict)

    # Content-based relevance for when IDs don't match (e.g., LTM generates its own IDs)
    relevant_content: list[str] = field(default_factory=list)  # Content snippets that indicate relevance

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkQuery":
        return cls(
            query_id=data.get("query_id", ""),
            query_text=data.get("query_text", ""),
            relevant_entry_ids=data.get("relevant_entry_ids", []),
            relevance_scores=data.get("relevance_scores", {}),
            query_type=data.get("query_type", "retrieval"),
            metadata=data.get("metadata", {}),
            relevant_content=data.get("relevant_content", []),
        )


@dataclass
class DatasetMetadata:
    """Metadata for a benchmark dataset."""
    name: str
    version: str
    source: str                                    # URL or path
    entry_count: int
    query_count: int
    entity_count: int
    temporal_range: tuple[float, float]            # (earliest, latest) timestamp
    domains: list[str] = field(default_factory=list)
    splits: dict[str, float] = field(default_factory=dict)  # {"train": 0.7, "val": 0.15, "test": 0.15}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    """Validation report per Section 7.7 reporting format."""
    dataset_name: str
    is_valid: bool
    total_entries: int
    total_queries: int
    missing_fields: dict[str, int]               # field_name -> count
    entity_extraction_coverage: float
    temporal_annotation_coverage: float
    cross_reference_errors: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    validated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


class DatasetPipeline:
    """
    Dataset Pipeline per Section 3.1 of TRS-AGEMEM-EVAL-001.

    Ingests and validates benchmark datasets for evaluation.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        default_splits: tuple[float, float, float] = (0.70, 0.15, 0.15),
    ) -> None:
        self._output_dir = output_dir or Path("evaluation/datasets")
        self._default_splits = default_splits
        self._entries: dict[str, BenchmarkEntry] = {}
        self._queries: dict[str, BenchmarkQuery] = {}
        self._metadata: Optional[DatasetMetadata] = None
        self._validation_report: Optional[ValidationReport] = None

    # ── Data Ingestion ───────────────────────────────────────────────────────

    def load_json(self, path: Path) -> dict[str, Any]:
        """Load dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_csv(self, path: Path) -> list[dict[str, Any]]:
        """Load dataset from CSV file."""
        import csv
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(dict(row))
        return entries

    def load_parquet(self, path: Path) -> list[dict[str, Any]]:
        """Load dataset from Parquet file."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            return table.to_pylist()
        except ImportError:
            logger.warning("pyarrow not installed, cannot load Parquet files")
            return []

    def ingest_dataset(
        self,
        path: Path,
        dataset_name: str,
        format: Optional[str] = None,
    ) -> tuple[list[BenchmarkEntry], list[BenchmarkQuery]]:
        """
        Ingest a benchmark dataset from file.

        Args:
            path: Path to dataset file
            dataset_name: Name of the dataset (longmemeval, locomo, convomem, or custom)
            format: File format (json, csv, parquet). Auto-detected if not specified.

        Returns:
            Tuple of (entries, queries)
        """
        # Auto-detect format
        if format is None:
            suffix = path.suffix.lower()
            format_map = {".json": "json", ".csv": "csv", ".parquet": "parquet"}
            format = format_map.get(suffix, "json")

        # Load data
        if format == "json":
            data = self.load_json(path)
        elif format == "csv":
            data = self.load_csv(path)
        elif format == "parquet":
            data = self.load_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Parse based on dataset type
        if dataset_name.lower() == "longmemeval":
            entries, queries = self._parse_longmemeval(data)
        elif dataset_name.lower() == "locomo":
            entries, queries = self._parse_locomo(data)
        elif dataset_name.lower() == "convomem":
            entries, queries = self._parse_convomem(data)
        else:
            entries, queries = self._parse_generic(data)

        # Store entries and queries
        for entry in entries:
            self._entries[entry.entry_id] = entry
        for query in queries:
            self._queries[query.query_id] = query

        logger.info(f"Ingested {len(entries)} entries and {len(queries)} queries from {path}")
        return entries, queries

    def _parse_longmemeval(self, data: dict | list) -> tuple[list[BenchmarkEntry], list[BenchmarkQuery]]:
        """
        Parse LongMemEval format dataset.

        Supports two formats:
        1. Official LongMemEval format (list): Each item contains haystack_sessions with
           chat history, a question, and answer. Evidence is marked with has_answer=True.
        2. Legacy/custom format (dict): {"memories": [...], "questions": [...]}
        """
        entries = []
        queries = []

        # Handle official LongMemEval format (list of evaluation instances)
        if isinstance(data, list):
            entry_id_counter = 0

            for instance in data:
                question_id = instance.get("question_id", f"q_{entry_id_counter}")
                question_text = instance.get("question", "")
                question_type = instance.get("question_type", "retrieval")
                question_date = instance.get("question_date")

                # Extract entries from haystack_sessions
                relevant_entry_ids = []
                haystack_sessions = instance.get("haystack_sessions", [])
                haystack_dates = instance.get("haystack_dates", [])
                session_ids = instance.get("haystack_session_ids", [])

                for session_idx, session in enumerate(haystack_sessions):
                    session_date = haystack_dates[session_idx] if session_idx < len(haystack_dates) else None
                    session_id = session_ids[session_idx] if session_idx < len(session_ids) else f"s_{session_idx}"

                    for turn_idx, turn in enumerate(session):
                        role = turn.get("role", "user")
                        content = turn.get("content", "")
                        has_answer = turn.get("has_answer", False)

                        # Create entry for each turn
                        entry_id = f"{question_id}_{session_id}_{turn_idx}"
                        entry = BenchmarkEntry(
                            content=f"[{role}] {content}",
                            entry_id=entry_id,
                            created_at=time.time(),
                            temporal_anchor=session_date,
                            tags=[question_type, f"session_{session_id}"],
                        )
                        entries.append(entry)

                        # Track relevant entries (evidence)
                        if has_answer:
                            relevant_entry_ids.append(entry_id)

                        entry_id_counter += 1

                # Create query for this instance
                query = BenchmarkQuery(
                    query_id=question_id,
                    query_text=question_text,
                    relevant_entry_ids=relevant_entry_ids,
                    query_type=question_type,
                    metadata={
                        "answer": instance.get("answer", ""),
                        "question_date": question_date,
                        "answer_session_ids": instance.get("answer_session_ids", []),
                    },
                )
                queries.append(query)

            logger.info(f"Parsed LongMemEval: {len(entries)} entries, {len(queries)} queries")
            return entries, queries

        # Handle legacy/custom format (dict with memories/questions)
        memories = data.get("memories", data.get("entries", []))
        questions = data.get("questions", data.get("queries", []))

        for mem in memories:
            entry = BenchmarkEntry(
                content=mem.get("content", mem.get("text", "")),
                entry_id=mem.get("id", mem.get("entry_id", "")),
                created_at=mem.get("timestamp", mem.get("created_at", time.time())),
                temporal_anchor=mem.get("temporal_anchor"),
                entities=mem.get("entities", {}),
                tags=mem.get("tags", []),
            )
            entries.append(entry)

        for q in questions:
            query = BenchmarkQuery(
                query_id=q.get("id", q.get("query_id", "")),
                query_text=q.get("question", q.get("query", q.get("query_text", q.get("text", "")))),
                relevant_entry_ids=q.get("relevant_ids", q.get("relevant_entry_ids", q.get("ground_truth", []))),
                query_type=q.get("type", "retrieval"),
            )
            queries.append(query)

        return entries, queries

    def _parse_locomo(self, data: dict) -> tuple[list[BenchmarkEntry], list[BenchmarkQuery]]:
        """Parse LoCoMo format dataset."""
        entries = []
        queries = []

        # LoCoMo format: {"context": [...], "queries": [...]}
        contexts = data.get("context", data.get("contexts", data.get("entries", [])))
        query_list = data.get("queries", data.get("questions", []))

        for ctx in contexts:
            entry = BenchmarkEntry(
                content=ctx.get("content", ctx.get("text", "")),
                entry_id=ctx.get("id", ctx.get("entry_id", "")),
                created_at=ctx.get("timestamp", time.time()),
                tags=ctx.get("tags", []),
            )
            entries.append(entry)

        for q in query_list:
            query = BenchmarkQuery(
                query_id=q.get("id", ""),
                query_text=q.get("query", q.get("question", "")),
                relevant_entry_ids=q.get("relevant", q.get("ground_truth", [])),
            )
            queries.append(query)

        return entries, queries

    def _parse_convomem(self, data: dict) -> tuple[list[BenchmarkEntry], list[BenchmarkQuery]]:
        """Parse ConvoMem format dataset."""
        entries = []
        queries = []

        # ConvoMem format: {"conversations": [...], "preferences": [...]}
        conversations = data.get("conversations", data.get("entries", []))
        preferences = data.get("preferences", [])

        for conv in conversations:
            entry = BenchmarkEntry(
                content=conv.get("content", conv.get("text", "")),
                entry_id=conv.get("id", ""),
                created_at=conv.get("timestamp", time.time()),
                entities=conv.get("entities", {}),
                preferences=conv.get("preferences", {}),
            )
            entries.append(entry)

        for pref in preferences:
            # Create query for preference recall
            query = BenchmarkQuery(
                query_id=f"pref_{pref.get('key', '')}",
                query_text=pref.get("query", f"What is the user's {pref.get('key', 'preference')}?"),
                relevant_entry_ids=pref.get("relevant_ids", []),
                query_type="preference",
            )
            queries.append(query)

        return entries, queries

    def _parse_generic(self, data: dict) -> tuple[list[BenchmarkEntry], list[BenchmarkQuery]]:
        """Parse generic dataset format."""
        entries = []
        queries = []

        # Try to extract entries
        entry_list = data.get("entries", data.get("memories", data.get("data", [])))
        if isinstance(data, list):
            entry_list = data

        for item in entry_list:
            if isinstance(item, dict):
                entry = BenchmarkEntry.from_dict(item)
                entries.append(entry)

        # Try to extract queries
        query_list = data.get("queries", data.get("questions", []))
        for q in query_list:
            if isinstance(q, dict):
                query = BenchmarkQuery.from_dict(q)
                queries.append(query)

        return entries, queries

    # ── Validation ───────────────────────────────────────────────────────────

    def validate(self) -> ValidationReport:
        """
        Validate the loaded dataset per Section 3.1.3 of TRS-AGEMEM-EVAL-001.

        Checks:
        - Completeness: all required fields present
        - Consistency: cross-reference integrity
        - Quality: entity extraction, temporal annotation coverage
        """
        missing_fields: dict[str, int] = {}
        errors = []
        warnings = []

        # Check entries
        for entry_id, entry in self._entries.items():
            if not entry.content:
                missing_fields["content"] = missing_fields.get("content", 0) + 1
            if not entry.entry_id:
                missing_fields["entry_id"] = missing_fields.get("entry_id", 0) + 1

        # Check cross-references
        cross_ref_errors = 0
        for query in self._queries.values():
            for ref_id in query.relevant_entry_ids:
                if ref_id not in self._entries:
                    cross_ref_errors += 1
                    errors.append(f"Query {query.query_id} references non-existent entry {ref_id}")

        # Calculate coverage metrics
        entity_count = sum(1 for e in self._entries.values() if e.entities)
        temporal_count = sum(1 for e in self._entries.values() if e.temporal_anchor is not None)

        entity_coverage = entity_count / len(self._entries) if self._entries else 0.0
        temporal_coverage = temporal_count / len(self._entries) if self._entries else 0.0

        # Create report
        is_valid = len(errors) == 0 and sum(missing_fields.values()) == 0

        report = ValidationReport(
            dataset_name=self._metadata.name if self._metadata else "unknown",
            is_valid=is_valid,
            total_entries=len(self._entries),
            total_queries=len(self._queries),
            missing_fields=missing_fields,
            entity_extraction_coverage=entity_coverage,
            temporal_annotation_coverage=temporal_coverage,
            cross_reference_errors=cross_ref_errors,
            warnings=warnings,
            errors=errors,
        )

        self._validation_report = report
        return report

    # ── Data Partitioning ─────────────────────────────────────────────────────

    def partition(
        self,
        splits: Optional[tuple[float, float, float]] = None,
    ) -> dict[str, list[BenchmarkEntry]]:
        """
        Partition dataset into train/val/test sets.

        Args:
            splits: Tuple of (train, val, test) ratios. Default: (0.70, 0.15, 0.15)

        Returns:
            Dictionary with "train", "val", "test" keys
        """
        splits = splits or self._default_splits
        train_ratio, val_ratio, test_ratio = splits

        entries = list(self._entries.values())
        total = len(entries)

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        return {
            "train": entries[:train_end],
            "val": entries[train_end:val_end],
            "test": entries[val_end:],
        }

    # ── Output ───────────────────────────────────────────────────────────────

    def export(self, output_path: Optional[Path] = None) -> Path:
        """Export validated dataset to JSON file."""
        output_path = output_path or self._output_dir / "validated_dataset.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entries": [e.to_dict() for e in self._entries.values()],
            "queries": [q.to_dict() for q in self._queries.values()],
            "metadata": self._metadata.to_dict() if self._metadata else {},
            "validation": self._validation_report.to_dict() if self._validation_report else {},
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported dataset to {output_path}")
        return output_path

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_entries(self) -> list[BenchmarkEntry]:
        return list(self._entries.values())

    def get_queries(self) -> list[BenchmarkQuery]:
        return list(self._queries.values())

    def get_entry(self, entry_id: str) -> Optional[BenchmarkEntry]:
        return self._entries.get(entry_id)

    def get_query(self, query_id: str) -> Optional[BenchmarkQuery]:
        return self._queries.get(query_id)

    def size(self) -> int:
        return len(self._entries)

    def query_count(self) -> int:
        return len(self._queries)