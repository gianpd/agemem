"""
Dataset loader for LongMemEval format.

Reconciles loading logic from run.py and batch_runner.py into a single interface.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loader for LongMemEval format datasets.

    This class reconciles the dataset loading implementations from:
    - run.py::load_dataset (lines 124-197)
    - batch_runner.py::_load_dataset (lines 556-607)

    Key reconciliations between the two implementations:

    1. LIMIT APPLICATION POINT:
       - run.py applies limit to raw_data BEFORE processing (queries-first limit)
       - batch_runner.py applies limit AFTER processing (entry-first limit)
       - CHOSEN: run.py's approach - limiting queries before processing ensures
         consistent entry/query pairing and avoids orphaned queries.

    2. RELEVANT ENTRY TRACKING:
       - run.py tracks has_answer and builds relevant_entry_ids/relevant_content
       - batch_runner.py always returns empty lists for these fields
       - CHOSEN: run.py's approach - this data is essential for evaluation metrics.

    3. SESSION ID NAMING:
       - run.py uses haystack_session_ids field when available, falls back to index
       - batch_runner.py only uses numeric index
       - CHOSEN: run.py's approach - preserves semantic session identity.

    4. QUESTION ID FALLBACK:
       - run.py generates "q_{counter}" as fallback
       - batch_runner.py uses empty string ""
       - CHOSEN: run.py's approach - non-empty IDs are required for downstream processing.

    5. LOAD_SESSIONS FLAG:
       - run.py has load_sessions flag to optionally skip session processing
       - batch_runner.py always loads sessions
       - CHOSEN: Include the flag, default True for backward compatibility.
    """

    def __init__(self):
        self._entry_id_counter = 0

    def load(
        self,
        path: Path | str,
        subset: str | None = None,
        limit: int = 0,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Load a LongMemEval dataset.

        Args:
            path: Path to the dataset JSON file.
            subset: Optional subset filter. Currently supports:
                    - None or "all": Load everything (default)
                    - "sessions": Load sessions (equivalent to load_sessions=True)
                    - Any other value: Treated as load_sessions=True for backward compat
            limit: Max number of query instances to load (0 = all).
                   Applied BEFORE processing to ensure entry/query consistency.

        Returns:
            Tuple of (entries, queries, raw_data):
            - entries: List of entry dicts with content, entry_id, and tags
            - queries: List of query dicts with query_id, query_text, relevant_entry_ids,
                       relevant_content, query_type, and expected_answer
            - raw_data: The original loaded JSON data (limited if limit > 0)
        """
        dataset_path = Path(path)
        logger.info(f"Loading dataset from {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # DIFFERENCE RESOLVED: run.py limits BEFORE processing, batch_runner.py limits AFTER.
        # Chosen run.py's approach: limiting queries first ensures entries are only built
        # for the queries we'll actually use, preventing mismatched entry/query counts.
        if limit > 0:
            raw_data = raw_data[:limit]
            logger.info(f"Limited to {len(raw_data)} query instances")

        # Determine if we should load sessions based on subset parameter
        # subset=None or "all" -> load sessions (default behavior for batch_runner compatibility)
        # subset="sessions" -> load sessions (explicit)
        # Any other value -> load sessions (backward compatible default)
        load_sessions = subset is None or subset != "queries_only"

        entries, queries = self._build_entries_and_queries(raw_data, load_sessions)

        logger.info(f"Loaded {len(entries)} entries and {len(queries)} queries")
        return entries, queries, raw_data

    def _build_entries_and_queries(
        self,
        raw_data: list[dict],
        load_sessions: bool,
    ) -> tuple[list[dict], list[dict]]:
        """
        Build entries and queries from raw LongMemEval format data.

        Args:
            raw_data: List of query instances from the dataset.
            load_sessions: If True, process haystack_sessions into entries.
                          If False, only build queries (faster for metadata-only operations).

        Returns:
            Tuple of (entries, queries).
        """
        entries = []
        queries = []
        self._entry_id_counter = 0

        for instance in raw_data:
            # DIFFERENCE RESOLVED: run.py generates "q_{counter}", batch_runner.py uses "".
            # Chosen run.py's approach: non-empty IDs are required for entry_id generation.
            question_id = instance.get("question_id") or f"q_{self._entry_id_counter}"
            question_text = instance.get("question", "")
            question_type = instance.get("question_type", "retrieval")
            answer = instance.get("answer", "")

            relevant_entry_ids = []
            relevant_content = []

            # DIFFERENCE RESOLVED: run.py uses haystack_session_ids for semantic naming,
            # batch_runner.py only uses numeric index.
            # Chosen run.py's approach: preserves session identity from source data.
            session_ids = instance.get("haystack_session_ids", [])

            if load_sessions:
                haystack_sessions = instance.get("haystack_sessions", [])
                for session_idx, session in enumerate(haystack_sessions):
                    # Use semantic session_id if available, otherwise use index
                    session_id = (
                        session_ids[session_idx]
                        if session_idx < len(session_ids)
                        else f"s_{session_idx}"
                    )

                    for turn_idx, turn in enumerate(session):
                        role = turn.get("role", "user")
                        content = turn.get("content", "")
                        has_answer = turn.get("has_answer", False)

                        entry_id = f"{question_id}_{session_id}_{turn_idx}"
                        entries.append({
                            "content": f"[{role}] {content}",
                            "entry_id": entry_id,
                            "tags": [question_type, f"session_{session_id}"],
                        })

                        # DIFFERENCE RESOLVED: run.py tracks has_answer for relevant entries,
                        # batch_runner.py omits this tracking.
                        # Chosen run.py's approach: essential for evaluation metrics.
                        if has_answer:
                            relevant_entry_ids.append(entry_id)
                            relevant_content.append(content)

                        self._entry_id_counter += 1

            queries.append({
                "query_id": question_id,
                "query_text": question_text,
                "relevant_entry_ids": relevant_entry_ids,
                "relevant_content": relevant_content,
                "query_type": question_type,
                "expected_answer": answer,
            })

        return entries, queries

    # Convenience methods for backward compatibility with callers

    def load_full(
        self,
        path: Path | str,
        limit: int = 0,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Load dataset with sessions (equivalent to run.py load_sessions=True).

        Convenience method for callers that always want session data.
        """
        return self.load(path, subset=None, limit=limit)

    def load_queries_only(
        self,
        path: Path | str,
        limit: int = 0,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Load dataset without session processing.

        Convenience method for metadata-only operations where entries aren't needed.
        Returns empty entries list but fully populated queries.
        """
        return self.load(path, subset="queries_only", limit=limit)


# Module-level function for drop-in compatibility with run.py::load_dataset
def load_dataset(
    dataset_path: Path,
    query_limit: int = 0,
    load_sessions: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load LongMemEval dataset.

    Drop-in replacement for run.py::load_dataset for backward compatibility.
    See DatasetLoader.load() for full documentation.
    """
    loader = DatasetLoader()
    subset = None if load_sessions else "queries_only"
    return loader.load(dataset_path, subset=subset, limit=query_limit)