"""
evaluation/checkpoint.py
──────────────────────────────
Checkpoint management for batch evaluation with crash recovery.

Provides atomic persistence of evaluation state, enabling resume
capability after interruptions.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


@dataclass
class BatchProgress:
    """Progress tracking for batch evaluation."""
    total_interactions: int = 0
    completed_interactions: int = 0
    completed_batches: int = 0
    last_batch_id: int = -1
    last_interaction_id: str = ""

    @property
    def percent_complete(self) -> float:
        if self.total_interactions == 0:
            return 0.0
        return (self.completed_interactions / self.total_interactions) * 100


@dataclass
class CheckpointState:
    """Complete checkpoint state for recovery."""
    session_id: str
    config: dict[str, Any]
    progress: BatchProgress
    aggregated_metrics: dict[str, Any]
    status: str = "running"  # "running" | "completed" | "failed"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "config": self.config,
            "progress": asdict(self.progress),
            "aggregated_metrics": self.aggregated_metrics,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        return cls(
            session_id=data["session_id"],
            config=data.get("config", {}),
            progress=BatchProgress(**data.get("progress", {})),
            aggregated_metrics=data.get("aggregated_metrics", {}),
            status=data.get("status", "running"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


class CheckpointManager:
    """
    Manages evaluation state for crash recovery.

    Provides atomic writes (write to temp file, then rename) to ensure
checkpoint consistency even if the process crashes during write.

    Usage:
        manager = CheckpointManager(output_dir=Path("evaluation/results"))

        # Save progress
        state = CheckpointState(
            session_id="eval_20260322_120000",
            config={"batch_size": 10},
            progress=BatchProgress(completed_batches=5),
            aggregated_metrics={"accuracy": 0.75},
        )
        manager.save_checkpoint(state)

        # Resume from checkpoint
        loaded = manager.load_checkpoint("eval_20260322_120000")
        if loaded:
            print(f"Resuming from batch {loaded.progress.completed_batches}")
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory where checkpoint files will be stored.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, session_id: str) -> Path:
        """Get the checkpoint file path for a session."""
        return self.output_dir / f"{session_id}_checkpoint.json"

    def save_checkpoint(self, state: CheckpointState) -> Path:
        """
        Save checkpoint atomically.

        Writes to a temporary file first, then renames to ensure
        atomicity. This prevents corrupt checkpoints if the process
        crashes during write.

        Args:
            state: Checkpoint state to save.

        Returns:
            Path to the saved checkpoint file.
        """
        state.updated_at = datetime.now().isoformat()

        checkpoint_path = self._checkpoint_path(state.session_id)
        temp_path = Path(tempfile.mktemp(
            suffix=".tmp",
            prefix=f"checkpoint_{state.session_id}_",
            dir=str(self.output_dir)
        ))

        try:
            # Write to temp file first
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)

            # Atomic rename (os.replace is atomic on POSIX and Windows)
            os.replace(str(temp_path), str(checkpoint_path))

            return checkpoint_path
        except Exception:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

    def load_checkpoint(self, session_id: str) -> Optional[CheckpointState]:
        """
        Load checkpoint for a session.

        Args:
            session_id: The evaluation session ID.

        Returns:
            CheckpointState if found, None otherwise.
        """
        checkpoint_path = self._checkpoint_path(session_id)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CheckpointState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            # Corrupted checkpoint - treat as non-existent
            print(f"[WARN] Corrupted checkpoint for {session_id}: {e}")
            return None

    def list_checkpoints(self) -> list[str]:
        """
        List all available checkpoint session IDs.

        Returns:
            List of session IDs that have checkpoints.
        """
        checkpoints = []
        for path in self.output_dir.glob("*_checkpoint.json"):
            # Extract session_id from filename (remove _checkpoint.json suffix)
            session_id = path.stem.replace("_checkpoint", "")
            checkpoints.append(session_id)
        return sorted(checkpoints)

    def list_completed_batches(self, session_id: str) -> list[int]:
        """
        List all completed batch IDs for a session.

        Args:
            session_id: The evaluation session ID.

        Returns:
            List of batch IDs that have been written to disk.
        """
        batch_ids = []
        pattern = f"{session_id}_batch_*.jsonl"
        for path in self.output_dir.glob(pattern):
            # Extract batch number from filename
            try:
                batch_str = path.stem.split("_batch_")[-1]
                batch_ids.append(int(batch_str))
            except ValueError:
                continue
        return sorted(batch_ids)

    def get_batch_path(self, session_id: str, batch_id: int) -> Path:
        """Get the file path for a specific batch result file."""
        return self.output_dir / f"{session_id}_batch_{batch_id}.jsonl"

    def mark_completed(self, session_id: str) -> Optional[Path]:
        """
        Mark a checkpoint as completed.

        Args:
            session_id: The evaluation session ID.

        Returns:
            Path to the updated checkpoint file, or None if not found.
        """
        state = self.load_checkpoint(session_id)
        if not state:
            return None

        state.status = "completed"
        return self.save_checkpoint(state)

    def mark_failed(self, session_id: str, error: str) -> Optional[Path]:
        """
        Mark a checkpoint as failed with error info.

        Args:
            session_id: The evaluation session ID.
            error: Error message to store in checkpoint.

        Returns:
            Path to the updated checkpoint file, or None if not found.
        """
        state = self.load_checkpoint(session_id)
        if not state:
            return None

        state.status = "failed"
        state.aggregated_metrics["error"] = error
        state.aggregated_metrics["failed_at"] = datetime.now().isoformat()
        return self.save_checkpoint(state)

    def cleanup(self, session_id: str, keep_checkpoint: bool = False) -> None:
        """
        Clean up checkpoint and batch files for a session.

        Args:
            session_id: The evaluation session ID.
            keep_checkpoint: If True, keep the checkpoint file but remove batches.
        """
        # Remove batch files
        for batch_id in self.list_completed_batches(session_id):
            batch_path = self.get_batch_path(session_id, batch_id)
            if batch_path.exists():
                batch_path.unlink()

        # Remove checkpoint if requested
        if not keep_checkpoint:
            checkpoint_path = self._checkpoint_path(session_id)
            if checkpoint_path.exists():
                checkpoint_path.unlink()