"""
Tests for evaluation/checkpoint.py and batch_checkpoint.py - CheckpointManager

Tests that checkpoint save/load works correctly, non-existent checkpoints
return None rather than corrupting state, and resume starts from correct offset.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime

from evaluation.batch_checkpoint import (
    CheckpointManager,
    CheckpointState,
    BatchProgress,
)


class TestCheckpointSaveLoad:
    """Test checkpoint save and load operations."""

    def test_save_and_load_returns_identical_data(self, tmp_path: Path, fake_checkpoint_state):
        """Saving and loading a checkpoint returns identical data."""
        manager = CheckpointManager(tmp_path)

        # Save
        saved_path = manager.save_checkpoint(fake_checkpoint_state)
        assert saved_path.exists()

        # Load
        loaded = manager.load_checkpoint(fake_checkpoint_state.session_id)

        assert loaded is not None
        assert loaded.session_id == fake_checkpoint_state.session_id
        assert loaded.config == fake_checkpoint_state.config
        assert loaded.progress.completed_batches == fake_checkpoint_state.progress.completed_batches
        assert loaded.progress.completed_interactions == fake_checkpoint_state.progress.completed_interactions
        assert loaded.aggregated_metrics == fake_checkpoint_state.aggregated_metrics
        assert loaded.status == fake_checkpoint_state.status

    def test_save_creates_file_with_correct_name(self, tmp_path: Path):
        """Checkpoint file is named correctly."""
        manager = CheckpointManager(tmp_path)

        state = CheckpointState(
            session_id="my_test_session",
            config={},
            progress=BatchProgress(),
            aggregated_metrics={},
        )

        path = manager.save_checkpoint(state)

        assert path.name == "my_test_session_checkpoint.json"

    def test_save_updates_timestamp(self, tmp_path: Path):
        """Saving a checkpoint updates the updated_at timestamp."""
        manager = CheckpointManager(tmp_path)

        state = CheckpointState(
            session_id="test",
            config={},
            progress=BatchProgress(),
            aggregated_metrics={},
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
        )

        path = manager.save_checkpoint(state)
        loaded = manager.load_checkpoint("test")

        # updated_at should be different from created_at
        assert loaded.updated_at != "2025-01-01T00:00:00"


class TestLoadNonExistent:
    """Test loading non-existent checkpoints."""

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        """Loading a non-existent checkpoint returns None, not an error."""
        manager = CheckpointManager(tmp_path)

        result = manager.load_checkpoint("does_not_exist")

        assert result is None, "CRITICAL: Non-existent checkpoint should return None, not raise"

    def test_load_nonexistent_does_not_create_file(self, tmp_path: Path):
        """Attempting to load non-existent checkpoint doesn't create files."""
        manager = CheckpointManager(tmp_path)

        manager.load_checkpoint("nonexistent")

        # Directory should still be empty
        files = list(tmp_path.glob("*"))
        assert len(files) == 0


class TestCorruptedCheckpoint:
    """Test handling of corrupted checkpoint files."""

    def test_load_corrupted_returns_none(self, tmp_path: Path):
        """Loading a corrupted JSON file returns None (not crash)."""
        manager = CheckpointManager(tmp_path)

        # Write invalid JSON
        bad_path = tmp_path / "corrupted_checkpoint.json"
        bad_path.write_text("{ invalid json }", encoding="utf-8")

        result = manager.load_checkpoint("corrupted")

        assert result is None


class TestCheckpointResume:
    """Test checkpoint resume functionality."""

    def test_resume_starts_from_saved_offset(self, tmp_path: Path):
        """A resumed run starts from the saved completed_interactions offset."""
        manager = CheckpointManager(tmp_path)

        # Save checkpoint at offset 15
        state = CheckpointState(
            session_id="resume_test",
            config={"batch_size": 10},
            progress=BatchProgress(
                total_interactions=30,
                completed_interactions=15,
                completed_batches=2,
            ),
            aggregated_metrics={"total_queries": 15},
        )
        manager.save_checkpoint(state)

        # Load and verify offset
        loaded = manager.load_checkpoint("resume_test")

        # The key indicator: completed_interactions = 15
        assert loaded.progress.completed_interactions == 15
        assert loaded.progress.total_interactions == 30

        # This tells the runner to start from index 15, not 0
        start_index = loaded.progress.completed_interactions
        assert start_index == 15, "Resume must start from saved offset, not zero"

    def test_list_checkpoints_returns_sorted(self, tmp_path: Path):
        """list_checkpoints returns sorted session IDs."""
        manager = CheckpointManager(tmp_path)

        # Create multiple checkpoints
        for session_id in ["gamma", "alpha", "beta"]:
            state = CheckpointState(
                session_id=session_id,
                config={},
                progress=BatchProgress(),
                aggregated_metrics={},
            )
            manager.save_checkpoint(state)

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3
        assert checkpoints == ["alpha", "beta", "gamma"]  # Sorted


class TestCheckpointStatus:
    """Test checkpoint status management."""

    def test_mark_completed(self, tmp_path: Path):
        """mark_completed updates status."""
        manager = CheckpointManager(tmp_path)

        state = CheckpointState(
            session_id="test_status",
            config={},
            progress=BatchProgress(),
            aggregated_metrics={},
            status="running",
        )
        manager.save_checkpoint(state)

        manager.mark_completed("test_status")

        loaded = manager.load_checkpoint("test_status")
        assert loaded.status == "completed"

    def test_mark_failed(self, tmp_path: Path):
        """mark_failed updates status and stores error."""
        manager = CheckpointManager(tmp_path)

        state = CheckpointState(
            session_id="test_fail",
            config={},
            progress=BatchProgress(),
            aggregated_metrics={},
            status="running",
        )
        manager.save_checkpoint(state)

        manager.mark_failed("test_fail", "Something went wrong")

        loaded = manager.load_checkpoint("test_fail")
        assert loaded.status == "failed"
        assert loaded.aggregated_metrics.get("error") == "Something went wrong"


class TestBatchProgress:
    """Test BatchProgress dataclass."""

    def test_percent_complete_zero_total(self):
        """percent_complete returns 0.0 when total is 0."""
        progress = BatchProgress(total_interactions=0)

        assert progress.percent_complete == 0.0

    def test_percent_complete_calculation(self):
        """percent_complete calculates correctly."""
        progress = BatchProgress(
            total_interactions=100,
            completed_interactions=25,
        )

        assert progress.percent_complete == 25.0


class TestCleanup:
    """Test checkpoint cleanup."""

    def test_cleanup_removes_checkpoint(self, tmp_path: Path):
        """cleanup removes checkpoint file."""
        manager = CheckpointManager(tmp_path)

        state = CheckpointState(
            session_id="cleanup_test",
            config={},
            progress=BatchProgress(),
            aggregated_metrics={},
        )
        manager.save_checkpoint(state)

        assert manager.load_checkpoint("cleanup_test") is not None

        manager.cleanup("cleanup_test")

        assert manager.load_checkpoint("cleanup_test") is None

    def test_cleanup_keeps_checkpoint_when_requested(self, tmp_path: Path):
        """cleanup with keep_checkpoint=True preserves checkpoint."""
        manager = CheckpointManager(tmp_path)

        state = CheckpointState(
            session_id="keep_test",
            config={},
            progress=BatchProgress(),
            aggregated_metrics={},
        )
        manager.save_checkpoint(state)

        manager.cleanup("keep_test", keep_checkpoint=True)

        # Checkpoint should still exist
        assert manager.load_checkpoint("keep_test") is not None


class TestBatchFiles:
    """Test batch file management."""

    def test_get_batch_path(self, tmp_path: Path):
        """get_batch_path returns correct path."""
        manager = CheckpointManager(tmp_path)

        path = manager.get_batch_path("session_123", 5)

        assert path.name == "session_123_batch_5.jsonl"

    def test_list_completed_batches(self, tmp_path: Path):
        """list_completed_batches finds batch files."""
        manager = CheckpointManager(tmp_path)

        # Create some batch files
        for batch_id in [1, 3, 5]:
            path = manager.get_batch_path("test_session", batch_id)
            path.write_text("{}", encoding="utf-8")

        batches = manager.list_completed_batches("test_session")

        assert sorted(batches) == [1, 3, 5]