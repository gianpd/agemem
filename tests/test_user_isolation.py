"""Test that from_yaml respects custom persist_dir for user isolation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents.llm_client import LLMClient
from core.factory import OrchestratorFactory
from core.yaml_config import load_config, to_config_overrides


def _mock_llm(response: str = "Mock response") -> LLMClient:
    """Returns a LLMClient whose underlying client always returns `response`."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=response))]
    )
    return LLMClient(mock_client, default_model="test-model")


class TestUserIsolation:
    """Tests for per-user persist_dir isolation via from_yaml."""

    def test_to_config_overrides_excludes_persist_dir(self):
        """PERSIST_DIR should NOT be in overrides to avoid overwriting user-specific path."""
        cfg = load_config()
        overrides = to_config_overrides(cfg)

        assert "PERSIST_DIR" not in overrides, (
            "PERSIST_DIR should not be in overrides - it's handled by persist_dir argument"
        )

    def test_build_config_uses_passed_persist_dir(self):
        """_build_config should use the passed persist_dir, not YAML's default."""
        factory_config = {"model": "test-model", "max_tokens": 2048, "temperature": 0.2}
        persist_dir = Path("agent_memory/users/alice")

        cfg = load_config()
        overrides = to_config_overrides(cfg)

        config = OrchestratorFactory._build_config(factory_config, persist_dir, overrides)

        assert config.PERSIST_DIR == "agent_memory/users/alice", (
            f"PERSIST_DIR should be 'agent_memory/users/alice', got '{config.PERSIST_DIR}'"
        )

    def test_from_yaml_persist_dir_override(self):
        """Orchestrator created via from_yaml should use passed persist_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = Path(tmpdir) / "agent_memory/users/bob"

            # Create orchestrator with custom persist_dir
            orch = OrchestratorFactory.build(
                llm_client=_mock_llm(),
                persist_dir=user_dir,
                use_learning_scorer=False,
                include_web_tools=False,
            )

            assert orch._config.PERSIST_DIR == str(user_dir), (
                f"Orchestrator PERSIST_DIR should be '{user_dir}', got '{orch._config.PERSIST_DIR}'"
            )

            assert orch._persist_dir == user_dir
            assert orch._stm_persist_path == user_dir / "stm_context.json"