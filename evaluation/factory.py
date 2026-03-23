"""
evaluation/factory.py
─────────────────────
DEPRECATED: Use core.factory.OrchestratorFactory instead.

This module is kept for backward compatibility only.
All functionality has been moved to core/factory.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator


# Re-export for backward compatibility
from core.factory import OrchestratorBuildConfig


class OrchestratorFactory:
    """DEPRECATED: Use core.factory.OrchestratorFactory instead.

    This class is kept for backward compatibility.
    All functionality has been consolidated in core.factory.OrchestratorFactory.
    """

    def __init__(self):
        warnings.warn(
            "evaluation.factory.OrchestratorFactory is deprecated. "
            "Use core.factory.OrchestratorFactory instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def build_for_evaluation(
        self,
        llm_client: Optional[LLMClient] = None,
        persist_dir: Optional[Path] = None,
        config_overrides: Optional[dict[str, Any]] = None,
        tools: Optional[list[dict]] = None,
        use_real_llm: bool = True,
        use_default_tools: bool = True,
        use_learning_scorer: bool = True,
    ) -> Orchestrator:
        """DEPRECATED: Use core.factory.OrchestratorFactory.build() instead.

        Args:
            llm_client: Pre-built LLMClient. If None, creates from environment.
            persist_dir: Directory for isolated storage.
            config_overrides: Optional dict of AgememConfig field values to override.
            tools: Optional list of tool definitions.
            use_real_llm: Ignored (kept for backward compatibility).
            use_default_tools: If True, includes standard tool set.
            use_learning_scorer: Whether to enable learning scorer.

        Returns:
            Orchestrator instance configured for evaluation.
        """
        warnings.warn(
            "build_for_evaluation() is deprecated. Use core.factory.OrchestratorFactory.build() "
            "with include_web_tools=False instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from core.factory import OrchestratorFactory as CoreFactory

        return CoreFactory.build(
            llm_client=llm_client,
            persist_dir=persist_dir,
            config_overrides=config_overrides,
            tools=tools,
            include_web_tools=False,  # Evaluation typically doesn't need web tools
            use_learning_scorer=use_learning_scorer,
        )

    def build_from_config(self, build_config: OrchestratorBuildConfig) -> Orchestrator:
        """DEPRECATED: Use core.factory.OrchestratorFactory.from_config() instead."""
        warnings.warn(
            "build_from_config() is deprecated. Use core.factory.OrchestratorFactory.from_config() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from core.factory import OrchestratorFactory as CoreFactory

        return CoreFactory.from_config(build_config)