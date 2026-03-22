"""
evaluation/factory.py
─────────────────────
Factory for building Orchestrator instances with dependency injection.

Enables isolated orchestrator creation for evaluation scenarios with
mock LLM clients and isolated storage paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from core.config import AgememConfig, DEFAULT_CONFIG
from core.llm_factory import LLMClientFactory
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator
from tools.corpus import tool_definitions as corpus_tool_definitions
from memory import introspection_tool_definitions
from dotenv import load_dotenv
load_dotenv()


def _get_default_tool_definitions() -> list[dict]:
    """Get the standard tool definitions matching production (main.py).

    This function handles lazy imports for web_tools to avoid dependency issues.
    """
    from tools.web_tools import tool_definitions as web_tool_definitions
    return (
        # web_tool_definitions +
        corpus_tool_definitions +
        introspection_tool_definitions
    )


@dataclass
class OrchestratorBuildConfig:
    """Configuration for building an Orchestrator instance.

    Attributes:
        llm_client: Optional pre-built LLMClient (for mocking in tests).
        persist_dir: Optional path for isolated storage (evaluation runs).
        config_overrides: Optional dict of config values to override.
        tools: Optional list of tool definitions to inject.
    """
    llm_client: Optional[LLMClient] = None
    persist_dir: Optional[Path] = None
    config_overrides: Optional[dict[str, Any]] = None
    tools: Optional[list[dict]] = None


class OrchestratorFactory:
    """Factory for creating Orchestrator instances with dependency injection.

    Primary use case: evaluation pipelines that need isolated orchestrators
    with mock LLMs and separate storage directories.

    Example:
        factory = OrchestratorFactory()
        orch = factory.build_for_evaluation(
            llm_client=mock_llm,
            persist_dir=Path("/tmp/eval_run_1"),
            config_overrides={"STM_TOKEN_LIMIT": 5000}
        )
    """

    def _create_real_llm_client(self) -> LLMClient:
        """Create a real LLM client using configured endpoint.

        Uses LLMClientFactory for consistent client creation across
        main application and evaluation pipeline.

        Returns:
            Configured LLMClient instance.
        """
        factory = LLMClientFactory()
        return factory.create()

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
        """Build an isolated Orchestrator for evaluation.

        Args:
            llm_client: Pre-built LLMClient. If None and use_real_llm=True, creates real client.
            persist_dir: Directory for isolated LTM/STM storage. If None, uses temp dir.
            config_overrides: Optional dict of AgememConfig field values to override.
            tools: Optional list of tool definitions. If None and use_default_tools=True,
                   uses the standard production tool set.
            use_real_llm: If True and llm_client is None, creates real LLM client.
                          Set to False to require explicit mock passing.
            use_default_tools: If True (default), includes web, corpus, and introspection
                               tools matching production behavior.
            use_learning_scorer: If True (default), creates a separate LLM client for
                                 learning scorer using OpenRouter (requires OPENROUTER_API_KEY).

        Returns:
            Orchestrator instance configured for isolated evaluation.
        """
        import tempfile

        # Create LLM client if not provided
        if llm_client is None:
            if use_real_llm:
                llm_client = self._create_real_llm_client()
            else:
                raise ValueError("llm_client is required when use_real_llm=False")

        # Get model from factory config for consistency with LLM client
        llm_factory = LLMClientFactory()
        model = llm_factory.config.model

        # Create learning scorer LLM client (separate from main LLM)
        learning_scorer_llm = None
        if use_learning_scorer and use_real_llm:
            try:
                learning_factory = LLMClientFactory.for_learning_scorer()
                learning_scorer_llm = learning_factory.create()
            except ValueError as e:
                # Learning scorer will fallback to main LLM if no API key
                pass

        # Create persist dir if not provided
        if persist_dir is None:
            persist_dir = Path(tempfile.mkdtemp(prefix="agemem_eval_"))
        config_values: dict[str, Any] = {
            "DEFAULT_MODEL": model,  # Use same model as LLM client
            "MEMORY_AGENT_MODEL": model,  # Consistency with main app
            "STM_TOKEN_LIMIT": DEFAULT_CONFIG.STM_TOKEN_LIMIT,
            "STM_WARNING_THRESHOLD": DEFAULT_CONFIG.STM_WARNING_THRESHOLD,
            "STM_CRITICAL_THRESHOLD": DEFAULT_CONFIG.STM_CRITICAL_THRESHOLD,
            "LTM_PROMOTE_THRESHOLD": DEFAULT_CONFIG.LTM_PROMOTE_THRESHOLD,
            "LEARNING_SCORE_PROMPT_EVERY_N": DEFAULT_CONFIG.LEARNING_SCORE_PROMPT_EVERY_N,
            "TRIGGER_EVERY_N_TURNS": DEFAULT_CONFIG.TRIGGER_EVERY_N_TURNS,
            "DEFAULT_MAX_TOKENS": llm_factory.config.max_tokens,  # Use factory config
            "DEFAULT_TEMPERATURE": llm_factory.config.temperature,  # Use factory config
        }

        # Override persist_dir for isolation
        config_values["PERSIST_DIR"] = str(persist_dir)

        # Apply any user-provided overrides
        if config_overrides:
            config_values.update(config_overrides)

        # Create the config
        cfg = AgememConfig(**config_values)

        # Build orchestrator with injected LLM client and learning scorer
        orch = Orchestrator(
            llm=llm_client,
            config=cfg,
            learning_scorer_llm=learning_scorer_llm,
        )

        # Set tools: use provided tools, or default tools if enabled
        if tools is not None:
            orch.set_tools(tools)
        elif use_default_tools:
            orch.set_tools(_get_default_tool_definitions())

        return orch

    def build_from_config(self, build_config: OrchestratorBuildConfig) -> Orchestrator:
        """Build an Orchestrator from an OrchestratorBuildConfig.

        Alternative entry point when you have a pre-built config object.

        Args:
            build_config: Configuration specifying how to build the orchestrator.

        Returns:
            Orchestrator instance.

        Raises:
            ValueError: If llm_client or persist_dir is not provided.
        """
        if build_config.llm_client is None:
            raise ValueError("llm_client is required for OrchestratorFactory.build_from_config()")
        if build_config.persist_dir is None:
            raise ValueError("persist_dir is required for OrchestratorFactory.build_from_config()")

        return self.build_for_evaluation(
            llm_client=build_config.llm_client,
            persist_dir=build_config.persist_dir,
            config_overrides=build_config.config_overrides,
            tools=build_config.tools,
        )