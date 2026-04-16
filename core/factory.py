"""
core/factory.py
───────────────
Factory for building Orchestrator instances with production-ready defaults.

Single entry point for:
- Production REPL (main.py)
- E2E evaluation (run_e2e_longmemeval.py)
- Unit tests with mocks
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.config import AgememConfig, DEFAULT_CONFIG
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator
from core.yaml_config import AgememYAMLConfig, load_config, to_llm_config, to_learning_scorer_llm_config, to_config_overrides, resolve_tool_list


@dataclass
class OrchestratorBuildConfig:
    """Immutable configuration for advanced orchestrator building.

    Most users should use OrchestratorFactory.build() directly with kwargs.
    This dataclass exists for cases where configuration needs to be passed
    around as data (e.g., CLI parsing, config files).
    """
    llm_client: Optional[LLMClient] = None
    persist_dir: Optional[Path] = None
    config_overrides: Optional[dict[str, Any]] = None
    tools: Optional[list[dict]] = None
    include_web_tools: bool = True
    use_learning_scorer: bool = True


class OrchestratorFactory:
    """Factory for building Orchestrator instances.

    Single entry point with sensible defaults:

    - Production: OrchestratorFactory.build()
    - Evaluation: OrchestratorFactory.build(persist_dir=Path(...), include_web_tools=False)
    - Tests: OrchestratorFactory.build(llm_client=mock_llm, persist_dir=...)

    The factory hides:
    - LLM client creation (delegates to LLMClientFactory)
    - Learning scorer client creation with graceful fallback
    - Config construction (syncs model/temperature/max_tokens from LLM factory)
    - Tool assembly (lazy imports, conditional web tools)
    - Environment variable reading (all inside factory)

    Example:
        # Production REPL
        orch = OrchestratorFactory.build()

        # E2E evaluation with isolated storage
        orch = OrchestratorFactory.build(
            persist_dir=Path("/tmp/eval_session"),
            include_web_tools=False,
        )

        # Unit test with mock LLM
        orch = OrchestratorFactory.build(
            llm_client=mock_llm,
            persist_dir=Path(tempfile.mkdtemp()),
            use_learning_scorer=False,
        )
    """

    @classmethod
    def build(
        cls,
        llm_client: Optional[LLMClient] = None,
        persist_dir: Optional[Path] = None,
        config_overrides: Optional[dict[str, Any]] = None,
        tools: Optional[list[dict]] = None,
        include_web_tools: bool = True,
        use_learning_scorer: bool = True,
    ) -> Orchestrator:
        """Build an Orchestrator with production-ready defaults.

        Args:
            llm_client: Pre-built LLMClient. If None, creates from environment.
            persist_dir: Storage directory. If None, uses PERSIST_DIR from env.
            config_overrides: Override specific AgememConfig fields.
            tools: Custom tool list. If None, assembles from include_web_tools flag.
            include_web_tools: Whether to include web tools (default True).
                              Set False for evaluation runs to avoid network calls.
            use_learning_scorer: Whether to enable learning scorer (default True).
                                Requires OPENROUTER_API_KEY for full functionality.

        Returns:
            Configured Orchestrator instance ready for use.
        """
        # Resolve dependencies
        main_llm, learning_scorer_llm, factory_config = cls._create_llm_clients(
            llm_client, use_learning_scorer
        )

        resolved_persist_dir = cls._resolve_persist_dir(persist_dir)
        config = cls._build_config(factory_config, resolved_persist_dir, config_overrides)
        resolved_tools = cls._resolve_tools(tools, include_web_tools)

        # Build orchestrator
        orch = Orchestrator(
            llm=main_llm,
            config=config,
            learning_scorer_llm=learning_scorer_llm,
        )
        orch.set_tools(resolved_tools)

        return orch

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
    ) -> Orchestrator:
        """Build an Orchestrator from a YAML config file.

        The YAML file is the single source of truth for API startup.
        Environment variables fill in null/missing values; explicit YAML
        values take precedence.

        Args:
            yaml_path: Path to config.yaml. None auto-discovers via AGEMEM_CONFIG env
                       or falls back to config.yaml in the project root.
            persist_dir: Optional override for persist_dir (e.g. per-user isolation).
                         If None, uses the value from the YAML config.

        Returns:
            Configured Orchestrator instance.
        """
        cfg = load_config(yaml_path)

        # Create LLM clients from YAML config
        from core.llm_factory import LLMClientFactory

        main_llm_config = to_llm_config(cfg)
        factory = LLMClientFactory(main_llm_config)
        main_llm = factory.create()

        factory_config = {
            "model": main_llm_config.model,
            "max_tokens": main_llm_config.max_tokens,
            "temperature": main_llm_config.temperature,
        }

        # Learning scorer (optional)
        learning_scorer_llm = None
        if cfg.learning_scorer.enabled:
            try:
                ls_config = to_learning_scorer_llm_config(cfg)
                ls_factory = LLMClientFactory(ls_config)
                learning_scorer_llm = ls_factory.create()
            except ValueError:
                pass  # Graceful degradation

        # Resolve persist dir
        resolved_persist = persist_dir or Path(cfg.memory.persist_dir)

        # Build AgememConfig with YAML overrides
        overrides = to_config_overrides(cfg)
        config = cls._build_config(factory_config, resolved_persist, overrides)

        # Resolve tools from YAML flags
        tools = resolve_tool_list(cfg)

        # Build orchestrator
        orch = Orchestrator(
            llm=main_llm,
            config=config,
            learning_scorer_llm=learning_scorer_llm,
        )
        orch.set_tools(tools)
        return orch

    @classmethod
    def from_config(cls, config: OrchestratorBuildConfig) -> Orchestrator:
        """Build from a pre-built config object.

        Alternative entry point for CLI tools or config-file-driven setups.

        Args:
            config: OrchestratorBuildConfig with all build parameters.

        Returns:
            Configured Orchestrator instance.
        """
        return cls.build(
            llm_client=config.llm_client,
            persist_dir=config.persist_dir,
            config_overrides=config.config_overrides,
            tools=config.tools,
            include_web_tools=config.include_web_tools,
            use_learning_scorer=config.use_learning_scorer,
        )

    # --- Internal helpers ---

    @classmethod
    def _create_llm_clients(
        cls,
        llm_client: Optional[LLMClient],
        use_learning_scorer: bool,
    ) -> tuple[LLMClient, Optional[LLMClient], dict]:
        """Create LLM clients and return config-derived values.

        Returns:
            Tuple of (main_llm, learning_scorer_llm, factory_config)
            where factory_config contains model, max_tokens, temperature
            for syncing with AgememConfig.
        """
        from core.llm_factory import LLMClientFactory

        # Create main LLM if not provided
        if llm_client is None:
            factory = LLMClientFactory()
            main_llm = factory.create()
            factory_config = {
                "model": factory.config.model,
                "max_tokens": factory.config.max_tokens,
                "temperature": factory.config.temperature,
            }
        else:
            main_llm = llm_client
            # Use defaults when LLM is injected
            factory_config = {
                "model": "injected",
                "max_tokens": 2048,
                "temperature": 0.2,
            }

        # Create learning scorer LLM if enabled
        learning_scorer_llm = None
        if use_learning_scorer and llm_client is None:
            try:
                learning_factory = LLMClientFactory.for_learning_scorer()
                learning_scorer_llm = learning_factory.create()
            except ValueError:
                # Graceful degradation - learning scorer will use main LLM
                pass

        return main_llm, learning_scorer_llm, factory_config

    @classmethod
    def _resolve_persist_dir(cls, persist_dir: Optional[Path]) -> Path:
        """Resolve persistence directory from arg or environment."""
        if persist_dir is not None:
            return persist_dir

        return Path(os.getenv("PERSIST_DIR", "agent_memory"))

    @classmethod
    def _build_config(
        cls,
        factory_config: dict,
        persist_dir: Path,
        overrides: Optional[dict[str, Any]],
    ) -> AgememConfig:
        """Build AgememConfig with synced defaults."""
        config_values: dict[str, Any] = {
            "DEFAULT_MODEL": factory_config["model"],
            "MEMORY_AGENT_MODEL": factory_config["model"],
            "STM_TOKEN_LIMIT": int(os.getenv("STM_TOKEN_LIMIT", "6000")),
            "STM_WARNING_THRESHOLD": DEFAULT_CONFIG.STM_WARNING_THRESHOLD,
            "STM_CRITICAL_THRESHOLD": DEFAULT_CONFIG.STM_CRITICAL_THRESHOLD,
            "LTM_PROMOTE_THRESHOLD": DEFAULT_CONFIG.LTM_PROMOTE_THRESHOLD,
            "LEARNING_SCORE_PROMPT_EVERY_N": DEFAULT_CONFIG.LEARNING_SCORE_PROMPT_EVERY_N,
            "TRIGGER_EVERY_N_TURNS": DEFAULT_CONFIG.TRIGGER_EVERY_N_TURNS,
            "DEFAULT_MAX_TOKENS": factory_config["max_tokens"],
            "DEFAULT_TEMPERATURE": factory_config["temperature"],
            "PERSIST_DIR": str(persist_dir),
        }

        if overrides:
            config_values.update(overrides)

        return AgememConfig(**config_values)

    @classmethod
    def _resolve_tools(
        cls,
        tools: Optional[list[dict]],
        include_web_tools: bool,
    ) -> list[dict]:
        """Resolve tool list from arg or assemble from flags."""
        if tools is not None:
            return tools

        from tools.corpus import tool_definitions as corpus_tools
        from memory import introspection_tool_definitions

        result = corpus_tools + introspection_tool_definitions

        if include_web_tools:
            # Lazy import to avoid dependency issues
            from tools.web_tools import tool_definitions as web_tools
            from tools.browser_tools import tool_definitions as browser_tools
            result = web_tools + browser_tools + result

        return result