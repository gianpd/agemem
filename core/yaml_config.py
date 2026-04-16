"""
core/yaml_config.py
────────────────────
YAML-based configuration loader for Agemem.

Reads config.yaml (or a custom path via AGEMEM_CONFIG env / --config flag)
and produces typed config objects consumed by OrchestratorFactory.

Priority order (highest wins):
  1. YAML explicit values (non-null)
  2. Environment variables (.env)
  3. Hardcoded defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ── Typed result of loading a YAML config ─────────────────────────────────────

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8010
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    reload: bool = False


@dataclass
class LLMSection:
    base_url: str = "http://localhost:8080"
    model: str = "Qwen3.5-9B-UD-Q4_K_XL.gguf"
    max_tokens: int = 12324
    temperature: float = 0.1
    api_key: Optional[str] = None
    timeout: float = 300.0


@dataclass
class LearningScorerSection:
    enabled: bool = True
    base_url: str = "https://openrouter.ai/api"
    model: str = "google/gemini-3-flash-preview"
    api_key: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.1


@dataclass
class ToolsSection:
    web_search: bool = True
    browser: bool = True
    corpus: bool = True
    introspection: bool = True


@dataclass
class MemorySection:
    persist_dir: str = "agent_memory"
    stm_token_limit: int = 6000
    enable_semantic_search: bool = True
    semantic_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    semantic_embedding_dim: int = 1024
    enable_query_expansion: bool = False
    query_expansion_n_variants: int = 3
    learning_score_prompt_every_n: int = 3
    trigger_every_n_turns: int = 5
    ltm_dedup_threshold: float = 0.92


@dataclass
class SearchSection:
    uwot_enabled: bool = True
    uwot_service_url: str = "http://localhost:8001"
    uwot_api_key: Optional[str] = None
    fetch_only_mentioned_urls: bool = True
    web_search_max_results: int = 5


@dataclass
class BrowserSection:
    cdp_endpoint: str = ""
    connect_over_cdp: bool = False


@dataclass
class AgememYAMLConfig:
    """Complete configuration loaded from YAML."""

    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMSection = field(default_factory=LLMSection)
    learning_scorer: LearningScorerSection = field(default_factory=LearningScorerSection)
    tools: ToolsSection = field(default_factory=ToolsSection)
    memory: MemorySection = field(default_factory=MemorySection)
    search: SearchSection = field(default_factory=SearchSection)
    browser: BrowserSection = field(default_factory=BrowserSection)


# ── Helper: resolve a value with env fallback ──────────────────────────────────

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read an env var. Returns None if unset and no default."""
    val = os.getenv(key)
    return val if val is not None else default


def _resolve_str(yaml_val: Optional[str], env_key: str, default: str) -> str:
    """YAML value → env var → default."""
    if yaml_val is not None:
        return yaml_val
    env_val = os.getenv(env_key)
    return env_val if env_val is not None else default


def _resolve_optional_str(yaml_val: Optional[str], env_key: str) -> Optional[str]:
    """YAML value → env var → None."""
    if yaml_val is not None:
        return yaml_val
    return os.getenv(env_key)


def _resolve_int(yaml_val: Optional[int], env_key: str, default: int) -> int:
    if yaml_val is not None:
        return yaml_val
    env_val = os.getenv(env_key)
    return int(env_val) if env_val is not None else default


def _resolve_float(yaml_val: Optional[float], env_key: str, default: float) -> float:
    if yaml_val is not None:
        return yaml_val
    env_val = os.getenv(env_key)
    return float(env_val) if env_val is not None else default


def _resolve_bool(yaml_val: Optional[bool], env_key: str, default: bool) -> bool:
    if yaml_val is not None:
        return yaml_val
    env_val = os.getenv(env_key)
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    return default


# ── Section parsers ────────────────────────────────────────────────────────────

def _parse_server(raw: dict[str, Any]) -> ServerConfig:
    s = raw.get("server", {})
    return ServerConfig(
        host=_resolve_str(s.get("host"), "API_HOST", "0.0.0.0"),
        port=_resolve_int(s.get("port"), "API_PORT", 8010),
        cors_origins=s.get("cors_origins") or os.getenv("API_CORS_ORIGINS", "*").split(","),
        reload=_resolve_bool(s.get("reload"), "API_RELOAD", False),
    )


def _parse_llm(raw: dict[str, Any]) -> LLMSection:
    s = raw.get("llm", {})
    return LLMSection(
        base_url=_resolve_str(
            s.get("base_url"), "BASE_URL",
            os.getenv("LLAMA_HOST", "http://localhost:8080"),
        ),
        model=_resolve_str(
            s.get("model"), "BASE_MODEL",
            os.getenv("LLAMA_MODEL", "Qwen3.5-9B-UD-Q4_K_XL.gguf"),
        ),
        max_tokens=_resolve_int(
            s.get("max_tokens"), "BASE_MAX_TOKENS",
            int(os.getenv("LLAMA_MAX_TOKENS", "12324")),
        ),
        temperature=_resolve_float(
            s.get("temperature"), "BASE_TEMPERATURE",
            float(os.getenv("LLAMA_TEMPERATURE", "0.1")),
        ),
        api_key=_resolve_optional_str(s.get("api_key"), "API_KEY") or os.getenv("OPENAI_API_KEY"),
        timeout=_resolve_float(s.get("timeout"), "LLM_TIMEOUT", 300.0),
    )


def _parse_learning_scorer(raw: dict[str, Any]) -> LearningScorerSection:
    s = raw.get("learning_scorer", {})
    return LearningScorerSection(
        enabled=_resolve_bool(s.get("enabled"), "LEARNING_SCORER_ENABLED", True),
        base_url=_resolve_str(s.get("base_url"), "LEARNING_SCORER_BASE_URL", "https://openrouter.ai/api"),
        model=_resolve_str(s.get("model"), "LEARNING_SCORER_MODEL", "google/gemini-3-flash-preview"),
        api_key=(
            _resolve_optional_str(s.get("api_key"), "LEARNING_SCORER_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        ),
        max_tokens=_resolve_int(s.get("max_tokens"), "LEARNING_SCORER_MAX_TOKENS", 1024),
        temperature=_resolve_float(s.get("temperature"), "LEARNING_SCORER_TEMPERATURE", 0.1),
    )


def _parse_tools(raw: dict[str, Any]) -> ToolsSection:
    s = raw.get("tools", {})
    return ToolsSection(
        web_search=s.get("web_search", True),
        browser=s.get("browser", True),
        corpus=s.get("corpus", True),
        introspection=s.get("introspection", True),
    )


def _parse_memory(raw: dict[str, Any]) -> MemorySection:
    s = raw.get("memory", {})
    return MemorySection(
        persist_dir=_resolve_str(s.get("persist_dir"), "PERSIST_DIR", "agent_memory"),
        stm_token_limit=_resolve_int(s.get("stm_token_limit"), "STM_TOKEN_LIMIT", 6000),
        enable_semantic_search=_resolve_bool(s.get("enable_semantic_search"), "ENABLE_SEMANTIC_SEARCH", True),
        semantic_embedding_model=s.get("semantic_embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        semantic_embedding_dim=s.get("semantic_embedding_dim", 1024),
        enable_query_expansion=_resolve_bool(s.get("enable_query_expansion"), "ENABLE_QUERY_EXPANSION", False),
        query_expansion_n_variants=s.get("query_expansion_n_variants", 3),
        learning_score_prompt_every_n=s.get("learning_score_prompt_every_n", 3),
        trigger_every_n_turns=s.get("trigger_every_n_turns", 5),
        ltm_dedup_threshold=s.get("ltm_dedup_threshold", 0.92),
    )


def _parse_search(raw: dict[str, Any]) -> SearchSection:
    s = raw.get("search", {})
    return SearchSection(
        uwot_enabled=_resolve_bool(s.get("uwot_enabled"), "UWOT_SEARCH_ENABLED", True),
        uwot_service_url=_resolve_str(s.get("uwot_service_url"), "UWOT_SEARCH_SERVICE_URL", "http://localhost:8001"),
        uwot_api_key=_resolve_optional_str(s.get("uwot_api_key"), "UWOT_API_KEY"),
        fetch_only_mentioned_urls=_resolve_bool(s.get("fetch_only_mentioned_urls"), "FETCH_ONLY_MENTIONED_URLS", True),
        web_search_max_results=_resolve_int(s.get("web_search_max_results"), "WEB_SEARCH_MAX_RESULTS", 5),
    )


def _parse_browser(raw: dict[str, Any]) -> BrowserSection:
    s = raw.get("browser", {})
    return BrowserSection(
        cdp_endpoint=_resolve_str(s.get("cdp_endpoint"), "BROWSER_CDP_ENDPOINT", ""),
        connect_over_cdp=_resolve_bool(s.get("connect_over_cdp"), "BROWSER_CONNECT_OVER_CDP", False),
    )


# ── Public API ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def find_config_path() -> Path:
    """Determine the YAML config path.

    Resolution order:
      1. AGEMEM_CONFIG environment variable
      2. config.yaml in project root (next to pyproject.toml)
    """
    env_path = os.getenv("AGEMEM_CONFIG")
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise FileNotFoundError(f"AGEMEM_CONFIG points to non-existent file: {p}")
        return p

    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH

    raise FileNotFoundError(
        f"No config.yaml found at {DEFAULT_CONFIG_PATH}. "
        "Create one or set AGEMEM_CONFIG to its path."
    )


def load_config(path: Optional[Path] = None) -> AgememYAMLConfig:
    """Load and parse the YAML config file.

    Args:
        path: Explicit path, or None to auto-discover.

    Returns:
        Fully resolved AgememYAMLConfig.
    """
    if path is None:
        path = find_config_path()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return AgememYAMLConfig(
        server=_parse_server(raw),
        llm=_parse_llm(raw),
        learning_scorer=_parse_learning_scorer(raw),
        tools=_parse_tools(raw),
        memory=_parse_memory(raw),
        search=_parse_search(raw),
        browser=_parse_browser(raw),
    )


def to_llm_config(cfg: AgememYAMLConfig) -> "LLMConfig":
    """Convert YAML config to LLMConfig for LLMClientFactory."""
    from core.llm_factory import LLMConfig
    return LLMConfig(
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
        api_key=cfg.llm.api_key,
        timeout=cfg.llm.timeout,
    )


def to_learning_scorer_llm_config(cfg: AgememYAMLConfig) -> "LLMConfig":
    """Convert YAML config to LLMConfig for learning scorer."""
    from core.llm_factory import LLMConfig
    return LLMConfig(
        base_url=cfg.learning_scorer.base_url,
        model=cfg.learning_scorer.model,
        max_tokens=cfg.learning_scorer.max_tokens,
        temperature=cfg.learning_scorer.temperature,
        api_key=cfg.learning_scorer.api_key,
        timeout=60.0,
    )


def to_config_overrides(cfg: AgememYAMLConfig) -> dict[str, Any]:
    """Produce AgememConfig field overrides from the YAML config.

    Note: PERSIST_DIR is NOT included here because it's handled separately
    via the persist_dir argument in OrchestratorFactory.from_yaml/_build_config.
    Including it here would overwrite the user-specific persist_dir passed to from_yaml().
    """
    overrides: dict[str, Any] = {
        "DEFAULT_MODEL": cfg.llm.model,
        "DEFAULT_MAX_TOKENS": cfg.llm.max_tokens,
        "DEFAULT_TEMPERATURE": cfg.llm.temperature,
        "MEMORY_AGENT_MODEL": cfg.llm.model,
        "STM_TOKEN_LIMIT": cfg.memory.stm_token_limit,
        # PERSIST_DIR intentionally excluded - handled by persist_dir argument
        "ENABLE_SEMANTIC_SEARCH": cfg.memory.enable_semantic_search,
        "SEMANTIC_EMBEDDING_MODEL": cfg.memory.semantic_embedding_model,
        "SEMANTIC_EMBEDDING_DIM": cfg.memory.semantic_embedding_dim,
        "ENABLE_QUERY_EXPANSION": cfg.memory.enable_query_expansion,
        "QUERY_EXPANSION_N_VARIANTS": cfg.memory.query_expansion_n_variants,
        "LEARNING_SCORE_PROMPT_EVERY_N": cfg.memory.learning_score_prompt_every_n,
        "TRIGGER_EVERY_N_TURNS": cfg.memory.trigger_every_n_turns,
        "LTM_DEDUP_THRESHOLD": cfg.memory.ltm_dedup_threshold,
        "LEARNING_SCORER_ENABLED": cfg.learning_scorer.enabled,
        "LEARNING_SCORER_BASE_URL": cfg.learning_scorer.base_url,
        "LEARNING_SCORER_MODEL": cfg.learning_scorer.model,
        "LEARNING_SCORER_API_KEY": cfg.learning_scorer.api_key or "",
    }
    return overrides


def resolve_tool_list(cfg: AgememYAMLConfig) -> list[dict]:
    """Assemble the tool list based on YAML tool flags."""
    result: list[dict] = []

    if cfg.tools.corpus:
        from tools.corpus import tool_definitions as corpus_tools
        result.extend(corpus_tools)

    if cfg.tools.introspection:
        from memory import introspection_tool_definitions
        result.extend(introspection_tool_definitions)

    if cfg.tools.web_search:
        from tools.web_tools import tool_definitions as web_tools
        result.extend(web_tools)

    if cfg.tools.browser:
        from tools.browser_tools import tool_definitions as browser_tools
        result.extend(browser_tools)

    return result
