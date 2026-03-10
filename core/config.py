"""
core/config.py
──────────────
All tunable thresholds in one place.

Acceptance criteria addressed:
  AC-1  Context does not explode   →  STM_TOKEN_LIMIT + WARNING_THRESHOLD
  AC-2  LTM/STM managed by scores →  LTM_PROMOTE_THRESHOLD + STM_EVICT_THRESHOLD
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os 
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG — Agent 1 (Local Qwen)
# ─────────────────────────────────────────────────────────────
def get_env_with_fallback(new_name: str, old_name: str, default: str) -> str:
    """Get env var with fallback to old name for backward compatibility."""
    import warnings
    value = os.getenv(new_name) or os.getenv(old_name, default)
    return value

BASE_URL = get_env_with_fallback("BASE_URL", "LLAMA_HOST", "http://localhost:8080")
MODEL_NAME = get_env_with_fallback("BASE_MODEL", "LLAMA_MODEL", "qwen3.5-9b")
MAX_TOKENS = int(get_env_with_fallback("BASE_MAX_TOKENS", "LLAMA_MAX_TOKENS", "10324"))
TEMPERATURE = float(get_env_with_fallback("BASE_TEMPERATURE", "LLAMA_TEMPERATURE", "0.1"))
MAX_STEPS = int(os.getenv("LLAMA_MAX_STEPS", "50"))
SHOW_THINKING = os.getenv("SHOW_THINKING", "false").lower() == "true"
CORPUS = Path("corpus")

# ─────────────────────────────────────────────────────────────
# CONFIG — Agent 2 & 3 (Oracle + Archivist, External API)
# ─────────────────────────────────────────────────────────────
ORACLE_ENABLED = os.getenv("ORACLE_ENABLED", "true").lower() == "true"
ORACLE_API_KEY = os.getenv("ORACLE_API_KEY", "")
ORACLE_BASE_URL = os.getenv("ORACLE_BASE_URL", "https://api.openai.com")
ORACLE_MODEL = os.getenv("ORACLE_MODEL", "gpt-4o-mini")

# ─────────────────────────────────────────────────────────────
# CONFIG — uWOT Search Service
# ─────────────────────────────────────────────────────────────
UWOT_SEARCH_ENABLED = os.getenv("UWOT_SEARCH_ENABLED", "true").lower() == "true"
UWOT_SEARCH_SERVICE_URL = os.getenv("UWOT_SEARCH_SERVICE_URL", "http://localhost:8001")
UWOT_API_KEY = os.getenv("UWOT_API_KEY")


SLIDING_WINDOW_TURNS = 4
SUMMARY_MAX_TOKENS = 1200
TOP_FACTS_COUNT = 5

# ─────────────────────────────────────────────────────────────
# CONFIG — Context window
# ─────────────────────────────────────────────────────────────
CTX_SIZE = int(os.getenv("BASE_CTX_SIZE") or os.getenv("LLAMA_CTX_SIZE", "49152"))
CTX_REPLY_RESERVE = MAX_TOKENS
CTX_PROMPT_BUDGET = CTX_SIZE - CTX_REPLY_RESERVE
CTX_WARN_THRESHOLD = int(CTX_PROMPT_BUDGET * 0.70)
CTX_COMPACT_TRIGGER = int(CTX_PROMPT_BUDGET * 0.80)
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "4000"))
CHARS_PER_TOKEN = 3.5
MAX_READ_LINES = 75


@dataclass
class AgememConfig:

    # ── STM / context window ──────────────────────────────────────────────────
    STM_TOKEN_LIMIT: int = 6_000
    """Hard upper bound on tokens in the active context (excluding system prompt)."""

    STM_WARNING_THRESHOLD: float = 0.75
    """Fraction of STM_TOKEN_LIMIT at which the system starts proactive SUMMARY."""

    STM_CRITICAL_THRESHOLD: float = 0.90
    """Fraction at which the system forces FILTER before the next LLM call."""

    STM_MIN_MESSAGES: int = 4
    """Minimum messages to keep even under critical pressure (last N turns)."""

    # ── LTM store ─────────────────────────────────────────────────────────────
    LTM_MAX_ENTRIES: int = 500
    """Maximum entries in the LTM store before least-scored entries are pruned."""

    LTM_PROMOTE_THRESHOLD: float = 0.65
    """LearningFeedback.score >= this value triggers LTM ADD candidacy."""

    LTM_UPDATE_THRESHOLD: float = 0.50
    """Score above this updates an existing similar LTM entry instead of adding."""

    LTM_SIMILARITY_WORDS: int = 6
    """Number of leading content words used for naive duplicate detection."""

    # ── STM eviction / FILTER ─────────────────────────────────────────────────
    STM_EVICT_THRESHOLD: float = 0.30
    """relevance_score <= this makes a message a FILTER candidate."""

    STM_SUMMARY_WINDOW: int = 8
    """Number of recent non-pinned messages the SUMMARY operation compresses."""

    # ── Trigger: system rule ──────────────────────────────────────────────────
    TRIGGER_EVERY_N_TURNS: int = 10
    """Every N turns the system forces a memory-agent review cycle."""

    TRIGGER_IDLE_SECONDS: float = 0.0
    """Unused in inference-only; kept for future async extension."""

    # ── Learning-score collection ─────────────────────────────────────────────
    LEARNING_SCORE_PROMPT_EVERY_N: int = 3
    """Ask the agent for a LearningFeedback every N turns."""

    LEARNING_SCORE_THRESHOLD_IMMEDIATE: float = 0.85
    """If agent self-reports >= this score, skip the N-turn cadence and act now."""

    # ── Memory agent ──────────────────────────────────────────────────────────
    MEMORY_AGENT_MODEL: str = "gpt-4o-mini"
    """Model used by the dedicated MemoryAgent.  Can differ from main model."""

    MEMORY_AGENT_MAX_TOKENS: int = 512

    # ── LLM client defaults ───────────────────────────────────────────────────
    DEFAULT_MODEL: str = "gpt-4o-mini"
    DEFAULT_MAX_TOKENS: int = 1024
    DEFAULT_TEMPERATURE: float = 0.2

    # ── Persistence ──────────────────────────────────────────────────────────
    PERSIST_DIR: Optional[str] = "agent_memory"
    """Directory for persistent storage. Set to None to disable persistence.

    Coherence note: This MUST match the default in main.py's LTM_PERSIST_PATH.
    Both LTM and STM are stored in this directory:
    - {PERSIST_DIR}/ltm_store.json  (LTM entries)
    - {PERSIST_DIR}/stm_context.json (STM context)
    """

    LTM_PERSIST_FILENAME: str = "ltm_store.json"
    """Filename for LTM store persistence within PERSIST_DIR.

    WARNING: This path must be kept in sync with main.py's LTM_PERSIST_PATH
    environment variable. If you change one, change the other.
    """

    STM_PERSIST_FILENAME: str = "stm_context.json"
    """Filename for STM context persistence within PERSIST_DIR."""

    # ── Skills ───────────────────────────────────────────────────────────────
    SKILL_DETECTION_ENABLED: bool = True
    """Enable automatic skill detection based on keywords."""

    SKILL_MAX_HINTS_PER_TURN: int = 3
    """Maximum number of skill hints to inject per turn."""

    SKILL_DEFAULT_RELEVANCE: float = 0.5
    """Default relevance score for skill hint messages."""

    SKILL_TRIGGER_MIN_MATCHES: int = 1
    """Minimum keyword matches required to trigger a skill."""

    SKILL_CORPUS_PATH: Optional[str] = "corpus"
    """Path to corpus directory for loading skill documents."""

    # ── Prompt Registry ───────────────────────────────────────────────────────
    PROMPT_REGISTRY_DIR: Optional[str] = None
    """Directory for prompt registry files. If None, uses default prompts/prompts/."""

    @property
    def SYSTEM_PROMPT_HEADER(self) -> str:
        """Get the main system prompt from the registry."""
        # Lazy import to avoid circular dependency
        from prompts import get_main_system_prompt
        try:
            return get_main_system_prompt()
        except Exception:
            # Fallback to minimal prompt if registry fails
            return "You are AgeMem, an intelligent assistant with memory capabilities."



# Singleton-style default config; callers can replace it.
DEFAULT_CONFIG = AgememConfig()