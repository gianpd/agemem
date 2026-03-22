"""
core/llm_factory.py
────────────────────
Unified factory for LLMClient instantiation.

Provides a single source of truth for creating LLM clients
across main application, evaluation pipeline, and tests.

Usage:
    # Simple usage (uses environment variables)
    factory = LLMClientFactory()
    client = factory.create()

    # With custom config
    config = LLMConfig(base_url="http://custom:8080", model="custom-model")
    client = factory.create(config=config)

    # Get raw OpenAI client for advanced use
    openai_client, model = factory.create_raw()
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from agents.llm_client import LLMClient


@dataclass
class LLMConfig:
    """Configuration for LLM client creation.

    All fields have environment variable fallbacks with deprecation
    handling for legacy LLAMA_* variable names.

    Attributes:
        base_url: LLM API endpoint URL
        model: Model name to use
        max_tokens: Maximum tokens for responses
        temperature: Sampling temperature
        api_key: API key (optional for local endpoints)
        timeout: Request timeout in seconds
    """

    base_url: str = field(
        default_factory=lambda: _get_env_with_fallback(
            "BASE_URL", "LLAMA_HOST", "http://localhost:8080"
        )
    )
    model: str = field(
        default_factory=lambda: _get_env_with_fallback(
            "BASE_MODEL", "LLAMA_MODEL", "Qwen3.5-9B-UD-Q4_K_XL.gguf"
        )
    )
    max_tokens: int = field(
        default_factory=lambda: int(
            _get_env_with_fallback("BASE_MAX_TOKENS", "LLAMA_MAX_TOKENS", "12324")
        )
    )
    temperature: float = field(
        default_factory=lambda: float(
            _get_env_with_fallback("BASE_TEMPERATURE", "LLAMA_TEMPERATURE", "0.1")
        )
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    timeout: float = field(
        default_factory=lambda: float(
            os.getenv("LLM_TIMEOUT", "300.0")
        )
    )


def _get_env_with_fallback(new_name: str, old_name: str, default: str) -> str:
    """Get env var with fallback to old name for backward compatibility."""
    value = os.getenv(new_name) or os.getenv(old_name, default)
    if os.getenv(old_name) and not os.getenv(new_name):
        warnings.warn(
            f"Environment variable '{old_name}' is deprecated. "
            f"Please use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return value


class LLMClientFactory:
    """Factory for creating LLMClient instances.

    Single entry point for all LLM client creation across:
    - Main REPL application
    - Evaluation pipeline
    - Integration tests

    This factory ensures consistent client configuration and eliminates
    duplicated endpoint detection logic across the codebase.

    Example:
        # Simple usage (uses environment variables)
        factory = LLMClientFactory()
        client = factory.create()

        # With custom config
        config = LLMConfig(base_url="http://custom:8080", model="custom-model")
        client = factory.create(config=config)

        # Get raw OpenAI client for advanced use
        openai_client, model = factory.create_raw()
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize factory with optional config override.

        Args:
            config: LLMConfig instance. If None, creates one from environment.
        """
        self._config = config or LLMConfig()

    @property
    def config(self) -> LLMConfig:
        """Get the current configuration."""
        return self._config

    def _get_api_key(self) -> str:
        """Determine API key based on endpoint type.

        Local endpoints (localhost/127.0.0.1) don't require a real API key.
        Remote endpoints require API_KEY or OPENAI_API_KEY env var.

        Returns:
            API key string

        Raises:
            ValueError: If remote endpoint has no API key configured
        """
        base_url = self._config.base_url
        is_local = "localhost" in base_url or "127.0.0.1" in base_url

        if is_local:
            return "not-needed"

        api_key = self._config.api_key
        if not api_key:
            raise ValueError(
                f"API_KEY or OPENAI_API_KEY environment variable is required "
                f"for non-local endpoint: {base_url}"
            )
        return api_key

    def _normalize_base_url(self, base_url: str) -> str:
        """Ensure base_url ends with /v1 for OpenAI compatibility.

        Args:
            base_url: Raw base URL from config

        Returns:
            Normalized URL ending with /v1
        """
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized

    def create_raw(self) -> tuple[OpenAI, str]:
        """Create raw OpenAI client with model name.

        Returns:
            Tuple of (OpenAI client instance, model name string)

        Use this when you need direct access to the OpenAI client
        for advanced operations not supported by LLMClient wrapper.
        """
        import httpx

        api_key = self._get_api_key()
        base_url = self._normalize_base_url(self._config.base_url)

        # OpenRouter requires specific headers
        openai_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": httpx.Timeout(self._config.timeout, connect=30.0),
        }
        if "openrouter.ai" in base_url:
            openai_kwargs["default_headers"] = {
                "HTTP-Referer": "https://github.com/agemem/agemem",
                "X-Title": "AgeMem",
            }

        openai_client = OpenAI(**openai_kwargs)

        return openai_client, self._config.model

    def create(self, config: Optional[LLMConfig] = None) -> LLMClient:
        """Create an LLMClient instance.

        Args:
            config: Optional config override. If provided, temporarily uses
                    this config instead of the factory's default.

        Returns:
            Configured LLMClient instance ready for use.
        """
        # Temporarily swap config if override provided
        if config is not None:
            original_config = self._config
            self._config = config

        try:
            openai_client, model = self.create_raw()
            return LLMClient(
                client=openai_client,
                default_model=model,
                default_temperature=self._config.temperature,
            )
        finally:
            # Restore original config if we swapped it
            if config is not None:
                self._config = original_config

    @classmethod
    def from_environment(cls) -> "LLMClientFactory":
        """Create factory from environment variables only.

        Convenience class method for the common case of using
        all defaults from environment.

        Returns:
            LLMClientFactory configured from environment
        """
        return cls(LLMConfig())

    @classmethod
    def for_evaluation(
        cls,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> "LLMClientFactory":
        """Create factory configured for evaluation runs.

        Provides explicit parameter control for reproducible
        evaluation scenarios. Any parameter not provided falls
        back to environment variable or default.

        Args:
            base_url: Override base URL
            model: Override model name
            max_tokens: Override max tokens
            temperature: Override temperature
            api_key: Override API key
            timeout: Override request timeout in seconds

        Returns:
            LLMClientFactory configured for evaluation
        """
        # Start with environment defaults
        env_config = LLMConfig()

        # Apply explicit overrides
        config = LLMConfig(
            base_url=base_url if base_url is not None else env_config.base_url,
            model=model if model is not None else env_config.model,
            max_tokens=max_tokens if max_tokens is not None else env_config.max_tokens,
            temperature=temperature if temperature is not None else env_config.temperature,
            api_key=api_key if api_key is not None else env_config.api_key,
            timeout=timeout if timeout is not None else env_config.timeout,
        )
        return cls(config)


    @classmethod
    def for_learning_scorer(
        cls,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> "LLMClientFactory":
        """Create factory configured for the learning scorer (always external API).

        The learning scorer uses an external API (OpenRouter by default) to ensure
        consistent, high-quality structured JSON output for learning feedback,
        regardless of whether the main model is local or external.

        Args:
            base_url: Override base URL (default: OpenRouter API)
            model: Override model name (default: google/gemini-3-flash-preview)
            api_key: API key for the external service (required)
            max_tokens: Max tokens for responses
            temperature: Sampling temperature (default: 0.1 for deterministic output)
            timeout: Request timeout in seconds (default: 60.0)

        Returns:
            LLMClientFactory configured for learning scorer

        Raises:
            ValueError: If no API key is provided and OPENROUTER_API_KEY env var is not set
        """
        import os

        # Get values with environment fallbacks
        final_base_url = base_url or os.getenv("LEARNING_SCORER_BASE_URL", "https://openrouter.ai/api")
        final_model = model or os.getenv("LEARNING_SCORER_MODEL", "google/gemini-3-flash-preview")
        final_api_key = api_key or os.getenv("LEARNING_SCORER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

        if not final_api_key:
            raise ValueError(
                "Learning scorer requires an API key. Set LEARNING_SCORER_API_KEY "
                "or OPENROUTER_API_KEY environment variable."
            )

        config = LLMConfig(
            base_url=final_base_url,
            model=final_model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=final_api_key,
            timeout=timeout,
        )
        return cls(config)
def get_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Create an LLMClient instance using default configuration.

    Convenience function that wraps LLMClientFactory for simple use cases.
    This provides backward compatibility with existing code that calls
    get_llm_client() directly.

    Args:
        config: Optional LLMConfig to override defaults

    Returns:
        Configured LLMClient instance
    """
    return LLMClientFactory(config).create()


# Export public API
__all__ = [
    "LLMConfig",
    "LLMClientFactory",
    "get_llm_client",
]