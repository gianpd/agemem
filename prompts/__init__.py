"""
prompts/__init__.py
───────────────────
Central Prompt Registry with Versioning for AgeMem.

This module provides a file-based registry for managing system prompts
with versioning, history tracking, and runtime modification capabilities.

Example:
    from prompts import registry

    # Get the current version of a prompt
    prompt = registry.get_prompt("main-system")
    print(prompt.content)

    # List all available prompts
    for meta in registry.list_prompts():
        print(f"{meta.prompt_id}: {meta.name} (v{meta.active_version})")
"""

from __future__ import annotations

from prompts.registry import PromptRegistry, get_registry
from prompts.loader import PromptLoader
from core.types import Prompt, PromptMetadata, PromptVersion

# Singleton registry instance
_registry: PromptRegistry | None = None


def get_prompt(prompt_id: str, version: str | None = None) -> Prompt:
    """Get a prompt by ID, optionally specifying a version."""
    global _registry
    if _registry is None:
        _registry = get_registry()
    return _registry.get_prompt(prompt_id, version)


def get_active_version(prompt_id: str) -> str:
    """Get the currently active version for a prompt."""
    global _registry
    if _registry is None:
        _registry = get_registry()
    return _registry.get_active_version(prompt_id)


def list_prompts() -> list[PromptMetadata]:
    """List all prompts in the registry."""
    global _registry
    if _registry is None:
        _registry = get_registry()
    return _registry.list_prompts()


def list_versions(prompt_id: str) -> list[PromptVersion]:
    """List all versions of a specific prompt."""
    global _registry
    if _registry is None:
        _registry = get_registry()
    return _registry.list_versions(prompt_id)


def activate_version(prompt_id: str, version: str) -> bool:
    """Activate a specific version of a prompt."""
    global _registry
    if _registry is None:
        _registry = get_registry()
    return _registry.activate_version(prompt_id, version)


def reload() -> None:
    """Reload the registry from disk."""
    global _registry
    # Also reset the global registry in registry.py to allow fresh start
    import prompts.registry
    prompts.registry._global_registry = None
    _registry = None
    _registry = get_registry()


# Convenience accessors
def get_main_system_prompt() -> str:
    """Get the main system prompt content."""
    return get_prompt("main-system").content


def get_memory_agent_prompt() -> str:
    """Get the MemoryAgent system prompt content."""
    return get_prompt("memory-agent").content


__all__ = [
    "PromptRegistry",
    "PromptLoader",
    "get_registry",
    "get_prompt",
    "get_active_version",
    "list_prompts",
    "list_versions",
    "activate_version",
    "reload",
    "get_main_system_prompt",
    "get_memory_agent_prompt",
    "Prompt",
    "PromptMetadata",
    "PromptVersion",
]
