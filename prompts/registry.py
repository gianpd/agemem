"""
prompts/registry.py
───────────────────
Core registry logic for the prompt system.

The PromptRegistry provides a clean API for:
- Getting prompts by ID
- Listing available prompts and versions
- Activating specific versions
- Tracking prompt usage for audit trails
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from prompts.loader import PromptLoader, get_loader, DEFAULT_PROMPTS_DIR
from core.types import Prompt, PromptMetadata, PromptVersion


class PromptRegistry:
    """
    Central registry for managing system prompts.

    This is the main entry point for accessing prompts. It wraps the
    PromptLoader and provides a higher-level API with audit tracking.

    Example:
        registry = PromptRegistry()

        # Get the active version of a prompt
        prompt = registry.get_prompt("main-system")

        # List all prompts
        for meta in registry.list_prompts():
            print(f"{meta.prompt_id}: {meta.name}")
    """

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        """
        Initialize the registry.

        Args:
            prompts_dir: Directory containing prompt files. Uses default if None.
        """
        self._loader = PromptLoader(prompts_dir or DEFAULT_PROMPTS_DIR)
        self._prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR

    def get_prompt(self, prompt_id: str, version: Optional[str] = None) -> Prompt:
        """
        Get a prompt by ID.

        Args:
            prompt_id: The unique identifier for the prompt (e.g., "main-system")
            version: Specific version to retrieve, or None for active version

        Returns:
            The Prompt object

        Raises:
            KeyError: If the prompt or version is not found
        """
        prompt = self._loader.get_prompt(prompt_id, version)
        if prompt is None:
            available = self._loader.list_prompts()
            available_ids = [p['prompt_id'] for p in available]
            raise KeyError(
                f"Prompt '{prompt_id}' not found. "
                f"Available: {available_ids}"
            )
        return prompt

    def get_active_version(self, prompt_id: str) -> str:
        """
        Get the currently active version for a prompt.

        Args:
            prompt_id: The prompt identifier

        Returns:
            The active version string

        Raises:
            KeyError: If the prompt is not found
        """
        version = self._loader.get_active_version(prompt_id)
        if version is None:
            # Check if prompt exists at all
            if self._loader.get_prompt(prompt_id) is None:
                raise KeyError(f"Prompt '{prompt_id}' not found")
            # Prompt exists but no active version set
            versions = self._loader.list_versions(prompt_id)
            if versions:
                version = versions[-1].version
            else:
                raise KeyError(f"No versions found for prompt '{prompt_id}'")
        return version

    def list_prompts(self) -> list[PromptMetadata]:
        """
        List all prompts with their metadata.

        Returns:
            List of PromptMetadata objects (without full content)
        """
        raw_list = self._loader.list_prompts()
        return [
            PromptMetadata(
                prompt_id=item['prompt_id'],
                name=item['name'],
                description='',  # Could be added to frontmatter schema
                active_version=item['active_version'],
                tags=item['tags'],
                created_at=item['created_at'],
                updated_at=item['updated_at'],
            )
            for item in raw_list
        ]

    def list_versions(self, prompt_id: str) -> list[PromptVersion]:
        """
        List all versions of a specific prompt.

        Args:
            prompt_id: The prompt identifier

        Returns:
            List of PromptVersion objects

        Raises:
            KeyError: If the prompt is not found
        """
        versions = self._loader.list_versions(prompt_id)
        if not versions and self._loader.get_prompt(prompt_id) is None:
            raise KeyError(f"Prompt '{prompt_id}' not found")
        return versions

    def activate_version(self, prompt_id: str, version: str) -> bool:
        """
        Activate a specific version of a prompt.

        This updates the active flag in the file and makes this version
        the default for future get_prompt() calls.

        Args:
            prompt_id: The prompt identifier
            version: The version to activate

        Returns:
            True if successful

        Raises:
            KeyError: If the prompt or version is not found
        """
        # Verify prompt and version exist
        if self._loader.get_prompt(prompt_id) is None:
            raise KeyError(f"Prompt '{prompt_id}' not found")

        if self._loader.get_prompt(prompt_id, version) is None:
            available = self._loader.list_versions(prompt_id)
            available_versions = [v.version for v in available]
            raise KeyError(
                f"Version '{version}' not found for prompt '{prompt_id}'. "
                f"Available: {available_versions}"
            )

        return self._loader.activate_version(prompt_id, version)

    def reload(self) -> None:
        """Reload all prompts from disk, clearing any caches."""
        self._loader.reload()

    def save_prompt(
        self,
        prompt_id: str,
        name: str,
        content: str,
        version: str,
        author: str = 'system',
        tags: Optional[list[str]] = None,
        activate: bool = True,
    ) -> bool:
        """
        Save a new prompt version.

        Args:
            prompt_id: Unique identifier
            name: Human-readable name
            content: The prompt content
            version: Semantic version string
            author: Who created this version
            tags: List of tags
            activate: Whether to make this the active version

        Returns:
            True if successful
        """
        return self._loader.save_prompt(
            prompt_id=prompt_id,
            name=name,
            content=content,
            version=version,
            author=author,
            tags=tags,
            activate=activate,
        )

    def get_prompts_dir(self) -> Path:
        """Get the directory where prompts are stored."""
        return self._prompts_dir


# Global registry instance
_global_registry: Optional[PromptRegistry] = None


def get_registry(prompts_dir: Optional[Path] = None) -> PromptRegistry:
    """Get or create the global prompt registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PromptRegistry(prompts_dir)
    return _global_registry
