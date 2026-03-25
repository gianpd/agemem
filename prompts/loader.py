"""
prompts/loader.py
─────────────────
Prompt loading with YAML frontmatter parsing and caching.

Each prompt file is a markdown file with YAML frontmatter:

---
prompt_id: main-system
name: Main System Prompt
version: 1.0.0
created_at: 2026-03-10
updated_at: 2026-03-10
author: system
tags: [system, core]
active: true
---

Prompt content here...
"""

from __future__ import annotations

import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from core.types import Prompt, PromptVersion


# Cache: prompt_id -> {version -> Prompt}
_prompt_cache: dict[str, dict[str, Prompt]] = {}
_active_versions: dict[str, str] = {}

# Default prompts directory
DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Allow override via environment variable for testing
_env_prompts_dir = os.environ.get("AGEMEM_PROMPTS_DIR")
if _env_prompts_dir:
    DEFAULT_PROMPTS_DIR = Path(_env_prompts_dir)


@dataclass
class ParsedFrontmatter:
    """Result of parsing YAML frontmatter from a prompt file."""
    metadata: dict
    content: str


def parse_frontmatter(text: str) -> ParsedFrontmatter:
    """
    Parse YAML frontmatter from markdown text.

    Frontmatter is delimited by --- at the start and end.
    Returns metadata dict and content string.
    """
    # Match frontmatter pattern: --- at start, then YAML, then ---
    pattern = r'^---\s*\n(.*?)\n---\s*\n?(.*)$'
    match = re.match(pattern, text, re.DOTALL)

    if not match:
        # No frontmatter - treat entire text as content
        return ParsedFrontmatter(metadata={}, content=text.strip())

    yaml_text = match.group(1)
    content = match.group(2).strip()

    # Simple YAML parser for our use case (no nested structures needed)
    metadata: dict = {}
    for line in yaml_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Handle key: value format
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            # Handle arrays like [item1, item2]
            if value.startswith('[') and value.endswith(']'):
                value = [
                    v.strip().strip('"\'') for v in value[1:-1].split(',') if v.strip()
                ]
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False

            metadata[key] = value

    return ParsedFrontmatter(metadata=metadata, content=content)


def render_frontmatter(metadata: dict, content: str) -> str:
    """Render metadata and content into a prompt file with YAML frontmatter."""
    lines = ['---']
    for key, value in metadata.items():
        if isinstance(value, list):
            value_str = '[' + ', '.join(f'"{v}"' for v in value) + ']'
        elif isinstance(value, bool):
            value_str = 'true' if value else 'false'
        elif isinstance(value, str) and any(c in value for c in [':', '"', "'", '[', ']']):
            value_str = f'"{value}"'
        else:
            value_str = str(value)
        lines.append(f'{key}: {value_str}')
    lines.append('---')
    lines.append('')
    lines.append(content)
    return '\n'.join(lines)


class PromptLoader:
    """
    Loads and caches prompts from the file system.

    Each prompt is stored as a versioned markdown file with YAML frontmatter.
    The loader maintains a cache for fast access and can reload from disk.
    """

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        self._prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR
        self._cache: dict[str, dict[str, Prompt]] = {}
        self._active_versions: dict[str, str] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy load all prompts on first access."""
        if not self._loaded:
            self.reload()

    def reload(self) -> None:
        """Reload all prompts from disk, clearing the cache."""
        self._cache.clear()
        self._active_versions.clear()

        if not self._prompts_dir.exists():
            self._prompts_dir.mkdir(parents=True, exist_ok=True)
            return

        # Scan all .md files in the prompts directory
        for file_path in self._prompts_dir.glob('*.md'):
            try:
                self._load_prompt_file(file_path)
            except Exception as e:
                print(f"[PromptLoader] Failed to load {file_path}: {e}")

        self._loaded = True

    def _load_prompt_file(self, file_path: Path) -> None:
        """Load a single prompt file and add to cache."""
        text = file_path.read_text(encoding='utf-8')
        parsed = parse_frontmatter(text)
        metadata = parsed.metadata
        content = parsed.content

        prompt_id = metadata.get('prompt_id')
        if not prompt_id:
            # Derive prompt_id from filename
            prompt_id = file_path.stem

        version = metadata.get('version', '1.0.0')

        prompt = Prompt(
            prompt_id=prompt_id,
            name=metadata.get('name', prompt_id),
            version=version,
            content=content,
            created_at=metadata.get('created_at', ''),
            updated_at=metadata.get('updated_at', ''),
            author=metadata.get('author', 'system'),
            tags=metadata.get('tags', []),
            active=metadata.get('active', False),
        )

        # Add to cache
        if prompt_id not in self._cache:
            self._cache[prompt_id] = {}
        self._cache[prompt_id][version] = prompt

        # Track active version
        if prompt.active:
            self._active_versions[prompt_id] = version

    def get_prompt(self, prompt_id: str, version: Optional[str] = None) -> Optional[Prompt]:
        """
        Get a prompt by ID.

        Args:
            prompt_id: The unique identifier for the prompt
            version: Specific version to retrieve, or None for active version

        Returns:
            The Prompt object, or None if not found
        """
        self._ensure_loaded()

        if prompt_id not in self._cache:
            return None

        versions = self._cache[prompt_id]

        if version is None:
            # Use active version
            version = self._active_versions.get(prompt_id)
            if version is None:
                # Fallback to latest version
                version = sorted(versions.keys())[-1] if versions else None

        if version is None or version not in versions:
            return None

        return versions[version]

    def get_active_version(self, prompt_id: str) -> Optional[str]:
        """Get the currently active version for a prompt."""
        self._ensure_loaded()
        return self._active_versions.get(prompt_id)

    def list_prompts(self) -> list[dict]:
        """List all prompts with their metadata (without full content)."""
        self._ensure_loaded()

        result = []
        for prompt_id, versions in self._cache.items():
            active_version = self._active_versions.get(prompt_id)
            latest = versions.get(active_version) if active_version else None

            if not latest and versions:
                latest = list(versions.values())[-1]

            if latest:
                result.append({
                    'prompt_id': prompt_id,
                    'name': latest.name,
                    'active_version': active_version or latest.version,
                    'tags': latest.tags,
                    'created_at': latest.created_at,
                    'updated_at': latest.updated_at,
                })

        return result

    def list_versions(self, prompt_id: str) -> list[PromptVersion]:
        """List all versions of a specific prompt."""
        self._ensure_loaded()

        if prompt_id not in self._cache:
            return []

        versions = []
        for version_str, prompt in self._cache[prompt_id].items():
            versions.append(PromptVersion(
                version=version_str,
                created_at=prompt.created_at,
                author=prompt.author,
            ))

        return sorted(versions, key=lambda v: v.version)

    def activate_version(self, prompt_id: str, version: str) -> bool:
        """
        Activate a specific version of a prompt.

        This updates the 'active' flag in the file frontmatter.
        """
        self._ensure_loaded()

        if prompt_id not in self._cache:
            return False

        if version not in self._cache[prompt_id]:
            return False

        # Update active version in cache
        self._active_versions[prompt_id] = version

        # Update files: mark all versions as inactive except the target
        for file_path in self._prompts_dir.glob('*.md'):
            try:
                text = file_path.read_text(encoding='utf-8')
                parsed = parse_frontmatter(text)

                file_prompt_id = parsed.metadata.get('prompt_id', file_path.stem)
                if file_prompt_id != prompt_id:
                    continue

                file_version = parsed.metadata.get('version', '1.0.0')
                should_be_active = (file_version == version)

                if parsed.metadata.get('active') != should_be_active:
                    parsed.metadata['active'] = should_be_active
                    new_text = render_frontmatter(parsed.metadata, parsed.content)
                    file_path.write_text(new_text, encoding='utf-8')

            except Exception as e:
                print(f"[PromptLoader] Failed to update {file_path}: {e}")

        return True

    def save_prompt(
        self,
        prompt_id: str,
        name: str,
        content: str,
        version: str,
        author: str = 'system',
        tags: Optional[list] = None,
        activate: bool = True,
    ) -> bool:
        """
        Save a new prompt version to disk.

        Args:
            prompt_id: Unique identifier for the prompt
            name: Human-readable name
            content: The prompt content
            version: Semantic version string
            author: Who created this version
            tags: List of tags
            activate: Whether to make this the active version

        Returns:
            True if successful
        """
        from datetime import datetime

        now = datetime.now().isoformat()[:10]  # YYYY-MM-DD

        metadata = {
            'prompt_id': prompt_id,
            'name': name,
            'version': version,
            'created_at': now,
            'updated_at': now,
            'author': author,
            'tags': tags or [],
            'active': activate,
        }

        # Ensure directory exists
        self._prompts_dir.mkdir(parents=True, exist_ok=True)

        # Filename includes version for history
        safe_version = version.replace('.', '_')
        filename = f"{prompt_id}-v{safe_version}.md"
        file_path = self._prompts_dir / filename

        # If this is being activated, deactivate others first
        if activate:
            for existing_file in self._prompts_dir.glob(f"{prompt_id}-v*.md"):
                try:
                    text = existing_file.read_text(encoding='utf-8')
                    parsed = parse_frontmatter(text)
                    if parsed.metadata.get('active'):
                        parsed.metadata['active'] = False
                        parsed.metadata['updated_at'] = now
                        new_text = render_frontmatter(parsed.metadata, parsed.content)
                        existing_file.write_text(new_text, encoding='utf-8')
                except Exception as e:
                    print(f"[PromptLoader] Failed to update {existing_file}: {e}")

        # Write new file
        file_text = render_frontmatter(metadata, content)
        file_path.write_text(file_text, encoding='utf-8')

        # Update cache
        prompt = Prompt(
            prompt_id=prompt_id,
            name=name,
            version=version,
            content=content,
            created_at=now,
            updated_at=now,
            author=author,
            tags=tags or [],
            active=activate,
        )

        if prompt_id not in self._cache:
            self._cache[prompt_id] = {}
        self._cache[prompt_id][version] = prompt

        if activate:
            self._active_versions[prompt_id] = version

        return True


# Global loader instance
_global_loader: Optional[PromptLoader] = None


def get_loader(prompts_dir: Optional[Path] = None) -> PromptLoader:
    """Get or create the global prompt loader."""
    global _global_loader
    if _global_loader is None:
        _global_loader = PromptLoader(prompts_dir)
    return _global_loader
