"""
skills/manager.py
─────────────────
Skill detection and management for AgeMem-Hybrid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.types import Skill
from core.config import AgememConfig
from skills.loader import load_skills_from_corpus


class SkillManager:
    """
    Manages skill registration, loading, and detection.

    Skills are loaded from:
    - Corpus documents with doc_type: skill
    - Built-in skills defined in code (optional)

    The manager detects relevant skills based on user input keywords and
    provides hints that can be injected into the conversation context.
    """

    def __init__(self, config: Optional[AgememConfig] = None):
        self._config = config or AgememConfig()
        self._skills: list[Skill] = []
        self._loaded = False

    def load_skills(
        self,
        corpus_path: Optional[Path | str] = None,
        skip_isolated: bool = False,
    ) -> None:
        """
        Load skills from corpus and built-in sources.

        Args:
            corpus_path: Path to corpus directory. If None, uses config.SKILL_CORPUS_PATH
            skip_isolated: If True, skip users/ and tenants/ subdirectories (for global corpus)
        """
        if corpus_path is None:
            corpus_path = self._config.SKILL_CORPUS_PATH

        corpus_skills = []  # Initialize before conditional to prevent NameError
        if corpus_path:
            corpus_skills = load_skills_from_corpus(corpus_path, skip_isolated=skip_isolated)

        # Deduplication: later load wins (tenant override semantics)
        skill_index = {s.skill_id: i for i, s in enumerate(self._skills)}
        for skill in corpus_skills:
            if skill.skill_id in skill_index:
                # Replace existing skill (tenant override)
                self._skills[skill_index[skill.skill_id]] = skill
            else:
                # Append new skill
                self._skills.append(skill)

        # Re-sort by priority after dedup
        self._skills.sort(key=lambda s: s.priority, reverse=True)

        self._loaded = True

        if corpus_skills:
            print(f"[SKILL MANAGER] Loaded {len(corpus_skills)} skills from corpus")

    def add_skill(self, skill: Skill) -> None:
        """Add a skill manually (e.g., built-in skills)."""
        self._skills.append(skill)
        # Re-sort by priority
        self._skills.sort(key=lambda s: s.priority, reverse=True)

    def detect_skills(self, user_input: str) -> list[Skill]:
        """
        Detect relevant skills based on user input.

        Args:
            user_input: The user's message text

        Returns:
            List of matching skills, sorted by priority (highest first)
        """
        if not self._config.SKILL_DETECTION_ENABLED:
            return []

        if not self._loaded:
            self.load_skills()

        matches = []
        min_matches = self._config.SKILL_TRIGGER_MIN_MATCHES

        for skill in self._skills:
            if skill.matches_input(user_input, min_matches=min_matches):
                matches.append(skill)

        # Limit to max hints per turn
        max_hints = self._config.SKILL_MAX_HINTS_PER_TURN
        return matches[:max_hints]

    def get_all_skills(self) -> list[Skill]:
        """Return all loaded skills."""
        return list(self._skills)

    def get_skill_by_id(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by its ID."""
        for skill in self._skills:
            if skill.skill_id == skill_id:
                return skill
        return None

    def clear_skills(self) -> None:
        """Clear all loaded skills."""
        self._skills.clear()
        self._loaded = False
