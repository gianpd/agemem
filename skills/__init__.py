"""
skills/__init__.py
─────────────────
SKILLS system for AgeMem-Hybrid.

Skills are learned capabilities that enhance the agent's effectiveness.
They are dynamically detected from user input and injected as context hints.
"""

from core.types import Skill
from skills.manager import SkillManager
from skills.loader import load_skills_from_corpus

__all__ = ["Skill", "SkillManager", "load_skills_from_corpus"]
