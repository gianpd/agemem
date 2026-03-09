"""
skills/loader.py
────────────────
Load skills from corpus documents with skill metadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from core.types import Skill


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter from markdown content.

    Args:
        content: Full markdown content possibly with YAML frontmatter

    Returns:
        Tuple of (frontmatter_dict, body_content)
    """
    # Check for YAML frontmatter (starts with ---)
    if not content.strip().startswith("---"):
        return {}, content

    # Find the end of frontmatter
    match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    body = content[match.end():]

    # Simple YAML parsing for common cases
    frontmatter = {}
    for line in frontmatter_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Handle key: value
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Handle list values (simple case)
            if value.startswith("[") and value.endswith("]"):
                # Parse simple list: [item1, item2, item3]
                items = value[1:-1].split(",")
                frontmatter[key] = [item.strip().strip('"\'') for item in items if item.strip()]
            elif value.startswith("-"):
                # This is actually the start of a list, handle below
                continue
            else:
                # Try to convert to int/float/bool
                if value.lower() == "true":
                    frontmatter[key] = True
                elif value.lower() == "false":
                    frontmatter[key] = False
                elif value.isdigit():
                    frontmatter[key] = int(value)
                else:
                    try:
                        frontmatter[key] = float(value)
                    except ValueError:
                        frontmatter[key] = value.strip('"\'')

    # Handle multi-line lists
    current_key = None
    current_list = []
    in_list = False

    for line in frontmatter_text.strip().split("\n"):
        line = line.rstrip()
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("-") and in_list and current_key:
            # List item
            item = stripped[1:].strip().strip('"\'')
            current_list.append(item)
        elif ":" in stripped:
            # Save previous list if any
            if in_list and current_key and current_list:
                frontmatter[current_key] = current_list

            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()

            if value.startswith("-") or value == "":
                # Start of a list
                current_key = key
                current_list = []
                in_list = True
                # Check if there's an inline first item
                if value.startswith("-"):
                    current_list.append(value[1:].strip().strip('"\''))
            else:
                in_list = False
                current_key = None

    # Save final list if any
    if in_list and current_key and current_list:
        frontmatter[current_key] = current_list

    return frontmatter, body


def extract_hint_message(frontmatter: dict, body: str) -> str:
    """
    Extract the hint message from frontmatter or document body.

    Priority:
    1. skill_hint field in frontmatter
    2. First paragraph of Core section in body
    3. First paragraph of body
    """
    # Check for explicit hint
    if "skill_hint" in frontmatter:
        return frontmatter["skill_hint"]

    # Try to find "Core" section
    core_match = re.search(
        r"##\s*Core.*?\n(.*?)(?=\n##|\Z)",
        body,
        re.DOTALL | re.IGNORECASE
    )
    if core_match:
        section = core_match.group(1).strip()
        # Get first non-empty paragraph
        for para in section.split("\n\n"):
            para = para.strip()
            if para and not para.startswith("#"):
                # Clean up markdown
                para = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", para)  # Remove links
                para = para.replace("**", "").replace("*", "")  # Remove bold/italic
                para = para.replace("`", "")  # Remove code markers
                return para[:500]  # Limit length

    # Fallback: first paragraph of body
    for para in body.split("\n\n"):
        para = para.strip()
        if para and not para.startswith("#") and not para.startswith("---"):
            para = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", para)
            para = para.replace("**", "").replace("*", "")
            para = para.replace("`", "")
            return para[:500]

    return ""


def load_skills_from_corpus(corpus_path: Path | str) -> list[Skill]:
    """
    Load skills from corpus documents with doc_type: skill or skill triggers.

    Args:
        corpus_path: Path to corpus directory

    Returns:
        List of Skill objects
    """
    skills = []
    corpus_path = Path(corpus_path)

    if not corpus_path.exists():
        return skills

    for doc_file in corpus_path.glob("*.md"):
        try:
            content = doc_file.read_text(encoding="utf-8")
            frontmatter, body = parse_frontmatter(content)

            # Check if this is a skill document
            is_skill = (
                frontmatter.get("doc_type") == "skill"
                or "skill_triggers" in frontmatter
            )

            if not is_skill:
                continue

            # Build skill from frontmatter
            skill_id = frontmatter.get("doc_id", doc_file.stem)
            name = frontmatter.get(
                "skill_name",
                frontmatter.get("doc_title", "Unknown Skill")
            )
            description = frontmatter.get("skill_description", "")
            trigger_keywords = frontmatter.get("skill_triggers", [])

            # If no explicit triggers, try to infer from description
            if not trigger_keywords and description:
                # Use description words as triggers
                trigger_keywords = description.lower().split()[:5]

            hint_message = extract_hint_message(frontmatter, body)

            # If still no hint, use description
            if not hint_message and description:
                hint_message = description

            priority = frontmatter.get("skill_priority", 0)

            skill = Skill(
                skill_id=skill_id,
                name=name,
                description=description,
                trigger_keywords=trigger_keywords,
                hint_message=hint_message,
                source_doc_id=frontmatter.get("doc_id"),
                priority=priority,
            )
            skills.append(skill)

        except Exception as e:
            # Log error but continue loading other skills
            print(f"[SKILL LOADER] Error loading {doc_file}: {e}")
            continue

    # Sort by priority (higher first)
    skills.sort(key=lambda s: s.priority, reverse=True)
    return skills
