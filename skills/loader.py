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

    Handles single or double frontmatter blocks (e.g., skill documents
    that have both ingestion metadata and skill definition blocks).

    Args:
        content: Full markdown content possibly with YAML frontmatter

    Returns:
        Tuple of (frontmatter_dict, body_content)
    """
    # Check for YAML frontmatter (starts with ---)
    if not content.strip().startswith("---"):
        return {}, content

    # Find the end of first frontmatter block
    match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}, content

    first_fm_text = match.group(1)
    body = content[match.end():]

    # Parse first frontmatter block
    frontmatter = _parse_yaml_block(first_fm_text)

    # Handle nested frontmatter (common in skill documents after ingestion)
    # If body starts with another --- block, parse it for skill definition
    if body.strip().startswith("---"):
        nested_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", body, re.DOTALL)
        if nested_match:
            nested_fm_text = nested_match.group(1)
            nested_fm = _parse_yaml_block(nested_fm_text)
            # Merge skill-specific fields from nested frontmatter
            skill_fields = [
                "name", "description", "trigger_keywords", "skill_triggers",
                "skill_name", "skill_description", "skill_hint", "skill_priority",
                "priority", "version", "author", "license", "compatibility",
            ]
            for field in skill_fields:
                if field in nested_fm and field not in frontmatter:
                    frontmatter[field] = nested_fm[field]
            # Update body to skip the nested frontmatter
            body = body[nested_match.end():]

    return frontmatter, body


def _parse_yaml_block(text: str) -> dict:
    """
    Parse a single YAML frontmatter block.

    Handles:
    - Simple key: value pairs
    - Inline lists: [item1, item2]
    - Multi-line lists with - prefix
    - Multi-line strings with > or | indicators

    Args:
        text: YAML frontmatter text (between --- markers)

    Returns:
        Dictionary of parsed key-value pairs
    """
    frontmatter = {}
    current_key = None
    current_list = []
    in_list = False
    multiline_key = None
    multiline_value = []
    in_multiline = False

    lines = text.strip().split("\n")

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        line_rstripped = line.rstrip()

        if not line_stripped or line_stripped.startswith("#"):
            continue

        # Handle multiline string continuation
        if in_multiline and multiline_key:
            # YAML multiline continues on indented lines
            # Check if line starts with whitespace (indentation) - that means continuation
            if line.startswith(" ") or line.startswith("\t"):
                # Continue multiline - this is indented content
                multiline_value.append(line_stripped)
                continue
            else:
                # Non-indented line - end multiline
                frontmatter[multiline_key] = " ".join(multiline_value).strip()
                multiline_key = None
                multiline_value = []
                in_multiline = False
                # Don't continue - process this line as normal

        # Handle list item continuation
        if line_stripped.startswith("-") and in_list and current_key:
            item = line_stripped[1:].strip().strip('"\'')
            current_list.append(item)
            continue

        # Handle key: value
        if ":" in line_stripped:
            # Save previous list if any
            if in_list and current_key and current_list:
                frontmatter[current_key] = current_list
                current_key = None
                current_list = []
                in_list = False

            key, value = line_stripped.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Handle multiline string indicators (> or |)
            if value in (">", "|"):
                multiline_key = key
                multiline_value = []
                in_multiline = True
                continue

            # Handle inline list
            if value.startswith("[") and value.endswith("]"):
                items = value[1:-1].split(",")
                frontmatter[key] = [item.strip().strip('"\'') for item in items if item.strip()]
                continue

            # Handle start of multi-line list
            if value == "" or value.startswith("-"):
                current_key = key
                current_list = []
                in_list = True
                # Check if there's an inline first item
                if value.startswith("-"):
                    current_list.append(value[1:].strip().strip('"\''))
                continue

            # Handle scalar value
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

    # Save final pending values
    if in_list and current_key and current_list:
        frontmatter[current_key] = current_list
    if in_multiline and multiline_key:
        frontmatter[multiline_key] = " ".join(multiline_value).strip()

    return frontmatter


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
                para = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", para)
                para = para.replace("**", "").replace("*", "")
                para = para.replace("`", "")
                return para[:500]

    # Fallback: first paragraph of body
    for para in body.split("\n\n"):
        para = para.strip()
        if para and not para.startswith("#") and not para.startswith("---"):
            para = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", para)
            para = para.replace("**", "").replace("*", "")
            para = para.replace("`", "")
            return para[:500]

    return ""


def load_skills_from_corpus(
    corpus_path: Path | str,
    skip_isolated: bool = False,
) -> list[Skill]:
    """
    Load skills from corpus documents with doc_type: skill or skill triggers.

    Args:
        corpus_path: Path to corpus directory
        skip_isolated: If True, skip users/ and tenants/ subdirectories (for global corpus)

    Returns:
        List of Skill objects
    """
    skills = []
    corpus_path = Path(corpus_path)

    if not corpus_path.exists():
        return skills

    # Paths that represent isolated tenant/user corpora - skip when loading global
    isolated_paths = ("users", "tenants")

    for doc_file in corpus_path.rglob("*.md"):
        try:
            # Skip isolated subdirectories when loading global corpus
            if skip_isolated:
                rel_path = doc_file.relative_to(corpus_path)
                if any(part in isolated_paths for part in rel_path.parts):
                    continue

            content = doc_file.read_text(encoding="utf-8")
            frontmatter, body = parse_frontmatter(content)

            # Check if this is a skill document
            is_skill = (
                frontmatter.get("doc_type") == "skill"
                or "skill_triggers" in frontmatter
                or frontmatter.get("name")  # skill definition has 'name' field
            )

            if not is_skill:
                continue

            # Build skill from merged frontmatter
            skill_id = frontmatter.get("doc_id", doc_file.stem)
            name = frontmatter.get(
                "skill_name",
                frontmatter.get("name", frontmatter.get("doc_title", "Unknown Skill"))
            )
            description = frontmatter.get(
                "skill_description",
                frontmatter.get("description", "")
            )
            trigger_keywords = frontmatter.get(
                "skill_triggers",
                frontmatter.get("trigger_keywords", [])
            )

            # If no explicit triggers, try to parse from description
            if not trigger_keywords and description:
                # Check for "Triggers for:" or "Triggers include:" pattern in description
                triggers_match = re.search(
                    r'Triggers\s*(?:for|include)\s*:\s*([^.]+)',
                    description,
                    re.IGNORECASE
                )
                if triggers_match:
                    # Parse quoted keywords: "keyword1", "keyword2"
                    triggers_text = triggers_match.group(1)
                    trigger_keywords = re.findall(r'"([^"]+)"', triggers_text)
                    if not trigger_keywords:
                        # Fallback: split by comma
                        trigger_keywords = [k.strip() for k in triggers_text.split(",") if k.strip()]
                # No fallback to "first 5 words" — that produced garbage keywords like
                # "of", "sub-skill", "full" that triggered on almost any input.
                # Skills without explicit triggers simply won't auto-trigger
                # (they can still be accessed via read_document).

            hint_message = extract_hint_message(frontmatter, body)

            # If still no hint, use description
            if not hint_message and description:
                hint_message = description

            priority = frontmatter.get("skill_priority", frontmatter.get("priority", 0))

            skill = Skill(
                skill_id=skill_id,
                name=name,
                description=description,
                trigger_keywords=trigger_keywords,
                hint_message=hint_message,
                source_doc_id=frontmatter.get("doc_id"),
                source_path=str(doc_file.relative_to(corpus_path)),
                priority=priority,
            )
            skills.append(skill)

        except Exception as e:
            print(f"[SKILL LOADER] Error loading {doc_file}: {e}")
            continue

    # Sort by priority (higher first)
    skills.sort(key=lambda s: s.priority, reverse=True)
    return skills


if __name__ == "__main__":
    skills = load_skills_from_corpus("corpus")
    for s in skills:
        print(f"{s.skill_id}: {s.name}")
        print(f"  triggers: {s.trigger_keywords}")