"""
tests/test_skills.py
────────────────────
Unit tests for skills loader and manager focusing on detect_skills
and load_skills_from_corpus methods.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import AgememConfig
from core.types import Skill
from skills.loader import load_skills_from_corpus, parse_frontmatter, extract_hint_message
from skills.manager import SkillManager


class TestParseFrontmatter(unittest.TestCase):
    """Tests for parse_frontmatter helper function."""

    def test_parse_empty_frontmatter(self):
        """parse_frontmatter returns empty dict for content without frontmatter."""
        content = "# Simple document\n\nThis has no frontmatter."
        frontmatter, body = parse_frontmatter(content)

        self.assertEqual(frontmatter, {})
        self.assertEqual(body, content)

    def test_parse_basic_frontmatter(self):
        """parse_frontmatter extracts basic key-value pairs."""
        content = """---
title: My Document
doc_type: skill
priority: 5
---

# Document body

Content here.
"""
        frontmatter, body = parse_frontmatter(content)

        self.assertEqual(frontmatter["title"], "My Document")
        self.assertEqual(frontmatter["doc_type"], "skill")
        self.assertEqual(frontmatter["priority"], 5)
        self.assertIn("# Document body", body)

    def test_parse_list_in_brackets(self):
        """parse_frontmatter handles list values in brackets."""
        content = """---
triggers: [python, coding, programming]
---

Body text.
"""
        frontmatter, body = parse_frontmatter(content)

        self.assertEqual(frontmatter["triggers"], ["python", "coding", "programming"])

    def test_parse_multiline_list(self):
        """parse_frontmatter handles multiline YAML lists."""
        content = """---
triggers:
  - python
  - coding
  - programming
---

Body text.
"""
        frontmatter, body = parse_frontmatter(content)

        self.assertEqual(frontmatter["triggers"], ["python", "coding", "programming"])

    def test_parse_boolean_values(self):
        """parse_frontmatter correctly parses boolean values."""
        content = """---
enabled: true
disabled: false
---

Body.
"""
        frontmatter, _ = parse_frontmatter(content)

        self.assertEqual(frontmatter["enabled"], True)
        self.assertEqual(frontmatter["disabled"], False)

    def test_parse_numeric_values(self):
        """parse_frontmatter correctly parses numeric values."""
        content = """---
count: 42
score: 3.14
---

Body.
"""
        frontmatter, _ = parse_frontmatter(content)

        self.assertEqual(frontmatter["count"], 42)
        self.assertEqual(frontmatter["score"], 3.14)

    def test_parse_quoted_strings(self):
        """parse_frontmatter handles quoted strings."""
        content = """---
title: "Quoted Title"
description: 'Single quoted'
---

Body.
"""
        frontmatter, _ = parse_frontmatter(content)

        self.assertEqual(frontmatter["title"], "Quoted Title")
        self.assertEqual(frontmatter["description"], "Single quoted")


class TestExtractHintMessage(unittest.TestCase):
    """Tests for extract_hint_message helper function."""

    def test_extract_from_skill_hint_field(self):
        """extract_hint_message prioritizes skill_hint field."""
        frontmatter = {"skill_hint": "This is the hint"}
        body = "# Some content\n\nOther text."

        hint = extract_hint_message(frontmatter, body)

        self.assertEqual(hint, "This is the hint")

    def test_extract_from_core_section(self):
        """extract_hint_message extracts from Core section when no skill_hint."""
        frontmatter = {}
        body = """# Title

## Core

This is the core paragraph.

More core content.

## Other Section

Other content.
"""
        hint = extract_hint_message(frontmatter, body)

        self.assertEqual(hint, "This is the core paragraph.")

    def test_extract_from_first_paragraph(self):
        """extract_hint_message falls back to first paragraph."""
        frontmatter = {}
        body = """# Title

This is the first paragraph.

Second paragraph.
"""
        hint = extract_hint_message(frontmatter, body)

        self.assertEqual(hint, "This is the first paragraph.")

    def test_extract_removes_markdown(self):
        """extract_hint_message removes markdown formatting."""
        frontmatter = {}
        body = """# Title

This has **bold** and *italic* and `code`.
"""
        hint = extract_hint_message(frontmatter, body)

        self.assertEqual(hint, "This has bold and italic and code.")

    def test_extract_removes_links(self):
        """extract_hint_message removes markdown links."""
        frontmatter = {}
        body = """# Title

Check out [this link](http://example.com) for more info.
"""
        hint = extract_hint_message(frontmatter, body)

        self.assertEqual(hint, "Check out this link for more info.")

    def test_extract_limits_length(self):
        """extract_hint_message limits hint to 500 characters."""
        frontmatter = {}
        long_text = "A" * 600
        body = f"# Title\n\n{long_text}"

        hint = extract_hint_message(frontmatter, body)

        self.assertEqual(len(hint), 500)


class TestLoadSkillsFromCorpus(unittest.TestCase):
    """Tests for load_skills_from_corpus function."""

    def setUp(self):
        """Create temporary directory for corpus files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.corpus_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_load_skills_empty_directory(self):
        """load_skills_from_corpus returns empty list for empty directory."""
        skills = load_skills_from_corpus(self.corpus_path)

        self.assertEqual(len(skills), 0)
        self.assertIsInstance(skills, list)

    def test_load_skills_nonexistent_directory(self):
        """load_skills_from_corpus returns empty list for nonexistent directory."""
        skills = load_skills_from_corpus("/nonexistent/path")

        self.assertEqual(len(skills), 0)

    def test_load_skills_by_doc_type(self):
        """load_skills_from_corpus loads documents with doc_type: skill."""
        skill_doc = """---
doc_id: python-skill
doc_type: skill
skill_name: Python Programming
skill_description: Help with Python code
skill_triggers:
  - python
  - coding
skill_hint: I can help with Python programming
skill_priority: 10
---

# Python Programming

This skill helps with Python code.
"""
        doc_path = self.corpus_path / "python_skill.md"
        doc_path.write_text(skill_doc)

        skills = load_skills_from_corpus(self.corpus_path)

        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].skill_id, "python-skill")
        self.assertEqual(skills[0].name, "Python Programming")
        self.assertEqual(skills[0].trigger_keywords, ["python", "coding"])

    def test_load_skills_by_skill_triggers(self):
        """load_skills_from_corpus loads documents with skill_triggers field."""
        skill_doc = """---
doc_id: web-skill
skill_name: Web Development
skill_description: Help with web dev
skill_triggers: [javascript, html, css]
---

# Web Development

This skill helps with web development.
"""
        doc_path = self.corpus_path / "web_skill.md"
        doc_path.write_text(skill_doc)

        skills = load_skills_from_corpus(self.corpus_path)

        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].skill_id, "web-skill")
        self.assertEqual(skills[0].trigger_keywords, ["javascript", "html", "css"])

    def test_load_skills_ignores_non_skill_docs(self):
        """load_skills_from_corpus ignores documents without skill markers."""
        regular_doc = """---
doc_id: regular-doc
doc_type: guide
---

# Regular Document

This is not a skill.
"""
        doc_path = self.corpus_path / "regular_doc.md"
        doc_path.write_text(regular_doc)

        skills = load_skills_from_corpus(self.corpus_path)

        self.assertEqual(len(skills), 0)

    def test_load_skills_sorts_by_priority(self):
        """load_skills_from_corpus sorts skills by priority descending."""
        low_priority = """---
doc_id: low-skill
doc_type: skill
skill_name: Low Priority
skill_triggers: [test]
skill_priority: 1
---

Content.
"""
        high_priority = """---
doc_id: high-skill
doc_type: skill
skill_name: High Priority
skill_triggers: [test]
skill_priority: 10
---

Content.
"""
        (self.corpus_path / "low.md").write_text(low_priority)
        (self.corpus_path / "high.md").write_text(high_priority)

        skills = load_skills_from_corpus(self.corpus_path)

        self.assertEqual(len(skills), 2)
        self.assertEqual(skills[0].name, "High Priority")
        self.assertEqual(skills[1].name, "Low Priority")

    def test_load_skills_uses_stem_as_fallback_id(self):
        """load_skills_from_corpus uses filename stem when doc_id missing."""
        skill_doc = """---
doc_type: skill
skill_name: Test Skill
skill_triggers: [test]
---

Content.
"""
        doc_path = self.corpus_path / "my_custom_skill.md"
        doc_path.write_text(skill_doc)

        skills = load_skills_from_corpus(self.corpus_path)

        self.assertEqual(skills[0].skill_id, "my_custom_skill")

    def test_load_skills_infers_triggers_from_description(self):
        """load_skills_from_corpus infers triggers from description if none provided."""
        skill_doc = """---
doc_type: skill
skill_name: Test Skill
skill_description: python machine learning ai
---

Content.
"""
        doc_path = self.corpus_path / "test_skill.md"
        doc_path.write_text(skill_doc)

        skills = load_skills_from_corpus(self.corpus_path)

        # Should take first 5 words from description
        self.assertEqual(skills[0].trigger_keywords, ["python", "machine", "learning", "ai"])

    def test_load_skills_uses_description_as_hint_fallback(self):
        """load_skills_from_corpus uses description when no hint extracted."""
        skill_doc = """---
doc_type: skill
skill_name: Test Skill
skill_description: This is the description
skill_triggers: [test]
---

---

"""
        doc_path = self.corpus_path / "test_skill.md"
        doc_path.write_text(skill_doc)

        skills = load_skills_from_corpus(self.corpus_path)

        self.assertEqual(skills[0].hint_message, "This is the description")

    def test_load_skills_handles_errors_gracefully(self):
        """load_skills_from_corpus continues loading when one file errors."""
        good_doc = """---
doc_type: skill
skill_name: Good Skill
skill_triggers: [good]
---

Content.
"""
        (self.corpus_path / "good.md").write_text(good_doc)
        (self.corpus_path / "bad.md").write_text("invalid content")

        skills = load_skills_from_corpus(self.corpus_path)

        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].name, "Good Skill")


class TestSkillManagerDetectSkills(unittest.TestCase):
    """Tests for SkillManager.detect_skills method."""

    def setUp(self):
        """Create skill manager with test config."""
        self.config = AgememConfig(
            SKILL_DETECTION_ENABLED=True,
            SKILL_MAX_HINTS_PER_TURN=3,
            SKILL_TRIGGER_MIN_MATCHES=1,
            SKILL_CORPUS_PATH=None,
        )
        self.manager = SkillManager(self.config)

    def test_detect_skills_disabled(self):
        """detect_skills returns empty list when disabled."""
        self.config.SKILL_DETECTION_ENABLED = False

        self.manager.add_skill(Skill(
            skill_id="test",
            name="Test",
            description="Test skill",
            trigger_keywords=["python"],
            hint_message="Hint",
        ))

        matches = self.manager.detect_skills("I love python")

        self.assertEqual(len(matches), 0)

    def test_detect_skills_single_match(self):
        """detect_skills returns matching skill."""
        self.manager._loaded = True  # Prevent auto-load
        self.manager.add_skill(Skill(
            skill_id="python",
            name="Python",
            description="Python help",
            trigger_keywords=["python", "coding"],
            hint_message="I can help with Python",
        ))

        matches = self.manager.detect_skills("I need help with python")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].skill_id, "python")

    def test_detect_skills_no_match(self):
        """detect_skills returns empty list when no skills match."""
        self.manager._loaded = True  # Prevent auto-load
        self.manager.add_skill(Skill(
            skill_id="python",
            name="Python",
            description="Python help",
            trigger_keywords=["python"],
            hint_message="Hint",
        ))

        matches = self.manager.detect_skills("I love javascript")

        self.assertEqual(len(matches), 0)

    def test_detect_skills_multiple_matches(self):
        """detect_skills returns multiple matching skills."""
        self.manager._loaded = True  # Prevent auto-load
        self.manager.add_skill(Skill(
            skill_id="python",
            name="Python",
            description="Python help",
            trigger_keywords=["python"],
            hint_message="Hint",
            priority=1,
        ))
        self.manager.add_skill(Skill(
            skill_id="coding",
            name="Coding",
            description="Coding help",
            trigger_keywords=["coding"],
            hint_message="Hint",
            priority=2,
        ))

        matches = self.manager.detect_skills("I do python coding")

        self.assertEqual(len(matches), 2)

    def test_detect_skills_respects_min_matches(self):
        """detect_skills respects SKILL_TRIGGER_MIN_MATCHES config."""
        self.manager._loaded = True  # Prevent auto-load
        self.config.SKILL_TRIGGER_MIN_MATCHES = 2

        self.manager.add_skill(Skill(
            skill_id="python",
            name="Python",
            description="Python help",
            trigger_keywords=["python", "programming"],
            hint_message="Hint",
        ))

        # Only one keyword matches
        matches = self.manager.detect_skills("I love python")
        self.assertEqual(len(matches), 0)

        # Two keywords match
        matches = self.manager.detect_skills("I love python programming")
        self.assertEqual(len(matches), 1)

    def test_detect_skills_respects_max_hints(self):
        """detect_skills limits results to SKILL_MAX_HINTS_PER_TURN."""
        self.manager._loaded = True  # Prevent auto-load
        self.config.SKILL_MAX_HINTS_PER_TURN = 2

        # Add 5 skills that would all match
        for i in range(5):
            self.manager.add_skill(Skill(
                skill_id=f"skill{i}",
                name=f"Skill {i}",
                description="Help",
                trigger_keywords=["help"],
                hint_message="Hint",
            ))

        matches = self.manager.detect_skills("I need help")

        self.assertEqual(len(matches), 2)

    def test_detect_skills_sorts_by_priority(self):
        """detect_skills returns skills sorted by priority."""
        self.manager._loaded = True  # Prevent auto-load
        self.manager.add_skill(Skill(
            skill_id="low",
            name="Low",
            description="Help",
            trigger_keywords=["help"],
            hint_message="Hint",
            priority=1,
        ))
        self.manager.add_skill(Skill(
            skill_id="high",
            name="High",
            description="Help",
            trigger_keywords=["help"],
            hint_message="Hint",
            priority=10,
        ))

        matches = self.manager.detect_skills("I need help")

        self.assertEqual(matches[0].skill_id, "high")
        self.assertEqual(matches[1].skill_id, "low")

    def test_detect_skills_auto_loads(self):
        """detect_skills auto-loads skills if not loaded."""
        # Create temp corpus
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_doc = """---
doc_type: skill
skill_name: Test Skill
skill_triggers: [test]
---

Content.
"""
            (Path(tmpdir) / "test_skill.md").write_text(skill_doc)

            self.config.SKILL_CORPUS_PATH = tmpdir
            manager = SkillManager(self.config)

            # Should auto-load before detecting
            matches = manager.detect_skills("test")

            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].name, "Test Skill")

    def test_detect_skills_case_insensitive(self):
        """detect_skills matching is case insensitive."""
        self.manager._loaded = True  # Prevent auto-load
        self.manager.add_skill(Skill(
            skill_id="python",
            name="Python",
            description="Python help",
            trigger_keywords=["python", "PYTHON"],
            hint_message="Hint",
        ))

        matches = self.manager.detect_skills("I love PYTHON programming")

        self.assertEqual(len(matches), 1)


class TestSkillManagerIntegration(unittest.TestCase):
    """Integration tests for SkillManager with corpus loading."""

    def test_load_and_detect_integration(self):
        """Full workflow: load skills from corpus then detect."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create skill documents
            python_skill = """---
doc_id: python-help
doc_type: skill
skill_name: Python Helper
skill_description: Help with Python code
skill_triggers:
  - python
  - coding
skill_hint: I can help you write Python code
skill_priority: 10
---

# Python Helper

This skill provides help with Python programming.
"""
            web_skill = """---
doc_id: web-dev
doc_type: skill
skill_name: Web Development
skill_description: Help with web development
skill_triggers:
  - javascript
  - html
  - css
  - web
skill_hint: I can help with web development
skill_priority: 5
---

# Web Development

This skill helps with web development.
"""
            (Path(tmpdir) / "python.md").write_text(python_skill)
            (Path(tmpdir) / "web.md").write_text(web_skill)

            # Create manager and load
            config = AgememConfig(SKILL_CORPUS_PATH=None)
            manager = SkillManager(config)
            manager.load_skills(tmpdir)

            # Verify loaded
            self.assertEqual(len(manager.get_all_skills()), 2)

            # Detect python-related skills
            matches = manager.detect_skills("I need help with python")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].skill_id, "python-help")

            # Detect web-related skills
            matches = manager.detect_skills("javascript and css help")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].skill_id, "web-dev")

    def test_get_skill_by_id(self):
        """get_skill_by_id retrieves specific skill."""
        self.manager = SkillManager(AgememConfig())
        self.manager.add_skill(Skill(
            skill_id="test-skill",
            name="Test",
            description="Test skill",
            trigger_keywords=["test"],
            hint_message="Hint",
        ))

        skill = self.manager.get_skill_by_id("test-skill")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "Test")

        not_found = self.manager.get_skill_by_id("nonexistent")
        self.assertIsNone(not_found)

    def test_clear_skills(self):
        """clear_skills removes all loaded skills."""
        self.manager = SkillManager(AgememConfig())
        self.manager.add_skill(Skill(
            skill_id="test",
            name="Test",
            description="Test",
            trigger_keywords=["test"],
            hint_message="Hint",
        ))

        self.assertEqual(len(self.manager.get_all_skills()), 1)

        self.manager.clear_skills()

        self.assertEqual(len(self.manager.get_all_skills()), 0)
        self.assertFalse(self.manager._loaded)


if __name__ == "__main__":
    unittest.main()
