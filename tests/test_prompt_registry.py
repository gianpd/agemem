"""
tests/test_prompt_registry.py
─────────────────────────────
Unit tests for the Prompt Registry with Versioning.

Tests cover:
- PromptLoader: loading, parsing, caching, saving prompts
- PromptRegistry: high-level API for prompt management
- Frontmatter parsing and rendering
- Version activation and switching
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.loader import PromptLoader, parse_frontmatter, render_frontmatter
from prompts.registry import PromptRegistry
from core.types import Prompt, PromptMetadata, PromptVersion


class TestParseFrontmatter(unittest.TestCase):
    """Tests for parse_frontmatter helper function."""

    def test_parse_no_frontmatter(self):
        """parse_frontmatter returns empty metadata for content without frontmatter."""
        content = "# Simple prompt\n\nThis has no frontmatter."
        parsed = parse_frontmatter(content)

        self.assertEqual(parsed.metadata, {})
        self.assertEqual(parsed.content, content.strip())

    def test_parse_basic_frontmatter(self):
        """parse_frontmatter extracts basic key-value pairs."""
        content = """---
prompt_id: main-system
name: Main System Prompt
version: 1.0.0
---

# System Prompt

You are a helpful assistant.
"""
        parsed = parse_frontmatter(content)

        self.assertEqual(parsed.metadata["prompt_id"], "main-system")
        self.assertEqual(parsed.metadata["name"], "Main System Prompt")
        self.assertEqual(parsed.metadata["version"], "1.0.0")
        self.assertIn("# System Prompt", parsed.content)

    def test_parse_list_in_brackets(self):
        """parse_frontmatter handles list values in brackets."""
        content = """---
prompt_id: test
tags: [system, core, test]
---

Body text.
"""
        parsed = parse_frontmatter(content)

        self.assertEqual(parsed.metadata["tags"], ["system", "core", "test"])

    def test_parse_boolean_values(self):
        """parse_frontmatter correctly parses boolean values."""
        content = """---
prompt_id: test
active: true
disabled: false
---

Body.
"""
        parsed = parse_frontmatter(content)

        self.assertEqual(parsed.metadata["active"], True)
        self.assertEqual(parsed.metadata["disabled"], False)

    def test_parse_quoted_strings(self):
        """parse_frontmatter handles quoted strings with special characters."""
        content = """---
prompt_id: test
description: "A test: with special [chars]"
---

Body.
"""
        parsed = parse_frontmatter(content)

        self.assertEqual(parsed.metadata["description"], "A test: with special [chars]")

    def test_parse_single_quotes(self):
        """parse_frontmatter handles single-quoted strings."""
        content = """---
prompt_id: test
name: 'Test Prompt'
---

Body.
"""
        parsed = parse_frontmatter(content)

        self.assertEqual(parsed.metadata["name"], "Test Prompt")


class TestRenderFrontmatter(unittest.TestCase):
    """Tests for render_frontmatter function."""

    def test_render_basic_metadata(self):
        """render_frontmatter creates valid frontmatter."""
        metadata = {
            "prompt_id": "test",
            "name": "Test Prompt",
            "version": "1.0.0",
        }
        content = "Test content."

        result = render_frontmatter(metadata, content)

        self.assertIn("---", result)
        self.assertIn("prompt_id: test", result)
        self.assertIn("name: Test Prompt", result)
        self.assertIn("version: 1.0.0", result)
        self.assertIn("Test content.", result)

    def test_render_list_values(self):
        """render_frontmatter correctly formats list values."""
        metadata = {
            "prompt_id": "test",
            "tags": ["a", "b", "c"],
        }
        content = "Body."

        result = render_frontmatter(metadata, content)

        self.assertIn('tags: ["a", "b", "c"]', result)

    def test_render_boolean_values(self):
        """render_frontmatter correctly formats boolean values."""
        metadata = {
            "prompt_id": "test",
            "active": True,
            "disabled": False,
        }
        content = "Body."

        result = render_frontmatter(metadata, content)

        self.assertIn("active: true", result)
        self.assertIn("disabled: false", result)

    def test_roundtrip_parsing(self):
        """Parsing then rendering preserves data."""
        original = """---
prompt_id: test-prompt
name: Test Prompt
version: 1.0.0
tags: [system, core]
active: true
---

# Test Content

This is the prompt content.
"""
        parsed = parse_frontmatter(original)
        rendered = render_frontmatter(parsed.metadata, parsed.content)
        reparsed = parse_frontmatter(rendered)

        self.assertEqual(parsed.metadata, reparsed.metadata)
        self.assertEqual(parsed.content, reparsed.content)


class TestPromptLoader(unittest.TestCase):
    """Tests for PromptLoader class."""

    def setUp(self):
        """Create temporary directory for test prompts."""
        self.temp_dir = tempfile.mkdtemp()
        self.prompts_dir = Path(self.temp_dir) / "prompts"
        self.prompts_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_file(self, filename: str, content: str) -> Path:
        """Helper to create a test prompt file."""
        file_path = self.prompts_dir / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def test_load_single_prompt(self):
        """PromptLoader loads a single prompt file."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test Prompt
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

Test prompt content.
""")
        loader = PromptLoader(self.prompts_dir)
        prompt = loader.get_prompt("test")

        self.assertIsNotNone(prompt)
        self.assertEqual(prompt.prompt_id, "test")
        self.assertEqual(prompt.name, "Test Prompt")
        self.assertEqual(prompt.version, "1.0.0")
        self.assertEqual(prompt.content, "Test prompt content.")

    def test_load_multiple_versions(self):
        """PromptLoader handles multiple versions of the same prompt."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test Prompt
version: 1.0.0
active: false
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

Version 1 content.
""")
        self._create_test_file("test-v2_0_0.md", """---
prompt_id: test
name: Test Prompt
version: 2.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

Version 2 content.
""")
        loader = PromptLoader(self.prompts_dir)

        # Get active version (should be 2.0.0)
        active = loader.get_prompt("test")
        self.assertEqual(active.version, "2.0.0")

        # Get specific version
        v1 = loader.get_prompt("test", "1.0.0")
        self.assertEqual(v1.version, "1.0.0")
        self.assertEqual(v1.content, "Version 1 content.")

    def test_get_active_version(self):
        """PromptLoader returns the correct active version."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: false
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

V1.
""")
        self._create_test_file("test-v2_0_0.md", """---
prompt_id: test
name: Test
version: 2.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

V2.
""")
        loader = PromptLoader(self.prompts_dir)

        self.assertEqual(loader.get_active_version("test"), "2.0.0")

    def test_list_prompts(self):
        """PromptLoader lists all available prompts."""
        self._create_test_file("prompt1-v1_0_0.md", """---
prompt_id: prompt1
name: Prompt One
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
tags: [a, b]
---

Content 1.
""")
        self._create_test_file("prompt2-v1_0_0.md", """---
prompt_id: prompt2
name: Prompt Two
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
tags: [c, d]
---

Content 2.
""")
        loader = PromptLoader(self.prompts_dir)
        prompts = loader.list_prompts()

        self.assertEqual(len(prompts), 2)
        prompt_ids = [p["prompt_id"] for p in prompts]
        self.assertIn("prompt1", prompt_ids)
        self.assertIn("prompt2", prompt_ids)

    def test_list_versions(self):
        """PromptLoader lists all versions of a prompt."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: false
created_at: 2026-03-09
updated_at: 2026-03-09
author: alice
---

V1.
""")
        self._create_test_file("test-v2_0_0.md", """---
prompt_id: test
name: Test
version: 2.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: bob
---

V2.
""")
        loader = PromptLoader(self.prompts_dir)
        versions = loader.list_versions("test")

        self.assertEqual(len(versions), 2)
        version_strings = [v.version for v in versions]
        self.assertIn("1.0.0", version_strings)
        self.assertIn("2.0.0", version_strings)

    def test_activate_version(self):
        """PromptLoader can activate a specific version."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

V1.
""")
        self._create_test_file("test-v2_0_0.md", """---
prompt_id: test
name: Test
version: 2.0.0
active: false
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

V2.
""")
        loader = PromptLoader(self.prompts_dir)

        # Initially 1.0.0 is active
        self.assertEqual(loader.get_active_version("test"), "1.0.0")

        # Activate 2.0.0
        result = loader.activate_version("test", "2.0.0")
        self.assertTrue(result)

        # Now 2.0.0 should be active
        self.assertEqual(loader.get_active_version("test"), "2.0.0")

    def test_save_prompt(self):
        """PromptLoader can save a new prompt."""
        loader = PromptLoader(self.prompts_dir)

        result = loader.save_prompt(
            prompt_id="new-prompt",
            name="New Prompt",
            content="This is a new prompt.",
            version="1.0.0",
            author="tester",
            tags=["test", "new"],
            activate=True,
        )

        self.assertTrue(result)

        # Verify it was saved
        saved_file = self.prompts_dir / "new-prompt-v1_0_0.md"
        self.assertTrue(saved_file.exists())

        # Verify content
        loaded = loader.get_prompt("new-prompt")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content, "This is a new prompt.")
        self.assertEqual(loaded.author, "tester")

    def test_get_prompt_not_found(self):
        """PromptLoader returns None for non-existent prompt."""
        loader = PromptLoader(self.prompts_dir)

        result = loader.get_prompt("non-existent")
        self.assertIsNone(result)

    def test_reload_clears_cache(self):
        """reload() clears the cache and reloads from disk."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

Original.
""")
        loader = PromptLoader(self.prompts_dir)

        # Load initial content
        original = loader.get_prompt("test")
        self.assertEqual(original.content, "Original.")

        # Modify file directly
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

Modified.
""")

        # Without reload, still get cached version
        cached = loader.get_prompt("test")
        self.assertEqual(cached.content, "Original.")

        # After reload, get new content
        loader.reload()
        refreshed = loader.get_prompt("test")
        self.assertEqual(refreshed.content, "Modified.")


class TestPromptRegistry(unittest.TestCase):
    """Tests for PromptRegistry high-level API."""

    def setUp(self):
        """Create temporary directory for test prompts."""
        self.temp_dir = tempfile.mkdtemp()
        self.prompts_dir = Path(self.temp_dir) / "prompts"
        self.prompts_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_file(self, filename: str, content: str) -> Path:
        """Helper to create a test prompt file."""
        file_path = self.prompts_dir / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def test_get_prompt(self):
        """Registry returns prompt by ID."""
        self._create_test_file("main-v1_0_0.md", """---
prompt_id: main
name: Main Prompt
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

Main content.
""")
        registry = PromptRegistry(self.prompts_dir)
        prompt = registry.get_prompt("main")

        self.assertIsInstance(prompt, Prompt)
        self.assertEqual(prompt.prompt_id, "main")
        self.assertEqual(prompt.content, "Main content.")

    def test_get_prompt_not_found_raises(self):
        """Registry raises KeyError for non-existent prompt."""
        registry = PromptRegistry(self.prompts_dir)

        with self.assertRaises(KeyError) as ctx:
            registry.get_prompt("non-existent")

        self.assertIn("non-existent", str(ctx.exception))

    def test_get_active_version(self):
        """Registry returns active version string."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: false
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

V1.
""")
        self._create_test_file("test-v2_0_0.md", """---
prompt_id: test
name: Test
version: 2.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

V2.
""")
        registry = PromptRegistry(self.prompts_dir)

        self.assertEqual(registry.get_active_version("test"), "2.0.0")

    def test_list_prompts_returns_metadata(self):
        """Registry list_prompts returns PromptMetadata objects."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test Prompt
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
tags: [a, b]
---

Content.
""")
        registry = PromptRegistry(self.prompts_dir)
        prompts = registry.list_prompts()

        self.assertEqual(len(prompts), 1)
        self.assertIsInstance(prompts[0], PromptMetadata)
        self.assertEqual(prompts[0].prompt_id, "test")
        self.assertEqual(prompts[0].name, "Test Prompt")

    def test_list_versions(self):
        """Registry list_versions returns PromptVersion objects."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: false
created_at: 2026-03-09
updated_at: 2026-03-09
author: alice
---

V1.
""")
        self._create_test_file("test-v2_0_0.md", """---
prompt_id: test
name: Test
version: 2.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: bob
---

V2.
""")
        registry = PromptRegistry(self.prompts_dir)
        versions = registry.list_versions("test")

        self.assertEqual(len(versions), 2)
        self.assertIsInstance(versions[0], PromptVersion)

        # Should be sorted by version
        self.assertEqual(versions[0].version, "1.0.0")
        self.assertEqual(versions[1].version, "2.0.0")

    def test_list_versions_not_found_raises(self):
        """Registry raises KeyError for non-existent prompt versions."""
        registry = PromptRegistry(self.prompts_dir)

        with self.assertRaises(KeyError) as ctx:
            registry.list_versions("non-existent")

        self.assertIn("non-existent", str(ctx.exception))

    def test_activate_version(self):
        """Registry can activate a specific version."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

V1.
""")
        self._create_test_file("test-v2_0_0.md", """---
prompt_id: test
name: Test
version: 2.0.0
active: false
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

V2.
""")
        registry = PromptRegistry(self.prompts_dir)

        # Initially 1.0.0 is active
        self.assertEqual(registry.get_active_version("test"), "1.0.0")

        # Activate 2.0.0
        result = registry.activate_version("test", "2.0.0")
        self.assertTrue(result)

        # Verify activation
        self.assertEqual(registry.get_active_version("test"), "2.0.0")
        active_prompt = registry.get_prompt("test")
        self.assertEqual(active_prompt.version, "2.0.0")

    def test_activate_version_not_found_raises(self):
        """Registry raises KeyError when activating non-existent version."""
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

V1.
""")
        registry = PromptRegistry(self.prompts_dir)

        with self.assertRaises(KeyError) as ctx:
            registry.activate_version("test", "9.9.9")

        self.assertIn("9.9.9", str(ctx.exception))

    def test_save_prompt_via_registry(self):
        """Registry can save new prompts."""
        registry = PromptRegistry(self.prompts_dir)

        result = registry.save_prompt(
            prompt_id="new-prompt",
            name="New Prompt",
            content="Registry test content.",
            version="1.0.0",
            author="registry-test",
            tags=["registry", "test"],
            activate=True,
        )

        self.assertTrue(result)

        # Verify it can be retrieved
        loaded = registry.get_prompt("new-prompt")
        self.assertEqual(loaded.content, "Registry test content.")
        self.assertEqual(loaded.author, "registry-test")

    def test_reload(self):
        """Registry reload refreshes from disk."""
        # Create initial version
        test_file = self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

Original content.
""")
        registry = PromptRegistry(self.prompts_dir)

        # Verify initial content
        original = registry.get_prompt("test")
        self.assertEqual(original.content, "Original content.")

        # Modify file directly (simulating external edit)
        test_file.write_text("""---
prompt_id: test
name: Test
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

Modified content.
""", encoding="utf-8")

        # After reload, should see modified content
        registry.reload()
        refreshed = registry.get_prompt("test")
        self.assertEqual(refreshed.content, "Modified content.")

    def test_get_prompts_dir(self):
        """Registry returns the prompts directory."""
        registry = PromptRegistry(self.prompts_dir)

        self.assertEqual(registry.get_prompts_dir(), self.prompts_dir)


class TestPromptTypes(unittest.TestCase):
    """Tests for Prompt dataclass types."""

    def test_prompt_to_dict(self):
        """Prompt.to_dict serializes correctly."""
        prompt = Prompt(
            prompt_id="test",
            name="Test",
            version="1.0.0",
            content="Test content",
            created_at="2026-03-10",
            updated_at="2026-03-10",
            author="test",
            tags=["a", "b"],
            active=True,
        )

        data = prompt.to_dict()

        self.assertEqual(data["prompt_id"], "test")
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["content"], "Test content")
        self.assertEqual(data["tags"], ["a", "b"])
        self.assertEqual(data["active"], True)


class TestSTMPinnedSystemMessageUpdate(unittest.TestCase):
    """Tests for STMContext.update_pinned_system_message() method."""

    def setUp(self):
        """Create STMContext with mocked token counter."""
        from memory.stm_context import STMContext
        from core.types import TokenCounter

        self.tc = TokenCounter()
        self.stm = STMContext(token_counter=self.tc)

    def test_update_pinned_system_message_success(self):
        """update_pinned_system_message updates existing pinned message."""
        # Add initial pinned system message
        self.stm.add_message(
            role="system",
            content="Original system prompt",
            is_pinned=True,
        )

        # Update the pinned message
        result = self.stm.update_pinned_system_message("Updated system prompt")

        self.assertTrue(result)
        messages = self.stm.messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "Updated system prompt")
        self.assertTrue(messages[0].is_pinned)

    def test_update_pinned_system_message_not_found(self):
        """update_pinned_system_message returns False when no pinned system msg."""
        # Add non-pinned message
        self.stm.add_message(
            role="user",
            content="User message",
            is_pinned=False,
        )

        result = self.stm.update_pinned_system_message("New system prompt")

        self.assertFalse(result)
        messages = self.stm.messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "User message")

    def test_update_preserves_other_pinned_messages(self):
        """update_pinned_system_message only updates the system prompt."""
        # Add multiple pinned messages
        self.stm.add_message(
            role="system",
            content="Main system prompt",
            is_pinned=True,
        )
        self.stm.add_message(
            role="system",
            content="[MEMORY:abc123] Memory content",
            is_pinned=True,
        )

        result = self.stm.update_pinned_system_message("Updated main prompt")

        self.assertTrue(result)
        messages = self.stm.messages()
        self.assertEqual(len(messages), 2)

        # Find the main system prompt (first one)
        main_prompt = messages[0]
        self.assertEqual(main_prompt.content, "Updated main prompt")
        self.assertTrue(main_prompt.is_pinned)

        # Memory message should be unchanged
        memory_msg = messages[1]
        self.assertEqual(memory_msg.content, "[MEMORY:abc123] Memory content")


class TestPromptReloadIntegration(unittest.TestCase):
    """Integration tests for prompt reload with STM update."""

    def setUp(self):
        """Create temporary directory for test prompts."""
        self.temp_dir = tempfile.mkdtemp()
        self.prompts_dir = Path(self.temp_dir) / "prompts"
        self.prompts_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_file(self, filename: str, content: str) -> Path:
        """Helper to create a test prompt file."""
        file_path = self.prompts_dir / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def test_reload_clears_registry_cache(self):
        """reload() clears registry cache and reloads from disk."""
        from prompts import get_prompt, reload as reload_prompts

        # Create initial version
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

Original content.
""")

        # Force fresh registry for this test
        import prompts
        prompts._registry = None

        # Load via prompts module (uses global registry)
        original = get_prompt("test")
        self.assertEqual(original.content, "Original content.")

        # Modify file directly
        self._create_test_file("test-v1_0_0.md", """---
prompt_id: test
name: Test
version: 1.0.0
active: true
created_at: 2026-03-10
updated_at: 2026-03-10
author: test
---

Modified content.
""")

        # Without reload, still get cached version
        cached = get_prompt("test")
        self.assertEqual(cached.content, "Original content.")

        # After reload, get new content
        reload_prompts()
        refreshed = get_prompt("test")
        self.assertEqual(refreshed.content, "Modified content.")


if __name__ == "__main__":
    unittest.main()
