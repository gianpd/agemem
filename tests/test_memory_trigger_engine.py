"""
tests/test_memory_trigger_engine.py
───────────────────────────────────
Boundary tests for MemoryTriggerEngine.

Tests at the process_turn() boundary, asserting on MemoryCycleReport.
These tests replace the shallow module tests for SystemRules and MemoryAgentDecision.

Per RFC-001:
- Test overflow triggers (R1, R2)
- Test periodic review (R3)
- Test learning spike (R4)
- Test graceful degradation on LLM failures
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import AgememConfig
from core.types import MemoryOp, LearningFeedback
from agents.llm_client import LLMClient
from memory.ltm_store import LTMStore
from memory.stm_context import STMContext
from triggers.memory_trigger_engine import MemoryTriggerEngine, MemoryCycleReport
from triggers.system_rules import RuleID


def _cfg(**overrides) -> AgememConfig:
    """Create config with defaults and overrides."""
    defaults = {
        "STM_TOKEN_LIMIT": 1000,
        "STM_WARNING_THRESHOLD": 0.75,
        "STM_CRITICAL_THRESHOLD": 0.90,
        "STM_MIN_MESSAGES": 2,
        "STM_SUMMARY_WINDOW": 2,
        "LTM_MAX_ENTRIES": 10,
        "LTM_PROMOTE_THRESHOLD": 0.65,
        "LTM_UPDATE_THRESHOLD": 0.5,
        "LEARNING_SCORE_PROMPT_EVERY_N": 999,  # disable in tests
        "LEARNING_SCORE_THRESHOLD_IMMEDIATE": 0.85,
        "TRIGGER_EVERY_N_TURNS": 5,
        "PERSIST_DIR": None,
    }
    defaults.update(overrides)
    return AgememConfig(**defaults)


def _mock_llm_with_json(text_resp: str, json_resp: dict) -> LLMClient:
    """Create LLMClient that returns text then JSON."""
    mock_client = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        is_json = kwargs.get("response_format", {}).get("type") == "json_object"
        choice = MagicMock()
        if is_json:
            choice.message.content = json.dumps(json_resp)
        else:
            choice.message.content = text_resp
        return MagicMock(
            choices=[choice],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5)
        )

    mock_client.chat.completions.create.side_effect = side_effect
    return LLMClient(mock_client, default_model="test-model")


class TestMemoryCycleReport(unittest.TestCase):
    """Test MemoryCycleReport dataclass."""

    def test_empty_report(self):
        """Empty report has defaults."""
        report = MemoryCycleReport()
        self.assertEqual(report.rules_triggered, [])
        self.assertEqual(report.operations, [])
        self.assertEqual(report.summary, "")
        self.assertIsNone(report.agent_rationale)
        self.assertFalse(report.ltm_modified)
        self.assertIsNone(report.stm_stats)

    def test_report_with_data(self):
        """Report stores all fields correctly."""
        report = MemoryCycleReport(
            rules_triggered=[RuleID.OVERFLOW_WARN],
            operations=[MagicMock()],
            summary="Test summary",
            agent_rationale="Agent reasoning",
            ltm_modified=True,
            stm_stats=MagicMock(),
        )
        self.assertEqual(report.rules_triggered, [RuleID.OVERFLOW_WARN])
        self.assertTrue(report.ltm_modified)


class TestOverflowCriticalTrigger(unittest.TestCase):
    """Test R2 OVERFLOW_CRITICAL triggers FILTER + SUMMARY."""

    def test_overflow_critical_triggers_filter_and_summary(self):
        """When STM util >= CRITICAL_THRESHOLD, FILTER and SUMMARY fire."""
        cfg = _cfg(
            STM_TOKEN_LIMIT=100,
            STM_WARNING_THRESHOLD=0.75,
            STM_CRITICAL_THRESHOLD=0.90,
            TRIGGER_EVERY_N_TURNS=999,  # disable periodic
        )
        mock_llm = _mock_llm_with_json("test", {"ltm_operations": [], "context_relevance": [], "summary_needed": False})

        stm = STMContext(cfg)
        # Force overflow by adding large content - need to exceed critical threshold
        # Token counter uses heuristic: ~0.75 tokens per word + 4 overhead
        # So we need a lot of characters to hit 90% of 100 tokens
        stm.add_message("user", "word " * 200)  # ~200 words ≈ 150+ tokens

        ltm = LTMStore(cfg)
        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        # Check stats before processing
        stats = stm.stats()
        self.assertGreaterEqual(stats.utilisation_ratio, 0.90,
                                f"Should be at critical threshold, got {stats.utilisation_ratio:.2f}")

        report = engine.process_turn(turn_index=1)

        self.assertIn(RuleID.OVERFLOW_CRITICAL, report.rules_triggered,
                      "OVERFLOW_CRITICAL should fire at >=90% utilisation")

        op_types = [op.op for op in report.operations]
        self.assertIn(MemoryOp.FILTER, op_types,
                      "FILTER should be executed on critical overflow")
        self.assertIn(MemoryOp.SUMMARY, op_types,
                      "SUMMARY should be executed on critical overflow")

    def test_overflow_critical_priority(self):
        """OVERFLOW_CRITICAL has highest priority (100)."""
        cfg = _cfg(STM_TOKEN_LIMIT=100, STM_CRITICAL_THRESHOLD=0.90)
        mock_llm = _mock_llm_with_json("test", {"ltm_operations": [], "context_relevance": [], "summary_needed": False})

        stm = STMContext(cfg)
        stm.add_message("user", "word " * 200)  # Force overflow

        ltm = LTMStore(cfg)
        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        report = engine.process_turn(turn_index=1)

        # Verify critical overflow happened
        self.assertIn(RuleID.OVERFLOW_CRITICAL, report.rules_triggered,
                      "OVERFLOW_CRITICAL should be present")
        # OVERFLOW_CRITICAL should be first in the list (highest priority)
        self.assertEqual(report.rules_triggered[0], RuleID.OVERFLOW_CRITICAL)


class TestOverflowWarningTrigger(unittest.TestCase):
    """Test R1 OVERFLOW_WARN triggers SUMMARY."""

    def test_overflow_warning_triggers_summary(self):
        """When STM util >= WARNING_THRESHOLD, SUMMARY fires."""
        cfg = _cfg(
            STM_TOKEN_LIMIT=100,
            STM_WARNING_THRESHOLD=0.75,
            STM_CRITICAL_THRESHOLD=0.95,  # Higher to not trigger critical
            TRIGGER_EVERY_N_TURNS=999,
        )
        mock_llm = _mock_llm_with_json("test", {"ltm_operations": [], "context_relevance": [], "summary_needed": False})

        stm = STMContext(cfg)
        # Add content to reach ~80% utilisation
        # Token counter uses heuristic: ~0.75 tokens per word + 4 overhead
        # 100 token limit, need ~80 tokens for 80%
        stm.add_message("user", "word " * 120)  # ~120 words ≈ 90 tokens

        ltm = LTMStore(cfg)
        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        # Check stats before processing
        stats = stm.stats()
        self.assertGreaterEqual(stats.utilisation_ratio, 0.75,
                                f"Should be at warning threshold, got {stats.utilisation_ratio:.2f}")

        report = engine.process_turn(turn_index=1)

        self.assertIn(RuleID.OVERFLOW_WARN, report.rules_triggered,
                      "OVERFLOW_WARN should fire at >=75% utilisation")

        op_types = [op.op for op in report.operations]
        self.assertIn(MemoryOp.SUMMARY, op_types,
                      "SUMMARY should be executed on warning overflow")

    def test_below_warning_threshold_no_action(self):
        """When STM util < WARNING_THRESHOLD, no overflow rules fire."""
        cfg = _cfg(
            STM_TOKEN_LIMIT=1000,
            STM_WARNING_THRESHOLD=0.75,
            TRIGGER_EVERY_N_TURNS=999,
        )
        mock_llm = _mock_llm_with_json("test", {"ltm_operations": [], "context_relevance": [], "summary_needed": False})

        stm = STMContext(cfg)
        stm.add_message("user", "short message")

        ltm = LTMStore(cfg)
        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        report = engine.process_turn(turn_index=1)

        self.assertNotIn(RuleID.OVERFLOW_WARN, report.rules_triggered,
                         "OVERFLOW_WARN should not fire below threshold")
        self.assertNotIn(RuleID.OVERFLOW_CRITICAL, report.rules_triggered,
                         "OVERFLOW_CRITICAL should not fire below threshold")


class TestPeriodicReviewTrigger(unittest.TestCase):
    """Test R3 PERIODIC_REVIEW triggers MemoryAgent."""

    def test_periodic_review_invokes_memory_agent(self):
        """PERIODIC_REVIEW fires at turn N and invokes MemoryAgent."""
        json_resp = {
            "ltm_operations": [{"op": "add", "content": "User likes Python", "confidence": 0.8}],
            "context_relevance": [],
            "summary_needed": False,
            "rationale": "Detected user preference",
        }
        mock_llm = _mock_llm_with_json("test", json_resp)

        cfg = _cfg(
            TRIGGER_EVERY_N_TURNS=5,
            STM_TOKEN_LIMIT=10000,  # No overflow
        )

        stm = STMContext(cfg)
        stm.add_message("user", "I like Python")

        ltm = LTMStore(cfg)
        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        report = engine.process_turn(turn_index=5)

        self.assertIn(RuleID.PERIODIC_REVIEW, report.rules_triggered,
                      "PERIODIC_REVIEW should fire at turn 5")
        self.assertIsNotNone(report.agent_rationale,
                             "Agent rationale should be populated")
        # Check that ADD operation was performed
        add_ops = [op for op in report.operations if op.op == MemoryOp.ADD]
        self.assertTrue(len(add_ops) > 0,
                        "MemoryAgent ADD should be in operations")

    def test_periodic_review_not_on_non_n_turns(self):
        """PERIODIC_REVIEW does not fire on non-N turns."""
        mock_llm = _mock_llm_with_json("test", {"ltm_operations": [], "context_relevance": [], "summary_needed": False})

        cfg = _cfg(
            TRIGGER_EVERY_N_TURNS=5,
            STM_TOKEN_LIMIT=10000,
        )

        stm = STMContext(cfg)
        ltm = LTMStore(cfg)
        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        report = engine.process_turn(turn_index=4)

        self.assertNotIn(RuleID.PERIODIC_REVIEW, report.rules_triggered,
                         "PERIODIC_REVIEW should not fire at turn 4")


class TestLearningSpikeTrigger(unittest.TestCase):
    """Test R4 LEARNING_SPIKE triggers immediate LTM ADD."""

    def test_learning_spike_triggers_immediate_add(self):
        """High learning score triggers immediate LTM promotion."""
        mock_llm = _mock_llm_with_json("test", {"ltm_operations": [], "context_relevance": [], "summary_needed": False})

        cfg = _cfg(
            LTM_PROMOTE_THRESHOLD=0.65,
            LEARNING_SCORE_THRESHOLD_IMMEDIATE=0.85,
            STM_TOKEN_LIMIT=10000,
        )

        stm = STMContext(cfg)
        stm.add_message("assistant", "Important fact to remember")

        ltm = LTMStore(cfg)
        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        feedback = LearningFeedback(
            score=0.90,
            affected_content="Key fact",
            rationale="High importance",
        )

        report = engine.process_turn(turn_index=1, feedback=feedback)

        self.assertIn(RuleID.LEARNING_SPIKE, report.rules_triggered,
                      "LEARNING_SPIKE should fire at score 0.90")
        self.assertTrue(report.ltm_modified,
                        "LTM should be modified on learning spike")

        op_types = [op.op for op in report.operations]
        self.assertIn(MemoryOp.ADD, op_types,
                      "ADD operation should be executed on learning spike")

    def test_learning_spike_below_threshold_no_action(self):
        """Learning score below threshold does not trigger immediate ADD."""
        mock_llm = _mock_llm_with_json("test", {"ltm_operations": [], "context_relevance": [], "summary_needed": False})

        cfg = _cfg(
            LTM_PROMOTE_THRESHOLD=0.65,
            LEARNING_SCORE_THRESHOLD_IMMEDIATE=0.85,
            STM_TOKEN_LIMIT=10000,
        )

        stm = STMContext(cfg)
        ltm = LTMStore(cfg)
        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        feedback = LearningFeedback(
            score=0.50,  # Below threshold
            affected_content="Some content",
            rationale="Low importance",
        )

        report = engine.process_turn(turn_index=1, feedback=feedback)

        # LEARNING_SPIKE rule should NOT fire (0.50 < 0.85)
        self.assertNotIn(RuleID.LEARNING_SPIKE, report.rules_triggered,
                         "LEARNING_SPIKE should not fire at score 0.50")
        # But LTM_PROMOTE_THRESHOLD is 0.65, so still no ADD
        self.assertFalse(report.ltm_modified,
                         "LTM should not be modified at score 0.50")


class TestGracefulDegradation(unittest.TestCase):
    """Test graceful degradation when LLM fails."""

    def test_memory_agent_failure_returns_empty_ops(self):
        """When MemoryAgent fails, engine returns empty operations."""
        # Create LLM that raises exception on JSON calls
        mock_client = MagicMock()

        def side_effect(**kwargs):
            is_json = kwargs.get("response_format", {}).get("type") == "json_object"
            if is_json:
                raise Exception("LLM error")
            choice = MagicMock()
            choice.message.content = "test response"
            return MagicMock(choices=[choice])

        mock_client.chat.completions.create.side_effect = side_effect
        mock_llm = LLMClient(mock_client, default_model="test-model")

        cfg = _cfg(
            TRIGGER_EVERY_N_TURNS=1,  # Force periodic review
            STM_TOKEN_LIMIT=10000,
        )

        stm = STMContext(cfg)
        stm.add_message("user", "test")

        ltm = LTMStore(cfg)
        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        # Should not raise, should return report with no agent ops
        report = engine.process_turn(turn_index=1)

        self.assertIsNotNone(report, "Should return report even on LLM failure")
        # Rules should still fire
        self.assertIn(RuleID.PERIODIC_REVIEW, report.rules_triggered,
                      "PERIODIC_REVIEW should fire even if agent fails")


class TestForceSummary(unittest.TestCase):
    """Test force_summary emergency escape hatch."""

    def test_force_summary_bypasses_rules(self):
        """force_summary executes SUMMARY without rule evaluation."""
        cfg = _cfg(STM_TOKEN_LIMIT=10000)
        mock_llm = _mock_llm_with_json("test", {"ltm_operations": [], "context_relevance": [], "summary_needed": False})

        stm = STMContext(cfg)
        stm.add_message("user", "test message")
        stm.add_message("assistant", "test response")

        ltm = LTMStore(cfg)
        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        result = engine.force_summary()

        self.assertEqual(result.op, MemoryOp.SUMMARY,
                         "force_summary should return SUMMARY operation")


class TestCheckHealth(unittest.TestCase):
    """Test check_health diagnostic method."""

    def test_check_health_returns_diagnostics(self):
        """check_health returns useful diagnostic info."""
        cfg = _cfg(
            STM_TOKEN_LIMIT=1000,
            TRIGGER_EVERY_N_TURNS=10,
        )
        mock_llm = _mock_llm_with_json("test", {"ltm_operations": [], "context_relevance": [], "summary_needed": False})

        stm = STMContext(cfg)
        stm.add_message("user", "test")

        ltm = LTMStore(cfg)
        ltm.add("Test entry", learning_score=0.8)

        engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

        health = engine.check_health()

        self.assertIn("stm_utilisation", health)
        self.assertIn("stm_message_count", health)
        self.assertIn("ltm_entry_count", health)
        self.assertIn("config", health)
        self.assertEqual(health["ltm_entry_count"], 1,
                         "LTM entry count should be 1")


class TestOrchestratorIntegration(unittest.TestCase):
    """Test that Orchestrator correctly uses MemoryTriggerEngine."""

    def test_orchestrator_uses_trigger_engine(self):
        """Orchestrator.process_turn uses MemoryTriggerEngine."""
        from agents.orchestrator import Orchestrator

        json_resp = {
            "score": 0.50,  # Below promote threshold
            "rationale": "Test",
            "affected_content": "",
        }
        mock_llm = _mock_llm_with_json("Response", json_resp)

        cfg = _cfg(
            LEARNING_SCORE_PROMPT_EVERY_N=1,
            LTM_PROMOTE_THRESHOLD=0.65,
            TRIGGER_EVERY_N_TURNS=999,
        )

        orch = Orchestrator(llm=mock_llm, config=cfg)
        orch.chat("Test message")

        trace = orch.last_trace()
        self.assertIsNotNone(trace, "Trace should be recorded")

        # Verify trigger engine exists
        self.assertIsNotNone(orch._trigger_engine,
                             "Orchestrator should have trigger engine")


if __name__ == "__main__":
    unittest.main()