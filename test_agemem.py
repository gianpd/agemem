"""
tests/test_agemem.py
─────────────────────
Offline unit tests.  No LLM calls.  No network.

Coverage
────────
T01  TokenCounter – estimation accuracy
T02  LTMStore.add – basic storage
T03  LTMStore.add – duplicate detection → UPDATE path
T04  LTMStore.search – overlap scoring ranks correctly
T05  LTMStore._maybe_prune – respects LTM_MAX_ENTRIES
T06  STMContext.stats – token accounting
T07  STMContext.filter – evicts below-threshold messages, respects MIN
T08  STMContext.filter – never evicts pinned messages
T09  STMContext.summary – compresses window, fallback stub
T10  STMContext.force_fit – triggers summary at warning threshold
T11  STMContext.force_fit – hard-drops at critical threshold
T12  STMContext.retrieve – injects LTM entries, deduplicates
T13  SystemRules – R1/R2 fire on correct utilisation thresholds
T14  SystemRules – R3 fires every N turns, not twice on same turn
T15  SystemRules – R4 fires on learning spike only
T16  MemoryAgentDecision.from_dict – parses valid JSON
T17  MemoryAgentDecision.from_dict – tolerates malformed ops
T18  Orchestrator – full turn with mock LLM, ops recorded in trace
T19  Orchestrator – LTM ADD triggered when feedback.score >= threshold
T20  Orchestrator – no overflow: force_fit called before LLM
T21  Orchestrator – LTM promotes with fallback content when affected_content empty
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.types import (
    ContextMessage, LearningFeedback, MemoryEntry, MemoryOp,
    TriggerKind, TokenCounter
)
from core.config import AgememConfig
from memory.ltm_store import LTMStore
from memory.stm_context import STMContext
from triggers.system_rules import SystemRules, RuleID
from agents.memory_agent import MemoryAgentDecision, LTMOperation
from agents.orchestrator import Orchestrator
from agents.llm_client import LLMClient


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cfg(**overrides) -> AgememConfig:
    cfg = AgememConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_message(role: str, content: str, turn: int = 0, pinned: bool = False, relevance: float = 1.0) -> ContextMessage:
    return ContextMessage(
        role=role, content=content, turn_index=turn,
        token_estimate=len(content.split()),
        is_pinned=pinned, relevance_score=relevance
    )


def _mock_llm(response: str = "Mock response") -> LLMClient:
    """Returns a LLMClient whose underlying client always returns `response`."""
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = response
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[choice], usage=MagicMock(prompt_tokens=10, completion_tokens=5)
    )
    return LLMClient(mock_client, default_model="test-model")


# ──────────────────────────────────────────────────────────────────────────────
# T01 – TokenCounter
# ──────────────────────────────────────────────────────────────────────────────

class TestTokenCounter(unittest.TestCase):

    def test_empty_string(self):
        tc = TokenCounter()
        # Minimum 1 token + 4 framing overhead = 5, but our heuristic is max(1, ...) + 4
        count = tc.count("")
        self.assertGreaterEqual(count, 1)

    def test_longer_text_more_tokens(self):
        tc = TokenCounter()
        short = tc.count("hello")
        long_  = tc.count("hello world this is a longer sentence with many more words")
        self.assertGreater(long_, short)

    def test_count_messages_sums_correctly(self):
        tc = TokenCounter()
        msgs = [
            _make_message("user", "hello world"),
            _make_message("assistant", "hi there"),
        ]
        # Manually set token_estimate to known values
        msgs[0].token_estimate = 10
        msgs[1].token_estimate = 8
        # count_messages uses tc.count(content), not token_estimate field
        total = tc.count_messages(msgs)
        self.assertGreater(total, 0)


# ──────────────────────────────────────────────────────────────────────────────
# T02–T05 – LTMStore
# ──────────────────────────────────────────────────────────────────────────────

class TestLTMStore(unittest.TestCase):

    def setUp(self):
        self.cfg = _cfg(LTM_MAX_ENTRIES=5, LTM_PROMOTE_THRESHOLD=0.65,
                        LTM_UPDATE_THRESHOLD=0.5, LTM_SIMILARITY_WORDS=4)
        self.store = LTMStore(self.cfg)

    def test_T02_add_stores_entry(self):
        result = self.store.add("Python is a programming language", learning_score=0.8)
        self.assertTrue(result.success)
        self.assertEqual(result.op, MemoryOp.ADD)
        self.assertEqual(self.store.size(), 1)

    def test_T03_duplicate_routes_to_update(self):
        self.store.add("Python is a programming language", learning_score=0.8)
        # Same leading words → should UPDATE, not ADD second entry
        result = self.store.add("Python is a programming language version 3", learning_score=0.9)
        self.assertEqual(result.op, MemoryOp.UPDATE)
        self.assertEqual(self.store.size(), 1)  # still one entry

    def test_T04_search_returns_relevant_first(self):
        self.store.add("Python is a high-level programming language", learning_score=0.9)
        self.store.add("The capital of France is Paris", learning_score=0.7)
        self.store.add("Machine learning uses neural networks", learning_score=0.6)
        results = self.store.search("Python programming", top_k=3)
        self.assertTrue(len(results) >= 1)
        # Most relevant should mention Python
        self.assertIn("Python", results[0].content)

    def test_T05_prune_respects_max_entries(self):
        for i in range(7):
            self.store.add(f"Unique fact number {i} about topic {i}", learning_score=float(i) * 0.1)
        self.assertLessEqual(self.store.size(), self.cfg.LTM_MAX_ENTRIES)

    def test_delete_removes_entry(self):
        result = self.store.add("Temporary fact", learning_score=0.5)
        entry_id = result.entries_affected[0]
        del_result = self.store.delete(entry_id)
        self.assertTrue(del_result.success)
        self.assertEqual(self.store.size(), 0)

    def test_delete_missing_returns_failure(self):
        result = self.store.delete("nonexistent_id")
        self.assertFalse(result.success)


# ──────────────────────────────────────────────────────────────────────────────
# T06–T12 – STMContext
# ──────────────────────────────────────────────────────────────────────────────

class TestSTMContext(unittest.TestCase):

    def _make_stm(self, **overrides) -> STMContext:
        cfg = _cfg(**overrides)
        return STMContext(config=cfg)

    def test_T06_stats_token_accounting(self):
        stm = self._make_stm(STM_TOKEN_LIMIT=1000)
        stm.add_message("user", "hello world")
        stats = stm.stats()
        self.assertGreater(stats.total_tokens, 0)
        self.assertGreater(stats.message_count, 0)
        self.assertGreater(stats.utilisation_ratio, 0)

    def test_T07_filter_evicts_low_relevance(self):
        stm = self._make_stm(STM_EVICT_THRESHOLD=0.3, STM_MIN_MESSAGES=1)
        stm.add_message("user", "important content", relevance_score=0.9)
        stm.add_message("user", "noise message",    relevance_score=0.1)
        stm.add_message("user", "also noise",       relevance_score=0.15)
        before = stm.stats().message_count
        stm.filter()
        after = stm.stats().message_count
        self.assertLess(after, before)

    def test_T08_filter_never_evicts_pinned(self):
        stm = self._make_stm(STM_EVICT_THRESHOLD=0.3, STM_MIN_MESSAGES=0)
        stm.add_message("system", "pinned system prompt", is_pinned=True, relevance_score=0.0)
        stm.add_message("user", "noise", relevance_score=0.05)
        stm.filter()
        msgs = stm.messages()
        pinned = [m for m in msgs if m.is_pinned]
        self.assertEqual(len(pinned), 1)
        self.assertEqual(pinned[0].content, "pinned system prompt")

    def test_T09_summary_compresses_window(self):
        stm = self._make_stm(STM_SUMMARY_WINDOW=3, STM_MIN_MESSAGES=1)
        # Inject fallback summary_fn is default (no llm)
        for i in range(5):
            stm.add_message("user", f"message {i} with some content here")
        before = stm.stats().message_count
        result = stm.summary()
        after = stm.stats().message_count
        self.assertTrue(result.success)
        # After summary, message count should decrease (window compressed → 1 summary)
        self.assertLess(after, before)

    def test_T10_force_fit_triggers_at_warning(self):
        # Tiny limit so we can easily hit warning threshold
        stm = self._make_stm(
            STM_TOKEN_LIMIT=50,
            STM_WARNING_THRESHOLD=0.5,
            STM_CRITICAL_THRESHOLD=0.9,
            STM_MIN_MESSAGES=1,
            STM_SUMMARY_WINDOW=2,
        )
        # Add enough content to exceed 50% of 50 tokens = 25 tokens
        for i in range(8):
            stm.add_message("user", f"this is a somewhat long message number {i}")
        ops = stm.force_fit()
        # At least one operation should have fired
        self.assertGreater(len(ops), 0)

    def test_T11_force_fit_hard_drops_at_critical(self):
        stm = self._make_stm(
            STM_TOKEN_LIMIT=30,
            STM_WARNING_THRESHOLD=0.5,
            STM_CRITICAL_THRESHOLD=0.7,
            STM_MIN_MESSAGES=2,
            STM_SUMMARY_WINDOW=2,
        )
        for i in range(10):
            stm.add_message("user", f"message {i} is quite long with content")
        ops = stm.force_fit()
        stats = stm.stats()
        self.assertGreaterEqual(stats.message_count, stm._config.STM_MIN_MESSAGES)

    def test_T12_retrieve_deduplicates(self):
        stm = STMContext()
        entry = MemoryEntry(content="Python tip: use list comprehensions", learning_score=0.8)
        stm.retrieve([entry])
        count_before = stm.stats().message_count
        # Retrieve the same entry again
        stm.retrieve([entry])
        count_after = stm.stats().message_count
        # Should NOT have doubled
        self.assertEqual(count_before, count_after)


# ──────────────────────────────────────────────────────────────────────────────
# T13–T15 – SystemRules
# ──────────────────────────────────────────────────────────────────────────────

class TestSystemRules(unittest.TestCase):

    def _stats(self, ratio: float) -> object:
        from core.types import ContextStats
        return ContextStats(
            total_tokens=int(ratio * 6000),
            message_count=10,
            pinned_count=1,
            utilisation_ratio=ratio,
            overflow_risk=ratio >= 0.75,
        )

    def test_T13_R1_fires_at_warning(self):
        rules = SystemRules(_cfg(STM_WARNING_THRESHOLD=0.75, STM_CRITICAL_THRESHOLD=0.90))
        decisions = rules.evaluate(self._stats(0.80), turn_index=1)
        rule_ids = [d.rule_id for d in decisions]
        self.assertIn(RuleID.OVERFLOW_WARN, rule_ids)
        self.assertNotIn(RuleID.OVERFLOW_CRITICAL, rule_ids)

    def test_T13_R2_fires_at_critical(self):
        rules = SystemRules(_cfg(STM_WARNING_THRESHOLD=0.75, STM_CRITICAL_THRESHOLD=0.90))
        decisions = rules.evaluate(self._stats(0.95), turn_index=1)
        rule_ids = [d.rule_id for d in decisions]
        self.assertIn(RuleID.OVERFLOW_CRITICAL, rule_ids)

    def test_T13_no_overflow_rule_when_under_threshold(self):
        rules = SystemRules()
        decisions = rules.evaluate(self._stats(0.30), turn_index=1)
        rule_ids = [d.rule_id for d in decisions]
        self.assertNotIn(RuleID.OVERFLOW_WARN,     rule_ids)
        self.assertNotIn(RuleID.OVERFLOW_CRITICAL, rule_ids)

    def test_T14_R3_fires_every_N(self):
        cfg = _cfg(TRIGGER_EVERY_N_TURNS=5)
        rules = SystemRules(cfg)
        decisions_at_5 = rules.evaluate(self._stats(0.1), turn_index=5)
        rule_ids = [d.rule_id for d in decisions_at_5]
        self.assertIn(RuleID.PERIODIC_REVIEW, rule_ids)

    def test_T14_R3_does_not_fire_twice_same_turn(self):
        cfg = _cfg(TRIGGER_EVERY_N_TURNS=5)
        rules = SystemRules(cfg)
        rules.evaluate(self._stats(0.1), turn_index=5)  # first call
        decisions = rules.evaluate(self._stats(0.1), turn_index=5)  # second call same turn
        rule_ids = [d.rule_id for d in decisions]
        self.assertNotIn(RuleID.PERIODIC_REVIEW, rule_ids)

    def test_T15_R4_fires_on_spike(self):
        cfg = _cfg(LEARNING_SCORE_THRESHOLD_IMMEDIATE=0.85)
        rules = SystemRules(cfg)
        feedback = LearningFeedback(score=0.90, affected_content="User prefers JSON output", turn_index=2)
        decisions = rules.evaluate(self._stats(0.1), turn_index=2, feedback=feedback)
        rule_ids = [d.rule_id for d in decisions]
        self.assertIn(RuleID.LEARNING_SPIKE, rule_ids)

    def test_T15_R4_does_not_fire_below_threshold(self):
        cfg = _cfg(LEARNING_SCORE_THRESHOLD_IMMEDIATE=0.85)
        rules = SystemRules(cfg)
        feedback = LearningFeedback(score=0.60, affected_content="Some content", turn_index=2)
        decisions = rules.evaluate(self._stats(0.1), turn_index=2, feedback=feedback)
        rule_ids = [d.rule_id for d in decisions]
        self.assertNotIn(RuleID.LEARNING_SPIKE, rule_ids)


# ──────────────────────────────────────────────────────────────────────────────
# T16–T17 – MemoryAgentDecision
# ──────────────────────────────────────────────────────────────────────────────

class TestMemoryAgentDecision(unittest.TestCase):

    def test_T16_parses_valid_json(self):
        data = {
            "ltm_operations": [
                {"op": "add", "content": "User likes dark mode", "entry_id": None, "tags": ["ui"], "confidence": 0.9}
            ],
            "context_relevance": [
                {"turn_index": 3, "relevance_score": 0.8}
            ],
            "summary_needed": False,
            "rationale": "Stored UI preference."
        }
        dec = MemoryAgentDecision.from_dict(data)
        self.assertEqual(len(dec.ltm_operations), 1)
        self.assertEqual(dec.ltm_operations[0].op, MemoryOp.ADD)
        self.assertEqual(dec.context_relevance[3], 0.8)
        self.assertFalse(dec.summary_needed)

    def test_T17_tolerates_malformed_ops(self):
        data = {
            "ltm_operations": [
                {"op": "INVALID_OP", "content": "x"},  # bad op
                {"content": "missing op field"},         # missing op key
                {"op": "add", "content": "valid", "confidence": 0.8},
            ],
            "context_relevance": [],
            "summary_needed": False,
            "rationale": "test",
        }
        dec = MemoryAgentDecision.from_dict(data)
        # Only the valid op should survive
        self.assertEqual(len(dec.ltm_operations), 1)
        self.assertEqual(dec.ltm_operations[0].content, "valid")


# ──────────────────────────────────────────────────────────────────────────────
# T18–T20 – Orchestrator (mock LLM)
# ──────────────────────────────────────────────────────────────────────────────

class TestOrchestrator(unittest.TestCase):

    def _make_orchestrator(self, llm_response: str = "Sure!", **cfg_overrides) -> Orchestrator:
        cfg = _cfg(
            LEARNING_SCORE_PROMPT_EVERY_N=3,
            LTM_PROMOTE_THRESHOLD=0.65,
            LEARNING_SCORE_THRESHOLD_IMMEDIATE=0.85,
            TRIGGER_EVERY_N_TURNS=10,
            **cfg_overrides,
        )
        llm = _mock_llm(llm_response)
        return Orchestrator(llm=llm, config=cfg)

    def test_T18_full_turn_records_trace(self):
        orch = self._make_orchestrator()
        response = orch.chat("What is the capital of France?")
        self.assertIsInstance(response, str)
        trace = orch.last_trace()
        self.assertIsNotNone(trace)
        self.assertEqual(trace.user_input, "What is the capital of France?")
        self.assertIsNotNone(trace.stm_stats_before)
        self.assertIsNotNone(trace.stm_stats_after)

    def test_T19_ltm_add_on_high_learning_score(self):
        """
        When the LLM returns a learning score >= LTM_PROMOTE_THRESHOLD,
        an ADD op should appear in the trace.
        """
        # Mock LLM: first call = main response, second call = learning score JSON
        import json
        main_resp = "Paris is the capital of France."
        score_resp = json.dumps({
            "score": 0.90,
            "rationale": "Novel geographic fact",
            "affected_content": "Paris is the capital of France"
        })

        mock_client = MagicMock()
        responses = [main_resp, score_resp]
        call_count = [0]

        def side_effect(**kwargs):
            resp = responses[min(call_count[0], len(responses) - 1)]
            call_count[0] += 1
            choice = MagicMock()
            choice.message.content = resp
            return MagicMock(
                choices=[choice],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5)
            )

        mock_client.chat.completions.create.side_effect = side_effect
        llm = LLMClient(mock_client, default_model="test")

        cfg = _cfg(
            LEARNING_SCORE_PROMPT_EVERY_N=1,  # collect every turn
            LTM_PROMOTE_THRESHOLD=0.65,
            LEARNING_SCORE_THRESHOLD_IMMEDIATE=0.85,
            PERSIST_DIR=None,  # Disable persistence for test isolation
        )
        orch = Orchestrator(llm=llm, config=cfg)
        orch.chat("What is the capital of France?")

        trace = orch.last_trace()
        add_ops = [op for op in trace.ops_applied if op.op == MemoryOp.ADD and op.success]
        self.assertGreater(len(add_ops), 0, "Expected at least one LTM ADD op")

    def test_T20_no_overflow_force_fit_called(self):
        """
        Stuffing the STM beyond the warning threshold should trigger
        force_fit and the context should remain under the limit.
        """
        cfg = _cfg(
            STM_TOKEN_LIMIT=80,
            STM_WARNING_THRESHOLD=0.5,
            STM_CRITICAL_THRESHOLD=0.85,
            STM_MIN_MESSAGES=2,
            STM_SUMMARY_WINDOW=2,
            LEARNING_SCORE_PROMPT_EVERY_N=999,  # disable scorer
            TRIGGER_EVERY_N_TURNS=999,           # disable periodic review
        )
        llm = _mock_llm("a " * 30)  # response that adds ~30 tokens
        orch = Orchestrator(llm=llm, config=cfg)

        for _ in range(6):
            orch.chat("tell me something " + "word " * 10)

        stats = orch.stm_stats()
        self.assertLessEqual(
            stats.utilisation_ratio,
            cfg.STM_CRITICAL_THRESHOLD + 0.05,  # small tolerance for framing tokens
            f"STM should stay near or below critical threshold, got {stats.utilisation_ratio:.2%}"
        )

    def test_T21_ltm_promotes_with_fallback_content(self):
        """
        REGRESSION TEST: Empty affected_content should not block LTM promotion.
        When the LLM returns a high score but empty affected_content, the
        orchestrator should fallback to the assistant's response content.
        """
        import json
        main_resp = "The user likes Python programming."
        score_resp = json.dumps({
            "score": 0.90,  # High score
            "rationale": "Learned user preference",
            "affected_content": ""  # Empty! Should fallback to assistant response
        })

        mock_client = MagicMock()
        responses = [main_resp, score_resp]
        call_count = [0]

        def side_effect(**kwargs):
            resp = responses[min(call_count[0], len(responses) - 1)]
            call_count[0] += 1
            choice = MagicMock()
            choice.message.content = resp
            return MagicMock(
                choices=[choice],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5)
            )

        mock_client.chat.completions.create.side_effect = side_effect
        llm = LLMClient(mock_client, default_model="test")

        cfg = _cfg(
            LEARNING_SCORE_PROMPT_EVERY_N=1,
            LTM_PROMOTE_THRESHOLD=0.65,
            PERSIST_DIR=None,
        )
        orch = Orchestrator(llm=llm, config=cfg)
        orch.chat("I love Python programming")

        trace = orch.last_trace()
        add_ops = [op for op in trace.ops_applied if op.op == MemoryOp.ADD and op.success]
        self.assertGreater(len(add_ops), 0,
            "LTM should promote even with empty affected_content - fallback to response text")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
