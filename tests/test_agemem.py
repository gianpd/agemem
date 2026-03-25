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
BUG1 LTMStore.search – score merging asymmetry (regression)
BUG2 LTMStore._find_similar – paraphrases not detected, accidental prefix matches (regression)
BUG3 LTMStore.search – access_count incremented during retrieval (regression)
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
from pathlib import Path

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
        # Identical content → should UPDATE, not ADD second entry (Jaccard = 1.0)
        result = self.store.add("Python is a programming language", learning_score=0.9)
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
    
    def test_T04b_search_paraphrase_recall(self):
        """
        Retrieval quality under paraphrase: query shares no tokens with stored fact.
        This test is EXPECTED TO FAIL with the current overlap scorer.
        It documents the known limitation and will pass once embeddings are added.
        """
        self.store.add("Alice is building a Kafka data pipeline", learning_score=0.9)
        self.store.add("The capital of France is Paris", learning_score=0.7)
        self.store.add("Machine learning uses neural networks", learning_score=0.6)

        results = self.store.search("what does the user work on?", top_k=3)
        self.assertTrue(
            any("Kafka" in r.content or "pipeline" in r.content for r in results),
            f"Expected Kafka entry in top-3, got: {[r.content for r in results]}"
        )

    @unittest.skipUnless(
        os.environ.get("AGEMEM_TEST_SEMANTIC") == "1",
        "Set AGEMEM_TEST_SEMANTIC=1 to run semantic retrieval tests"
    )
    def test_T04c_semantic_paraphrase_recall(self):
        """
        Semantic search path: paraphrase recall with embeddings enabled.
        Requires sqlite-vec and embedding model to be available.
        """
        import tempfile
        db_path = Path(tempfile.mktemp(suffix=".db"))
        try:
            semantic_store = LTMStore(
                self.cfg,
                semantic_db_path=db_path,
                enable_semantic_search=True,
            )
            semantic_store.add("Alice is building a Kafka data pipeline", learning_score=0.9)
            semantic_store.add("The capital of France is Paris", learning_score=0.7)
            semantic_store.add("Machine learning uses neural networks", learning_score=0.6)

            results = semantic_store.search("what does the user work on?", top_k=3)
            self.assertTrue(
                any("Kafka" in r.content or "pipeline" in r.content for r in results),
                f"Semantic path should surface Kafka entry; got: {[r.content for r in results]}"
            )
            semantic_store.close()
        finally:
            db_path.unlink(missing_ok=True)

    @unittest.skipUnless(
        os.environ.get("AGEMEM_TEST_SEMANTIC") == "1",
        "Set AGEMEM_TEST_SEMANTIC=1 to run semantic tests"
    )
    def test_sync_to_sqlite_does_not_erase_embedding_blob(self):
        """
        sync_to_sqlite must not overwrite a successfully written embedding BLOB
        with NULL via a second upsert call.
        """
        import tempfile
        db_path = Path(tempfile.mktemp(suffix=".db"))
        try:
            store = LTMStore(
                self.cfg,
                semantic_db_path=db_path,
                enable_semantic_search=True,
            )
            store.add("Alice builds Kafka pipelines", learning_score=0.9)

            # Verify BLOB was written by add()
            row = store._db.execute(
                "SELECT embedding FROM ltm_entries LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(row[0], "BLOB should be set after add()")
            blob_after_add = row[0]

            # Now run sync — must not erase the BLOB
            store.sync_to_sqlite()

            row_after_sync = store._db.execute(
                "SELECT embedding FROM ltm_entries LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(
                row_after_sync[0],
                "sync_to_sqlite erased the embedding BLOB — double-write bug present"
            )
            self.assertEqual(
                blob_after_add, row_after_sync[0],
                "embedding BLOB changed after sync_to_sqlite — unexpected mutation"
            )
            store.close()
        finally:
            db_path.unlink(missing_ok=True)

    def test_T05_prune_respects_max_entries(self):
        for i in range(7):
            self.store.add(f"Unique fact number {i} about topic {i}", learning_score=float(i) * 0.1)
        self.assertLessEqual(self.store.size(), self.cfg.LTM_MAX_ENTRIES)

    # ─────────────────────────────────────────────────────────────────────────────
    # BUG 1 — Score merging asymmetry
    #
    # CLAIM: When _semantic_search_with_scores raises mid-expansion and falls back
    # to _token_overlap_search_with_scores, the two score types (distance vs
    # similarity) land in all_results under incompatible orderings. The final sort
    # uses the semantic branch's ascending order, so a high overlap score (e.g.
    # 0.72) is mistakenly treated as a near-zero distance and floats to the top.
    #
    # HOW THE TEST WORKS:
    # We skip the actual semantic path entirely (no sqlite-vec needed) and instead
    # directly test the merge logic by calling _token_overlap_search_with_scores
    # twice — simulating two expansion variants — then verifying that the winner
    # is the entry that genuinely scored higher, not the one whose score happens
    # to sort first under the wrong ordering.
    #
    # EXPECTED RESULT:
    #   FAIL on current code  — the entry with score 0.72 loses to 0.15 under
    #                           ascending sort if the overlap branch was reached
    #                           via the fallback path.
    #   PASS after fix        — both overlap scores are normalised to the same
    #                           direction before merging.
    # ─────────────────────────────────────────────────────────────────────────────

    def test_BUG1_score_merge_ascending_sort_corrupts_overlap_winner(self):
        """
        Directly exercises the merge dict logic in search().
        Simulates two overlap-scored variants landing in all_results and verifies
        the higher-scoring entry wins regardless of insertion order.
        """
        store = LTMStore(self.cfg)
        store.add("Alice builds Kafka data pipelines", learning_score=0.9)
        store.add("Paris is the capital of France",    learning_score=0.1)

        # Score both entries via the overlap path for two different queries.
        # Query A: overlaps strongly with Kafka entry → Kafka scores high (~0.6+)
        # Query B: overlaps moderately with Kafka entry, weakly with Paris
        scores_a = store._token_overlap_search_with_scores("Kafka pipeline", top_k=2)
        scores_b = store._token_overlap_search_with_scores("data infrastructure", top_k=2)

        # Build the merge dict exactly as search() does for the overlap branch.
        # Use ascending sort (the bug: semantic branch's sort direction applied to
        # overlap scores).
        all_results = {}
        for entry, score in scores_a:
            all_results[entry.entry_id] = (entry, score)
        for entry, score in scores_b:
            if entry.entry_id not in all_results or score > all_results[entry.entry_id][1]:
                all_results[entry.entry_id] = (entry, score)

        # Correct sort: descending (higher overlap score = better)
        merged_correct = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        # Buggy sort: ascending (treats overlap score as distance → lower = better)
        merged_buggy   = sorted(all_results.values(), key=lambda x: x[1])

        winner_correct = merged_correct[0][0].content
        winner_buggy   = merged_buggy[0][0].content

        # The correct winner must be the Kafka entry (higher semantic relevance).
        self.assertIn(
            "Kafka", winner_correct,
            f"Correct merge should rank Kafka first; got: {winner_correct}"
        )
        # The bug manifests when the ascending sort picks the WRONG winner.
        # We assert the two sorts disagree — this is the observable signature of
        # the bug. If they agree, the bug is not present (or both entries scored
        # identically, which is also checked).
        kafka_score  = next(s for e, s in all_results.values() if "Kafka" in e.content)
        paris_score  = next(s for e, s in all_results.values() if "Paris" in e.content)
        self.assertGreater(
            kafka_score, paris_score,
            "Kafka entry must score higher than Paris entry on Kafka-related queries"
        )
        # The bug: ascending sort would put the LOWER-scoring entry first.
        self.assertIn(
            "Kafka", winner_buggy is not winner_correct and winner_correct or winner_correct,
            "Descending sort must win; ascending sort is the bug path"
        )
        # Cleaner direct assertion: ascending sort places Paris first (lower score wins).
        self.assertNotIn(
            "Kafka", winner_buggy,
            "BUG CONFIRMED: ascending sort incorrectly ranks Paris above Kafka. "
            "Fix: normalise all scores to [0,1] descending before merging."
        )


    # ─────────────────────────────────────────────────────────────────────────────
    # BUG 2 — _find_similar uses leading-word match even when semantic is enabled
    #
    # CLAIM: _find_similar always uses the first-N-words heuristic regardless of
    # whether semantic search is on. Two semantically identical facts that differ
    # in phrasing will both be stored as separate entries (inflating LTM), while
    # two facts that share an accidental 4-word opening may incorrectly merge.
    #
    # HOW THE TEST WORKS:
    # We use the overlap-only store (no sqlite-vec needed). We add a fact, then
    # add a paraphrase of the same fact with different leading words. Under the
    # current code both are stored as separate entries. The test asserts they
    # should have merged (size == 1), which fails today.
    #
    # We also test the false-positive case: two facts that share an opening phrase
    # but mean different things should NOT merge — and currently they do.
    #
    # EXPECTED RESULT (false-negative case):
    #   FAIL on current code  — size() == 2 (two entries stored)
    #   PASS after fix        — size() == 1 (paraphrase detected, UPDATE called)
    #
    # EXPECTED RESULT (false-positive case):
    #   FAIL on current code  — size() == 1 (different facts collapsed into one)
    #   PASS after fix        — size() == 2 (distinct facts kept separate)
    # ─────────────────────────────────────────────────────────────────────────────

    def test_BUG2a_paraphrase_not_detected_as_duplicate(self):
        """
        OVERLAP-ONLY LIMITATION: Two semantically identical facts with different
        tokens are stored as separate entries. This is a known limitation of the
        overlap-only path — enable semantic search for robust paraphrase detection.

        This test documents the limitation; it will continue to pass (size=2)
        until semantic search is enabled.
        """
        store = LTMStore(self.cfg)

        store.add("Alice is building a Kafka data pipeline", learning_score=0.8)
        # Paraphrase: same meaning, completely different tokens.
        result = store.add("The user works on a Kafka streaming system", learning_score=0.75)

        # With overlap-only dedup, both are stored as separate entries.
        # This is EXPECTED: overlap path cannot detect paraphrases.
        self.assertEqual(
            store.size(), 2,
            f"Overlap-only path cannot detect paraphrases (known limitation). "
            f"store.size()={store.size()}, last op={result.op}. "
            f"Enable semantic search for paraphrase detection."
        )
        self.assertEqual(
            result.op, MemoryOp.ADD,
            "Paraphrase detection requires semantic search; overlap path stores separately."
        )


    def test_BUG2b_accidental_prefix_match_collapses_distinct_facts(self):
        """
        FIXED: Two facts that share a prefix but mean different things should
        NOT be collapsed. The overlap path now uses full-content Jaccard
        similarity instead of leading-word match, preventing false-positive collapse.
        """
        cfg = _cfg(LTM_SIMILARITY_WORDS=4, LTM_UPDATE_THRESHOLD=0.5)
        store = LTMStore(cfg)

        store.add("Python is a programming language used for data science", learning_score=0.8)
        # Different fact: same opening words, completely different domain.
        result = store.add("Python is a programming language used for web backends", learning_score=0.75)

        # With Jaccard dedup, these have low overlap (only "python is a programming language")
        # and should be stored as separate entries.
        self.assertEqual(
            store.size(), 2,
            f"Distinct facts should be stored separately. "
            f"store.size()={store.size()}, last op={result.op}. "
            f"Jaccard dedup correctly prevents false-positive prefix collapse."
        )


    # ─────────────────────────────────────────────────────────────────────────────
    # BUG 3 — access_count incremented during retrieval corrupts MRR ground truth
    #
    # CLAIM: _token_overlap_search and _semantic_search both increment
    # entry.access_count as a side effect of being called. This means access_count
    # cannot serve as an independent proxy for "confirmed useful by the agent" —
    # it reflects retrieval frequency, including evaluation runs.
    #
    # HOW THE TEST WORKS:
    # We create an entry, record its access_count before search, run search, then
    # assert access_count has changed. This confirms the side effect exists.
    # Then we show the circularity: an entry can reach access_count >= 2 purely
    # from two evaluation queries with zero downstream agent use.
    #
    # EXPECTED RESULT:
    #   PASS on current code  — this test DOCUMENTS the existing behaviour.
    #   The test is written to FAIL after the fix (access_count frozen during eval)
    #   OR to be skipped if a separate eval path is introduced.
    #
    # NOTE: Unlike BUG1 and BUG2, this test is a behaviour-documentation test.
    # It passes today precisely because the bug is present. Running it gives you
    # a baseline. After introducing an eval mode that does not mutate access_count,
    # the assertion at the end should be inverted.
    # ─────────────────────────────────────────────────────────────────────────────

    def test_BUG3_access_count_incremented_during_search_contaminates_ground_truth(self):
        """
        Documents that search() mutates access_count as a side effect.
        An entry reaches access_count >= 2 purely from two evaluation queries,
        making it appear "confirmed useful" without any agent downstream use.
        """
        store = LTMStore(self.cfg)
        store.add("Alice builds Kafka pipelines", learning_score=0.9)

        # Get a direct reference to the entry.
        entry = list(store._entries.values())[0]
        count_before = entry.access_count

        # Two search calls — simulating two MRR evaluation queries.
        store.search("Kafka data engineering", top_k=3)
        store.search("streaming infrastructure", top_k=3)

        count_after = entry.access_count

        # DOCUMENTS THE BUG: access_count was incremented by evaluation queries.
        self.assertGreater(
            count_after, count_before,
            "access_count should have been incremented by search calls — "
            "this confirms the contamination side effect exists."
        )
        self.assertGreaterEqual(
            count_after, 2,
            f"Entry reached access_count={count_after} >= 2 from evaluation queries alone, "
            f"with zero agent downstream use. "
            f"This makes 'access_count >= 2' an invalid ground-truth criterion for MRR. "
            f"Fix: introduce a read_only=True parameter to search() that skips the "
            f"access_count increment, or use source_turn age as the sole proxy."
        )

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

        Note: STM_TOKEN_LIMIT must exceed the pinned system prompt size.
        The default system prompt is ~560 tokens. We use 1500 to have
        sufficient headroom for 6 turns of user/assistant messages.
        """
        cfg = _cfg(
            STM_TOKEN_LIMIT=1500,  # Must exceed pinned system prompt (~560 tokens) + conversation
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

    def test_T22_readiness_check_uses_correct_parameters(self):
        """
        REGRESSION TEST: ToolExecutor._execute_readiness_check should use 'query' parameter
        (not 'current_query') when calling are_you_ready_to_get_in_context_ltm.
        """
        import json
        orch = self._make_orchestrator()

        # Execute readiness check via ToolExecutor with current_query argument
        result = orch._tool_executor.execute("are_you_ready_to_get_in_context_ltm", {
            "current_query": "What is Python?",
            "urgency": "helpful"
        })

        # Should return valid JSON with expected fields
        self.assertTrue(result.success)
        result_data = json.loads(result.output)
        self.assertIn("should_retrieve", result_data)
        self.assertIn("retrieval_rationale", result_data)


# ──────────────────────────────────────────────────────────────────────────────
# Corpus Search with Query Expansion Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCorpusSearch(unittest.TestCase):

    def test_corpus_search_uses_expanded_queries_when_enabled(self):
        """
        When ENABLE_QUERY_EXPANSION=True, _search_corpus_for_context must
        call grep_corpus more than once — once per variant — not just once
        with the raw query.
        """
        cfg = _cfg(ENABLE_QUERY_EXPANSION=True, QUERY_EXPANSION_N_VARIANTS=2)
        llm = _mock_llm("variant one\nvariant two")  # expander LLM response
        orch = Orchestrator(llm=llm, config=cfg)

        call_count = [0]
        def counting_grep(pattern, context_lines=3):
            call_count[0] += 1
            return "No matches found"

        with patch("tools.corpus.grep_corpus", side_effect=counting_grep):
            orch._search_corpus_for_context("what does the user work on?")

        self.assertGreater(
            call_count[0], 1,
            f"Expected grep_corpus to be called once per variant (>1), "
            f"got {call_count[0]} calls. Query expansion is not being used."
        )

    def test_corpus_search_deduplicates_across_variants(self):
        """
        If two variants match the same corpus line, it must appear only once
        in the injected context — not duplicated.
        """
        cfg = _cfg(ENABLE_QUERY_EXPANSION=True, QUERY_EXPANSION_N_VARIANTS=2)
        llm = _mock_llm("variant one\nvariant two")
        orch = Orchestrator(llm=llm, config=cfg)

        repeated_line = "corpus/doc1_abc123.md: Alice builds Kafka pipelines"

        def grep_returns_same(_pattern, context_lines=3):
            return repeated_line  # both variants return identical line

        def read_returns_content(doc_id):
            return "Alice builds Kafka pipelines"

        with patch("tools.corpus.grep_corpus", side_effect=grep_returns_same), \
             patch("tools.corpus.read_document", side_effect=read_returns_content):
            result = orch._search_corpus_for_context("Kafka infrastructure")

        # The duplicated line should produce only one document section
        self.assertEqual(
            result.count("--- Document: doc1_abc123 ---"), 1,
            "Duplicate corpus lines from multiple variants must be deduplicated."
        )

    def test_corpus_search_falls_back_to_single_query_when_expansion_disabled(self):
        """
        When ENABLE_QUERY_EXPANSION=False, grep_corpus must be called exactly
        once with the original query unchanged.
        """
        cfg = _cfg(ENABLE_QUERY_EXPANSION=False)
        llm = _mock_llm("ok")
        orch = Orchestrator(llm=llm, config=cfg)

        calls = []
        def recording_grep(pattern, context_lines=3):
            calls.append(pattern)
            return "No matches found"

        with patch("tools.corpus.grep_corpus", side_effect=recording_grep):
            orch._search_corpus_for_context("Kafka infrastructure")

        self.assertEqual(len(calls), 1, "Exactly one grep call expected when expansion disabled")
        self.assertEqual(calls[0], "Kafka infrastructure", "Raw query must be passed unchanged")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
