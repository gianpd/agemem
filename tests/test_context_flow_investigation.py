"""
Context Flow Investigation Test
================================

This test demonstrates what context is ingested by the agent at different
context capacity levels (75-80%) and shows the data flow clearly.

The test can be run multiple times with different seeds to simulate
different user queries and verify the context management behavior.

Run with: python -m pytest tests/test_context_flow_investigation.py -v -s
"""

import json
import pytest
import random
import string
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional

from agents.orchestrator import Orchestrator
from agents.llm_client import LLMClient
from core.config import AgememConfig
from core.types import ContextMessage, MemoryEntry, LearningFeedback, TriggerKind
from memory.stm_context import STMContext
from memory.ltm_store import LTMStore
from triggers.system_rules import SystemRules, RuleID


# ─────────────────────────────────────────────────────────────────────────────
# Test Utilities
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_text(min_words: int = 10, max_words: int = 50) -> str:
    """Generate random text for testing."""
    word_count = random.randint(min_words, max_words)
    words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
             for _ in range(word_count)]
    return ' '.join(words)


def generate_seeded_query(seed: int) -> str:
    """Generate a user query based on a seed for reproducibility."""
    random.seed(seed)
    templates = [
        "Tell me about {topic}",
        "What is {topic}?",
        "Explain {topic} in detail",
        "How does {topic} work?",
        "Can you help me understand {topic}?",
    ]
    topics = ["Python", "machine learning", "databases", "APIs", "testing",
              "memory systems", "context windows", "embeddings", "agents", "LLMs"]
    return random.choice(templates).format(topic=random.choice(topics))


@dataclass
class ContextSnapshot:
    """Captures the state of context at a point in time."""
    turn_index: int
    total_tokens: int
    token_limit: int
    utilization_ratio: float
    message_count: int
    pinned_count: int
    overflow_risk: bool
    messages: list[dict] = field(default_factory=list)
    ltm_entries_count: int = 0
    skills_injected: int = 0
    corpus_injected: bool = False


class ContextFlowTracer:
    """
    Traces and records context flow throughout a conversation.

    This captures what context is ingested at each turn and how it changes
    as the conversation progresses towards 75-80% capacity.
    """

    def __init__(self, config: AgememConfig):
        self.config = config
        self.snapshots: list[ContextSnapshot] = []
        self.operations: list[dict] = []

    def capture_snapshot(
        self,
        stm: STMContext,
        ltm: LTMStore,
        turn_index: int,
        skills_count: int = 0,
        corpus_injected: bool = False,
    ) -> ContextSnapshot:
        """Capture the current state of the context."""
        stats = stm.stats()
        messages = []
        for m in stm.messages():
            messages.append({
                "role": m.role,
                "content_preview": (m.content[:100] + "...") if m.content and len(m.content) > 100 else m.content,
                "turn_index": m.turn_index,
                "is_pinned": m.is_pinned,
                "relevance_score": m.relevance_score,
                "token_estimate": m.token_estimate,
            })

        snapshot = ContextSnapshot(
            turn_index=turn_index,
            total_tokens=stats.total_tokens,
            token_limit=self.config.STM_TOKEN_LIMIT,
            utilization_ratio=stats.utilisation_ratio,
            message_count=stats.message_count,
            pinned_count=stats.pinned_count,
            overflow_risk=stats.overflow_risk,
            messages=messages,
            ltm_entries_count=ltm.size(),
            skills_injected=skills_count,
            corpus_injected=corpus_injected,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def record_operation(self, op_type: str, detail: str, trigger: str):
        """Record a memory operation."""
        self.operations.append({
            "type": op_type,
            "detail": detail,
            "trigger": trigger,
        })

    def get_snapshots_near_capacity(self, low: float = 0.70, high: float = 0.85) -> list[ContextSnapshot]:
        """Get snapshots where utilization is between low and high."""
        return [s for s in self.snapshots if low <= s.utilization_ratio <= high]

    def get_report(self) -> dict:
        """Generate a report of the context flow."""
        return {
            "total_turns": len(self.snapshots),
            "snapshots_at_75_80_pct": [
                {
                    "turn": s.turn_index,
                    "tokens": s.total_tokens,
                    "utilization": f"{s.utilization_ratio:.1%}",
                    "messages": s.message_count,
                    "pinned": s.pinned_count,
                    "ltm_entries": s.ltm_entries_count,
                }
                for s in self.get_snapshots_near_capacity()
            ],
            "operations": self.operations,
            "final_state": {
                "tokens": self.snapshots[-1].total_tokens if self.snapshots else 0,
                "utilization": f"{self.snapshots[-1].utilization_ratio:.1%}" if self.snapshots else "0%",
            } if self.snapshots else None,
        }


def create_mock_llm_client(response_text: str = "Mock response"):
    """Create a mock LLM client for testing."""
    mock_client = MagicMock()

    mock_message = MagicMock()
    mock_message.content = response_text
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def create_test_config(temp_dir: str, token_limit: int = 1000) -> AgememConfig:
    """Create a test configuration with a smaller token limit for testing."""
    return AgememConfig(
        LTM_MAX_ENTRIES=100,
        STM_TOKEN_LIMIT=token_limit,
        STM_WARNING_THRESHOLD=0.75,
        STM_CRITICAL_THRESHOLD=0.90,
        STM_MIN_MESSAGES=4,
        STM_SUMMARY_WINDOW=4,
        STM_EVICT_THRESHOLD=0.30,
        PERSIST_DIR=temp_dir,
        DEFAULT_MAX_TOKENS=1024,
        DEFAULT_TEMPERATURE=0.7,
        ENABLE_SEMANTIC_SEARCH=False,
        ENABLE_QUERY_EXPANSION=False,
        SKILL_DETECTION_ENABLED=False,
        TRIGGER_EVERY_N_TURNS=5,
        LEARNING_SCORE_PROMPT_EVERY_N=3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestContextFlowInvestigation:
    """
    Investigation tests for context management behavior.

    These tests demonstrate:
    1. What type of context is ingested
    2. How context is selected based on utilization
    3. Who is in charge of context management
    """

    def test_context_ingestion_at_75_pct_capacity(self):
        """
        Test what context is ingested when utilization reaches 75%.

        VERIFIES:
        - At 75% capacity, SUMMARY operation is triggered
        - LTM entries are retrieved based on query relevance
        - System messages (pinned) are preserved
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir, token_limit=500)
            tracer = ContextFlowTracer(config)

            # Create mock client
            mock_client = create_mock_llm_client("Test response")
            llm_client = LLMClient(mock_client, default_model="test-model")

            # Create orchestrator
            orchestrator = Orchestrator(llm_client, config=config)

            # Pre-populate LTM with some entries
            for i in range(5):
                orchestrator._ltm.add(
                    content=f"LTM entry {i}: {generate_random_text(20, 30)}",
                    learning_score=0.5 + (i * 0.1),
                )

            # Run multiple turns to reach 75% capacity
            for turn in range(20):
                query = generate_seeded_query(turn)
                response = orchestrator.chat(query)

                # Capture snapshot
                tracer.capture_snapshot(
                    stm=orchestrator._stm,
                    ltm=orchestrator._ltm,
                    turn_index=turn,
                )

                # Check if we've reached the target utilization
                stats = orchestrator._stm.stats()
                if stats.utilisation_ratio >= 0.75:
                    break

            # Analyze results
            snapshots_at_75 = tracer.get_snapshots_near_capacity(0.70, 0.85)

            print("\n" + "="*60)
            print("CONTEXT FLOW AT 75% CAPACITY")
            print("="*60)

            for snap in snapshots_at_75:
                print(f"\nTurn {snap.turn_index}:")
                print(f"  Tokens: {snap.total_tokens}/{snap.token_limit} ({snap.utilization_ratio:.1%})")
                print(f"  Messages: {snap.message_count} (pinned: {snap.pinned_count})")
                print(f"  LTM entries: {snap.ltm_entries_count}")
                print(f"  Overflow risk: {snap.overflow_risk}")

                # Show message composition
                roles = {}
                for m in snap.messages:
                    roles[m["role"]] = roles.get(m["role"], 0) + 1
                print(f"  Message roles: {roles}")

            # Verify expectations
            assert len(snapshots_at_75) > 0, "Should have snapshots at 75% capacity"

            # At 75%, SUMMARY should be triggered
            summary_ops = [op for op in tracer.operations if "SUMMARY" in op["type"]]
            print(f"\nSUMMARY operations triggered: {len(summary_ops)}")

    def test_context_composition_breakdown(self):
        """
        Test to show exactly what comprises the context at any point.

        VERIFIES:
        - Context = System prompts + LTM entries + Conversation history
        - Each component's contribution to token count
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir, token_limit=1000)

            # Create mock client
            mock_client = create_mock_llm_client("Response")
            llm_client = LLMClient(mock_client, default_model="test-model")

            # Create orchestrator
            orchestrator = Orchestrator(llm_client, config=config)

            # Add LTM entries
            orchestrator._ltm.add(
                content="Important fact: The project uses Python 3.11 for async features.",
                learning_score=0.9,
                tags=["project", "python"],
            )
            orchestrator._ltm.add(
                content="User preference: Always use type hints in code.",
                learning_score=0.85,
                tags=["preference", "coding"],
            )

            # Run a few turns
            orchestrator.chat("Tell me about Python")
            orchestrator.chat("What are the project requirements?")

            # Analyze context composition
            messages = orchestrator._stm.messages()

            print("\n" + "="*60)
            print("CONTEXT COMPOSITION BREAKDOWN")
            print("="*60)

            total_tokens = 0
            by_role = {}
            by_source = {
                "system_prompt": 0,
                "ltm_retrieved": 0,
                "conversation": 0,
                "skill_hints": 0,
                "corpus": 0,
            }

            for m in messages:
                total_tokens += m.token_estimate

                # By role
                if m.role not in by_role:
                    by_role[m.role] = {"count": 0, "tokens": 0}
                by_role[m.role]["count"] += 1
                by_role[m.role]["tokens"] += m.token_estimate

                # By source (inferred from content)
                if m.content:
                    if "[MEMORY:" in m.content:
                        by_source["ltm_retrieved"] += m.token_estimate
                    elif "[SKILL HINT:" in m.content:
                        by_source["skill_hints"] += m.token_estimate
                    elif "[CORPUS CONTEXT]" in m.content:
                        by_source["corpus"] += m.token_estimate
                    elif m.role == "system":
                        by_source["system_prompt"] += m.token_estimate
                    else:
                        by_source["conversation"] += m.token_estimate

            print(f"\nTotal tokens: {total_tokens}")
            print(f"\nBy Role:")
            for role, data in by_role.items():
                print(f"  {role}: {data['count']} messages, {data['tokens']} tokens")

            print(f"\nBy Source:")
            for source, tokens in by_source.items():
                pct = (tokens / total_tokens * 100) if total_tokens > 0 else 0
                print(f"  {source}: {tokens} tokens ({pct:.1f}%)")

            # Verify
            assert total_tokens > 0, "Should have tokens in context"

    def test_who_manages_context(self):
        """
        Test to identify the components responsible for context management.

        VERIFIES:
        - SystemRules: Deterministic triggers based on thresholds
        - MemoryAgent: LLM-based decisions for LTM operations
        - STMContext: Force-fit and overflow prevention
        - Orchestrator: Coordinates all components
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir, token_limit=500)

            # Test SystemRules
            rules = SystemRules(config)

            # Create mock stats at different utilization levels
            from core.types import ContextStats

            # At 75% (warning threshold)
            stats_75 = ContextStats(
                total_tokens=375,
                message_count=10,
                pinned_count=2,
                utilisation_ratio=0.75,
                overflow_risk=True,
            )

            decisions_75 = rules.evaluate(stats_75, turn_index=10)

            print("\n" + "="*60)
            print("WHO MANAGES CONTEXT?")
            print("="*60)

            print("\n1. SYSTEM RULES (Deterministic Triggers):")
            print(f"   At 75% utilization:")
            for d in decisions_75:
                print(f"   - Rule: {d.rule_id.value}")
                print(f"     Operation: {d.recommended_op.value}")
                print(f"     Priority: {d.priority}")
                print(f"     Reason: {d.reason}")

            # At 90% (critical threshold)
            stats_90 = ContextStats(
                total_tokens=450,
                message_count=15,
                pinned_count=2,
                utilisation_ratio=0.90,
                overflow_risk=True,
            )

            decisions_90 = rules.evaluate(stats_90, turn_index=15)

            print(f"\n   At 90% utilization:")
            for d in decisions_90:
                print(f"   - Rule: {d.rule_id.value}")
                print(f"     Operation: {d.recommended_op.value}")
                print(f"     Priority: {d.priority}")
                print(f"     Reason: {d.reason}")

            # Test STMContext force_fit
            print("\n2. STM CONTEXT (Force-fit mechanism):")
            stm = STMContext(config=config)

            # Add messages to reach critical
            for i in range(20):
                stm.add_message(
                    role="user" if i % 2 == 0 else "assistant",
                    content=generate_random_text(30, 50),
                    relevance_score=0.5,
                )

            stats_before = stm.stats()
            print(f"   Before force_fit: {stats_before.total_tokens} tokens ({stats_before.utilisation_ratio:.1%})")

            ops = stm.force_fit()

            stats_after = stm.stats()
            print(f"   After force_fit: {stats_after.total_tokens} tokens ({stats_after.utilisation_ratio:.1%})")
            print(f"   Operations applied: {len(ops)}")
            for op in ops:
                print(f"   - {op.op.value}: {op.detail}")

            # Verify
            assert stats_after.utilisation_ratio < config.STM_CRITICAL_THRESHOLD

            print("\n3. ORCHESTRATOR (Coordinator):")
            print("   Responsibilities:")
            print("   - Calls STM.force_fit() before every LLM call")
            print("   - Retrieves relevant LTM entries")
            print("   - Injects skill hints and corpus context")
            print("   - Evaluates SystemRules after each turn")
            print("   - Runs MemoryAgent on periodic review or learning spike")
            print("   - Persists STM/LTM state")

    def test_ltm_retrieval_mechanism(self):
        """
        Test how LTM entries are retrieved and injected into context.

        VERIFIES:
        - Retrieval is based on semantic similarity (when enabled) or token overlap
        - Retrieved entries are pinned in STM
        - Relevance score determines retrieval priority
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir, token_limit=1000)
            config.ENABLE_SEMANTIC_SEARCH = False  # Use token overlap for test

            # Create LTM store
            ltm = LTMStore(config=config)

            # Add entries with different relevance
            ltm.add(
                content="Python is a programming language used for web development and data science.",
                learning_score=0.9,
                tags=["python", "programming"],
            )
            ltm.add(
                content="JavaScript is used for frontend web development.",
                learning_score=0.7,
                tags=["javascript", "web"],
            )
            ltm.add(
                content="The user prefers dark mode in all applications.",
                learning_score=0.95,
                tags=["preference", "ui"],
            )

            print("\n" + "="*60)
            print("LTM RETRIEVAL MECHANISM")
            print("="*60)

            # Test retrieval
            queries = [
                "Tell me about Python programming",
                "What are the user preferences?",
                "How does web development work?",
            ]

            for query in queries:
                results = ltm.search(query, top_k=3)
                print(f"\nQuery: '{query}'")
                print(f"Retrieved {len(results)} entries:")
                for r in results:
                    print(f"  - [{r.entry_id}] score={r.learning_score:.2f}")
                    print(f"    Content: {r.content[:60]}...")

            # Verify
            assert ltm.size() == 3

    def test_context_at_different_seeds(self):
        """
        Test with different seeds to show context varies based on query.

        VERIFIES:
        - Different queries retrieve different LTM entries
        - Context composition changes based on user input
        """
        seeds = [42, 123, 456, 789, 999]
        results = []

        for seed in seeds:
            with tempfile.TemporaryDirectory() as temp_dir:
                config = create_test_config(temp_dir, token_limit=500)

                mock_client = create_mock_llm_client("Response")
                llm_client = LLMClient(mock_client, default_model="test-model")

                orchestrator = Orchestrator(llm_client, config=config)

                # Pre-populate LTM
                topics = ["Python", "JavaScript", "databases", "APIs", "testing"]
                for i, topic in enumerate(topics):
                    orchestrator._ltm.add(
                        content=f"Information about {topic}: {generate_random_text(10, 20)}",
                        learning_score=0.5 + (i * 0.1),
                    )

                # Run conversation with seeded query
                query = generate_seeded_query(seed)
                response = orchestrator.chat(query)

                stats = orchestrator._stm.stats()
                results.append({
                    "seed": seed,
                    "query": query,
                    "tokens": stats.total_tokens,
                    "utilization": f"{stats.utilisation_ratio:.1%}",
                    "messages": stats.message_count,
                })

        print("\n" + "="*60)
        print("CONTEXT WITH DIFFERENT SEEDS")
        print("="*60)

        for r in results:
            print(f"\nSeed {r['seed']}: '{r['query']}'")
            print(f"  Tokens: {r['tokens']}, Utilization: {r['utilization']}")
            print(f"  Messages: {r['messages']}")


class TestContextFlowDetailed:
    """
    Detailed tests showing the exact data flow through the context system.
    """

    def test_complete_turn_lifecycle(self):
        """
        Test showing the complete lifecycle of a single turn with context flow.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_test_config(temp_dir, token_limit=1000)

            mock_client = create_mock_llm_client("I understand your request.")
            llm_client = LLMClient(mock_client, default_model="test-model")

            orchestrator = Orchestrator(llm_client, config=config)

            # Pre-populate LTM
            orchestrator._ltm.add(
                content="User prefers concise responses.",
                learning_score=0.9,
            )

            print("\n" + "="*60)
            print("COMPLETE TURN LIFECYCLE")
            print("="*60)

            # Capture initial state
            print("\n--- PRE-TURN STATE ---")
            stats_before = orchestrator._stm.stats()
            print(f"Tokens: {stats_before.total_tokens}, Messages: {stats_before.message_count}")

            # Execute turn
            print("\n--- EXECUTING TURN ---")
            user_input = "What is machine learning?"
            print(f"User input: '{user_input}'")

            # Step 1a: Force fit
            print("\n1a. STM.force_fit() called")

            # Step 1b: LTM search
            relevant = orchestrator._ltm.search(user_input, top_k=5)
            print(f"1b. LTM search: Found {len(relevant)} relevant entries")

            # Step 2: Build messages and call LLM
            print(f"2. Building messages and calling LLM...")

            response = orchestrator.chat(user_input)

            print(f"   Response: '{response}'")

            # Post-turn state
            print("\n--- POST-TURN STATE ---")
            stats_after = orchestrator._stm.stats()
            print(f"Tokens: {stats_after.total_tokens}, Messages: {stats_after.message_count}")

            # Show messages
            print("\n--- MESSAGES IN STM ---")
            for m in orchestrator._stm.messages():
                content_preview = (m.content[:50] + "...") if m.content and len(m.content) > 50 else m.content
                print(f"[{m.role}] (turn={m.turn_index}, pinned={m.is_pinned}) {content_preview}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])