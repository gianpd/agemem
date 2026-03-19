"""
evaluation/pipeline/simulation.py
─────────────────────────────────
Simulation utilities for Phase 2 testing without live LLM.

Provides mock LLM responses and deterministic behavior for testing
memory operations and learning scores when an actual LLM is not available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Any
from core.types import LearningFeedback


# ──────────────────────────────────────────────────────────────────────────────
# Memory Operation Simulation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulatedOperation:
    """A simulated memory operation for testing."""
    op: str  # ADD, UPDATE, DELETE
    turn: int
    content_hint: str = ""
    expected: bool = True


class MemoryOperationSimulator:
    """
    Simulates memory operations based on conversation content analysis.

    Uses deterministic rules to predict what operations should occur,
    enabling testing without a live LLM.
    """

    # Patterns that should trigger ADD operations
    ADD_PATTERNS = [
        r"my name is (\w+)",
        r"I (?:prefer|like|want|need) (.+)",
        r"remember (?:that )?(.+)",
        r"my (?:favorite|preferred) (.+) is",
        r"I work (?:at|on|with) (.+)",
        r"my (?:email|phone|address) is",
    ]

    # Patterns that should trigger UPDATE operations (knowledge updates)
    UPDATE_PATTERNS = [
        r"actually,? my (.+) is now",
        r"I changed my mind",
        r"no[wt],? (?:it's|my|I) (.+)",
        r"forget (?:that )?(.+)",
        r"the new (.+) is",
    ]

    # Patterns for content importance scoring
    IMPORTANCE_PATTERNS = [
        (r"important", 0.9),
        (r"critical", 1.0),
        (r"remember", 0.8),
        (r"don't forget", 0.95),
        (r"always", 0.85),
        (r"never", 0.85),
    ]

    def __init__(self, promotion_threshold: float = 0.8):
        self._promotion_threshold = promotion_threshold
        self._operations: list[SimulatedOperation] = []

    def analyze_turn(
        self,
        turn: int,
        user_content: str,
        assistant_content: str = "",
    ) -> list[SimulatedOperation]:
        """
        Analyze a conversation turn and predict memory operations.

        Args:
            turn: Turn index
            user_content: User message content
            assistant_content: Assistant response (for context)

        Returns:
            List of predicted operations for this turn
        """
        operations = []

        # Check for ADD patterns
        for pattern in self.ADD_PATTERNS:
            matches = re.findall(pattern, user_content, re.IGNORECASE)
            for match in matches:
                operations.append(SimulatedOperation(
                    op="ADD",
                    turn=turn,
                    content_hint=str(match),
                    expected=True,
                ))

        # Check for UPDATE patterns
        for pattern in self.UPDATE_PATTERNS:
            matches = re.findall(pattern, user_content, re.IGNORECASE)
            for match in matches:
                operations.append(SimulatedOperation(
                    op="UPDATE",
                    turn=turn,
                    content_hint=str(match),
                    expected=True,
                ))

        self._operations.extend(operations)
        return operations

    def get_expected_operations(self) -> list[dict]:
        """Get all expected operations as dict list for Phase 2 testing."""
        return [
            {
                'op': op.op,
                'turn': op.turn,
                'content_hint': op.content_hint,
                'expected': op.expected,
            }
            for op in self._operations
        ]

    def clear(self) -> None:
        """Clear recorded operations."""
        self._operations = []


# ──────────────────────────────────────────────────────────────────────────────
# Learning Score Simulation
# ──────────────────────────────────────────────────────────────────────────────

class LearningScoreSimulator:
    """
    Simulates learning scores based on content analysis.

    Uses deterministic rules to predict learning scores,
    enabling testing without a live LLM.
    """

    def __init__(self, promotion_threshold: float = 0.8):
        self._promotion_threshold = promotion_threshold
        self._scores: list[dict] = []

    def score_turn(
        self,
        turn: int,
        user_content: str,
        assistant_content: str = "",
    ) -> LearningFeedback:
        """
        Calculate a simulated learning score for a turn.

        Uses content analysis rules that mirror the actual
        LearningScorer's deterministic scoring matrix.

        Args:
            turn: Turn index
            user_content: User message content
            assistant_content: Assistant response

        Returns:
            LearningFeedback with simulated score
        """
        score = 0.0
        rationale = ""
        affected_content = ""

        combined = f"{user_content} {assistant_content}".lower()

        # Score 1.0: Explicit declarations
        explicit_patterns = [
            r"my name is",
            r"I (?:prefer|like|want|need)",
            r"remember",
            r"my (?:favorite|preferred)",
            r"I work (?:at|on|with)",
            r"my (?:email|phone|address) is",
        ]

        for pattern in explicit_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                score = 1.0
                rationale = "Explicit user declaration detected."
                # Extract affected content
                match = re.search(pattern + r" (.+?)(?:\.|,|$)", combined, re.IGNORECASE)
                if match:
                    affected_content = match.group(0)[:80]
                break

        # Score 0.7: Temporary operational state
        if score == 0.0:
            operational_patterns = [
                r"for this session",
                r"right now",
                r"currently",
                r"for now",
            ]
            for pattern in operational_patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    score = 0.7
                    rationale = "Temporary operational state detected."
                    break

        # Score 0.4: Inferred goals
        if score == 0.0:
            inferred_patterns = [
                r"I think",
                r"maybe",
                r"possibly",
                r"I'm (?:thinking|considering)",
            ]
            for pattern in inferred_patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    score = 0.4
                    rationale = "Inferred goals without explicit declarations."
                    break

        # Score 0.0: Procedural/generic
        if score == 0.0:
            procedural_indicators = [
                "done",
                "understood",
                "ok",
                "thanks",
                "please",
            ]
            if any(ind in combined for ind in procedural_indicators):
                score = 0.0
                rationale = "Procedural or generic dialogue."

        feedback = LearningFeedback(
            score=score,
            rationale=rationale,
            affected_content=affected_content,
            turn_index=turn,
        )

        self._scores.append({
            'turn': turn,
            'score': score,
            'rationale': rationale,
            'affected': affected_content,
            'promoted': score >= self._promotion_threshold,
        })

        return feedback

    def get_score_observations(self) -> list[dict]:
        """Get all score observations for Phase 2 testing."""
        return self._scores.copy()

    def clear(self) -> None:
        """Clear recorded scores."""
        self._scores = []


# ──────────────────────────────────────────────────────────────────────────────
# Conversation Simulator for LongMemEval Data
# ──────────────────────────────────────────────────────────────────────────────

class LongMemEvalConversationSimulator:
    """
    Simulates conversation turns from LongMemEval dataset structure.

    LongMemEval contains sessions with dialogue that should trigger
    memory operations. This simulator extracts and formats that data.
    """

    def __init__(self):
        self._turns: list[dict] = []

    def extract_turns_from_entries(
        self,
        entries: list[Any],  # BenchmarkEntry objects
    ) -> list[dict]:
        """
        Extract conversation turns from benchmark entries.

        Args:
            entries: List of BenchmarkEntry objects from dataset

        Returns:
            List of turn dictionaries with user/assistant content
        """
        self._turns = []

        for entry in entries:
            turn = {
                'turn': getattr(entry, 'source_turn', 0),
                'content': getattr(entry, 'content', ''),
                'learning_score': getattr(entry, 'learning_score', 0.5),
                'tags': getattr(entry, 'tags', []),
            }
            self._turns.append(turn)

        # Sort by turn number
        self._turns.sort(key=lambda x: x['turn'])

        return self._turns

    def get_conversation_turns(self) -> list[dict]:
        """Get formatted conversation turns for memory operation testing."""
        return [
            {
                'user': t['content'],
                'assistant': '',
                'learning_score': t['learning_score'],
                'turn': t['turn'],
            }
            for t in self._turns
        ]

    def get_expected_operations(
        self,
        queries: list[Any],  # BenchmarkQuery objects
    ) -> list[dict]:
        """
        Infer expected memory operations from query types.

        Knowledge-update queries suggest UPDATE operations.
        Preference questions suggest ADD operations.

        Args:
            queries: List of BenchmarkQuery objects

        Returns:
            List of expected operations
        """
        operations = []

        for query in queries:
            query_type = getattr(query, 'query_type', '')
            source_turn = getattr(query, 'source_turn', 0)

            if query_type == 'knowledge-update':
                operations.append({
                    'op': 'UPDATE',
                    'turn': source_turn,
                    'expected': True,
                })
            elif query_type == 'preference':
                operations.append({
                    'op': 'ADD',
                    'turn': source_turn,
                    'expected': True,
                })

        return operations


# ──────────────────────────────────────────────────────────────────────────────
# Test Data Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_phase2_test_data(
    entries: list[Any],
    queries: list[Any],
) -> dict:
    """
    Generate comprehensive test data for Phase 2 evaluation.

    Args:
        entries: BenchmarkEntry objects from dataset
        queries: BenchmarkQuery objects from dataset

    Returns:
        Dictionary with conversation_turns, expected_operations, and score_observations
    """
    # Simulate conversations
    conv_sim = LongMemEvalConversationSimulator()
    conversation_turns = conv_sim.extract_turns_from_entries(entries)
    expected_operations = conv_sim.get_expected_operations(queries)

    # Simulate learning scores
    score_sim = LearningScoreSimulator()
    score_observations = []

    for turn in conversation_turns:
        if turn['content']:
            score_sim.score_turn(
                turn=turn['turn'],
                user_content=turn['content'],
            )

    score_observations = score_sim.get_score_observations()

    return {
        'conversation_turns': conversation_turns,
        'expected_operations': expected_operations,
        'score_observations': score_observations,
    }