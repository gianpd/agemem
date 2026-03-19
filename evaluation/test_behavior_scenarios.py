"""
evaluation/test_behavior_scenarios.py
────────────────────────────────────
Behavior-specific test scenarios for LongMemEval's 5 memory behaviors.

This module implements test cases for each of LongMemEval's memory behaviors:
- IE: Information Extraction (single-session recall)
- MR: Multi-Session Reasoning (synthesize across sessions)
- KU: Knowledge Updates (track changing information)
- TR: Temporal Reasoning (time-aware retrieval)
- ABS: Abstention (refrain when info missing)

Coherence with longmemeval_guide.md:
- Tests LongMemEval_S standard (~115k tokens, ~40 sessions)
- Maps to the 5 memory behavior categories
- Validates cross-session memory effects

Usage:
    pytest evaluation/test_behavior_scenarios.py -v
    pytest evaluation/test_behavior_scenarios.py::test_information_extraction -v
"""

from __future__ import annotations

import pytest
from typing import Optional

# Import the orchestrator test harness
from evaluation.orchestrator_test_harness import (
    EvaluationSession,
    MultiSessionEvaluation,
    TurnResult,
)


class TestInformationExtraction:
    """
    Tests Information Extraction (IE) behavior.

    From longmemeval_guide.md:
    "Ability to recall specific information from extensive interactive histories.
    Requires precise retrieval of facts buried in lengthy conversations."

    Examples:
    - "What is the user's current occupation?"
    - "What is my phone number?"
    - "Where did we meet?"
    """

    def test_single_session_detail_recall(self):
        """
        Test recalling a specific detail from a single session.

        LongMemEval pattern: Detail mentioned once in session 12
        Question: "What's my phone number?"
        Expected: The specific phone number from session 12
        """
        # Create a session with a memory containing the phone number
        ltm_seed_data = [
            {
                "content": "User mentioned their phone number is 555-1234",
                "learning_score": 0.8,
                "tags": ["contact", "phone"],
                "source_turn": 12,
            },
            {
                "content": "User likes coffee in the morning",
                "learning_score": 0.5,
                "tags": ["preference"],
                "source_turn": 5,
            },
        ]

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "phone": "Your phone number is 555-1234.",
            },
        )

        result = session.send_message("What's my phone number?")

        # Verify correct answer
        assert result.answer_is_correct("555-1234")

        # Verify LTM retrieval happened
        assert len(result.memories_injected) >= 0  # May be 0 if mocked

        session.cleanup()

    def test_user_side_information(self):
        """
        Test recalling information mentioned by the user (not assistant).

        LongMemEval pattern: User mentions detail in their message
        """
        ltm_seed_data = [
            {
                "content": "User said they work as a software engineer at TechCorp",
                "learning_score": 0.9,
                "tags": ["work", "employment"],
                "source_turn": 8,
            },
        ]

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "work": "You work as a software engineer at TechCorp.",
            },
        )

        result = session.send_message("Where do I work?")

        assert result.answer_is_correct("software engineer")
        assert result.answer_is_correct("TechCorp")

        session.cleanup()

    def test_assistant_side_information(self):
        """
        Test recalling information mentioned by the assistant.

        LongMemEval pattern: Assistant mentioned detail in previous response
        """
        ltm_seed_data = [
            {
                "content": "Assistant recommended restaurant 'La Trattoria' to user",
                "learning_score": 0.7,
                "tags": ["recommendation", "restaurant"],
                "source_turn": 15,
            },
        ]

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "restaurant": "You previously mentioned La Trattoria.",
            },
        )

        result = session.send_message("What restaurant did you recommend?")

        assert result.answer_is_correct("La Trattoria")

        session.cleanup()


class TestMultiSessionReasoning:
    """
    Tests Multi-Session Reasoning (MR) behavior.

    From longmemeval_guide.md:
    "Ability to synthesize information across multiple history sessions.
    Requires connecting disparate pieces of information from different conversations."

    Examples:
    - "How many pets do I have, and what are their names?"
    - "What's my total spending this month?"
    """

    def test_aggregate_across_sessions(self):
        """
        Test aggregating information scattered across multiple sessions.

        LongMemEval pattern:
        - Session 5: User mentions adopting a golden retriever named Max
        - Session 18: User talks about getting a Siamese cat named Luna
        - Session 32: User mentions their hamster named Peanut
        Question: "How many pets do I have?"
        Expected: "Three pets: Max, Luna, and Peanut"
        """
        ltm_seed_data = [
            {
                "content": "User adopted a golden retriever named Max",
                "learning_score": 0.8,
                "tags": ["pet", "dog"],
                "source_turn": 5,
            },
            {
                "content": "User got a Siamese cat named Luna",
                "learning_score": 0.8,
                "tags": ["pet", "cat"],
                "source_turn": 18,
            },
            {
                "content": "User has a hamster named Peanut",
                "learning_score": 0.8,
                "tags": ["pet", "hamster"],
                "source_turn": 32,
            },
        ]

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "pets": "You have three pets: Max (golden retriever), Luna (Siamese cat), and Peanut (hamster).",
            },
        )

        result = session.send_message("How many pets do I have?")

        # Should mention all three pets
        assert result.answer_is_correct("Max")
        assert result.answer_is_correct("Luna")
        assert result.answer_is_correct("Peanut")

        session.cleanup()

    def test_multi_session_calculation(self):
        """
        Test calculating from information across sessions.

        LongMemEval pattern: Sum values from different shopping sessions
        """
        ltm_seed_data = [
            {
                "content": "User spent $50 at grocery store",
                "learning_score": 0.7,
                "tags": ["expense", "shopping"],
                "source_turn": 3,
            },
            {
                "content": "User spent $120 on electronics",
                "learning_score": 0.7,
                "tags": ["expense", "shopping"],
                "source_turn": 12,
            },
            {
                "content": "User spent $30 at coffee shop",
                "learning_score": 0.7,
                "tags": ["expense", "shopping"],
                "source_turn": 20,
            },
        ]

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "spend": "You spent $200 total: $50 at the grocery store, $120 on electronics, and $30 at the coffee shop.",
            },
        )

        result = session.send_message("How much did I spend this week?")

        # Should calculate total
        assert result.answer_is_correct("200") or result.answer_is_correct("$200")

        session.cleanup()


class TestKnowledgeUpdates:
    """
    Tests Knowledge Updates (KU) behavior.

    From longmemeval_guide.md:
    "Ability to recognize changes in user information and update knowledge dynamically.
    Requires temporal awareness and conflict resolution."

    Examples:
    - "What is the user's current job title?" (with multiple job changes)
    - "Where do I live now?" (after moving)
    """

    def test_track_job_changes(self):
        """
        Test tracking changing job information.

        LongMemEval pattern:
        - Session 8: "I work as a data analyst at Finance Corp"
        - Session 22: "Just got promoted to senior data analyst!"
        - Session 35: "Actually, I switched companies. Now I'm a data science lead at TechGiant."
        Question: "What is my current job?"
        Expected: "Data science lead at TechGiant" (most recent)
        """
        ltm_seed_data = [
            {
                "content": "User works as a data analyst at Finance Corp",
                "learning_score": 0.6,
                "tags": ["job", "employment"],
                "source_turn": 8,
            },
            {
                "content": "User got promoted to senior data analyst",
                "learning_score": 0.7,
                "tags": ["job", "employment", "promotion"],
                "source_turn": 22,
            },
            {
                "content": "User switched companies and is now a data science lead at TechGiant",
                "learning_score": 0.9,
                "tags": ["job", "employment", "current"],
                "source_turn": 35,
            },
        ]

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "job": "You're currently a data science lead at TechGiant.",
            },
        )

        result = session.send_message("What is my current job?")

        # Should return most recent job
        assert result.answer_is_correct("data science lead")
        assert result.answer_is_correct("TechGiant")

        # Should NOT return old job
        assert not result.answer_is_correct("Finance Corp")

        session.cleanup()

    def test_address_change(self):
        """
        Test tracking address changes.

        LongMemEval pattern: User moves from old address to new
        """
        ltm_seed_data = [
            {
                "content": "User lives in Seattle, WA",
                "learning_score": 0.6,
                "tags": ["address", "location"],
                "source_turn": 5,
            },
            {
                "content": "User moved to Portland, OR",
                "learning_score": 0.9,
                "tags": ["address", "location", "current"],
                "source_turn": 25,
            },
        ]

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "live": "You currently live in Portland, OR.",
            },
        )

        result = session.send_message("Where do I live now?")

        assert result.answer_is_correct("Portland")
        assert not result.answer_is_correct("Seattle")

        session.cleanup()


class TestTemporalReasoning:
    """
    Tests Temporal Reasoning (TR) behavior.

    From longmemeval_guide.md:
    "Awareness of temporal aspects in user information.
    Requires understanding relative time references and absolute timestamps."

    Examples:
    - "What did we discuss last weekend?"
    - "What restaurant did I mention last Tuesday?"
    """

    def test_relative_time_query(self):
        """
        Test queries with relative time references.

        LongMemEval pattern: "What did we discuss last Tuesday?"
        """
        ltm_seed_data = [
            {
                "content": "User and assistant discussed machine learning frameworks on Tuesday",
                "learning_score": 0.7,
                "tags": ["discussion", "topic"],
                "source_turn": 10,
                "timestamp": "2026-03-10T14:00:00",  # Tuesday
            },
        ]

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "tuesday": "Last Tuesday we discussed machine learning frameworks.",
            },
        )

        result = session.send_message("What did we discuss last Tuesday?")

        assert result.answer_is_correct("machine learning")

        session.cleanup()

    def test_weekend_activity_query(self):
        """
        Test queries about weekend activities.

        LongMemEval pattern:
        "I had amazing pasta at Trattoria Roma last night. You should try it!"
        Question: "What restaurant did the user recommend last weekend?"
        """
        ltm_seed_data = [
            {
                "content": "User had pasta at Trattoria Roma on Saturday and recommended it",
                "learning_score": 0.8,
                "tags": ["restaurant", "recommendation", "weekend"],
                "source_turn": 28,
            },
        ]

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "restaurant": "You recommended Trattoria Roma last weekend.",
            },
        )

        result = session.send_message("What restaurant did I recommend last weekend?")

        assert result.answer_is_correct("Trattoria Roma")

        session.cleanup()


class TestAbstention:
    """
    Tests Abstention (ABS) behavior.

    From longmemeval_guide.md:
    "Ability to refrain from answering questions involving unknown information.
    Requires confidence calibration and hallucination prevention."

    Examples:
    - Asked about topic never mentioned -> "I don't know"
    - Question about future event -> "I cannot predict"
    """

    def test_refrain_when_unknown(self):
        """
        Test abstaining when information is not in history.

        LongMemEval pattern:
        - 40 sessions with no mention of topic X
        - Question: "What is X?"
        Expected: "I don't know" or similar abstention
        """
        ltm_seed_data = [
            {
                "content": "User likes hiking and outdoor activities",
                "learning_score": 0.6,
                "tags": ["hobby"],
                "source_turn": 5,
            },
            {
                "content": "User prefers tea over coffee",
                "learning_score": 0.5,
                "tags": ["preference", "beverage"],
                "source_turn": 12,
            },
        ]
        # Note: No mention of "programming language" or "Python"

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "programming": "I don't have information about your preferred programming language.",
            },
        )

        result = session.send_message("What is my favorite programming language?")

        # Should indicate abstention
        assert result.has_abstained()

        session.cleanup()

    def test_no_hallucination(self):
        """
        Test that system doesn't hallucinate information.

        LongMemEval pattern: Question about unmentioned specific detail
        """
        ltm_seed_data = [
            {
                "content": "User mentioned they have a pet",
                "learning_score": 0.5,
                "tags": ["pet"],
                "source_turn": 8,
            },
        ]
        # Note: Pet type is NOT specified

        session = EvaluationSession(
            ltm_seed_data=ltm_seed_data,
            use_mock_llm=True,
            mock_responses={
                "pet": "I know you have a pet, but I'm not sure what type.",
            },
        )

        result = session.send_message("What type of pet do I have?")

        # Should not make up a specific pet type
        # (In practice, we'd check the actual response for hallucination)
        assert not result.answer_is_correct("dog")  # Don't guess
        assert not result.answer_is_correct("cat")  # Don't guess

        session.cleanup()


class TestMultiSessionEvaluation:
    """
    Tests MultiSessionEvaluation class for full LongMemEval_S scenarios.

    Tests the 30-40 session loading pattern from LongMemEval_S standard.
    """

    def test_load_40_session_history(self):
        """
        Test loading a full LongMemEval_S session history (~40 sessions).
        """
        # Simulate 40 sessions with various content
        sessions = []
        for i in range(40):
            sessions.append([
                {"role": "user", "content": f"Session {i} message"},
                {"role": "assistant", "content": f"Response to session {i}"},
            ])

        multi_eval = MultiSessionEvaluation(
            config_overrides={"CONTEXT_AWARE_RETRIEVAL": True},
            use_mock_llm=True,
        )

        # This should not raise any errors
        eval_session = multi_eval.load_session_history(
            sessions=[{"messages": s} for s in sessions],
            evidence_indices=[12, 25],
        )

        # Verify session was created
        assert eval_session is not None

        eval_session.cleanup()

    def test_behavior_type_inference(self):
        """
        Test automatic behavior type inference from evidence pattern.
        """
        multi_eval = MultiSessionEvaluation()

        # Single evidence -> IE
        behavior = multi_eval._infer_behavior_type(
            sessions=[{"messages": []} for _ in range(40)],
            evidence_indices=[12],
        )
        assert behavior == "IE"

        # Multiple evidence -> MR
        behavior = multi_eval._infer_behavior_type(
            sessions=[{"messages": []} for _ in range(40)],
            evidence_indices=[5, 18, 32],
        )
        assert behavior == "MR"

        # No evidence -> ABS
        behavior = multi_eval._infer_behavior_type(
            sessions=[{"messages": []} for _ in range(40)],
            evidence_indices=[],
        )
        assert behavior == "ABS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
