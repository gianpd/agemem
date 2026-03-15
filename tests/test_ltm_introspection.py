"""
tests/test_ltm_introspection.py
────────────────────────────────
Comprehensive tests for the LTM Self-Management Toolkit.

Test Coverage
─────────────
* Unit tests for each individual tool (10 tools)
* Integration test for the full 8-step execution flow
* Edge case tests (retry caps, non-retrieval logging, etc.)
* Type validation tests

Design decisions
────────────────
* Tests use mocking to avoid requiring actual LLM/embeddings.
* State is cleared between test groups to ensure isolation.
* Tests validate both return type structure and behavior.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Import types
from core.types import MemoryEntry, ContextMessage
from memory.ltm_introspection_types import (
    DriftType, ConfidenceLevel, ExpectedValue, UrgencyLevel,
    RetrievalMode, FailureMode,
    DriftReport, ConfidenceReport, ReadinessAssessment,
    Paraphrase, LTMInjection,
    ValidatedBatch, RefinedQuery, CompressedContext,
    RetrievalDecision, ConversationProfile, StrategyRecommendation,
    Turn, RetrievedMemory, MemoryValidationResult,
    RetrievalAttempt, ConfidenceDimensionScore,
    # Tier 5 types
    PersistenceNeed, PersistenceResult, PersistenceValidation,
    PersistenceFailure, MemoryCommandPattern, PersistenceUrgency,
    PersistenceStatus, FailureCategory, ValidationCheck,
)

# Import tools
from memory.ltm_introspection import (
    assess_conversation_drift,
    self_assess_confidence,
    are_you_ready_to_get_in_context_ltm,
    paraphrase_for_coverage,
    trigger_contextual_ltm_retrieval,
    validate_ltm_relevance,
    refine_retrieval_target,
    compress_conversation_for_ltm,
    log_retrieval_decision,
    suggest_retrieval_strategy,
    set_anchor_from_context,
    get_decision_history,
    clear_state,
    # Tier 5 tools
    assess_persistence_need,
    force_memory_persistence,
    validate_memory_commit,
    log_persistence_failure,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_state():
    """Reset introspection state before each test."""
    clear_state()
    yield
    clear_state()


@pytest.fixture
def sample_memory_entries() -> List[MemoryEntry]:
    """Create sample memory entries for testing."""
    return [
        MemoryEntry(
            content="Python is a programming language",
            learning_score=0.8,
            tags=["python", "programming"],
            source_turn=1,
        ),
        MemoryEntry(
            content="FastAPI is a modern web framework",
            learning_score=0.9,
            tags=["fastapi", "web"],
            source_turn=2,
        ),
        MemoryEntry(
            content="Docker containers are lightweight",
            learning_score=0.7,
            tags=["docker", "containers"],
            source_turn=3,
        ),
    ]


@pytest.fixture
def sample_context_messages() -> List[ContextMessage]:
    """Create sample context messages for testing."""
    return [
        ContextMessage(role="user", content="How do I use Python?", turn_index=1),
        ContextMessage(role="assistant", content="Python is easy to learn.", turn_index=2),
        ContextMessage(role="user", content="Tell me about FastAPI", turn_index=3),
    ]


@pytest.fixture
def mock_ltm_store(sample_memory_entries):
    """Create a mock LTM store."""
    store = Mock()
    store.search.return_value = sample_memory_entries
    store.search_by_vector.return_value = sample_memory_entries
    return store


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 Tests — State Assessment (Introspection)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssessConversationDrift:
    """Tests for assess_conversation_drift tool."""

    def test_returns_drift_report(self, sample_context_messages):
        """Drift assessment returns structured DriftReport."""
        report = assess_conversation_drift(
            window_turns=3,
            against_anchor=False,  # No anchor set yet
            current_messages=sample_context_messages,
            current_turn=3,
        )

        assert isinstance(report, DriftReport)
        assert hasattr(report, 'topic_drift_score')
        assert hasattr(report, 'drift_type')
        assert hasattr(report, 'confidence')

    def test_no_anchor_returns_low_drift(self, sample_context_messages):
        """Without anchor, drift is classified as NONE with LOW confidence."""
        report = assess_conversation_drift(
            window_turns=3,
            against_anchor=True,
            current_messages=sample_context_messages,
        )

        assert report.drift_type == DriftType.NONE
        assert report.confidence == ConfidenceLevel.LOW

    def test_drift_type_classification(self, sample_context_messages):
        """Drift type is correctly classified based on scores."""
        # Set up an anchor first
        set_anchor_from_context(sample_context_messages[:1], turn_index=1)

        # Now assess with different context
        different_messages = [
            ContextMessage(role="user", content="How do I cook pasta?", turn_index=10),
            ContextMessage(role="user", content="Italian cuisine recipes", turn_index=11),
        ]

        report = assess_conversation_drift(
            window_turns=2,
            against_anchor=True,
            current_messages=different_messages,
        )

        # Should detect drift (coding -> cooking)
        assert report.drift_type in (DriftType.HARD_PIVOT, DriftType.GRADUAL_SLOPE, DriftType.SOFT_PIVOT)
        assert report.topic_drift_score > 0

    def test_drift_report_to_dict(self, sample_context_messages):
        """DriftReport can be serialized to dict."""
        report = assess_conversation_drift(
            window_turns=3,
            against_anchor=False,
            current_messages=sample_context_messages,
        )

        d = report.to_dict()
        assert isinstance(d, dict)
        assert 'topic_drift_score' in d
        assert 'drift_type' in d
        assert 'confidence' in d


class TestSelfAssessConfidence:
    """Tests for self_assess_confidence tool."""

    def test_returns_confidence_report(self, mock_ltm_store):
        """Confidence assessment returns structured ConfidenceReport."""
        report = self_assess_confidence(
            check_dimensions=["factual", "contextual"],
            current_context="How do I use Python with FastAPI?",
            ltm_store=mock_ltm_store,
        )

        assert isinstance(report, ConfidenceReport)
        assert hasattr(report, 'dimensions')
        assert hasattr(report, 'overall_score')
        assert hasattr(report, 'overall_confidence')
        assert hasattr(report, 'knowledge_gaps')

    def test_per_dimension_scores(self, mock_ltm_store):
        """Report includes per-dimension scores."""
        report = self_assess_confidence(
            check_dimensions=["factual", "contextual", "temporal"],
            current_context="Test context",
            ltm_store=mock_ltm_store,
        )

        assert len(report.dimensions) == 3
        for dim in report.dimensions:
            assert 0 <= dim.score <= 1
            assert dim.dimension.value in ["factual", "contextual", "temporal"]

    def test_confidence_levels(self, mock_ltm_store):
        """Overall confidence is classified correctly."""
        # Test with empty context (should be LOW)
        report = self_assess_confidence(
            check_dimensions=["factual"],
            current_context="",
            ltm_store=mock_ltm_store,
        )

        assert report.overall_confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]

    def test_confidence_report_to_dict(self, mock_ltm_store):
        """ConfidenceReport can be serialized to dict."""
        report = self_assess_confidence(
            check_dimensions=["factual"],
            current_context="Test",
            ltm_store=mock_ltm_store,
        )

        d = report.to_dict()
        assert isinstance(d, dict)
        assert 'dimensions' in d
        assert 'overall_score' in d
        assert 'overall_confidence' in d


class TestAreYouReadyToGetInContextLTM:
    """Tests for are_you_ready_to_get_in_context_ltm tool."""

    def test_returns_readiness_assessment(self, sample_context_messages, mock_ltm_store):
        """Readiness check returns structured ReadinessAssessment."""
        assessment = are_you_ready_to_get_in_context_ltm(
            query="How do I use Python?",
            urgency="helpful",
            current_messages=sample_context_messages,
            current_turn=3,
            ltm_store=mock_ltm_store,
        )

        assert isinstance(assessment, ReadinessAssessment)
        assert hasattr(assessment, 'should_retrieve')
        assert hasattr(assessment, 'retrieval_rationale')
        assert hasattr(assessment, 'suggested_retrieval_strategy')
        assert hasattr(assessment, 'expected_value')

    def test_blocking_urgency_always_retrieves(self, sample_context_messages, mock_ltm_store):
        """Blocking urgency always triggers retrieval."""
        assessment = are_you_ready_to_get_in_context_ltm(
            query="Test query",
            urgency="blocking",
            current_messages=sample_context_messages,
            ltm_store=mock_ltm_store,
        )

        assert assessment.should_retrieve is True
        assert assessment.expected_value == ExpectedValue.HIGH

    def test_exploratory_urgency_may_skip(self, mock_ltm_store):
        """Exploratory urgency may skip retrieval if no drift."""
        # Messages with no drift (simple, short)
        simple_messages = [
            ContextMessage(role="user", content="Hi", turn_index=1),
        ]

        assessment = are_you_ready_to_get_in_context_ltm(
            query="Hello",
            urgency="exploratory",
            current_messages=simple_messages,
            ltm_store=mock_ltm_store,
        )

        # Should provide a rationale either way
        assert len(assessment.retrieval_rationale) > 0

    def test_includes_supporting_reports(self, sample_context_messages, mock_ltm_store):
        """Assessment includes drift and confidence reports."""
        assessment = are_you_ready_to_get_in_context_ltm(
            query="Test",
            urgency="helpful",
            current_messages=sample_context_messages,
            ltm_store=mock_ltm_store,
        )

        assert assessment.drift_report is not None
        assert assessment.confidence_report is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 Tests — Retrieval Orchestration (Action)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParaphraseForCoverage:
    """Tests for paraphrase_for_coverage tool."""

    def test_returns_list_of_paraphrases(self):
        """Paraphrase generation returns list of Paraphrase objects."""
        paraphrases = paraphrase_for_coverage(
            core_concept="deploy python app",
            coverage_goals=["technical", "tutorial"],
            semantic_distance_target=0.3,
        )

        assert isinstance(paraphrases, list)
        assert len(paraphrases) > 0
        assert all(isinstance(p, Paraphrase) for p in paraphrases)

    def test_includes_original(self):
        """First paraphrase is always the original."""
        paraphrases = paraphrase_for_coverage(
            core_concept="test concept",
            coverage_goals=["technical"],
        )

        assert paraphrases[0].text == "test concept"
        assert paraphrases[0].source == "original"

    def test_paraphrase_metadata(self):
        """Paraphrases include coverage goal metadata."""
        paraphrases = paraphrase_for_coverage(
            core_concept="deploy app",
            coverage_goals=["technical", "tutorial", "troubleshooting"],
        )

        for p in paraphrases[1:]:  # Skip original
            assert p.coverage_goal in ["technical", "tutorial", "troubleshooting", "unknown"]
            assert p.semantic_distance >= 0


class TestTriggerContextualLTMRetrieval:
    """Tests for trigger_contextual_ltm_retrieval tool."""

    def test_returns_ltm_injection(self, mock_ltm_store):
        """Retrieval returns structured LTMInjection."""
        result = trigger_contextual_ltm_retrieval(
            retrieval_mode="single_query",
            query_or_concept="python fastapi",
            ltm_store=mock_ltm_store,
            top_k=5,
        )

        assert isinstance(result, LTMInjection)
        assert hasattr(result, 'memories')
        assert hasattr(result, 'retrieval_mode')
        assert hasattr(result, 'execution_time_ms')

    def test_single_query_mode(self, mock_ltm_store):
        """Single query mode executes one query."""
        result = trigger_contextual_ltm_retrieval(
            retrieval_mode="single_query",
            query_or_concept="test",
            ltm_store=mock_ltm_store,
        )

        assert result.retrieval_mode == RetrievalMode.SINGLE_QUERY
        assert result.queries_executed == 1
        mock_ltm_store.search.assert_called()

    def test_retrieval_mode_values(self, mock_ltm_store):
        """All retrieval modes are supported."""
        for mode in ["single_query", "multi_paraphrase", "anchored"]:
            result = trigger_contextual_ltm_retrieval(
                retrieval_mode=mode,
                query_or_concept="test",
                ltm_store=mock_ltm_store,
            )
            assert result.retrieval_mode.value == mode

    def test_ltm_injection_to_dict(self, mock_ltm_store):
        """LTMInjection can be serialized to dict."""
        result = trigger_contextual_ltm_retrieval(
            retrieval_mode="single_query",
            query_or_concept="test",
            ltm_store=mock_ltm_store,
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'memories' in d
        assert 'retrieval_mode' in d


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3 Tests — Validation & Refinement (Quality Control)
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateLTMRelevance:
    """Tests for validate_ltm_relevance tool."""

    def test_returns_validated_batch(self, sample_memory_entries, sample_context_messages):
        """Validation returns structured ValidatedBatch."""
        candidates = [
            RetrievedMemory(entry=e, retrieval_score=0.8, source_query="test", rank=i)
            for i, e in enumerate(sample_memory_entries)
        ]

        result = validate_ltm_relevance(
            candidate_memories=candidates,
            against_turns=[-3, -2, -1],
            match_dimensions=["entity", "intent"],
            recent_messages=sample_context_messages,
        )

        assert isinstance(result, ValidatedBatch)
        assert hasattr(result, 'validated_memories')
        assert hasattr(result, 'coverage_score')
        assert hasattr(result, 'coverage_sufficient')
        assert hasattr(result, 'recommendation')

    def test_per_memory_validation(self, sample_memory_entries, sample_context_messages):
        """Each memory gets individual validation result."""
        candidates = [
            RetrievedMemory(entry=e, retrieval_score=0.8, source_query="test", rank=i)
            for i, e in enumerate(sample_memory_entries)
        ]

        result = validate_ltm_relevance(
            candidate_memories=candidates,
            recent_messages=sample_context_messages,
        )

        assert len(result.validated_memories) == len(candidates)
        for v in result.validated_memories:
            assert hasattr(v, 'relevance_score')
            assert hasattr(v, 'is_relevant')
            assert 0 <= v.relevance_score <= 1

    def test_coverage_recommendation(self, sample_memory_entries, sample_context_messages):
        """Provides coverage recommendation."""
        candidates = [
            RetrievedMemory(entry=e, retrieval_score=0.8, source_query="test", rank=i)
            for i, e in enumerate(sample_memory_entries)
        ]

        result = validate_ltm_relevance(
            candidate_memories=candidates,
            recent_messages=sample_context_messages,
        )

        assert result.recommendation in ["proceed", "refine", "abort"]


class TestRefineRetrievalTarget:
    """Tests for refine_retrieval_target tool."""

    def test_returns_refined_query(self):
        """Refinement returns structured RefinedQuery."""
        failed = RetrievalAttempt(
            query="python deployment",
            retrieval_mode=RetrievalMode.SINGLE_QUERY,
            results_count=0,
        )

        result = refine_retrieval_target(
            failed_retrieval=failed,
            failure_mode="too_broad",
        )

        assert isinstance(result, RefinedQuery)
        assert hasattr(result, 'original_query')
        assert hasattr(result, 'refined_query')
        assert hasattr(result, 'refinement_strategy')
        assert hasattr(result, 'can_retry')

    def test_failure_mode_specific_refinement(self):
        """Different failure modes produce different refinements."""
        # Test with TOO_BROAD which definitely changes the query
        failed = RetrievalAttempt(
            query="python",
            retrieval_mode=RetrievalMode.SINGLE_QUERY,
            results_count=0,
        )

        result = refine_retrieval_target(
            failed_retrieval=failed,
            failure_mode="too_broad",
        )

        assert result.failure_mode == FailureMode.TOO_BROAD
        assert result.refined_query != result.original_query
        assert "specific details about" in result.refined_query

        # Test STALE which also changes query
        failed2 = RetrievalAttempt(
            query="python deployment",
            retrieval_mode=RetrievalMode.SINGLE_QUERY,
            results_count=0,
        )
        result2 = refine_retrieval_target(
            failed_retrieval=failed2,
            failure_mode="stale",
        )
        assert result2.failure_mode == FailureMode.STALE
        assert result2.refined_query.startswith("latest ")

    def test_retry_count_capped(self):
        """Retry count is capped at maximum."""
        # Note: retry count is tracked per-query, so we need to
        # simulate retries on the same base query
        base_query = "retry test query"

        for i in range(5):
            failed = RetrievalAttempt(
                query=base_query,  # Same query each time
                retrieval_mode=RetrievalMode.SINGLE_QUERY,
                results_count=0,
            )
            result = refine_retrieval_target(
                failed_retrieval=failed,
                failure_mode="too_broad",
                max_retries=2,
            )

            if i >= 2:
                assert result.can_retry is False
                assert result.retry_count > 2
                break

    def test_refined_query_to_dict(self):
        """RefinedQuery can be serialized to dict."""
        failed = RetrievalAttempt(
            query="test",
            retrieval_mode=RetrievalMode.SINGLE_QUERY,
            results_count=0,
        )

        result = refine_retrieval_target(
            failed_retrieval=failed,
            failure_mode="too_broad",
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'original_query' in d
        assert 'refined_query' in d
        assert 'failure_mode' in d


class TestCompressConversationForLTM:
    """Tests for compress_conversation_for_ltm tool."""

    def test_returns_compressed_context(self):
        """Compression returns structured CompressedContext."""
        turns = [
            Turn(role="user", content="How do I use Python?", turn_index=1),
            Turn(role="assistant", content="Python is a programming language.", turn_index=2),
            Turn(role="user", content="Can you show me an example?", turn_index=3),
        ]

        result = compress_conversation_for_ltm(
            turns=turns,
            target_length=1,
        )

        assert isinstance(result, CompressedContext)
        assert hasattr(result, 'compressed_text')
        assert hasattr(result, 'original_turns')
        assert hasattr(result, 'compression_ratio')

    def test_preserves_key_elements(self):
        """Compression preserves key elements based on priority."""
        turns = [
            Turn(role="user", content="I decided to use FastAPI.", turn_index=1),
            Turn(role="user", content="It must be production ready.", turn_index=2),
            Turn(role="user", content="What about deployment?", turn_index=3),
        ]

        result = compress_conversation_for_ltm(
            turns=turns,
            target_length=1,
            preservation_priority=["decisions", "constraints", "open_questions"],
        )

        assert len(result.preserved_elements) > 0

    def test_compression_ratio(self):
        """Compression ratio is calculated correctly."""
        turns = [
            Turn(role="user", content="This is a longer message with many words to compress.", turn_index=1),
            Turn(role="user", content="Another message with content that should be summarized.", turn_index=2),
        ]

        result = compress_conversation_for_ltm(
            turns=turns,
            target_length=1,
        )

        assert 0 <= result.compression_ratio <= 1
        assert result.original_turns == 2

    def test_empty_turns(self):
        """Handles empty turn list gracefully."""
        result = compress_conversation_for_ltm(turns=[], target_length=1)

        assert result.compressed_text == ""
        assert result.original_turns == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 4 Tests — Meta-Cognitive Tools (Learning)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogRetrievalDecision:
    """Tests for log_retrieval_decision tool."""

    def test_logs_retrieval_decision(self, sample_memory_entries):
        """Decision is logged and can be retrieved."""
        drift = DriftReport(topic_drift_score=0.3, drift_type=DriftType.SOFT_PIVOT)

        log_retrieval_decision(
            trigger="user_query",
            drift_scores=drift,
            retrieved_memories=sample_memory_entries,
            utility_score=0.8,
            was_retrieved=True,
            strategy_used="single_query",
        )

        history = get_decision_history()
        assert len(history) == 1
        assert history[0].trigger == "user_query"
        assert history[0].was_retrieved is True

    def test_logs_non_retrieval(self):
        """Non-retrieval decisions are also logged."""
        drift = DriftReport(topic_drift_score=0.1, drift_type=DriftType.NONE)

        log_retrieval_decision(
            trigger="low_drift",
            drift_scores=drift,
            retrieved_memories=[],
            utility_score=0.0,
            was_retrieved=False,
            strategy_used="",
        )

        history = get_decision_history()
        assert len(history) == 1
        assert history[0].was_retrieved is False

    def test_retrieval_decision_to_dict(self, sample_memory_entries):
        """RetrievalDecision can be serialized to dict."""
        drift = DriftReport()

        log_retrieval_decision(
            trigger="test",
            drift_scores=drift,
            retrieved_memories=sample_memory_entries,
            utility_score=0.5,
        )

        history = get_decision_history()
        d = history[0].to_dict()
        assert isinstance(d, dict)
        assert 'trigger' in d
        assert 'utility_score' in d


class TestSuggestRetrievalStrategy:
    """Tests for suggest_retrieval_strategy tool."""

    def test_returns_strategy_recommendation(self):
        """Strategy suggestion returns structured StrategyRecommendation."""
        profile = ConversationProfile(
            domain="coding",
            user_intent="debugging",
            session_length_turns=25,
            complexity_score=0.8,
        )

        result = suggest_retrieval_strategy(
            conversation_profile=profile,
        )

        assert isinstance(result, StrategyRecommendation)
        assert hasattr(result, 'recommended_strategy')
        assert hasattr(result, 'recommended_mode')
        assert hasattr(result, 'rationale')

    def test_complex_session_suggests_multi_paraphrase(self):
        """Complex sessions suggest multi_paraphrase."""
        profile = ConversationProfile(
            domain="technical",
            complexity_score=0.8,
            session_length_turns=25,
        )

        result = suggest_retrieval_strategy(profile)

        assert result.recommended_mode == RetrievalMode.MULTI_PARAPHRASE

    def test_early_session_suggests_single_query(self):
        """Early sessions suggest single_query."""
        profile = ConversationProfile(
            domain="general",
            complexity_score=0.3,
            session_length_turns=3,
            has_established_context=False,
        )

        result = suggest_retrieval_strategy(profile)

        assert result.recommended_mode == RetrievalMode.SINGLE_QUERY


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Test — Full 8-Step Flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestFull8StepFlow:
    """Integration test for the full 8-step execution flow."""

    def test_standard_8_step_flow(self, mock_ltm_store, sample_context_messages):
        """
        Test the canonical 8-step execution flow.

        Flow:
        1. assess_conversation_drift
        2. are_you_ready_to_get_in_context_ltm
        3. compress_conversation_for_ltm
        4. paraphrase_for_coverage
        5. trigger_contextual_ltm_retrieval
        6. validate_ltm_relevance
        7. (Inject and respond)
        8. log_retrieval_decision
        """
        clear_state()

        # Step 1: Assess drift
        drift_report = assess_conversation_drift(
            window_turns=3,
            against_anchor=False,
            current_messages=sample_context_messages,
            current_turn=3,
        )
        assert isinstance(drift_report, DriftReport)

        # Step 2: Readiness check
        query = "How do I use Python with FastAPI?"
        readiness = are_you_ready_to_get_in_context_ltm(
            query=query,
            urgency="helpful",
            current_messages=sample_context_messages,
            current_turn=3,
            ltm_store=mock_ltm_store,
        )
        assert isinstance(readiness, ReadinessAssessment)

        # Skip if not ready (for this test, we'll proceed anyway)
        if not readiness.should_retrieve:
            pytest.skip("Readiness assessment recommended no retrieval")

        # Step 3: Compress conversation
        turns = [
            Turn(role=m.role, content=m.content, turn_index=m.turn_index)
            for m in sample_context_messages
        ]
        compressed = compress_conversation_for_ltm(
            turns=turns,
            target_length=1,
        )
        assert isinstance(compressed, CompressedContext)

        # Step 4: Generate paraphrases
        paraphrases = paraphrase_for_coverage(
            core_concept=compressed.compressed_text or query,
            coverage_goals=["technical", "tutorial"],
        )
        assert len(paraphrases) > 0

        # Step 5: Execute retrieval
        injection = trigger_contextual_ltm_retrieval(
            retrieval_mode="multi_paraphrase",
            query_or_concept=paraphrases[0].text,
            ltm_store=mock_ltm_store,
            paraphrase_count=min(3, len(paraphrases)),
            top_k=5,
        )
        assert isinstance(injection, LTMInjection)

        # Step 6: Validate results
        validation = validate_ltm_relevance(
            candidate_memories=injection.memories,
            against_turns=[-3, -2, -1],
            recent_messages=sample_context_messages,
        )
        assert isinstance(validation, ValidatedBatch)

        # If validation fails, could refine here
        if validation.recommendation == "refine":
            failed = RetrievalAttempt(
                query=query,
                retrieval_mode=RetrievalMode.MULTI_PARAPHRASE,
                results_count=len(injection.memories),
            )
            refined = refine_retrieval_target(
                failed_retrieval=failed,
                failure_mode="too_broad",
            )
            assert isinstance(refined, RefinedQuery)

        # Step 7: (Inject memories and generate response - simulated)
        # In real usage: inject validated memories into context

        # Step 8: Log decision
        log_retrieval_decision(
            trigger="user_query",
            drift_scores=drift_report,
            retrieved_memories=[m.entry for m in injection.memories],
            utility_score=0.8 if validation.coverage_sufficient else 0.4,
            was_retrieved=True,
            strategy_used="multi_paraphrase",
            execution_time_ms=injection.execution_time_ms,
        )

        # Verify decision was logged
        history = get_decision_history()
        assert len(history) == 1
        assert history[0].trigger == "user_query"

    def test_flow_with_intent_shift(self, mock_ltm_store):
        """Test flow simulating an intent shift scenario."""
        clear_state()

        # Initial anchor
        initial_messages = [
            ContextMessage(role="user", content="How do I learn Python?", turn_index=1),
        ]
        set_anchor_from_context(initial_messages, turn_index=1)

        # Intent shift to debugging
        shifted_messages = [
            ContextMessage(role="user", content="I'm getting an error", turn_index=10),
            ContextMessage(role="user", content="How do I fix this bug?", turn_index=11),
        ]

        # Step 1: Should detect drift
        drift = assess_conversation_drift(
            window_turns=2,
            against_anchor=True,
            current_messages=shifted_messages,
        )

        # Should detect intent shift (learning → debugging)
        assert drift.intent_delta is not None or drift.drift_type != DriftType.NONE

        # Step 2: Should recommend retrieval due to drift
        readiness = are_you_ready_to_get_in_context_ltm(
            query="How do I fix this bug?",
            urgency="blocking",
            current_messages=shifted_messages,
            ltm_store=mock_ltm_store,
        )

        # Blocking urgency should always retrieve
        assert readiness.should_retrieve is True
        assert readiness.urgency == UrgencyLevel.BLOCKING


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_retry_cap_prevents_infinite_loops(self):
        """Retry mechanism caps at maximum to prevent loops."""
        base_query = "test query"

        for i in range(5):
            failed = RetrievalAttempt(
                query=base_query,  # Same query each time
                retrieval_mode=RetrievalMode.SINGLE_QUERY,
                results_count=0,
            )
            result = refine_retrieval_target(
                failed_retrieval=failed,
                failure_mode="too_broad",
                max_retries=2,
            )

            if i >= 2:
                assert result.can_retry is False
                break

    def test_empty_memory_list_validation(self, sample_context_messages):
        """Validation handles empty memory list."""
        result = validate_ltm_relevance(
            candidate_memories=[],
            recent_messages=sample_context_messages,
        )

        assert result.coverage_sufficient is False
        assert result.recommendation == "abort"
        assert len(result.coverage_gaps) > 0

    def test_none_context_messages(self):
        """Drift assessment handles None messages gracefully."""
        report = assess_conversation_drift(
            window_turns=3,
            current_messages=None,
        )

        assert isinstance(report, DriftReport)
        assert report.drift_type == DriftType.NONE

    def test_all_return_types_are_serializable(self, mock_ltm_store, sample_context_messages):
        """All tool return types can be serialized to dict."""
        # Tier 1
        drift = assess_conversation_drift(current_messages=sample_context_messages)
        assert isinstance(drift.to_dict(), dict)

        conf = self_assess_confidence(ltm_store=mock_ltm_store)
        assert isinstance(conf.to_dict(), dict)

        ready = are_you_ready_to_get_in_context_ltm(
            query="test",
            ltm_store=mock_ltm_store,
        )
        assert isinstance(ready.to_dict(), dict)

        # Tier 2
        paraphrases = paraphrase_for_coverage(core_concept="test")
        for p in paraphrases:
            assert isinstance(p.to_dict(), dict)

        injection = trigger_contextual_ltm_retrieval(
            retrieval_mode="single_query",
            query_or_concept="test",
            ltm_store=mock_ltm_store,
        )
        assert isinstance(injection.to_dict(), dict)

        # Tier 3
        sample_memory_entries = [MemoryEntry(content="test")]
        candidates = [
            RetrievedMemory(entry=e, retrieval_score=0.8, source_query="test", rank=i)
            for i, e in enumerate(sample_memory_entries)
        ]
        validation = validate_ltm_relevance(
            candidate_memories=candidates,
            recent_messages=sample_context_messages,
        )
        assert isinstance(validation.to_dict(), dict)

        failed = RetrievalAttempt(
            query="test",
            retrieval_mode=RetrievalMode.SINGLE_QUERY,
            results_count=0,
        )
        refined = refine_retrieval_target(failed_retrieval=failed, failure_mode="too_broad")
        assert isinstance(refined.to_dict(), dict)

        compressed = compress_conversation_for_ltm(turns=[])
        assert isinstance(compressed.to_dict(), dict)

        # Tier 4
        profile = ConversationProfile()
        strategy = suggest_retrieval_strategy(profile)
        assert isinstance(strategy.to_dict(), dict)


# ═══════════════════════════════════════════════════════════════════════════════
# State Management Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateManagement:
    """Tests for introspection state management."""

    def test_anchor_is_set(self):
        """Anchor snapshot can be set and retrieved."""
        messages = [
            ContextMessage(role="user", content="Test message", turn_index=1),
        ]

        set_anchor_from_context(messages, turn_index=1)

        # Drift assessment should now use anchor
        drift = assess_conversation_drift(
            window_turns=1,
            against_anchor=True,
            current_messages=messages,
        )

        assert drift.anchor_timestamp is not None

    def test_decision_history_accumulates(self):
        """Decision history accumulates multiple decisions."""
        clear_state()

        for i in range(3):
            log_retrieval_decision(
                trigger=f"test_{i}",
                drift_scores=DriftReport(),
                retrieved_memories=[],
                utility_score=0.5,
            )

        history = get_decision_history()
        assert len(history) == 3

    def test_clear_state_resets(self):
        """Clear state resets all accumulated state."""
        log_retrieval_decision(
            trigger="test",
            drift_scores=DriftReport(),
            retrieved_memories=[],
            utility_score=0.5,
        )

        clear_state()

        history = get_decision_history()
        assert len(history) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 5 Tests — Persistence Assurance (Memory Integrity)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssessPersistenceNeed:
    """Tests for assess_persistence_need tool."""

    def test_detects_explicit_remember(self):
        """Detects explicit 'remember that...' commands."""
        need = assess_persistence_need(
            user_input="Remember that uwot-swarm is a framework for agent orchestration",
        )

        assert isinstance(need, PersistenceNeed)
        assert need.should_persist is True
        assert need.urgency == PersistenceUrgency.IMMEDIATE
        assert any(p.pattern_type == "explicit_remember" for p in need.detected_patterns)
        assert "uwot-swarm" in need.suggested_content

    def test_detects_store_commands(self):
        """Detects 'store this in your memory...' commands."""
        need = assess_persistence_need(
            user_input="Store this in your memory: the password is 12345",
        )

        assert need.should_persist is True
        assert any(p.pattern_type in ["explicit_remember", "implied_store"]
                  for p in need.detected_patterns)

    def test_detects_save_commands(self):
        """Detects 'save this...' commands."""
        need = assess_persistence_need(
            user_input="Save this: my favorite color is blue",
        )

        assert need.should_persist is True
        assert need.urgency == PersistenceUrgency.IMMEDIATE
        assert any(p.pattern_type == "explicit_remember" for p in need.detected_patterns)

    def test_detects_forget_commands(self):
        """Detects 'forget that...' commands."""
        need = assess_persistence_need(
            user_input="Forget that I told you my password",
        )

        assert need.should_persist is True
        assert need.urgency == PersistenceUrgency.IMMEDIATE
        assert any(p.pattern_type == "explicit_forget" for p in need.detected_patterns)

    def test_detects_confirmation_requests(self):
        """Detects 'did you remember...' confirmation requests."""
        need = assess_persistence_need(
            user_input="Did you remember what I told you about the project?",
        )

        assert any(p.pattern_type == "persistence_confirm" for p in need.detected_patterns)

    def test_no_patterns_returns_false(self):
        """No persistence needed for normal conversation."""
        need = assess_persistence_need(
            user_input="What's the weather like today?",
        )

        assert need.should_persist is False
        assert need.urgency == PersistenceUrgency.BACKGROUND
        assert len(need.detected_patterns) == 0

    def test_pattern_confidence_scoring(self):
        """Explicit patterns have higher confidence than implied."""
        explicit_need = assess_persistence_need(
            user_input="Remember that X is important",
        )
        implied_need = assess_persistence_need(
            user_input="This is important: X",
        )

        explicit_confidences = [p.confidence for p in explicit_need.detected_patterns
                               if p.pattern_type == "explicit_remember"]
        implied_confidences = [p.confidence for p in implied_need.detected_patterns
                              if p.pattern_type == "implied_store"]

        if explicit_confidences and implied_confidences:
            assert explicit_confidences[0] > implied_confidences[0]

    def test_content_extraction(self):
        """Content is extracted from matched patterns."""
        need = assess_persistence_need(
            user_input="Remember that the API key is secret123",
        )

        assert "secret123" in need.suggested_content

    def test_returns_structured_object(self):
        """Returns PersistenceNeed, not raw string."""
        need = assess_persistence_need(user_input="Remember this")

        assert isinstance(need, PersistenceNeed)
        assert hasattr(need, 'to_dict')
        assert isinstance(need.to_dict(), dict)


class TestForceMemoryPersistence:
    """Tests for force_memory_persistence tool."""

    def test_returns_persistence_result(self):
        """Returns structured PersistenceResult."""
        result = force_memory_persistence(
            content="Test content to persist",
            learning_score=0.9,
            trigger="user_command",
        )

        assert isinstance(result, PersistenceResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'status')

    def test_no_ltm_store_returns_pending(self):
        """Without LTM store, returns PENDING status."""
        result = force_memory_persistence(
            content="Test content",
        )

        assert result.status == PersistenceStatus.PENDING
        assert result.success is False

    def test_content_preview_truncated(self):
        """Long content is previewed with truncation."""
        long_content = "x" * 500

        result = force_memory_persistence(content=long_content)

        assert len(result.content_preview) < 300
        assert "..." in result.content_preview

    def test_defaults_for_explicit_commands(self):
        """Sets appropriate defaults for explicit user commands."""
        result = force_memory_persistence(
            content="Important fact",
        )

        # Uses defaults: learning_score=0.8, trigger=user_command
        assert result.learning_score == 0.8
        assert result.trigger == "user_command"

    def test_returns_structured_object(self):
        """Returns PersistenceResult with to_dict method."""
        result = force_memory_persistence(content="Test")

        assert isinstance(result, PersistenceResult)
        assert isinstance(result.to_dict(), dict)


class TestValidateMemoryCommit:
    """Tests for validate_memory_commit tool."""

    def test_returns_validation_object(self):
        """Returns structured PersistenceValidation."""
        validation = validate_memory_commit(
            memory_id="test-memory-id",
            expected_content="Test content",
        )

        assert isinstance(validation, PersistenceValidation)
        assert hasattr(validation, 'is_validated')
        assert hasattr(validation, 'validation_checks')

    def test_no_memory_id_reports_error(self):
        """Validation without memory_id reports appropriate error."""
        validation = validate_memory_commit(
            memory_id=None,
            expected_content="Test",
        )

        existence_check = [c for c in validation.validation_checks
                          if c.check_name == "existence"][0]
        assert existence_check.passed is False
        assert "no memory id" in existence_check.details.lower()

    def test_no_ltm_store_returns_not_found(self):
        """Without LTM store, memory is not found."""
        validation = validate_memory_commit(
            memory_id="test-id",
            expected_content="Test content",
        )

        assert validation.memory_found is False
        assert validation.is_validated is False

    def test_validation_checks_structure(self):
        """All validation checks have required fields."""
        validation = validate_memory_commit(memory_id="test")

        for check in validation.validation_checks:
            assert hasattr(check, 'check_name')
            assert hasattr(check, 'passed')
            assert hasattr(check, 'details')
            assert isinstance(check.to_dict(), dict)

    def test_returns_structured_object(self):
        """Returns PersistenceValidation with to_dict method."""
        validation = validate_memory_commit(memory_id="test")

        assert isinstance(validation, PersistenceValidation)
        assert isinstance(validation.to_dict(), dict)


class TestLogPersistenceFailure:
    """Tests for log_persistence_failure tool."""

    def test_returns_failure_object(self):
        """Returns structured PersistenceFailure."""
        error = Exception("Test error")
        failure = log_persistence_failure(
            content="Test content",
            error=error,
            retry_count=2,
        )

        assert isinstance(failure, PersistenceFailure)
        assert failure.error_message == "Test error"
        assert failure.retry_count == 2

    def test_classifies_network_errors(self):
        """Correctly classifies network-related errors."""
        error = Exception("Network connection timeout")
        failure = log_persistence_failure(content="Test", error=error)

        assert failure.failure_category == FailureCategory.NETWORK

    def test_classifies_rate_limit_errors(self):
        """Correctly classifies rate limit errors."""
        error = Exception("Rate limit exceeded, quota exhausted")
        failure = log_persistence_failure(content="Test", error=error)

        assert failure.failure_category == FailureCategory.RATE_LIMIT

    def test_classifies_validation_errors(self):
        """Correctly classifies validation errors."""
        error = Exception("Content validation failed: invalid schema")
        failure = log_persistence_failure(content="Test", error=error)

        assert failure.failure_category == FailureCategory.VALIDATION

    def test_provides_recovery_action(self):
        """Failure includes recommended recovery action."""
        error = Exception("Network timeout")
        failure = log_persistence_failure(content="Test", error=error)

        assert len(failure.recovery_action) > 0
        assert "retry" in failure.recovery_action.lower()

    def test_content_preview_truncated(self):
        """Long content is previewed with truncation."""
        long_content = "x" * 500
        error = Exception("Test")

        failure = log_persistence_failure(content=long_content, error=error)

        assert len(failure.content_preview) < 300
        assert "..." in failure.content_preview

    def test_logs_to_state(self):
        """Failure is logged to introspection state."""
        clear_state()

        error = Exception("Test error")
        failure = log_persistence_failure(
            content="Test",
            error=error,
        )

        from memory.ltm_introspection import get_introspection_state
        state = get_introspection_state()
        failures = state.get_persistence_failures()

        assert len(failures) > 0

    def test_returns_structured_object(self):
        """Returns PersistenceFailure with to_dict method."""
        error = Exception("Test")
        failure = log_persistence_failure(content="Test", error=error)

        assert isinstance(failure, PersistenceFailure)
        assert isinstance(failure.to_dict(), dict)


class TestTier5ReturnTypes:
    """Tests for Tier 5 return type consistency."""

    def test_all_persistence_types_serializable(self):
        """All Tier 5 types can be serialized to dict."""
        # PersistenceNeed
        need = assess_persistence_need(user_input="Remember this")
        assert isinstance(need.to_dict(), dict)

        # PersistenceResult
        result = force_memory_persistence(content="Test")
        assert isinstance(result.to_dict(), dict)

        # PersistenceValidation
        validation = validate_memory_commit(memory_id="test")
        assert isinstance(validation.to_dict(), dict)

        # PersistenceFailure
        error = Exception("Test")
        failure = log_persistence_failure(content="Test", error=error)
        assert isinstance(failure.to_dict(), dict)

    def test_persistence_need_serialization(self):
        """PersistenceNeed serializes correctly."""
        need = assess_persistence_need(user_input="Remember that X")
        d = need.to_dict()

        assert "should_persist" in d
        assert "urgency" in d
        assert "detected_patterns" in d
        assert "priority_score" in d

    def test_persistence_result_serialization(self):
        """PersistenceResult serializes correctly."""
        result = force_memory_persistence(content="Test")
        d = result.to_dict()

        assert "success" in d
        assert "status" in d
        assert "content_preview" in d
        assert "timestamp" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
