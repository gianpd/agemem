"""
memory/ltm_introspection.py
───────────────────────────
LTM Self-Management Toolkit — Agent Introspection API.

This module provides a self-directed introspection toolkit that enables an agent
to reason about, orchestrate, and validate its own long-term memory retrieval.

Rather than relying on automatic, time-based triggers, the agent calls explicit
tools to:
  1. Assess its own state (drift, confidence, readiness)
  2. Decide whether retrieval is warranted
  3. Execute retrieval with semantic coverage
  4. Validate results
  5. Log decisions for future calibration

Every retrieval event produces a traceable chain of reasoning:
  what signal fired → what assessment said → what was retrieved →
  whether it was validated → what utility was observed

Tool Organization (4 Tiers)
───────────────────────────
Tier 1 — State Assessment (Introspection):
  • assess_conversation_drift
  • self_assess_confidence
  • are_you_ready_to_get_in_context_ltm

Tier 2 — Retrieval Orchestration (Action):
  • paraphrase_for_coverage
  • trigger_contextual_ltm_retrieval

Tier 3 — Validation & Refinement (Quality Control):
  • validate_ltm_relevance
  • refine_retrieval_target
  • compress_conversation_for_ltm

Tier 4 — Meta-Cognitive Tools (Learning):
  • log_retrieval_decision
  • suggest_retrieval_strategy

Design decisions
────────────────
* All tools return structured objects, never raw strings.
* Each tier's tools are independently callable — the agent may skip tiers.
* Retry logic is capped at 2 retries to prevent retrieval loops.
* Logging happens even when retrieval is skipped (non-retrieval is informative).
* Anchor snapshots are stored after each successful retrieval cycle.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from core.types import MemoryEntry, ContextMessage
from core.config import get_settings
from memory.ltm_introspection_types import (
    # Enums
    DriftType, ConfidenceLevel, ExpectedValue, UrgencyLevel,
    RetrievalMode, FailureMode, MatchDimension, ConfidenceDimension,
    # Tier 1
    DriftReport, ConfidenceDimensionScore, ConfidenceReport, ReadinessAssessment,
    # Tier 2
    Paraphrase, RetrievedMemory, LTMInjection,
    # Tier 3
    MemoryValidationResult, ValidatedBatch, RetrievalAttempt, RefinedQuery,
    CompressedContext, Turn,
    # Tier 4
    RetrievalDecision, ConversationProfile, StrategyRecommendation,
    # Supporting
    AnchorSnapshot,
)

if TYPE_CHECKING:
    from memory.ltm_store import LTMStore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Global State (per-session)
# ═══════════════════════════════════════════════════════════════════════════════

class _IntrospectionState:
    """
    Session-scoped state for the introspection toolkit.

    This state maintains:
    * The current anchor snapshot for drift detection
    * Retrieval decision history for calibration
    * Retry counters to prevent loops
    """

    def __init__(self):
        self.anchor: Optional[AnchorSnapshot] = None
        self.decision_history: List[RetrievalDecision] = []
        self.retry_counts: Dict[str, int] = {}  # query -> retry count
        self.strategy_effectiveness: Dict[str, float] = {}

    def set_anchor(self, anchor: AnchorSnapshot) -> None:
        """Set a new anchor snapshot after successful retrieval."""
        self.anchor = anchor
        logger.debug(f"Anchor set at turn {anchor.turn_index}")

    def get_retry_count(self, query: str) -> int:
        """Get retry count for a query."""
        return self.retry_counts.get(query, 0)

    def increment_retry_count(self, query: str) -> int:
        """Increment retry count and return new value."""
        self.retry_counts[query] = self.retry_counts.get(query, 0) + 1
        return self.retry_counts[query]

    def reset_retry_count(self, query: str) -> None:
        """Reset retry count for a query."""
        self.retry_counts.pop(query, None)

    def log_decision(self, decision: RetrievalDecision) -> None:
        """Log a retrieval decision."""
        self.decision_history.append(decision)
        logger.debug(f"Logged retrieval decision: {decision.trigger}, "
                    f"retrieved={decision.was_retrieved}, utility={decision.utility_score}")

    def get_historical_effectiveness(self, strategy: str) -> float:
        """Calculate historical effectiveness of a strategy."""
        relevant = [d for d in self.decision_history
                   if d.strategy_used == strategy and d.was_retrieved]
        if not relevant:
            return 0.5  # Default neutral effectiveness
        return sum(d.utility_score for d in relevant) / len(relevant)


# Module-level thread-local state storage
_thread_local_state = threading.local()


def _get_state() -> _IntrospectionState:
    """
    Get thread-local introspection state.

    Returns the _IntrospectionState for the current thread, creating it if necessary.
    This ensures thread safety when multiple sessions run concurrently.
    """
    if not hasattr(_thread_local_state, 'state'):
        _thread_local_state.state = _IntrospectionState()
    return _thread_local_state.state


def get_introspection_state() -> _IntrospectionState:
    """
    Public accessor for the current thread's introspection state.

    Use this to inspect or reset state manually (e.g., for testing).
    """
    return _get_state()


def reset_introspection_state() -> None:
    """Reset the current thread's introspection state to initial values."""
    _thread_local_state.state = _IntrospectionState()


# Keep for backward compatibility during migration
# TODO: Remove after all callers migrated to _get_state()
_state = _IntrospectionState()


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — State Assessment (Introspection)
# ═══════════════════════════════════════════════════════════════════════════════

def assess_conversation_drift(
    window_turns: int = 3,
    against_anchor: bool = True,
    metrics: List[str] = None,
    current_messages: Optional[List[ContextMessage]] = None,
    current_turn: int = 0,
) -> DriftReport:
    """
    Assess conversation drift from the stored anchor.

    Returns a structured DriftReport (not a boolean) so the agent can reason
    about the type and degree of drift before deciding to retrieve.

    Drift Detection Mechanism:
    --------------------------
    This function primarily uses semantic embeddings to detect topic drift.
    When embeddings are available (anchor.embedding is set), it computes
    cosine similarity between the anchor and current conversation window.

    Fallback Behavior:
    ------------------
    If embeddings are NOT available (e.g., embedding service unavailable or
    anchor has no embedding), the system falls back to lexical overlap via
    entity_continuity scoring. This measures Jaccard similarity of extracted
    named entities between anchor and current text.

    Note: Lexical overlap is less accurate than semantic embeddings for
    detecting paraphrases and semantic shifts. Consider this when interpreting
    drift scores in environments without embedding support.

    Args:
        window_turns: Number of recent turns to analyze.
        against_anchor: Whether to compare against stored anchor (vs rolling window).
        metrics: List of metrics to compute ("embedding", "intent", "entity_continuity").
        current_messages: Current conversation messages for analysis.
        current_turn: Current turn index.

    Returns:
        DriftReport with drift scores and classification.
    """
    if metrics is None:
        metrics = ["embedding", "intent", "entity_continuity"]

    report = DriftReport(window_turns=window_turns)

    # If no anchor exists and against_anchor is True, we can't compute drift
    if against_anchor and _get_state().anchor is None:
        report.drift_type = DriftType.NONE
        report.confidence = ConfidenceLevel.LOW
        report.retrieval_rationale = "No anchor set; assuming fresh conversation"
        return report

    try:
        # Compute embedding-based drift if requested
        if "embedding" in metrics and current_messages:
            report.topic_drift_score = _compute_embedding_drift(
                current_messages, window_turns
            )

        # Compute intent delta if requested
        if "intent" in metrics and current_messages:
            report.intent_delta = _compute_intent_delta(current_messages, window_turns)

        # Compute entity continuity if requested
        if "entity_continuity" in metrics and current_messages:
            report.entity_continuity = _compute_entity_continuity(
                current_messages, window_turns
            )

        # Classify drift type based on scores
        report.drift_type = _classify_drift(report)
        report.confidence = _assess_drift_confidence(report)

        if _get_state().anchor:
            report.anchor_timestamp = _get_state().anchor.timestamp

    except Exception as e:
        logger.warning(f"Drift assessment failed: {e}")
        report.drift_type = DriftType.NONE
        report.confidence = ConfidenceLevel.LOW

    return report


def _compute_embedding_drift(
    messages: List[ContextMessage],
    window_turns: int
) -> float:
    """Compute embedding-based drift score (0-1, higher = more drift)."""
    if not _get_state().anchor or _get_state().anchor.embedding is None:
        return 0.0

    try:
        from memory.embedding import embed_text

        # Get recent user messages
        recent_user_msgs = [
            m.content for m in reversed(messages)
            if m.role == "user" and m.content
        ][:window_turns]

        if not recent_user_msgs:
            return 0.0

        # Compute embedding of recent context
        recent_text = " ".join(recent_user_msgs)
        recent_embedding = embed_text(recent_text)

        if recent_embedding is None:
            return 0.0

        # Compute cosine distance from anchor
        anchor_emb = _get_state().anchor.embedding
        if isinstance(anchor_emb, list):
            anchor_emb = np.array(anchor_emb, dtype=np.float32)
        if isinstance(recent_embedding, list):
            recent_embedding = np.array(recent_embedding, dtype=np.float32)

        # Cosine distance = 1 - cosine similarity
        similarity = np.dot(anchor_emb, recent_embedding) / (
            np.linalg.norm(anchor_emb) * np.linalg.norm(recent_embedding)
        )
        distance = 1.0 - float(similarity)

        # Normalize to 0-1 (distance is already in this range for normalized vectors)
        return max(0.0, min(1.0, distance))

    except Exception as e:
        logger.debug(f"Embedding drift computation failed: {e}")
        return 0.0


def _compute_intent_delta(messages: List[ContextMessage], window_turns: int) -> Optional[str]:
    """Compute intent delta string (e.g., 'learning → debugging')."""
    if not _get_state().anchor:
        return None

    # Simple heuristic: classify intent based on keyword patterns
    anchor_intent = _classify_intent(_get_state().anchor.summary)

    recent_msgs = [
        m.content for m in reversed(messages)
        if m.role == "user" and m.content
    ][:window_turns]

    if not recent_msgs:
        return None

    recent_text = " ".join(recent_msgs)
    recent_intent = _classify_intent(recent_text)

    if anchor_intent != recent_intent:
        return f"{anchor_intent} → {recent_intent}"
    return None


def _classify_intent(text: str) -> str:
    """Classify intent based on keyword patterns."""
    text_lower = text.lower()

    # Simple keyword-based classification
    if any(w in text_lower for w in ["how", "what is", "explain", "?"]):
        return "learning"
    elif any(w in text_lower for w in ["error", "bug", "fix", "issue", "problem"]):
        return "debugging"
    elif any(w in text_lower for w in ["implement", "create", "build", "write"]):
        return "building"
    elif any(w in text_lower for w in ["compare", "difference", "vs", "or"]):
        return "comparing"
    elif any(w in text_lower for w in ["optimize", "improve", "faster", "better"]):
        return "optimizing"
    else:
        return "general"


def _compute_entity_continuity(messages: List[ContextMessage], window_turns: int) -> float:
    """Compute entity continuity score (overlap of named entities)."""
    if not _get_state().anchor:
        return 1.0  # No anchor = assume full continuity

    anchor_entities = set(_get_state().anchor.entities)
    if not anchor_entities:
        return 1.0

    # Extract entities from recent messages
    recent_text = " ".join([
        m.content for m in reversed(messages)
        if m.role == "user" and m.content
    ][:window_turns])

    recent_entities = set(_extract_entities(recent_text))

    if not recent_entities:
        return 0.0

    # Jaccard similarity of entity sets
    intersection = anchor_entities & recent_entities
    union = anchor_entities | recent_entities

    if not union:
        return 1.0

    return len(intersection) / len(union)


def _extract_entities(text: str) -> List[str]:
    """Simple entity extraction based on capitalized words and patterns."""
    # Capitalized word sequences (potential proper nouns)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

    # Technical terms (camelCase, snake_case, dotted.paths)
    technical = re.findall(r'\b[a-z]+(?:_+[a-z]+)+\b|\b[a-z]+(?:\.[a-z]+)+\b', text)

    # Quoted strings
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    quoted = [q[0] or q[1] for q in quoted]

    entities = capitalized + technical + quoted
    return [e.lower() for e in entities if len(e) > 2]


def _classify_drift(report: DriftReport) -> DriftType:
    """Classify drift type based on scores using configured thresholds."""
    drift_score = report.topic_drift_score
    continuity = report.entity_continuity
    config = get_settings()

    # Use config thresholds for drift classification
    low_threshold = config.DRIFT_LOW_THRESHOLD  # 0.3
    medium_threshold = config.DRIFT_MEDIUM_THRESHOLD  # 0.7

    if drift_score < low_threshold * 0.67 and continuity > medium_threshold:
        return DriftType.NONE
    elif drift_score < low_threshold * 1.33 and continuity > 0.5:
        return DriftType.SOFT_PIVOT
    elif drift_score > medium_threshold * 0.86 or continuity < low_threshold:
        return DriftType.HARD_PIVOT
    else:
        return DriftType.GRADUAL_SLOPE


def _assess_drift_confidence(report: DriftReport) -> ConfidenceLevel:
    """Assess confidence in drift classification using configured thresholds."""
    config = get_settings()
    high_threshold = config.CONFIDENCE_HIGH_THRESHOLD  # 0.8
    low_threshold = config.CONFIDENCE_LOW_THRESHOLD  # 0.5

    # Higher drift scores and clear classifications = higher confidence
    if report.drift_type == DriftType.HARD_PIVOT and report.topic_drift_score > high_threshold:
        return ConfidenceLevel.HIGH
    elif report.drift_type == DriftType.NONE and report.topic_drift_score < 0.1:
        return ConfidenceLevel.HIGH
    elif report.topic_drift_score > low_threshold or report.topic_drift_score < 0.15:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def self_assess_confidence(
    check_dimensions: Optional[List[str]] = None,
    current_context: Optional[str] = None,
    ltm_store: Optional[LTMStore] = None,
) -> ConfidenceReport:
    """
    Self-assess confidence in knowledge to answer well.

    Answers: "Do I actually know what I need to know to respond well?"
    Catches same-topic, deeper-layer gaps that drift detection misses.

    Args:
        check_dimensions: Dimensions to check ("factual", "contextual", "temporal").
        current_context: Current conversation context for assessment.
        ltm_store: LTM store to check coverage.

    Returns:
        ConfidenceReport with per-dimension and overall scores.
    """
    if check_dimensions is None:
        check_dimensions = ["factual", "contextual", "temporal"]

    dimensions: List[ConfidenceDimensionScore] = []
    knowledge_gaps: List[str] = []

    # Factual confidence
    if "factual" in check_dimensions:
        factual_score = _assess_factual_confidence(current_context, ltm_store)
        dimensions.append(ConfidenceDimensionScore(
            dimension=ConfidenceDimension.FACTUAL,
            score=factual_score,
            rationale="Based on LTM coverage of key terms"
        ))
        if factual_score < 0.5:
            knowledge_gaps.append("Factual knowledge may be incomplete")

    # Contextual confidence
    if "contextual" in check_dimensions:
        context_score = _assess_contextual_confidence(current_context)
        dimensions.append(ConfidenceDimensionScore(
            dimension=ConfidenceDimension.CONTEXTUAL,
            score=context_score,
            rationale="Based on conversation coherence"
        ))
        if context_score < 0.5:
            knowledge_gaps.append("Context understanding may be limited")

    # Temporal confidence
    if "temporal" in check_dimensions:
        temporal_score = _assess_temporal_confidence(current_context, ltm_store)
        dimensions.append(ConfidenceDimensionScore(
            dimension=ConfidenceDimension.TEMPORAL,
            score=temporal_score,
            rationale="Based on recency of relevant memories"
        ))
        if temporal_score < 0.5:
            knowledge_gaps.append("Temporal context may be outdated")

    # Compute overall score
    if dimensions:
        overall_score = sum(d.score for d in dimensions) / len(dimensions)
    else:
        overall_score = 0.5

    # Classify overall confidence using configured thresholds
    config = get_settings()
    if overall_score >= config.CONFIDENCE_HIGH_THRESHOLD:
        overall_confidence = ConfidenceLevel.HIGH
    elif overall_score >= config.CONFIDENCE_LOW_THRESHOLD:
        overall_confidence = ConfidenceLevel.MEDIUM
    else:
        overall_confidence = ConfidenceLevel.LOW

    return ConfidenceReport(
        dimensions=dimensions,
        overall_score=overall_score,
        overall_confidence=overall_confidence,
        knowledge_gaps=knowledge_gaps,
    )


def _assess_factual_confidence(
    context: Optional[str],
    ltm_store: Optional[LTMStore]
) -> float:
    """Assess confidence in factual knowledge."""
    if not context or not ltm_store:
        return 0.5

    try:
        # Check if LTM has relevant entries
        results = ltm_store.search(context, top_k=3)
        if not results:
            return 0.3  # Low confidence if no memories

        # Score based on relevance and learning scores
        avg_learning = sum(r.learning_score for r in results) / len(results)
        return 0.5 + (avg_learning * 0.5)  # Scale to 0.5-1.0
    except Exception:
        return 0.5


def _assess_contextual_confidence(context: Optional[str]) -> float:
    """Assess confidence in context understanding."""
    if not context:
        return 0.5

    # Simple heuristic: longer context with clear structure = higher confidence
    words = len(context.split())
    if words < 10:
        return 0.4
    elif words < 50:
        return 0.6
    else:
        return 0.8


def _assess_temporal_confidence(
    context: Optional[str],
    ltm_store: Optional[LTMStore]
) -> float:
    """Assess confidence in temporal relevance."""
    if not context or not ltm_store:
        return 0.5

    try:
        results = ltm_store.search(context, top_k=3)
        if not results:
            return 0.5

        # Check recency of results
        now = time.time()
        max_age_days = 30

        total_recency = 0.0
        for r in results:
            age_days = (now - r.updated_at) / 86400
            recency = max(0, 1 - (age_days / max_age_days))
            total_recency += recency

        return total_recency / len(results)
    except Exception:
        return 0.5


def are_you_ready_to_get_in_context_ltm(
    query: str,
    urgency: str = "helpful",
    current_messages: Optional[List[ContextMessage]] = None,
    current_turn: int = 0,
    ltm_store: Optional[LTMStore] = None,
) -> ReadinessAssessment:
    """
    Pre-flight readiness check before retrieval.

    Combines drift and confidence signals into a single go/no-go
    recommendation with explicit rationale.

    Args:
        query: The query being considered for retrieval.
        urgency: Urgency level ("blocking", "helpful", "exploratory").
        current_messages: Current conversation messages.
        current_turn: Current turn index.
        ltm_store: LTM store for confidence assessment.

    Returns:
        ReadinessAssessment with recommendation and rationale.
    """
    urgency_enum = UrgencyLevel(urgency.lower())

    # Assess drift
    drift_report = assess_conversation_drift(
        window_turns=3,
        against_anchor=True,
        current_messages=current_messages,
        current_turn=current_turn,
    )

    # Assess confidence
    context_text = query
    if current_messages:
        context_text = " ".join([
            m.content for m in current_messages[-3:]
            if m.content
        ])

    confidence_report = self_assess_confidence(
        current_context=context_text,
        ltm_store=ltm_store,
    )

    # Determine recommendation
    should_retrieve = False
    rationale = ""
    suggested_strategy = "single_query"
    expected_value = ExpectedValue.MEDIUM

    # Decision logic
    if urgency_enum == UrgencyLevel.BLOCKING:
        should_retrieve = True
        rationale = "Urgency is blocking; retrieval required"
        suggested_strategy = "multi_paraphrase_with_validation"
        expected_value = ExpectedValue.HIGH
    elif drift_report.drift_type in (DriftType.HARD_PIVOT, DriftType.GRADUAL_SLOPE):
        should_retrieve = True
        rationale = f"Significant drift detected ({drift_report.drift_type.value}); retrieval recommended"
        suggested_strategy = "multi_paraphrase"
        expected_value = ExpectedValue.HIGH
    elif confidence_report.overall_confidence == ConfidenceLevel.LOW:
        should_retrieve = True
        rationale = f"Low confidence ({confidence_report.overall_score:.2f}); retrieval may help"
        suggested_strategy = "anchored"
        expected_value = ExpectedValue.MEDIUM
    elif drift_report.drift_type == DriftType.SOFT_PIVOT:
        should_retrieve = urgency_enum != UrgencyLevel.EXPLORATORY
        rationale = f"Soft pivot detected; retrieval {'recommended' if should_retrieve else 'optional'}"
        suggested_strategy = "single_query"
        expected_value = ExpectedValue.MEDIUM
    else:
        should_retrieve = urgency_enum == UrgencyLevel.BLOCKING
        rationale = "No significant drift; retrieval not necessary unless blocking"
        suggested_strategy = "single_query"
        expected_value = ExpectedValue.LOW if not should_retrieve else ExpectedValue.MEDIUM

    return ReadinessAssessment(
        should_retrieve=should_retrieve,
        retrieval_rationale=rationale,
        suggested_retrieval_strategy=suggested_strategy,
        expected_value=expected_value,
        drift_report=drift_report,
        confidence_report=confidence_report,
        urgency=urgency_enum,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Retrieval Orchestration (Action)
# ═══════════════════════════════════════════════════════════════════════════════

def paraphrase_for_coverage(
    core_concept: str,
    coverage_goals: Optional[List[str]] = None,
    semantic_distance_target: float = 0.3,
    llm_client: Optional[Any] = None,
    model: Optional[str] = None,
) -> List[Paraphrase]:
    """
    Generate semantically diverse paraphrases of a query.

    The agent can inspect and optionally edit these variants before
    executing multi-paraphrase retrieval.

    Fallback Behavior:
    ------------------
    This function prefers LLM-based paraphrasing for semantic diversity.
    When llm_client is not available, it falls back to regex-based template
    matching. The regex fallback recognizes common query patterns (e.g.,
    "how do I...", "what is...") and applies predefined transformations.

    Limitations of regex fallback:
    - Limited to known patterns; complex queries may not match
    - Less semantic diversity than LLM-generated variants
    - Returns source="regex" to distinguish from LLM-generated (source="llm")

    Consider enabling LLM client for production use where semantic coverage
    is critical.

    Args:
        core_concept: The core concept to paraphrase.
        coverage_goals: Target coverage types (e.g., ["technical", "tutorial"]).
        semantic_distance_target: Target semantic distance from original (0-1).
        llm_client: Optional LLM client for LLM-based expansion.
        model: Model name for LLM-based expansion.

    Returns:
        List of Paraphrase objects with metadata.
    """
    if coverage_goals is None:
        coverage_goals = ["technical", "tutorial", "troubleshooting"]

    paraphrases: List[Paraphrase] = []

    # Always include original
    paraphrases.append(Paraphrase(
        text=core_concept,
        coverage_goal="original",
        semantic_distance=0.0,
        source="original",
    ))

    # Try LLM-based expansion if available
    if llm_client and model:
        try:
            llm_paraphrases = _generate_llm_paraphrases(
                core_concept, coverage_goals, llm_client, model
            )
            paraphrases.extend(llm_paraphrases)
        except Exception as e:
            logger.debug(f"LLM paraphrase generation failed: {e}")

    # Fallback: regex-based transformations
    if len(paraphrases) < len(coverage_goals) + 1:
        regex_paraphrases = _generate_regex_paraphrases(
            core_concept, coverage_goals, semantic_distance_target
        )
        # Add unique paraphrases
        existing_texts = {p.text.lower() for p in paraphrases}
        for p in regex_paraphrases:
            if p.text.lower() not in existing_texts:
                paraphrases.append(p)
                existing_texts.add(p.text.lower())

    return paraphrases[:len(coverage_goals) + 1]


def _generate_llm_paraphrases(
    concept: str,
    coverage_goals: List[str],
    llm_client: Any,
    model: str,
) -> List[Paraphrase]:
    """Generate paraphrases using LLM."""
    system_prompt = """You are a query paraphrasing assistant.
Generate alternative phrasings of the given concept for different coverage goals.
Return a JSON array of objects with 'text' and 'goal' fields.
Example: [{"text": "how to deploy", "goal": "tutorial"}]"""

    user_prompt = f"""Concept: {concept}
Coverage goals: {', '.join(coverage_goals)}
Generate {len(coverage_goals)} paraphrases, one for each goal."""

    try:
        response = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            max_tokens=256,
            temperature=0.7,
        )

        # Parse response
        import json
        data = json.loads(response)

        paraphrases = []
        for item in data:
            if isinstance(item, dict) and "text" in item:
                paraphrases.append(Paraphrase(
                    text=item["text"],
                    coverage_goal=item.get("goal", "unknown"),
                    semantic_distance=0.3,  # Estimated
                    source="llm",
                ))

        return paraphrases
    except Exception as e:
        raise RuntimeError(f"LLM paraphrase generation failed: {e}")


def _generate_regex_paraphrases(
    concept: str,
    coverage_goals: List[str],
    distance_target: float,
) -> List[Paraphrase]:
    """Generate paraphrases using regex transformations."""
    paraphrases = []
    concept_lower = concept.lower()

    # Goal-specific transformations
    transformations = {
        "technical": lambda t: f"technical implementation of {t}",
        "tutorial": lambda t: f"how to {t}" if not t.startswith("how to") else t,
        "troubleshooting": lambda t: f"fixing issues with {t}",
        "conceptual": lambda t: f"understanding {t}",
        "reference": lambda t: f"{t} documentation",
    }

    for goal in coverage_goals:
        if goal in transformations:
            transformed = transformations[goal](concept_lower)
            if transformed != concept_lower:
                paraphrases.append(Paraphrase(
                    text=transformed,
                    coverage_goal=goal,
                    semantic_distance=distance_target,
                    source="regex",
                ))

    return paraphrases


def trigger_contextual_ltm_retrieval(
    retrieval_mode: str,
    query_or_concept: str,
    ltm_store: LTMStore,
    paraphrase_count: int = 3,
    recency_bias: float = 0.7,
    diversity_constraint: bool = True,
    llm_client: Optional[Any] = None,
    model: Optional[str] = None,
    top_k: int = 5,
) -> LTMInjection:
    """
    Execute contextual LTM retrieval.

    In multi_paraphrase mode, runs N queries in parallel and deduplicates.
    Applies recency bias and diversity constraints.

    Args:
        retrieval_mode: Mode - "single_query", "multi_paraphrase", or "anchored".
        query_or_concept: Query or concept to retrieve for.
        ltm_store: LTM store to search.
        paraphrase_count: Number of paraphrases for multi_paraphrase mode.
        recency_bias: Weight for recency (0-1).
        diversity_constraint: Whether to ensure semantic diversity in results.
        llm_client: LLM client for paraphrase generation.
        model: Model name for paraphrase generation.
        top_k: Number of results to return.

    Returns:
        LTMInjection with retrieved memories and metadata.
    """
    start_time = time.time()
    mode = RetrievalMode(retrieval_mode.lower())

    result = LTMInjection(
        retrieval_mode=mode,
        recency_bias_applied=recency_bias,
    )

    try:
        if mode == RetrievalMode.SINGLE_QUERY:
            memories = _retrieve_single_query(
                query_or_concept, ltm_store, top_k, recency_bias
            )
            result.queries_executed = 1

        elif mode == RetrievalMode.MULTI_PARAPHRASE:
            memories = _retrieve_multi_paraphrase(
                query_or_concept, ltm_store, paraphrase_count,
                recency_bias, diversity_constraint,
                llm_client, model, top_k
            )
            result.queries_executed = paraphrase_count

        elif mode == RetrievalMode.ANCHORED:
            memories = _retrieve_anchored(
                query_or_concept, ltm_store, top_k, recency_bias
            )
            result.queries_executed = 1

        else:
            memories = []

        result.memories = memories
        result.total_candidates = len(memories)
        result.deduplication_count = 0  # Set by retrieval functions

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        result.memories = []

    result.execution_time_ms = (time.time() - start_time) * 1000
    return result


def _retrieve_single_query(
    query: str,
    ltm_store: LTMStore,
    top_k: int,
    recency_bias: float,
) -> List[RetrievedMemory]:
    """Retrieve using single query."""
    entries = ltm_store.search(query, top_k=top_k)

    # Apply recency re-ranking
    scored = _apply_recency_bias(entries, recency_bias)

    return [
        RetrievedMemory(
            entry=entry,
            retrieval_score=score,
            source_query=query,
            rank=i + 1,
        )
        for i, (entry, score) in enumerate(scored)
    ]


def _retrieve_multi_paraphrase(
    query: str,
    ltm_store: LTMStore,
    paraphrase_count: int,
    recency_bias: float,
    diversity_constraint: bool,
    llm_client: Optional[Any],
    model: Optional[str],
    top_k: int,
) -> List[RetrievedMemory]:
    """Retrieve using multiple paraphrased queries in parallel."""
    # Generate paraphrases
    paraphrases = paraphrase_for_coverage(
        core_concept=query,
        coverage_goals=["technical", "tutorial", "troubleshooting"][:paraphrase_count - 1],
        llm_client=llm_client,
        model=model,
    )

    # Execute queries in parallel
    all_results: Dict[str, tuple[MemoryEntry, float, str]] = {}

    with ThreadPoolExecutor(max_workers=min(paraphrase_count, 4)) as executor:
        futures = {
            executor.submit(
                _retrieve_single_query_for_merge,
                p.text, ltm_store, top_k * 2, recency_bias
            ): p.text
            for p in paraphrases
        }

        for future in as_completed(futures):
            source_query = futures[future]
            try:
                results = future.result()
                for entry, score in results:
                    if entry.entry_id not in all_results:
                        all_results[entry.entry_id] = (entry, score, source_query)
                    else:
                        # Keep higher score
                        existing_score = all_results[entry.entry_id][1]
                        if score > existing_score:
                            all_results[entry.entry_id] = (entry, score, source_query)
            except Exception as e:
                logger.debug(f"Paraphrase query failed: {e}")

    # Sort by score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x[1],
        reverse=True
    )

    # Apply diversity constraint if enabled
    if diversity_constraint and len(sorted_results) > top_k:
        sorted_results = _apply_diversity_constraint(sorted_results, top_k)

    return [
        RetrievedMemory(
            entry=entry,
            retrieval_score=score,
            source_query=source_query,
            rank=i + 1,
        )
        for i, (entry, score, source_query) in enumerate(sorted_results[:top_k])
    ]


def _retrieve_single_query_for_merge(
    query: str,
    ltm_store: LTMStore,
    top_k: int,
    recency_bias: float,
) -> List[tuple[MemoryEntry, float]]:
    """Helper for parallel retrieval - returns list of (entry, score)."""
    entries = ltm_store.search(query, top_k=top_k)
    return _apply_recency_bias(entries, recency_bias)


def _retrieve_anchored(
    query: str,
    ltm_store: LTMStore,
    top_k: int,
    recency_bias: float,
) -> List[RetrievedMemory]:
    """Retrieve using anchor-based context."""
    if _get_state().anchor and _get_state().anchor.embedding is not None:
        # Use anchor embedding as context
        try:
            from memory.embedding import embed_text

            query_emb = embed_text(query)
            if query_emb is not None:
                # Combine with anchor
                anchor_emb = _get_state().anchor.embedding
                if isinstance(anchor_emb, list):
                    anchor_emb = np.array(anchor_emb, dtype=np.float32)

                # Weighted combination: 70% query, 30% anchor
                combined = 0.7 * query_emb + 0.3 * anchor_emb
                combined = combined / np.linalg.norm(combined)

                entries = ltm_store.search_by_vector(combined, top_k=top_k)
                scored = _apply_recency_bias(entries, recency_bias)

                return [
                    RetrievedMemory(
                        entry=entry,
                        retrieval_score=score,
                        source_query=f"anchored:{query}",
                        rank=i + 1,
                    )
                    for i, (entry, score) in enumerate(scored)
                ]
        except Exception as e:
            logger.debug(f"Anchored retrieval failed: {e}")

    # Fallback to single query
    return _retrieve_single_query(query, ltm_store, top_k, recency_bias)


def _apply_recency_bias(
    entries: List[MemoryEntry],
    recency_bias: float,
) -> List[tuple[MemoryEntry, float]]:
    """Apply recency bias to entry scores."""
    now = time.time()
    scored = []

    for entry in entries:
        # Base score from learning score
        base_score = entry.learning_score

        # Recency factor (exponential decay)
        age_days = (now - entry.updated_at) / 86400
        recency_factor = math.exp(-age_days / 30)  # 30-day half-life

        # Combined score
        score = (base_score * (1 - recency_bias)) + (recency_factor * recency_bias)
        scored.append((entry, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _apply_diversity_constraint(
    results: List[tuple],
    top_k: int,
    min_similarity_threshold: float = 0.8,
) -> List[tuple]:
    """Apply diversity constraint to ensure varied results."""
    diverse = [results[0]] if results else []

    for result in results[1:]:
        if len(diverse) >= top_k:
            break

        entry = result[0]
        is_diverse = True

        # Check similarity with already selected results
        for selected in diverse:
            similarity = _estimate_content_similarity(entry.content, selected[0].content)
            if similarity > min_similarity_threshold:
                is_diverse = False
                break

        if is_diverse:
            diverse.append(result)

    return diverse


def _estimate_content_similarity(text1: str, text2: str) -> float:
    """Estimate content similarity using token overlap."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3 — Validation & Refinement (Quality Control)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_ltm_relevance(
    candidate_memories: List[RetrievedMemory],
    against_turns: Optional[List[int]] = None,
    match_dimensions: Optional[List[str]] = None,
    recent_messages: Optional[List[ContextMessage]] = None,
    relevance_threshold: float = 0.5,
) -> ValidatedBatch:
    """
    Validate retrieved memories post-retrieval.

    Returns per-memory relevance scores plus aggregate coverage analysis.
    If coverage is insufficient, the agent should call refine_retrieval_target.

    Args:
        candidate_memories: Memories to validate.
        against_turns: Turn indices to validate against (e.g., [-3, -2, -1]).
        match_dimensions: Dimensions to check ("entity", "intent", "temporal").
        recent_messages: Recent messages for context.
        relevance_threshold: Minimum relevance score to pass.

    Returns:
        ValidatedBatch with validation results and coverage analysis.
    """
    if match_dimensions is None:
        match_dimensions = ["entity", "intent", "temporal"]

    if against_turns is None:
        against_turns = [-3, -2, -1]

    # Extract context from recent messages
    context_text = ""
    if recent_messages:
        # Filter to requested turns
        turn_indices = set(against_turns)
        context_parts = [
            m.content for m in recent_messages
            if m.turn_index in turn_indices or
            (m.turn_index < 0 and len(recent_messages) + m.turn_index in turn_indices)
        ]
        context_text = " ".join(filter(None, context_parts))

    validated: List[MemoryValidationResult] = []
    total_relevance = 0.0

    for mem in candidate_memories:
        # Compute per-dimension scores
        match_scores: Dict[str, float] = {}

        if "entity" in match_dimensions:
            match_scores["entity"] = _compute_entity_match(
                mem.entry.content, context_text
            )

        if "intent" in match_dimensions:
            match_scores["intent"] = _compute_intent_match(
                mem.entry.content, context_text
            )

        if "temporal" in match_dimensions:
            match_scores["temporal"] = _compute_temporal_match(mem.entry)

        # Overall relevance score (average of dimensions)
        relevance_score = sum(match_scores.values()) / len(match_scores) if match_scores else 0.5

        is_relevant = relevance_score >= relevance_threshold

        result = MemoryValidationResult(
            entry=mem.entry,
            relevance_score=relevance_score,
            match_dimensions=match_scores,
            is_relevant=is_relevant,
            exclusion_reason=None if is_relevant else f"Score {relevance_score:.2f} below threshold",
        )

        validated.append(result)
        if is_relevant:
            total_relevance += relevance_score

    # Compute aggregate coverage
    relevant_count = sum(1 for v in validated if v.is_relevant)
    total_count = len(validated)

    config = get_settings()
    coverage_threshold = config.VALIDATION_COVERAGE_THRESHOLD
    coverage_score = total_relevance / max(1, relevant_count)
    coverage_sufficient = relevant_count >= max(2, top_k // 2) and coverage_score >= coverage_threshold

    # Identify coverage gaps
    coverage_gaps: List[str] = []
    if not coverage_sufficient:
        if relevant_count < 2:
            coverage_gaps.append("Insufficient relevant memories")
        if coverage_score < coverage_threshold:
            coverage_gaps.append("Low average relevance")

    # Recommendation
    if coverage_sufficient:
        recommendation = "proceed"
    elif relevant_count > 0:
        recommendation = "refine"
    else:
        recommendation = "abort"

    return ValidatedBatch(
        validated_memories=validated,
        coverage_score=coverage_score,
        coverage_sufficient=coverage_sufficient,
        relevant_count=relevant_count,
        total_count=total_count,
        coverage_gaps=coverage_gaps,
        recommendation=recommendation,
    )


# Fix: Define top_k at module level for validate_ltm_relevance
top_k = 5  # Default value


def _compute_entity_match(memory_content: str, context_text: str) -> float:
    """Compute entity match score between memory and context."""
    mem_entities = set(_extract_entities(memory_content))
    ctx_entities = set(_extract_entities(context_text))

    if not mem_entities or not ctx_entities:
        return 0.5  # Neutral if no entities

    intersection = mem_entities & ctx_entities
    if not intersection:
        return 0.0

    # Jaccard similarity
    union = mem_entities | ctx_entities
    return len(intersection) / len(union)


def _compute_intent_match(memory_content: str, context_text: str) -> float:
    """Compute intent match score."""
    mem_intent = _classify_intent(memory_content)
    ctx_intent = _classify_intent(context_text)

    if mem_intent == ctx_intent:
        return 1.0

    # Partial match for related intents
    related_intents = {
        "learning": ["debugging", "building"],
        "debugging": ["learning", "optimizing"],
        "building": ["learning", "optimizing"],
        "optimizing": ["debugging", "building"],
    }

    if ctx_intent in related_intents.get(mem_intent, []):
        return 0.5

    return 0.2


def _compute_temporal_match(entry: MemoryEntry) -> float:
    """Compute temporal relevance score."""
    age_days = (time.time() - entry.updated_at) / 86400

    # Exponential decay
    recency = math.exp(-age_days / 30)

    # Boost for frequently accessed entries
    access_boost = min(0.2, entry.access_count * 0.05)

    return min(1.0, recency + access_boost)


def refine_retrieval_target(
    failed_retrieval: RetrievalAttempt,
    failure_mode: str,
    max_retries: int = 2,
) -> RefinedQuery:
    """
    Produce a revised query strategy when retrieved memories fail validation.

    Failure mode must be explicitly classified by the agent — this forces
    auditability of retry decisions. Retry count is capped at max_retries.

    Args:
        failed_retrieval: The failed retrieval attempt.
        failure_mode: Classification of failure ("too_broad", "too_narrow", etc.).
        max_retries: Maximum allowed retries (default 2).

    Returns:
        RefinedQuery with revised strategy.
    """
    mode = FailureMode(failure_mode.lower())
    retry_count = _get_state().get_retry_count(failed_retrieval.query) + 1

    can_retry = retry_count <= max_retries

    # Generate refined query based on failure mode
    original = failed_retrieval.query
    refined = original
    strategy = ""
    additional_params: Dict[str, Any] = {}

    if mode == FailureMode.TOO_BROAD:
        refined = f"specific details about {original}"
        strategy = "Added specificity modifier"
        additional_params["top_k"] = 3  # Fewer, more focused results

    elif mode == FailureMode.TOO_NARROW:
        refined = original.replace("specific ", "").replace("exact ", "")
        strategy = "Removed narrow modifiers"
        additional_params["top_k"] = 10  # More results for broader coverage

    elif mode == FailureMode.OFF_TOPIC:
        # Try to extract key terms and reformulate
        key_terms = _extract_entities(original)
        if key_terms:
            refined = " ".join(key_terms[:3])
        strategy = "Reformulated around key entities"
        additional_params["recency_bias"] = 0.9  # More weight on recent

    elif mode == FailureMode.STALE:
        refined = f"latest {original}"
        strategy = "Added recency indicator"
        additional_params["recency_bias"] = 1.0  # Maximum recency

    if can_retry:
        _get_state().increment_retry_count(failed_retrieval.query)
    else:
        strategy += " (max retries reached)"

    return RefinedQuery(
        original_query=original,
        refined_query=refined,
        refinement_strategy=strategy,
        failure_mode=mode,
        retry_count=retry_count,
        can_retry=can_retry,
        additional_params=additional_params,
    )


def compress_conversation_for_ltm(
    turns: List[Turn],
    target_length: int = 1,
    preservation_priority: Optional[List[str]] = None,
) -> CompressedContext:
    """
    Compress recent conversation turns into a clean LTM-optimized query.

    Used both before retrieval (to sharpen the query) and before writing
    to LTM (to reduce noise at storage time).

    Args:
        turns: Conversation turns to compress.
        target_length: Target length in 'turns equivalent'.
        preservation_priority: Key elements to preserve.

    Returns:
        CompressedContext with compressed text and metadata.
    """
    if preservation_priority is None:
        preservation_priority = ["decisions", "constraints", "open_questions"]

    original_turns = len(turns)

    if not turns:
        return CompressedContext(
            compressed_text="",
            original_turns=0,
            target_length=target_length,
            preserved_elements=[],
            compression_ratio=1.0,
        )

    # Extract key elements based on preservation priority
    preserved: List[str] = []

    for turn in turns:
        content = turn.content or ""

        if "decisions" in preservation_priority:
            # Look for decision patterns
            decisions = re.findall(
                r'\b(decided|agreed|concluded|chose|selected)\b[^.]+',
                content,
                re.IGNORECASE
            )
            preserved.extend(decisions)

        if "constraints" in preservation_priority:
            # Look for constraint patterns
            constraints = re.findall(
                r'\b(must|should|need|required|cannot)\b[^.]+',
                content,
                re.IGNORECASE
            )
            preserved.extend(constraints)

        if "open_questions" in preservation_priority:
            # Look for question patterns
            questions = re.findall(r'\b(what|how|why|when|where)\b[^?]+\?', content, re.IGNORECASE)
            preserved.extend(questions)

    # Build compressed text
    # Strategy: Concatenate user messages with preserved elements
    user_contents = [
        t.content for t in turns
        if t.role == "user" and t.content
    ]

    if len(user_contents) > target_length:
        # Compress by taking most recent and key preserved elements
        compressed = " ".join(user_contents[-target_length:])
        if preserved:
            compressed += " | Key points: " + "; ".join(preserved[:5])
    else:
        compressed = " ".join(user_contents)

    # Calculate compression ratio
    original_length = sum(len(t.content or "") for t in turns)
    compressed_length = len(compressed)
    ratio = compressed_length / max(1, original_length)

    return CompressedContext(
        compressed_text=compressed,
        original_turns=original_turns,
        target_length=target_length,
        preserved_elements=preserved[:10],  # Limit preserved elements
        compression_ratio=ratio,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 4 — Meta-Cognitive Tools (Learning)
# ═══════════════════════════════════════════════════════════════════════════════

def log_retrieval_decision(
    trigger: str,
    drift_scores: DriftReport,
    retrieved_memories: List[MemoryEntry],
    utility_score: float,
    was_retrieved: bool = True,
    strategy_used: str = "",
    execution_time_ms: float = 0.0,
) -> None:
    """
    Log the full retrieval event for future calibration.

    The utility_score is the agent's self-assessment of whether the
    retrieved memories improved its response. This feed is used to
    tune drift thresholds over time.

    Must be called even when should_retrieve=False — non-retrieval
    decisions are equally important for calibration.

    Args:
        trigger: What triggered this decision.
        drift_scores: Drift report at decision time.
        retrieved_memories: Memories that were retrieved (empty if skipped).
        utility_score: Self-assessed utility (0-1).
        was_retrieved: Whether retrieval was actually performed.
        strategy_used: Strategy that was used.
        execution_time_ms: Execution time.
    """
    decision = RetrievalDecision(
        trigger=trigger,
        drift_scores=drift_scores,
        retrieved_memories=retrieved_memories,
        memory_count=len(retrieved_memories),
        utility_score=utility_score,
        was_retrieved=was_retrieved,
        strategy_used=strategy_used,
        execution_time_ms=execution_time_ms,
    )

    _get_state().log_decision(decision)

    # Also log to Python logger
    logger.info(
        f"Retrieval decision logged: trigger={trigger}, "
        f"retrieved={was_retrieved}, memories={len(retrieved_memories)}, "
        f"utility={utility_score:.2f}"
    )


def suggest_retrieval_strategy(
    conversation_profile: ConversationProfile,
    historical_effectiveness: Optional[Dict[str, float]] = None,
) -> StrategyRecommendation:
    """
    Recommend retrieval strategy based on conversation characteristics.

    Given this conversation's profile (domain, user intent, session length),
    recommends the retrieval strategy that has historically performed best.
    Useful as a warm-start for long or complex sessions.

    Args:
        conversation_profile: Profile of the current conversation.
        historical_effectiveness: Optional effectiveness scores by strategy.

    Returns:
        StrategyRecommendation with suggested strategy and rationale.
    """
    if historical_effectiveness is None:
        historical_effectiveness = {}

    profile = conversation_profile

    # Default strategy
    recommended_mode = RetrievalMode.SINGLE_QUERY
    strategy = "single_query"
    rationale = "Default strategy for general conversations"
    confidence = ConfidenceLevel.MEDIUM

    # Strategy selection based on profile
    if profile.complexity_score > 0.7 or profile.session_length_turns > 20:
        recommended_mode = RetrievalMode.MULTI_PARAPHRASE
        strategy = "multi_paraphrase_with_validation"
        rationale = "High complexity or long session; diverse coverage needed"
        confidence = ConfidenceLevel.HIGH

    elif profile.domain in ["coding", "technical"]:
        recommended_mode = RetrievalMode.ANCHORED
        strategy = "anchored_technical"
        rationale = "Technical domain; anchor-based retrieval for context stability"
        confidence = ConfidenceLevel.HIGH

    elif profile.user_intent in ["debugging", "troubleshooting"]:
        recommended_mode = RetrievalMode.MULTI_PARAPHRASE
        strategy = "multi_paraphrase_troubleshooting"
        rationale = "Troubleshooting intent; multiple angles improve coverage"
        confidence = ConfidenceLevel.MEDIUM

    elif not profile.has_established_context:
        recommended_mode = RetrievalMode.SINGLE_QUERY
        strategy = "single_query_exploratory"
        rationale = "Early session; simple retrieval to establish baseline"
        confidence = ConfidenceLevel.MEDIUM

    # Check historical effectiveness
    historical = historical_effectiveness.get(strategy)
    if historical is None:
        historical = _get_state().get_historical_effectiveness(strategy)

    # Adjust confidence based on historical data
    if historical > 0.8:
        confidence = ConfidenceLevel.HIGH
    elif historical < 0.4:
        confidence = ConfidenceLevel.LOW
        rationale += " (historically low effectiveness)"

    suggested_params: Dict[str, Any] = {}
    if recommended_mode == RetrievalMode.MULTI_PARAPHRASE:
        suggested_params["paraphrase_count"] = 3
        suggested_params["diversity_constraint"] = True
    elif recommended_mode == RetrievalMode.ANCHORED:
        suggested_params["recency_bias"] = 0.5  # Balanced

    return StrategyRecommendation(
        recommended_strategy=strategy,
        recommended_mode=recommended_mode,
        confidence=confidence,
        rationale=rationale,
        historical_effectiveness=historical,
        suggested_params=suggested_params,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def set_anchor_from_context(
    messages: List[ContextMessage],
    turn_index: int,
) -> None:
    """
    Set a new anchor snapshot from current context.

    Should be called after each successful retrieval cycle.

    Args:
        messages: Current conversation messages.
        turn_index: Current turn index.
    """
    try:
        from memory.embedding import embed_text

        # Build summary text
        user_contents = [
            m.content for m in messages
            if m.role == "user" and m.content
        ][-5:]  # Last 5 user messages

        summary = " ".join(user_contents)

        # Generate embedding
        embedding = embed_text(summary)

        # Extract entities
        entities = _extract_entities(summary)

        # Classify intent
        intent = _classify_intent(summary)

        anchor = AnchorSnapshot(
            embedding=embedding,
            entities=entities,
            intent=intent,
            turn_index=turn_index,
            summary=summary[:200],  # Truncate for storage
        )

        _get_state().set_anchor(anchor)

    except Exception as e:
        logger.warning(f"Failed to set anchor: {e}")


def get_decision_history(limit: int = 100) -> List[RetrievalDecision]:
    """
    Get recent retrieval decision history.

    Args:
        limit: Maximum number of decisions to return.

    Returns:
        List of recent RetrievalDecision objects.
    """
    return _get_state().decision_history[-limit:]


def clear_state() -> None:
    """Clear all introspection state for the current thread (for testing)."""
    _thread_local_state.state = _IntrospectionState()
    logger.debug("Introspection state cleared")
