"""
memory/ltm_introspection_types.py
─────────────────────────────────
Typed data models for the LTM Self-Management Toolkit.

All return types are structured dataclasses (not raw strings) to enable
agent reasoning about individual fields. This supports auditability and
traceable decision chains for every retrieval event.

Design decisions
────────────────
* All enums are string-based for JSON serialization compatibility.
* All dataclasses include to_dict() for structured logging.
* Optional fields allow progressive enrichment as tools are composed.
* Types are organized by tier to match the tool organization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

from core.types import MemoryEntry, ContextMessage


# ═══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════════

class DriftType(str, Enum):
    """Classification of conversation drift severity."""
    NONE = "none"              # No meaningful drift detected
    SOFT_PIVOT = "soft_pivot"  # Minor topic shift, related context
    HARD_PIVOT = "hard_pivot"  # Major topic change, new domain
    GRADUAL_SLOPE = "gradual_slope"  # Slow drift over multiple turns


class ConfidenceLevel(str, Enum):
    """Confidence levels for assessments."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExpectedValue(str, Enum):
    """Expected value of retrieval for decision-making."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UrgencyLevel(str, Enum):
    """Urgency levels for retrieval requests."""
    BLOCKING = "blocking"      # Cannot proceed without context
    HELPFUL = "helpful"        # Would improve response quality
    EXPLORATORY = "exploratory"  # Nice to have, not critical


class RetrievalMode(str, Enum):
    """Modes for contextual LTM retrieval."""
    SINGLE_QUERY = "single_query"           # Single query retrieval
    MULTI_PARAPHRASE = "multi_paraphrase"   # Multiple paraphrased queries
    ANCHORED = "anchored"                   # Anchor-based retrieval


class FailureMode(str, Enum):
    """Classification of retrieval failure modes."""
    TOO_BROAD = "too_broad"      # Results too general
    TOO_NARROW = "too_narrow"    # Results too specific
    OFF_TOPIC = "off_topic"      # Results irrelevant
    STALE = "stale"              # Results outdated


class MatchDimension(str, Enum):
    """Dimensions for relevance matching."""
    ENTITY = "entity"        # Named entity overlap
    INTENT = "intent"        # Intent similarity
    TEMPORAL = "temporal"    # Temporal relevance


class ConfidenceDimension(str, Enum):
    """Dimensions for confidence self-assessment."""
    FACTUAL = "factual"        # Factual knowledge confidence
    CONTEXTUAL = "contextual"  # Context understanding confidence
    TEMPORAL = "temporal"      # Temporal relevance confidence


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — State Assessment (Introspection)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DriftReport:
    """
    Structured report from conversation drift assessment.

    The agent reasons about these fields before deciding to retrieve,
    enabling nuanced decisions beyond simple boolean triggers.
    """
    topic_drift_score: float = 0.0
    """0-1 score; embedding distance from anchor (higher = more drift)."""

    intent_delta: Optional[str] = None
    """e.g. 'learning → debugging' describing the intent shift."""

    entity_continuity: float = 0.0
    """0-1 score; overlap of named entities vs anchor."""

    drift_type: DriftType = DriftType.NONE
    """Classification of drift severity."""

    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    """Confidence in the drift assessment."""

    # Additional metadata for debugging
    window_turns: int = 3
    """Number of turns analyzed."""

    anchor_timestamp: Optional[float] = None
    """Timestamp of the anchor snapshot."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "topic_drift_score": self.topic_drift_score,
            "intent_delta": self.intent_delta,
            "entity_continuity": self.entity_continuity,
            "drift_type": self.drift_type.value,
            "confidence": self.confidence.value,
            "window_turns": self.window_turns,
            "anchor_timestamp": self.anchor_timestamp,
        }


@dataclass
class ConfidenceDimensionScore:
    """Score for a single confidence dimension."""
    dimension: ConfidenceDimension
    score: float  # 0-1
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "rationale": self.rationale,
        }


@dataclass
class ConfidenceReport:
    """
    Self-assessment of agent's confidence in its knowledge.

    Answers: "Do I actually know what I need to know to respond well?"
    Catches same-topic, deeper-layer gaps that drift detection misses.
    """
    dimensions: List[ConfidenceDimensionScore] = field(default_factory=list)
    """Per-dimension confidence scores."""

    overall_score: float = 0.0
    """Aggregated confidence score (0-1)."""

    overall_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    """Overall confidence classification."""

    knowledge_gaps: List[str] = field(default_factory=list)
    """Identified areas where knowledge may be insufficient."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": [d.to_dict() for d in self.dimensions],
            "overall_score": self.overall_score,
            "overall_confidence": self.overall_confidence.value,
            "knowledge_gaps": self.knowledge_gaps,
        }


@dataclass
class ReadinessAssessment:
    """
    Pre-flight readiness check result.

    Combines drift and confidence signals into a go/no-go recommendation
    with explicit rationale for auditability.
    """
    should_retrieve: bool = False
    """Primary recommendation: proceed with retrieval?"""

    retrieval_rationale: str = ""
    """Human-readable reason for the recommendation."""

    suggested_retrieval_strategy: str = "single_query"
    """Recommended strategy e.g. 'multi_paraphrase_with_validation'."""

    expected_value: ExpectedValue = ExpectedValue.MEDIUM
    """Expected value of retrieval if executed."""

    # Supporting data for agent reasoning
    drift_report: Optional[DriftReport] = None
    """The drift assessment that contributed to this decision."""

    confidence_report: Optional[ConfidenceReport] = None
    """The confidence assessment that contributed to this decision."""

    urgency: UrgencyLevel = UrgencyLevel.HELPFUL
    """Urgency level of the retrieval request."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_retrieve": self.should_retrieve,
            "retrieval_rationale": self.retrieval_rationale,
            "suggested_retrieval_strategy": self.suggested_retrieval_strategy,
            "expected_value": self.expected_value.value,
            "drift_report": self.drift_report.to_dict() if self.drift_report else None,
            "confidence_report": self.confidence_report.to_dict() if self.confidence_report else None,
            "urgency": self.urgency.value,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Retrieval Orchestration (Action)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Paraphrase:
    """
    A single paraphrase variant with metadata.

    Allows the agent to inspect and optionally edit variants before
    executing multi-paraphrase retrieval.
    """
    text: str
    """The paraphrased query text."""

    coverage_goal: str = ""
    """Target coverage type (e.g., 'technical', 'tutorial')."""

    semantic_distance: float = 0.0
    """Estimated semantic distance from original (0-1)."""

    source: str = "llm"  # 'llm' | 'regex' | 'user_edit'
    """How this paraphrase was generated."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "coverage_goal": self.coverage_goal,
            "semantic_distance": self.semantic_distance,
            "source": self.source,
        }


@dataclass
class RetrievedMemory:
    """A memory with its retrieval metadata."""
    entry: MemoryEntry
    retrieval_score: float
    source_query: str  # Which query variant produced this
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "retrieval_score": self.retrieval_score,
            "source_query": self.source_query,
            "rank": self.rank,
        }


@dataclass
class LTMInjection:
    """
    Result of contextual LTM retrieval.

    Contains the deduplicated memory set and execution metadata.
    """
    memories: List[RetrievedMemory] = field(default_factory=list)
    """Retrieved and deduplicated memories."""

    retrieval_mode: RetrievalMode = RetrievalMode.SINGLE_QUERY
    """Mode used for this retrieval."""

    queries_executed: int = 0
    """Number of queries executed (for multi_paraphrase mode)."""

    total_candidates: int = 0
    """Total candidates before deduplication."""

    deduplication_count: int = 0
    """Number of duplicates removed."""

    recency_bias_applied: float = 0.7
    """Recency bias weight used."""

    execution_time_ms: float = 0.0
    """Execution time in milliseconds."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "retrieval_mode": self.retrieval_mode.value,
            "queries_executed": self.queries_executed,
            "total_candidates": self.total_candidates,
            "deduplication_count": self.deduplication_count,
            "recency_bias_applied": self.recency_bias_applied,
            "execution_time_ms": self.execution_time_ms,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3 — Validation & Refinement (Quality Control)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryValidationResult:
    """Validation result for a single memory."""
    entry: MemoryEntry
    relevance_score: float  # 0-1
    match_dimensions: Dict[str, float] = field(default_factory=dict)
    """Per-dimension match scores."""

    is_relevant: bool = True
    """Whether this memory passes relevance threshold."""

    exclusion_reason: Optional[str] = None
    """Reason for exclusion if not relevant."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "relevance_score": self.relevance_score,
            "match_dimensions": self.match_dimensions,
            "is_relevant": self.is_relevant,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass
class ValidatedBatch:
    """
    Result of post-retrieval validation.

    Provides per-memory relevance scores plus aggregate coverage analysis.
    """
    validated_memories: List[MemoryValidationResult] = field(default_factory=list)
    """Individual memory validation results."""

    coverage_score: float = 0.0
    """Aggregate coverage score (0-1)."""

    coverage_sufficient: bool = True
    """Whether coverage meets threshold."""

    relevant_count: int = 0
    """Number of memories passing relevance threshold."""

    total_count: int = 0
    """Total memories evaluated."""

    coverage_gaps: List[str] = field(default_factory=list)
    """Identified coverage gaps if insufficient."""

    recommendation: str = "proceed"
    """Recommendation: 'proceed', 'refine', or 'abort'."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validated_memories": [m.to_dict() for m in self.validated_memories],
            "coverage_score": self.coverage_score,
            "coverage_sufficient": self.coverage_sufficient,
            "relevant_count": self.relevant_count,
            "total_count": self.total_count,
            "coverage_gaps": self.coverage_gaps,
            "recommendation": self.recommendation,
        }


@dataclass
class RetrievalAttempt:
    """Record of a retrieval attempt for refinement analysis."""
    query: str
    retrieval_mode: RetrievalMode
    results_count: int
    top_scores: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: __import__('time').time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieval_mode": self.retrieval_mode.value,
            "results_count": self.results_count,
            "top_scores": self.top_scores,
            "timestamp": self.timestamp,
        }


@dataclass
class RefinedQuery:
    """
    Refined query strategy for retry.

    Produced when retrieved memories fail validation.
    """
    original_query: str = ""
    """Original query that failed."""

    refined_query: str = ""
    """New query with refined strategy."""

    refinement_strategy: str = ""
    """Description of refinement applied."""

    failure_mode: FailureMode = FailureMode.TOO_BROAD
    """Classified failure mode."""

    retry_count: int = 0
    """Current retry count (capped at 2)."""

    can_retry: bool = True
    """Whether another retry is allowed."""

    additional_params: Dict[str, Any] = field(default_factory=dict)
    """Additional parameters for the refined query."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "refined_query": self.refined_query,
            "refinement_strategy": self.refinement_strategy,
            "failure_mode": self.failure_mode.value,
            "retry_count": self.retry_count,
            "can_retry": self.can_retry,
            "additional_params": self.additional_params,
        }


@dataclass
class CompressedContext:
    """
    Compressed conversation turns for LTM-optimized query.

    Used both before retrieval (to sharpen the query) and before
    writing to LTM (to reduce noise at storage time).
    """
    compressed_text: str = ""
    """The compressed query text."""

    original_turns: int = 0
    """Number of original turns compressed."""

    target_length: int = 1
    """Target compression length in 'turns equivalent'."""

    preserved_elements: List[str] = field(default_factory=list)
    """Key elements preserved (decisions, constraints, open questions)."""

    compression_ratio: float = 0.0
    """Ratio of compressed to original length."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compressed_text": self.compressed_text,
            "original_turns": self.original_turns,
            "target_length": self.target_length,
            "preserved_elements": self.preserved_elements,
            "compression_ratio": self.compression_ratio,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 4 — Meta-Cognitive Tools (Learning)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalDecision:
    """
    Logged record of a retrieval decision.

    Every retrieval event is logged with full context for future
    calibration and threshold tuning.
    """
    trigger: str = ""
    """What triggered this retrieval decision."""

    drift_scores: Optional[DriftReport] = None
    """Drift assessment at decision time."""

    retrieved_memories: List[MemoryEntry] = field(default_factory=list)
    """Memories that were retrieved (empty if skipped)."""

    memory_count: int = 0
    """Number of memories retrieved."""

    utility_score: float = 0.0
    """Agent's self-assessment of utility (0-1)."""

    was_retrieved: bool = False
    """Whether retrieval was actually performed."""

    strategy_used: str = ""
    """Strategy that was used."""

    execution_time_ms: float = 0.0
    """Total execution time."""

    timestamp: float = field(default_factory=lambda: __import__('time').time())
    """When this decision was made."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger": self.trigger,
            "drift_scores": self.drift_scores.to_dict() if self.drift_scores else None,
            "retrieved_memories": [m.to_dict() for m in self.retrieved_memories],
            "memory_count": self.memory_count,
            "utility_score": self.utility_score,
            "was_retrieved": self.was_retrieved,
            "strategy_used": self.strategy_used,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class ConversationProfile:
    """
    Characteristics of a conversation for strategy recommendation.
    """
    domain: str = "general"
    """Detected domain (e.g., 'coding', 'medical', 'creative')."""

    user_intent: str = "informational"
    """Classified user intent."""

    session_length_turns: int = 0
    """Number of turns in session so far."""

    complexity_score: float = 0.5
    """Estimated complexity (0-1)."""

    has_established_context: bool = False
    """Whether conversation has established shared context."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "user_intent": self.user_intent,
            "session_length_turns": self.session_length_turns,
            "complexity_score": self.complexity_score,
            "has_established_context": self.has_established_context,
        }


@dataclass
class StrategyRecommendation:
    """
    Recommended retrieval strategy based on conversation profile.

    Provides warm-start recommendations for long or complex sessions.
    """
    recommended_strategy: str = "single_query"
    """Recommended strategy name."""

    recommended_mode: RetrievalMode = RetrievalMode.SINGLE_QUERY
    """Recommended retrieval mode."""

    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    """Confidence in this recommendation."""

    rationale: str = ""
    """Why this strategy is recommended."""

    historical_effectiveness: float = 0.0
    """Historical effectiveness score for this strategy (0-1)."""

    suggested_params: Dict[str, Any] = field(default_factory=dict)
    """Suggested parameters for the strategy."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_strategy": self.recommended_strategy,
            "recommended_mode": self.recommended_mode.value,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "historical_effectiveness": self.historical_effectiveness,
            "suggested_params": self.suggested_params,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Supporting Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    """Simplified turn representation for compression."""
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    turn_index: int
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp,
        }


@dataclass
class AnchorSnapshot:
    """
    Anchor snapshot for drift detection.

    Captures the conversation state at a stable point to compare
    subsequent turns against.
    """
    embedding: Optional[Any] = None
    """Vector embedding of anchor state."""

    entities: List[str] = field(default_factory=list)
    """Named entities present at anchor."""

    intent: str = ""
    """Classified intent at anchor."""

    timestamp: float = field(default_factory=lambda: __import__('time').time())
    """When this anchor was set."""

    turn_index: int = 0
    """Turn index when anchor was set."""

    summary: str = ""
    """Text summary of anchor state."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": self.entities,
            "intent": self.intent,
            "timestamp": self.timestamp,
            "turn_index": self.turn_index,
            "summary": self.summary,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 5 — Persistence Assurance (Memory Integrity)
# ═══════════════════════════════════════════════════════════════════════════════

class PersistenceUrgency(str, Enum):
    """Urgency levels for persistence operations."""
    IMMEDIATE = "immediate"    # Must persist before responding
    BATCH = "batch"            # Can persist with other operations
    BACKGROUND = "background"  # Deferred persistence acceptable


class PersistenceStatus(str, Enum):
    """Status of a persistence operation."""
    PENDING = "pending"        # Not yet attempted
    CONFIRMED = "confirmed"    # Successfully persisted to LTM
    FAILED = "failed"          # Persistence failed
    VALIDATED = "validated"    # Confirmed and validated


class FailureCategory(str, Enum):
    """Categories of persistence failures."""
    NETWORK = "network"              # Network/storage connectivity issues
    VALIDATION = "validation"        # Content validation failed
    RATE_LIMIT = "rate_limit"        # Rate limiting or quota exceeded
    CORRUPTION = "corruption"        # Data corruption detected
    UNKNOWN = "unknown"              # Unclassified failure


@dataclass
class MemoryCommandPattern:
    """Detected memory command pattern in user input."""
    pattern_type: str = ""
    """Type of pattern detected (e.g., 'explicit_remember', 'implied_store')."""

    matched_phrase: str = ""
    """The exact phrase that matched."""

    confidence: float = 0.0
    """Confidence in pattern detection (0-1)."""

    content_to_persist: str = ""
    """Extracted content that should be persisted."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "matched_phrase": self.matched_phrase,
            "confidence": self.confidence,
            "content_to_persist": self.content_to_persist,
        }


@dataclass
class PersistenceNeed:
    """
    Assessment of whether persistence is needed and how urgently.

    Analyzes user input for explicit memory commands and conversation
    context to determine if immediate persistence is required.
    """
    should_persist: bool = False
    """Whether persistence is recommended."""

    urgency: PersistenceUrgency = PersistenceUrgency.BATCH
    """Urgency level for persistence."""

    detected_patterns: List[MemoryCommandPattern] = field(default_factory=list)
    """Memory command patterns detected in user input."""

    persistence_rationale: str = ""
    """Human-readable reason for persistence recommendation."""

    suggested_content: str = ""
    """Content that should be persisted (if extraction succeeded)."""

    priority_score: float = 0.0
    """Priority score (0-1) for this persistence operation."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_persist": self.should_persist,
            "urgency": self.urgency.value,
            "detected_patterns": [p.to_dict() for p in self.detected_patterns],
            "persistence_rationale": self.persistence_rationale,
            "suggested_content": self.suggested_content,
            "priority_score": self.priority_score,
        }


@dataclass
class PersistenceResult:
    """Result of a persistence operation."""
    success: bool = False
    """Whether persistence succeeded."""

    memory_id: Optional[str] = None
    """ID of the persisted memory entry (if successful)."""

    status: PersistenceStatus = PersistenceStatus.PENDING
    """Current status of the persistence operation."""

    content_preview: str = ""
    """Preview of persisted content (truncated)."""

    learning_score: float = 0.0
    """Learning score assigned to the memory."""

    trigger: str = ""
    """What triggered this persistence (e.g., 'user_command', 'learning_spike')."""

    timestamp: float = field(default_factory=lambda: __import__('time').time())
    """When persistence was attempted."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "memory_id": self.memory_id,
            "status": self.status.value,
            "content_preview": self.content_preview,
            "learning_score": self.learning_score,
            "trigger": self.trigger,
            "timestamp": self.timestamp,
        }


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    check_name: str = ""
    """Name of the validation check."""

    passed: bool = False
    """Whether the check passed."""

    details: str = ""
    """Details about the check result."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class PersistenceValidation:
    """
    Validation that a memory was successfully persisted.

    Performs multiple checks to confirm the memory exists in LTM
    and matches the expected content.
    """
    is_validated: bool = False
    """Whether persistence was confirmed."""

    memory_found: bool = False
    """Whether the memory exists in LTM."""

    content_matches: bool = False
    """Whether content matches what was intended."""

    validation_checks: List[ValidationCheck] = field(default_factory=list)
    """Individual validation check results."""

    memory_id: Optional[str] = None
    """ID of the validated memory (if found)."""

    validation_rationale: str = ""
    """Human-readable validation summary."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_validated": self.is_validated,
            "memory_found": self.memory_found,
            "content_matches": self.content_matches,
            "validation_checks": [c.to_dict() for c in self.validation_checks],
            "memory_id": self.memory_id,
            "validation_rationale": self.validation_rationale,
        }


@dataclass
class PersistenceFailure:
    """
    Logged record of a persistence failure.

    Captures details about why persistence failed for debugging
    and future policy improvement.
    """
    content_preview: str = ""
    """Preview of content that failed to persist."""

    failure_category: FailureCategory = FailureCategory.UNKNOWN
    """Category of the failure."""

    error_message: str = ""
    """Error message or description."""

    retry_count: int = 0
    """Number of retry attempts made."""

    recovery_action: str = ""
    """Recommended recovery action."""

    timestamp: float = field(default_factory=lambda: __import__('time').time())
    """When the failure occurred."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_preview": self.content_preview,
            "failure_category": self.failure_category.value,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "recovery_action": self.recovery_action,
            "timestamp": self.timestamp,
        }
