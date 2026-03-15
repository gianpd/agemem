"""
memory package
---------------
Memory storage modules for AgeMem.
"""

from memory.vector_index import (
    ensure_table_exists,
    insert_embedding,
    update_embedding,
    delete_embedding,
    query_similar,
    get_embedding_count,
    entry_exists,
)

from memory.retrieval import (
    retrieve_relevant_ltm,
    retrieve_by_tags,
    retrieve_recent,
)

from memory.embedding import (
    EmbeddingModule,
    embed_text,
    embed_batch,
    cosine_similarity,
)

# CONTEXT_AWARE_RETRIEVAL: Export context-aware retrieval classes
from memory.context_retrieval import (
    ContextAwareRetriever,
    ContextRetrievalConfig,
    retrieve_with_context,
)

# LTM_INTROSPECTION: Export LTM self-management toolkit
from memory.ltm_introspection_types import (
    # Enums
    DriftType,
    ConfidenceLevel,
    ExpectedValue,
    UrgencyLevel,
    RetrievalMode,
    FailureMode,
    MatchDimension,
    ConfidenceDimension,
    # Tier 1
    DriftReport,
    ConfidenceDimensionScore,
    ConfidenceReport,
    ReadinessAssessment,
    # Tier 2
    Paraphrase,
    RetrievedMemory,
    LTMInjection,
    # Tier 3
    MemoryValidationResult,
    ValidatedBatch,
    RetrievalAttempt,
    RefinedQuery,
    CompressedContext,
    Turn,
    # Tier 4
    RetrievalDecision,
    ConversationProfile,
    StrategyRecommendation,
    # Supporting
    AnchorSnapshot,
)

from memory.ltm_introspection import (
    # Tier 1
    assess_conversation_drift,
    self_assess_confidence,
    are_you_ready_to_get_in_context_ltm,
    # Tier 2
    paraphrase_for_coverage,
    trigger_contextual_ltm_retrieval,
    # Tier 3
    validate_ltm_relevance,
    refine_retrieval_target,
    compress_conversation_for_ltm,
    # Tier 4
    log_retrieval_decision,
    suggest_retrieval_strategy,
    # Utilities
    set_anchor_from_context,
    get_decision_history,
    clear_state,
)

# QUERY_EXPANSION: Export QueryExpander for external use
from tools.query_expansion import QueryExpander

# Introspection tool definitions for agent use
from memory.ltm_introspection_tools import introspection_tool_definitions

__all__ = [
    # Vector index
    "ensure_table_exists",
    "insert_embedding",
    "update_embedding",
    "delete_embedding",
    "query_similar",
    "get_embedding_count",
    "entry_exists",
    # Retrieval
    "retrieve_relevant_ltm",
    "retrieve_by_tags",
    "retrieve_recent",
    # Embedding
    "EmbeddingModule",
    "embed_text",
    "embed_batch",
    "cosine_similarity",
    # Context-aware retrieval
    "ContextAwareRetriever",
    "ContextRetrievalConfig",
    "retrieve_with_context",
    # LTM Introspection - Enums
    "DriftType",
    "ConfidenceLevel",
    "ExpectedValue",
    "UrgencyLevel",
    "RetrievalMode",
    "FailureMode",
    "MatchDimension",
    "ConfidenceDimension",
    # LTM Introspection - Tier 1 Types
    "DriftReport",
    "ConfidenceDimensionScore",
    "ConfidenceReport",
    "ReadinessAssessment",
    # LTM Introspection - Tier 2 Types
    "Paraphrase",
    "RetrievedMemory",
    "LTMInjection",
    # LTM Introspection - Tier 3 Types
    "MemoryValidationResult",
    "ValidatedBatch",
    "RetrievalAttempt",
    "RefinedQuery",
    "CompressedContext",
    "Turn",
    # LTM Introspection - Tier 4 Types
    "RetrievalDecision",
    "ConversationProfile",
    "StrategyRecommendation",
    # LTM Introspection - Supporting
    "AnchorSnapshot",
    # LTM Introspection - Tier 1 Tools
    "assess_conversation_drift",
    "self_assess_confidence",
    "are_you_ready_to_get_in_context_ltm",
    # LTM Introspection - Tier 2 Tools
    "paraphrase_for_coverage",
    "trigger_contextual_ltm_retrieval",
    # LTM Introspection - Tier 3 Tools
    "validate_ltm_relevance",
    "refine_retrieval_target",
    "compress_conversation_for_ltm",
    # LTM Introspection - Tier 4 Tools
    "log_retrieval_decision",
    "suggest_retrieval_strategy",
    # LTM Introspection - Utilities
    "set_anchor_from_context",
    "get_decision_history",
    "clear_state",
    # Query expansion
    "QueryExpander",
    # Introspection tool definitions
    "introspection_tool_definitions",
]
