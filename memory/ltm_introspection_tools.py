"""
memory/ltm_introspection_tools.py
─────────────────────────────────
OpenAI-compatible tool definitions for the LTM Self-Management Toolkit.

These are pure JSON schemas (no execution logic) to avoid circular imports.
Execution handlers live in agents/orchestrator.py with access to STM/LTM state.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1: State Assessment
# ═══════════════════════════════════════════════════════════════════════════════

ASSESS_CONVERSATION_DRIFT_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_conversation_drift",
        "description": (
            "Analyze the current conversation for topic drift compared to the static anchor. "
            "Returns a DriftReport with drift type, confidence scores, and continuity metrics. "
            "Call this when you suspect the conversation may have shifted topics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "current_query": {
                    "type": "string",
                    "description": "The most recent user message."
                },
                "recent_context": {
                    "type": "string",
                    "description": "Last 2-3 turns of conversation for comparison."
                }
            },
            "required": ["current_query"]
        }
    }
}

ARE_YOU_READY_TO_GET_IN_CONTEXT_LTM_TOOL = {
    "type": "function",
    "function": {
        "name": "are_you_ready_to_get_in_context_ltm",
        "description": (
            "Pre-flight check combining drift and confidence signals into a go/no-go recommendation. "
            "Returns a ReadinessAssessment with should_retrieve boolean and reasoning. "
            "Call this BEFORE attempting LTM retrieval to validate the need."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "current_query": {"type": "string"},
                "current_confidence": {
                    "type": "number",
                    "description": "Your current confidence level (0.0-1.0)."
                }
            },
            "required": ["current_query"]
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2: Retrieval Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

PARAPHRASE_FOR_COVERAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "paraphrase_for_coverage",
        "description": (
            "Generate semantic variants of a query to maximize search coverage. "
            "Creates Technical, Tutorial, and Troubleshooting variants. "
            "Returns a Paraphrase object with variants list. "
            "Use this when initial LTM retrieval returns few results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The original query to paraphrase."
                },
                "n_variants": {
                    "type": "integer",
                    "description": "Number of variants to generate (default 3, max 5).",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}

TRIGGER_CONTEXTUAL_LTM_RETRIEVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "trigger_contextual_ltm_retrieval",
        "description": (
            "Execute LTM retrieval with specified mode. "
            "Modes: 'single_query' (fast), 'multi_paraphrase' (thorough), 'anchored' (context-aware). "
            "Returns LTMInjection with retrieved memories and metadata. "
            "Call ONLY after are_you_ready_to_get_in_context_ltm returns should_retrieve=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["single_query", "multi_paraphrase", "anchored"],
                    "description": "Retrieval mode."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum memories to retrieve (default 5).",
                    "default": 5
                }
            },
            "required": ["query", "mode"]
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3: Validation & Refinement
# ═══════════════════════════════════════════════════════════════════════════════

VALIDATE_LTM_RELEVANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "validate_ltm_relevance",
        "description": (
            "Validate retrieved LTM entries for relevance, groundedness, and faithfulness. "
            "Returns a ValidatedBatch with pass/fail status per memory. "
            "Call this AFTER retrieval to ensure quality before using memories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "retrieved_memories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of retrieved memory contents to validate."
                },
                "current_query": {
                    "type": "string",
                    "description": "The query that produced these memories."
                }
            },
            "required": ["retrieved_memories", "current_query"]
        }
    }
}

REFINE_RETRIEVAL_TARGET_TOOL = {
    "type": "function",
    "function": {
        "name": "refine_retrieval_target",
        "description": (
            "If validation fails, generate a refined query strategy. "
            "Returns a RefinedQuery with failure classification and new strategy. "
            "Capped at 2 retries to prevent infinite loops. "
            "Call this when validate_ltm_relevance indicates poor quality."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "original_query": {"type": "string"},
                "failure_mode": {
                    "type": "string",
                    "enum": ["TOO_NARROW", "TOO_BROAD", "OFF_TOPIC", "STALE"],
                    "description": "Why the previous retrieval failed."
                }
            },
            "required": ["original_query", "failure_mode"]
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 4: Meta-Cognitive
# ═══════════════════════════════════════════════════════════════════════════════

LOG_RETRIEVAL_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "log_retrieval_decision",
        "description": (
            "Log a retrieval decision for future policy calibration. "
            "Records the decision chain (assessment → retrieval → validation). "
            "Returns confirmation with decision_id. "
            "Call this after ANY retrieval attempt (success or failure)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision_chain": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tools called in sequence."
                },
                "utility_score": {
                    "type": "number",
                    "description": "Self-assessed utility of this retrieval (0.0-1.0).",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "was_retrieval_skipped": {
                    "type": "boolean",
                    "description": "Whether retrieval was intentionally skipped."
                }
            },
            "required": ["decision_chain", "utility_score", "was_retrieval_skipped"]
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════

introspection_tool_definitions = [
    ASSESS_CONVERSATION_DRIFT_TOOL,
    ARE_YOU_READY_TO_GET_IN_CONTEXT_LTM_TOOL,
    PARAPHRASE_FOR_COVERAGE_TOOL,
    TRIGGER_CONTEXTUAL_LTM_RETRIEVAL_TOOL,
    VALIDATE_LTM_RELEVANCE_TOOL,
    REFINE_RETRIEVAL_TARGET_TOOL,
    LOG_RETRIEVAL_DECISION_TOOL,
]

__all__ = [
    "introspection_tool_definitions",
    "ASSESS_CONVERSATION_DRIFT_TOOL",
    "ARE_YOU_READY_TO_GET_IN_CONTEXT_LTM_TOOL",
    "PARAPHRASE_FOR_COVERAGE_TOOL",
    "TRIGGER_CONTEXTUAL_LTM_RETRIEVAL_TOOL",
    "VALIDATE_LTM_RELEVANCE_TOOL",
    "REFINE_RETRIEVAL_TARGET_TOOL",
    "LOG_RETRIEVAL_DECISION_TOOL",
]
