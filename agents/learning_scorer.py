"""
agents/learning_scorer.py
──────────────────────────
Collects LearningFeedback from the main agent after a turn.

Protocol
────────
After every LEARNING_SCORE_PROMPT_EVERY_N turns (or immediately on a
learning spike), the Orchestrator calls `collect()`.

The LLM is asked: "On a 0–1 scale, how much new, reusable information did
you just encounter?  Return JSON: {score, rationale, affected_content}."

Design note — why ask the *main* agent and not a separate call?
───────────────────────────────────────────────────────────────
The main agent has already processed the last exchange and has the
richest signal about what was novel.  A separate meta-call would duplicate
context.  We therefore append a lightweight self-assessment prompt to the
existing message list (without storing it in STM).
"""

from __future__ import annotations

from typing import Optional

from core.types import LearningFeedback
from core.config import AgememConfig, DEFAULT_CONFIG
from agents.llm_client import LLMClient
from agents.response_handler import ResponseHandler


_LEARNING_PROMPT = """\
Analyze the immediately preceding interaction to extract persistent memory artifacts. 

You MUST return ONLY a strictly valid JSON object. Do not include markdown formatting or conversational filler. 

Schema:
{
  "score": <float>,
  "rationale": "<string>",
  "affected_content": "<string>"
}

### 1. Deterministic Scoring Matrix
You MUST assign the "score" field exactly one of the following discrete values by evaluating these mutually exclusive conditions in descending order:

- IF the interaction contains explicit declarations of user attributes (e.g., Names, roles), permanent project architectures, file paths, or explicit user preferences ("I want", "always do X")
  -> ASSIGN score: 1.0

- IF the interaction establishes a temporary operational state required for the current workflow (e.g., a chosen algorithm, a specific target directory for this session, a transient constraint)
  -> ASSIGN score: 0.7

- IF the interaction contains inferred user goals without explicit declarations, OR mentions concepts that lack concrete operational parameters
  -> ASSIGN score: 0.4

- IF the interaction consists strictly of tool executions, procedural acknowledgments ("Done", "Understood"), formatting operations, or generic dialogue
  -> ASSIGN score: 0.0

### 2. Output Constraints

Rule A: "rationale"
- MUST be exactly ONE sentence.
- MUST explicitly state which scoring condition was triggered (e.g., "Explicit user file path declared.").

Rule B: "affected_content"
- IF score >= 0.7: MUST be an exact substring extraction (verbatim quote) of the newly established fact from the text. Maximum length: 50 words.
- IF score < 0.7: MUST be exactly an empty string "". 
"""


class LearningScorer:

    def __init__(
        self,
        llm: LLMClient,
        config: AgememConfig = DEFAULT_CONFIG,
    ) -> None:
        self._llm = llm
        self._config = config
        self._turns_since_last_collect: int = 0
        self._response_handler = ResponseHandler(llm, max_retries=2, enable_validation=True)

    def should_collect(self, turn_index: int) -> bool:
        """Returns True if it is time to ask for feedback this turn."""
        n = self._config.LEARNING_SCORE_PROMPT_EVERY_N
        return n > 0 and turn_index > 0 and turn_index % n == 0

    def collect(
        self,
        context_messages: list[dict],
        turn_index: int,
    ) -> Optional[LearningFeedback]:
        """
        Append the scoring prompt to the existing context and ask the LLM.
        Does NOT modify the caller's context list.

        Returns None on failure so callers can safely ignore errors.
        """
        probe_messages = list(context_messages) + [
            {"role": "user", "content": _LEARNING_PROMPT}
        ]
        print(f"[DEBUG] LearningScorer: Collecting feedback at turn {turn_index}...", flush=True)
        try:
            raw, metrics = self._response_handler.chat_json_with_recovery(
                messages=probe_messages,
                model=self._config.LEARNING_SCORER_MODEL,
                max_tokens=self._config.LEARNING_SCORER_MAX_TOKENS,
            )
            # Handle case where LLM returns just a float instead of a dict
            if isinstance(raw, (int, float)):
                score = max(0.0, min(1.0, float(raw)))
                rationale = ""
                affected = ""
            elif isinstance(raw, dict):
                score = max(0.0, min(1.0, float(raw.get("score", 0.0))))
                rationale = raw.get("rationale", "")
                affected = raw.get("affected_content", "")[:80]
            else:
                score = 0.0
                rationale = ""
                affected = ""
            print(f"[DEBUG] LearningScorer: score={score:.2f}, rationale='{rationale[:50]}...', content='{affected}...'", flush=True)
            return LearningFeedback(
                score=score,
                rationale=rationale,
                affected_content=affected,
                turn_index=turn_index,
            )
        except Exception as e:
            print(f"[DEBUG] LearningScorer.collect() failed: {e}", flush=True)
            # Log metrics for debugging
            if hasattr(self._response_handler, 'get_metrics'):
                recent_metrics = self._response_handler.get_metrics()[-3:]  # Last 3 attempts
                for m in recent_metrics:
                    print(f"[DEBUG] Response metrics: type={m.response_type.value}, latency={m.latency_ms:.0f}ms, quality={m.quality_score:.2f}", flush=True)
            return None
