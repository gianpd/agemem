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


_LEARNING_PROMPT = """\
You just responded to a user. Now evaluate your own response from a memory perspective.

Return ONLY valid JSON with these fields:
{
  "score": <float 0.0 to 1.0>,
  "rationale": "<one sentence>",
  "affected_content": "<quote the specific new fact or concept to remember>"
}

Scoring guide:
  1.0  — Highly novel, specific, reusable fact (e.g. user's name, project details, preference)
  0.7  — Useful context likely needed later in this session
  0.4  — Potentially relevant but uncertain
  0.1  — Routine exchange, no new persistent knowledge
  0.0  — Pure procedural step, nothing to retain

IMPORTANT:
- If score >= 0.65: affected_content MUST contain the specific fact/concept (truncated if needed)
- If score < 0.65: affected_content can be brief or empty
- Never return empty affected_content when score is high

Be honest and calibrated. Do not inflate scores.
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
            raw = self._llm.chat_json(
                messages=probe_messages,
                max_tokens=200,
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
            return None
