"""
ltm_probe.py
────────────
Interactive LTM Probe for NODE 01.
Runs 7 simulated turns with properly mocked LLM.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from core.config import AgememConfig
from core.types import MemoryOp
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator
from memory.ltm_store import LTMStore


@dataclass
class TurnObservation:
    turn_index: int
    user_input: str
    assistant_response: str
    stm_tokens: int
    stm_utilization: float
    ltm_entries_count: int
    learning_score: Optional[float] = None
    learning_rationale: str = ""
    memory_agent_rationale: str = ""
    ops_applied: list[dict] = field(default_factory=list)
    feedback_collected: bool = False
    bugs_observed: list[str] = field(default_factory=list)


def create_mock_llm_with_responses(responses: list[str], json_responses: list[str]) -> LLMClient:
    """Create LLMClient with mock responses that support chat and chat_json."""
    mock_client = MagicMock()
    text_call_count = [0]
    json_call_count = [0]

    def side_effect(**kwargs):
        is_json = kwargs.get('response_format', {}).get('type') == 'json_object'
        if is_json:
            resp = json_responses[min(json_call_count[0], len(json_responses) - 1)]
            json_call_count[0] += 1
        else:
            resp = responses[min(text_call_count[0], len(responses) - 1)]
            text_call_count[0] += 1
        choice = MagicMock()
        choice.message.content = resp
        return MagicMock(
            choices=[choice],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5)
        )

    mock_client.chat.completions.create.side_effect = side_effect
    return LLMClient(mock_client, default_model="test-model"), (text_call_count, json_call_count)


def run_probe():
    """Run 7-turn probe and report results."""
    print("=" * 60)
    print("AgeMem-Hybrid LTM Verification Probe")
    print("=" * 60)

    # Prepare responses for 7 turns
    # Text responses for main chat
    text_responses = [
        "Hello! I'm your assistant. I can help with various tasks.",
        "I understand. Let me help you with that project.",
        "That's interesting information about your work.",
        "I've noted your preferences for future reference.",
        "Based on what you've shared, here's my recommendation.",
        "I remember you mentioned that earlier. Let me build on it.",
        "Let me recall what we discussed previously.",
    ]

    # JSON responses for learning scorer (turns 3 and 6)
    json_responses = [
        json.dumps({"score": 0.8, "rationale": "User introduced themselves", "affected_content": "Marco is a civil engineer working on bridge restoration"}),
        json.dumps({"score": 0.9, "rationale": "Important project details", "affected_content": "Historic bridge restoration in Florence, 6 month deadline"}),
    ]

    llm, (text_count, json_count) = create_mock_llm_with_responses(text_responses, json_responses)

    cfg = AgememConfig(
        DEFAULT_MODEL="test-model",
        MEMORY_AGENT_MODEL="test-model",
        STM_TOKEN_LIMIT=2000,
        STM_WARNING_THRESHOLD=0.75,
        STM_CRITICAL_THRESHOLD=0.90,
        LTM_PROMOTE_THRESHOLD=0.65,
        LEARNING_SCORE_PROMPT_EVERY_N=3,
        TRIGGER_EVERY_N_TURNS=5,
        DEFAULT_MAX_TOKENS=200,
        DEFAULT_TEMPERATURE=0.2,
        PERSIST_DIR=None,  # No persistence for clean test
    )

    ltm_store = LTMStore(cfg, persist_path=None)
    orch = Orchestrator(llm=llm, config=cfg, ltm_store=ltm_store)

    test_inputs = [
        "Hi, I'm Marco and I work as a civil engineer on bridge restoration projects.",
        "I'm currently preparing a bid for a historic bridge restoration in Florence.",
        "What are the SOA requirements for participating in public tenders in Italy?",
        "I prefer working with reinforced concrete and have 15 years of experience.",
        "The project deadline is tight - only 6 months for full restoration.",
        "Can you help me research best practices for historic bridge preservation?",
        "Please remember that I always work with the same team of 5 specialists.",
    ]

    observations = []

    print(f"\nRunning {len(test_inputs)} simulated turns...\n")

    for turn_num, user_input in enumerate(test_inputs, 1):
        print(f"--- Turn {turn_num} ---")
        print(f"Input: {user_input[:60]}...")

        response = orch.chat(user_input)
        trace = orch.last_trace()
        stats = trace.stm_stats_after if trace else orch.stm_stats()
        ltm_count = len(orch.ltm_snapshot())

        feedback = trace.feedback if trace else None
        learning_score = feedback.score if feedback else None
        learning_rationale = feedback.rationale if feedback else ""
        ma_rationale = trace.memory_agent_rationale if trace else ""

        ops = trace.ops_applied if trace else []
        ops_list = [
            {
                "op": op.op.value if hasattr(op.op, 'value') else str(op.op),
                "success": op.success,
                "trigger": op.trigger.value if hasattr(op.trigger, 'value') else str(op.trigger),
            }
            for op in ops
        ]

        bugs = []

        # Check learning score collection at turns 3, 6
        if turn_num in [3, 6]:
            if feedback is None:
                bugs.append(f"LTM-05 FAIL: Learning score should collect at turn {turn_num} but returned None")
            elif learning_score < 0.65:
                bugs.append(f"LTM-04 NOTE: Score {learning_score:.2f} below threshold, no spike trigger")

        # Check LTM add on high scores
        if feedback and learning_score >= 0.65:
            add_found = any(op.get('op') == 'add' for op in ops_list)
            if not add_found:
                bugs.append(f"LTM-06 FAIL: Score {learning_score:.2f} >= 0.65 but no LTM ADD")

        obs = TurnObservation(
            turn_index=turn_num,
            user_input=user_input,
            assistant_response=response,
            stm_tokens=stats.total_tokens,
            stm_utilization=stats.utilisation_ratio,
            ltm_entries_count=ltm_count,
            learning_score=learning_score,
            learning_rationale=learning_rationale,
            memory_agent_rationale=ma_rationale,
            ops_applied=ops_list,
            feedback_collected=feedback is not None,
            bugs_observed=bugs,
        )
        observations.append(obs)

        print(f"  STM: {stats.utilisation_ratio:.0%} ({stats.total_tokens} tokens)")
        print(f"  LTM entries: {ltm_count}")
        if learning_score:
            print(f"  Learning: {learning_score:.2f} - {learning_rationale[:40]}...")
        print(f"  Ops: {[op['op'] for op in ops_list]}")
        if bugs:
            for bug in bugs:
                print(f"  ⚠️ {bug}")
        print()

    # Final report
    print("=" * 60)
    print("PROBE COMPLETE")
    print("=" * 60)
    print(f"\nTotal turns: {len(observations)}")
    print(f"Final LTM entries: {len(orch.ltm_snapshot())}")
    print(f"Total LLM calls: text={text_count[0]}, json={json_count[0]}")

    all_bugs = [bug for obs in observations for bug in obs.bugs_observed]
    print(f"Bugs found: {len(all_bugs)}")
    for bug in all_bugs:
        print(f"  - {bug}")

    # Write report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_turns": len(observations),
        "final_ltm_entries": len(orch.ltm_snapshot()),
        "total_text_calls": text_count[0],
        "total_json_calls": json_count[0],
        "bugs": all_bugs,
        "observations": [
            {
                "turn": obs.turn_index,
                "stm_util": obs.stm_utilization,
                "ltm_count": obs.ltm_entries_count,
                "learning_score": obs.learning_score,
                "ops": obs.ops_applied,
                "bugs": obs.bugs_observed,
            }
            for obs in observations
        ]
    }

    with open("probe_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote probe_report.json")

    return 0 if not all_bugs else 1


if __name__ == "__main__":
    sys.exit(run_probe())
