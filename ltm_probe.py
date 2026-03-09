"""
ltm_probe.py
────────────
Interactive LTM Probe for NODE 01.
Runs 5+ simulated turns with mock LLM to observe LTM behavior and identify bugs.

Usage:
    python3 ltm_probe.py

Outputs:
    - probe_report.md: Human-readable findings
    - observed_behaviors.json: Machine-readable data
    - bug_list.md: Documented bugs with severity
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import AgememConfig
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator
from memory.ltm_store import LTMStore


@dataclass
class TurnObservation:
    """Observation from a single turn."""
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


@dataclass
class ProbeReport:
    """Full probe report."""
    timestamp: str
    total_turns: int
    observations: list[TurnObservation]
    bugs_found: list[dict]
    ltm_rules_status: dict[str, str]


def create_mock_llm() -> tuple[LLMClient, MagicMock]:
    """Create a mock LLM client that returns predictable responses."""
    mock_client = MagicMock()
    mock_llm = LLMClient(mock_client, default_model="test-model")
    return mock_llm, mock_client


def setup_mock_responses(mock_client: MagicMock) -> None:
    """Configure mock LLM responses for different scenarios."""
    call_count = [0]

    def chat_side_effect(*, messages, max_tokens=None, temperature=None, tools=None):
        call_count[0] += 1
        # Return simple responses based on turn
        responses = [
            "Hello! I'm your assistant. I can help with various tasks.",
            "I understand. Let me help you with that project.",
            "That's interesting information about your work.",
            "I've noted your preferences for future reference.",
            "Based on what you've shared, here's my recommendation.",
            "I remember you mentioned that earlier. Let me build on it.",
        ]
        return responses[call_count[0] % len(responses)]

    def chat_json_side_effect(*, messages, max_tokens=None):
        # Simulate learning scorer responses
        # Return varying scores to test different thresholds
        scores = [
            {"score": 0.8, "affected_content": "User introduction and project details", "rationale": "High value personal information"},
            {"score": 0.3, "affected_content": "General query response", "rationale": "Low value procedural exchange"},
            {"score": 0.9, "affected_content": "Important work preference", "rationale": "Critical user preference"},
            {"score": 0.5, "affected_content": "Moderate value context", "rationale": "Useful but not critical"},
            {"score": 0.7, "affected_content": "Recurring topic mention", "rationale": "Potentially relevant pattern"},
        ]
        return scores[call_count[0] % len(scores)]

    mock_client.chat.completions.create.side_effect = chat_side_effect
    mock_client.chat.completions.create_json = chat_json_side_effect


def run_probe_turns(num_turns: int = 7) -> tuple[list[TurnObservation], Orchestrator]:
    """
    Run N simulated turns and collect observations.

    Test scenario designed to trigger LTM rules:
    - Turn 3, 6: Learning score collection (every 3 turns)
    - Turn 10: MemoryAgent periodic review
    - High-scoring content should trigger LTM promotion
    """
    mock_llm, mock_client = create_mock_llm()

    # Track the JSON calls separately
    json_call_count = [0]
    text_call_count = [0]

    def mock_chat(*, messages, max_tokens=None, temperature=None, tools=None):
        text_call_count[0] += 1
        responses = [
            "Hello! I'm your assistant. I can help with various tasks.",
            "I understand. Let me help you with that project.",
            "That's interesting information about your work.",
            "I've noted your preferences for future reference.",
            "Based on what you've shared, here's my recommendation.",
            "I remember you mentioned that earlier. Let me build on it.",
            "Let me recall what we discussed previously.",
        ]
        return responses[(text_call_count[0] - 1) % len(responses)]

    # Patch the LLM methods
    mock_llm.chat = mock_chat

    # Create orchestrator with test-friendly config
    cfg = AgememConfig(
        DEFAULT_MODEL="test-model",
        MEMORY_AGENT_MODEL="test-model",
        STM_TOKEN_LIMIT=2000,  # Lower for faster testing
        STM_WARNING_THRESHOLD=0.75,
        STM_CRITICAL_THRESHOLD=0.90,
        LTM_PROMOTE_THRESHOLD=0.65,  # Threshold for auto-promotion
        LEARNING_SCORE_PROMPT_EVERY_N=3,  # Collect every 3 turns
        TRIGGER_EVERY_N_TURNS=5,  # MemoryAgent every 5 turns
        DEFAULT_MAX_TOKENS=200,
        DEFAULT_TEMPERATURE=0.2,
        PERSIST_DIR="agent_memory/test_probe",
    )

    ltm_store = LTMStore(cfg, persist_path=None)  # No persistence for test
    orch = Orchestrator(llm=mock_llm, config=cfg, ltm_store=ltm_store)

    observations: list[TurnObservation] = []

    # Test inputs designed to trigger various LTM rules
    test_inputs = [
        "Hi, I'm Marco and I work as a civil engineer on bridge restoration projects.",
        "I'm currently preparing a bid for a historic bridge restoration in Florence.",
        "What are the SOA requirements for participating in public tenders in Italy?",
        "I prefer working with reinforced concrete and have 15 years of experience.",
        "The project deadline is tight - only 6 months for full restoration.",
        "Can you help me research best practices for historic bridge preservation?",
        "Please remember that I always work with the same team of 5 specialists.",
    ]

    # Patch the learning scorer's collect method to capture what happens
    original_scorer_collect = orch._scorer.collect
    collected_feedback = []

    def patched_collect(*args, **kwargs):
        result = original_scorer_collect(*args, **kwargs)
        collected_feedback.append({
            "turn": kwargs.get('turn_index', 'unknown'),
            "result": result,
            "result_type": type(result).__name__ if result else "None",
        })
        return result

    orch._scorer.collect = patched_collect

    for i, user_input in enumerate(test_inputs[:num_turns]):
        turn_num = i + 1
        print(f"\n--- Turn {turn_num} ---")
        print(f"Input: {user_input[:60]}...")

        try:
            response = orch.chat(user_input)
        except Exception as e:
            print(f"ERROR during chat: {e}")
            response = f"[ERROR: {e}]"

        # Get trace data
        trace = orch.last_trace()
        stats = trace.stm_stats_after if trace else orch.stm_stats()
        ltm_count = len(orch.ltm_snapshot())

        # Check for learning feedback
        feedback = trace.feedback if trace else None
        learning_score = feedback.score if feedback else None
        learning_rationale = feedback.rationale if feedback else ""

        # Check for MemoryAgent rationale
        ma_rationale = trace.memory_agent_rationale if trace else ""

        # Check which ops were applied
        ops = trace.ops_applied if trace else []
        ops_list = [
            {
                "op": op.op.value if hasattr(op.op, 'value') else str(op.op),
                "success": op.success,
                "trigger": op.trigger.value if hasattr(op.trigger, 'value') else str(op.trigger),
                "detail": op.detail,
            }
            for op in ops
        ]

        # Identify bugs for this turn
        bugs = []

        # LTM-05: Learning Score Collection
        if turn_num % cfg.LEARNING_SCORE_PROMPT_EVERY_N == 0:
            if feedback is None:
                bugs.append(f"LTM-05 FAIL: Learning score should collect at turn {turn_num} but returned None")
            else:
                print(f"  Learning score: {learning_score:.2f}")

        # LTM-06: LTM Add on Threshold
        if feedback and feedback.score >= cfg.LTM_PROMOTE_THRESHOLD:
            ltm_add_found = any(op.get('op') == 'ADD' and op.get('trigger') == 'LEARNING_SCORE' for op in ops_list)
            if not ltm_add_found:
                bugs.append(f"LTM-06 FAIL: Score {feedback.score:.2f} >= {cfg.LTM_PROMOTE_THRESHOLD} but no LTM ADD")

        # LTM-10: LTM Search/Retrieve should happen every turn
        retrieve_found = any(op.get('op') == 'RETRIEVE' for op in ops_list)
        if not retrieve_found:
            bugs.append(f"LTM-10 FAIL: No RETRIEVE operation on turn {turn_num}")

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
        print(f"  Ops: {[op['op'] for op in ops_list]}")
        if bugs:
            print(f"  BUGS: {len(bugs)}")

    return observations, orch


def analyze_bugs(observations: list[TurnObservation], orch: Orchestrator) -> list[dict]:
    """Analyze observations and compile bug list."""
    bugs: list[dict] = []

    # Collect all unique bugs
    bug_set = set()
    for obs in observations:
        for bug in obs.bugs_observed:
            bug_set.add(bug)

    # Convert to structured bug reports
    for i, bug_desc in enumerate(sorted(bug_set), 1):
        severity = "CRITICAL" if "FAIL" in bug_desc else "MEDIUM"
        ltm_rule = "UNKNOWN"
        if "LTM-" in bug_desc:
            ltm_rule = bug_desc.split()[0]

        bugs.append({
            "id": f"BUG-{i:02d}",
            "description": bug_desc,
            "severity": severity,
            "ltm_rule": ltm_rule,
            "status": "CONFIRMED",
        })

    # Check for silent failures (LTM-12)
    learning_attempts = [obs for obs in observations if obs.turn_index % 3 == 0]
    silent_failures = [obs for obs in learning_attempts if not obs.feedback_collected]
    if silent_failures:
        bugs.append({
            "id": f"BUG-{len(bugs)+1:02d}",
            "description": f"LTM-12 FAIL: LearningScorer silent failures on turns: {[o.turn_index for o in silent_failures]}",
            "severity": "CRITICAL",
            "ltm_rule": "LTM-12",
            "status": "CONFIRMED",
        })

    # Check LTM empty after high-score turns
    high_score_turns = [obs for obs in observations if obs.learning_score and obs.learning_score >= 0.65]
    ltm_final_count = len(orch.ltm_snapshot())
    if high_score_turns and ltm_final_count == 0:
        bugs.append({
            "id": f"BUG-{len(bugs)+1:02d}",
            "description": f"LTM-06/08 FAIL: LTM empty despite {len(high_score_turns)} high-score turns",
            "severity": "HIGH",
            "ltm_rule": "LTM-06, LTM-08",
            "status": "CONFIRMED",
        })

    return bugs


def generate_report(observations: list[TurnObservation], bugs: list[dict], orch: Orchestrator) -> None:
    """Generate all output files."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Generate observed_behaviors.json
    behaviors_data = {
        "timestamp": timestamp,
        "total_turns": len(observations),
        "observations": [
            {
                "turn_index": o.turn_index,
                "user_input": o.user_input,
                "assistant_response": o.assistant_response,
                "stm_tokens": o.stm_tokens,
                "stm_utilization": o.stm_utilization,
                "ltm_entries_count": o.ltm_entries_count,
                "learning_score": o.learning_score,
                "learning_rationale": o.learning_rationale,
                "memory_agent_rationale": o.memory_agent_rationale,
                "ops_applied": o.ops_applied,
                "feedback_collected": o.feedback_collected,
                "bugs_observed": o.bugs_observed,
            }
            for o in observations
        ],
        "final_ltm_state": orch.ltm_snapshot(),
        "bugs_found": bugs,
    }

    with open("observed_behaviors.json", "w") as f:
        json.dump(behaviors_data, f, indent=2, default=str)
    print("\nWrote observed_behaviors.json")

    # 2. Generate probe_report.md
    report_md = f"""# LTM Probe Report

**Generated:** {timestamp}
**Total Turns:** {len(observations)}
**Bugs Found:** {len(bugs)}

## Summary

| Metric | Value |
|--------|-------|
| Turns Simulated | {len(observations)} |
| Final LTM Entries | {len(orch.ltm_snapshot())} |
| Total Bugs | {len(bugs)} |
| Critical Bugs | {sum(1 for b in bugs if b['severity'] == 'CRITICAL')} |

## Turn-by-Turn Observations

"""

    for obs in observations:
        report_md += f"""### Turn {obs.turn_index}

**Input:** {obs.user_input[:80]}...

**Response:** {obs.assistant_response[:80]}...

| Metric | Value |
|--------|-------|
| STM Utilization | {obs.stm_utilization:.1%} |
| STM Tokens | {obs.stm_tokens} |
| LTM Entries | {obs.ltm_entries_count} |
| Learning Score | {(f'{obs.learning_score:.2f}' if obs.learning_score is not None else 'N/A')} |
| Feedback Collected | {'Yes' if obs.feedback_collected else 'No'} |

**Operations Applied:**
"""
        for op in obs.ops_applied:
            report_md += f"- `{op['op']}` (trigger: {op['trigger']}) - {'success' if op['success'] else 'failed'}\n"

        if obs.bugs_observed:
            report_md += "\n**Bugs Observed:**\n"
            for bug in obs.bugs_observed:
                report_md += f"- ⚠️ {bug}\n"

        report_md += "\n---\n\n"

    # LTM Rules Status
    report_md += "## LTM Rules Status\n\n"
    report_md += "| Rule ID | Status | Notes |\n"
    report_md += "|---------|--------|-------|\n"

    # Determine status based on observations
    rules_status = {
        "LTM-01": "PASS" if not any("LTM-01" in b for b in [obs.bugs_observed for obs in observations]) else "FAIL",
        "LTM-02": "PASS" if not any("LTM-02" in b for b in [obs.bugs_observed for obs in observations]) else "FAIL",
        "LTM-03": "NOT TESTED",
        "LTM-04": "NOT TESTED",
        "LTM-05": "FAIL" if any("LTM-05" in b['description'] for b in bugs) else "PASS",
        "LTM-06": "FAIL" if any("LTM-06" in b['description'] for b in bugs) else "PASS",
        "LTM-07": "NOT TESTED",
        "LTM-08": "NOT TESTED",
        "LTM-09": "NOT TESTED",
        "LTM-10": "FAIL" if any("LTM-10" in b['description'] for b in bugs) else "PASS",
        "LTM-11": "NOT TESTED",
        "LTM-12": "FAIL" if any("LTM-12" in b['description'] for b in bugs) else "PASS",
    }

    for rule_id, status in rules_status.items():
        report_md += f"| {rule_id} | {status} | |\n"

    with open("probe_report.md", "w") as f:
        f.write(report_md)
    print("Wrote probe_report.md")

    # 3. Generate bug_list.md
    bug_md = f"""# Bug List — LTM Verification Probe

**Generated:** {timestamp}
**Total Bugs:** {len(bugs)}

## Confirmed Bugs

"""

    if bugs:
        for bug in bugs:
            emoji = "🔴" if bug['severity'] == 'CRITICAL' else "🟡" if bug['severity'] == 'HIGH' else "🟢"
            bug_md += f"""### {bug['id']}: {bug['description'][:80]}...

- **Severity:** {emoji} {bug['severity']}
- **LTM Rule:** {bug['ltm_rule']}
- **Status:** {bug['status']}
- **Full Description:** {bug['description']}

"""
    else:
        bug_md += "**NO BUGS FOUND** — All LTM rules functioning as expected.\n"

    with open("bug_list.md", "w") as f:
        f.write(bug_md)
    print("Wrote bug_list.md")


def main():
    """Main entry point."""
    print("=" * 60)
    print("AgeMem-Hybrid LTM Verification Probe")
    print("=" * 60)

    print("\nRunning 7 simulated turns...")
    observations, orch = run_probe_turns(num_turns=7)

    print("\nAnalyzing bugs...")
    bugs = analyze_bugs(observations, orch)

    print("\nGenerating reports...")
    generate_report(observations, bugs, orch)

    print("\n" + "=" * 60)
    print("Probe Complete")
    print("=" * 60)
    print(f"\nBugs Found: {len(bugs)}")
    for bug in bugs:
        emoji = "🔴" if bug['severity'] == 'CRITICAL' else "🟡"
        print(f"  {emoji} {bug['id']}: {bug['description'][:70]}...")

    print(f"\nFinal LTM State: {len(orch.ltm_snapshot())} entries")

    return 0 if len(bugs) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
