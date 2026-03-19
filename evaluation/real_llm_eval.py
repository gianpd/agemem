#!/usr/bin/env python3
"""
evaluation/real_llm_eval.py
---------------------------
Run a single query evaluation with real LLM integration (no mock).

Usage:
    python evaluation/real_llm_eval.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator
from evaluation.factory import OrchestratorFactory
from core.config import BASE_URL, MODEL_NAME, MAX_TOKENS, TEMPERATURE
from core.llm_factory import LLMClientFactory


def load_one_query(dataset_path: Path) -> dict:
    """Load the first query from the dataset."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data[0][:3]


def run_evaluation():
    """Run a single query evaluation with real LLM."""
    print("=" * 60)
    print("REAL LLM EVALUATION - Single Query Test")
    print("=" * 60)

    # Load dataset
    dataset_path = Path(__file__).parent / "data" / "longmemeval_oracle.json"
    print(f"\nLoading query from: {dataset_path}")

    instance = load_one_query(dataset_path)

    question_id = instance["question_id"]
    question_text = instance["question"]
    expected_answer = instance["answer"]
    question_type = instance["question_type"]

    print(f"\nQuery ID: {question_id}")
    print(f"Type: {question_type}")
    print(f"Question: {question_text}")
    print(f"Expected Answer: {expected_answer}")

    # Count sessions and turns
    sessions = instance.get("haystack_sessions", [])
    total_turns = sum(len(s) for s in sessions)
    print(f"\nSessions: {len(sessions)}, Total turns: {total_turns}")

    # Create temp directory for this evaluation
    temp_dir = tempfile.mkdtemp(prefix="agemem_real_eval_")
    persist_dir = Path(temp_dir)
    print(f"Persist dir: {persist_dir}")

    # Initialize real LLM client using unified factory
    print(f"\nInitializing real LLM client...")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Model: {MODEL_NAME}")

    llm_client = LLMClientFactory().create()

    # Build orchestrator
    print("\nBuilding orchestrator...")
    orchestrator = OrchestratorFactory().build_for_evaluation(
        llm_client=llm_client,
        persist_dir=persist_dir,
        config_overrides={
            "STM_TOKEN_LIMIT": 8000,
            "LTM_PROMOTE_THRESHOLD": 0.5,
        },
    )

    # Phase 1: Replay sessions to build up memory
    print("\n" + "=" * 60)
    print("PHASE 1: REPLAYING SESSIONS")
    print("=" * 60)

    session_ids = instance.get("haystack_session_ids", [])
    for idx, session in enumerate(sessions):
        session_id = session_ids[idx] if idx < len(session_ids) else f"session_{idx}"
        print(f"\n--- Session {idx + 1}/{len(sessions)} ({session_id}) ---")

        for turn_idx, turn in enumerate(session):
            role = turn["role"]
            content = turn["content"]
            has_answer = turn.get("has_answer", False)

            if role == "user":
                print(f"  Turn {turn_idx + 1}: User (has_answer={has_answer})")

                # Send through orchestrator
                t0 = time.time()
                response = orchestrator.chat(content)
                latency_ms = (time.time() - t0) * 1000

                print(f"    Response time: {latency_ms:.0f}ms")
                print(f"    Response preview: {response[:100]}...")

                # Check if this turn contains the answer
                if has_answer:
                    print(f"    *** THIS TURN CONTAINS ANSWER EVIDENCE ***")

    # Get STM/LTM stats after replay
    last_trace = orchestrator.last_trace()
    if last_trace:
        print(f"\nSTM stats after replay:")
        print(f"  Before: {last_trace.stm_stats_before}")
        print(f"  After: {last_trace.stm_stats_after}")

    # Phase 2: Ask the test question
    print("\n" + "=" * 60)
    print("PHASE 2: EVALUATING TEST QUESTION")
    print("=" * 60)

    print(f"\nQuestion: {question_text}")
    print(f"Expected: {expected_answer}")

    t0 = time.time()
    response = orchestrator.chat(question_text)
    latency_ms = (time.time() - t0) * 1000

    print(f"\nResponse ({latency_ms:.0f}ms):")
    print("-" * 40)
    print(response)
    print("-" * 40)

    # Check if answer is in response
    response_lower = response.lower()
    expected_lower = expected_answer.lower()

    # Simple match
    is_correct = expected_lower in response_lower

    # Token overlap match
    if not is_correct:
        expected_tokens = set(expected_lower.split())
        response_tokens = set(response_lower.split())
        overlap = len(expected_tokens & response_tokens)
        overlap_ratio = overlap / len(expected_tokens) if expected_tokens else 0
        is_correct = overlap_ratio >= 0.7
        print(f"\nToken overlap: {overlap_ratio:.0%}")

    # Get final trace
    final_trace = orchestrator.last_trace()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Question ID: {question_id}")
    print(f"Question Type: {question_type}")
    print(f"Latency: {latency_ms:.0f}ms")
    print(f"Correct: {is_correct}")

    if final_trace:
        print(f"\nMemory ops applied:")
        for op in final_trace.ops_applied:
            print(f"  - {op.op.value}: success={op.success}")

        print(f"\nSTM stats:")
        print(f"  Token count: {final_trace.stm_stats_after.token_count}")

    # Print LLM usage
    stats = llm_client.usage_stats()
    print(f"\nLLM calls: {stats['total_calls']}")

    # Cleanup
    import shutil
    shutil.rmtree(persist_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return is_correct


if __name__ == "__main__":
    try:
        result = run_evaluation()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)