"""
evaluation/quick_test.py
------------------------
Quick evaluation test - 3 LLM calls from a single session.

This is a minimal test that loads one LongMemEval instance and makes
3 quick LLM calls to verify basic functionality without loading
the full orchestrator.

Usage:
    python evaluation/quick_test.py
"""

import json
import time
from pathlib import Path

# Try to use the mock LLM for quick testing
try:
    from evaluation.mock_llm import StatefulMockLLM
    HAS_MOCK = True
except ImportError:
    HAS_MOCK = False


def format_history(sessions, dates):
    """Format sessions into readable history string."""
    lines = []
    for session, date in zip(sessions, dates):
        lines.append(f"\n--- Session on {date} ---")
        for turn in session:
            lines.append(f"{turn['role'].upper()}: {turn['content']}")
    return "\n".join(lines)


def quick_test(dataset_path: str = "evaluation/data/longmemeval_s_cleaned.json"):
    """
    Run a quick test with 3 LLM calls from a single session.

    This mimics the pattern from the user's example but uses
    the mock LLM for deterministic testing.
    """
    print("=" * 60)
    print("Quick Evaluation Test - 3 LLM Calls")
    print("=" * 60)

    # 1. Load dataset and pick one instance
    print(f"\n[1] Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instance = data[0]  # Just grab the first instance
    question_id = instance.get("question_id", "unknown")
    print(f"    Loaded instance: {question_id}")
    print(f"    Question: {instance['question'][:60]}...")
    print(f"    Answer: {instance['answer'][:60]}...")
    print(f"    Sessions: {len(instance['haystack_sessions'])}")
    print(f"    Question type: {instance.get('question_type', 'unknown')}")

    # 2. Format the chat history
    print("\n[2] Formatting chat history...")
    history_text = format_history(
        instance["haystack_sessions"],
        instance["haystack_dates"]
    )
    print(f"    History length: {len(history_text)} chars")
    print(f"    (~{len(history_text) // 4} tokens estimated)")

    # 3. Make 3 LLM calls
    print("\n[3] Making 3 LLM calls...")

    if HAS_MOCK:
        # Use mock LLM
        mock = StatefulMockLLM(strategy="template")
        mock.add_response_template("commute", "Your commute is about 30 minutes by train.")
        mock.add_response_template("work", "You work at TechCorp as a software engineer.")
        mock.add_response_template("name", "Your name is Alex.")

        def ask(question, history):
            messages = [
                {"role": "system", "content": "You are a helpful assistant.\n\n" + history},
                {"role": "user", "content": question}
            ]
            return mock.chat(messages)
    else:
        # Fallback to echo
        def ask(question, history):
            return f"[ECHO] Received question: {question[:40]}..."

    questions = [
        ("benchmark", instance["question"]),
        ("probe", "Summarize what you know about this user."),
        ("recency", "What is the most recent thing the user mentioned?"),
    ]

    results = []
    for qtype, qtext in questions:
        t0 = time.time()
        response = ask(qtext, history_text)
        latency_ms = (time.time() - t0) * 1000

        print(f"\n    [{qtype.upper()}]")
        print(f"    Q: {qtext[:60]}...")
        print(f"    A: {response[:80]}...")
        print(f"    Latency: {latency_ms:.1f}ms")

        results.append({
            "type": qtype,
            "question": qtext,
            "response": response,
            "latency_ms": latency_ms,
        })

    # 4. Compare against ground truth
    print("\n[4] Ground Truth Comparison")
    print(f"    Expected Answer: {instance['answer']}")

    # Simple answer matching
    benchmark_response = results[0]["response"].lower()
    expected_answer = instance["answer"].lower()

    if expected_answer in benchmark_response or any(word in benchmark_response for word in expected_answer.split()[:3]):
        print("    ✓ Benchmark question: PASS (answer found in response)")
    else:
        print("    ✗ Benchmark question: FAIL (answer not found)")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

    return {
        "instance_id": question_id,
        "questions": results,
        "expected_answer": instance["answer"],
    }


if __name__ == "__main__":
    quick_test()
