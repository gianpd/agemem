"""
test_llm_judge.py
-----------------
Standalone test for LLM-as-Judge integration.

Creates a fake dataset with 3 questions and tests the LLMJudge
evaluation without requiring a real llama.cpp server (uses mocking).

Usage:
    python test_llm_judge.py

Requirements:
    - pytest (optional, for running as pytest)
    - unittest.mock (built-in)
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.llm_judge import LLMJudge, JudgeResult
from evaluation.evaluators import Evaluator, QuestionResult
from evaluation.metrics import calculate_metrics


# =============================================================================
# FAKE DATASET (3 questions of different types)
# =============================================================================

FAKE_DATASET = [
    {
        "question_id": "test_001_ie",
        "question_type": "single-session-user",
        "question": "What is my favorite color?",
        "question_date": "2024/01/15 (Mon) 10:00",
        "answer": "Blue",
        "answer_session_ids": ["session_001"],
        "haystack_dates": ["2024/01/10 (Wed) 14:00"],
        "haystack_session_ids": ["session_001"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "I just bought a new car! It's a beautiful blue color.",
                    "has_answer": True
                },
                {
                    "role": "assistant",
                    "content": "Congratulations on your new car! Blue is a great color choice.",
                    "has_answer": False
                }
            ]
        ]
    },
    {
        "question_id": "test_002_mr",
        "question_type": "multi-session",
        "question": "What are my two favorite hobbies based on our conversations?",
        "question_date": "2024/01/20 (Sat) 15:00",
        "answer": "Painting and hiking",
        "answer_session_ids": ["session_002", "session_003"],
        "haystack_dates": ["2024/01/12 (Fri) 09:00", "2024/01/18 (Thu) 16:00"],
        "haystack_session_ids": ["session_002", "session_003"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "I spent the weekend painting landscapes. It's so relaxing!",
                    "has_answer": True
                },
                {
                    "role": "assistant",
                    "content": "That sounds wonderful! Painting is a great creative outlet.",
                    "has_answer": False
                }
            ],
            [
                {
                    "role": "user",
                    "content": "I went hiking in the mountains yesterday. The views were amazing!",
                    "has_answer": True
                },
                {
                    "role": "assistant",
                    "content": "Hiking in the mountains must have been breathtaking!",
                    "has_answer": False
                }
            ]
        ]
    },
    {
        "question_id": "test_003_tr",
        "question_type": "temporal-reasoning",
        "question": "How many days ago did I start my new job?",
        "question_date": "2024/01/25 (Thu) 12:00",
        "answer": "10 days",
        "answer_session_ids": ["session_004"],
        "haystack_dates": ["2024/01/15 (Mon) 09:00"],
        "haystack_session_ids": ["session_004"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "Today is my first day at the new job! Started on January 15th.",
                    "has_answer": True
                },
                {
                    "role": "assistant",
                    "content": "Best of luck on your first day! Hope it goes well.",
                    "has_answer": False
                }
            ]
        ]
    }
]


# =============================================================================
# MOCK LLM CLIENT FOR ORCHESTRATOR
# =============================================================================

class MockLLMClient:
    """Mock LLM that returns predetermined responses for testing."""

    def __init__(self):
        self.responses = {
            "favorite color": "Your favorite color is blue.",
            "hobbies": "Based on our conversations, your favorite hobbies are painting and hiking.",
            "new job": "You started your new job 10 days ago, on January 15th.",
        }
        self.call_count = 0

    def complete(self, prompt: str, **kwargs) -> str:
        """Return a mock response based on prompt content."""
        self.call_count += 1
        prompt_lower = prompt.lower()

        if "color" in prompt_lower:
            return self.responses["favorite color"]
        elif "hobbies" in prompt_lower:
            return self.responses["hobbies"]
        elif "job" in prompt_lower:
            return self.responses["new job"]
        return "I don't have that information."

    def get_embedding(self, text: str) -> list[float]:
        """Return a mock embedding."""
        # Simple hash-based mock embedding (384 dims for all-MiniLM-L6-v2)
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val % 100) / 100.0 for _ in range(384)]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def create_test_dataset_file() -> Path:
    """Create a temporary JSON file with the fake dataset."""
    temp_dir = tempfile.mkdtemp(prefix="llm_judge_test_")
    dataset_path = Path(temp_dir) / "test_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(FAKE_DATASET, f, indent=2)
    return dataset_path


def test_llm_judge_class():
    """Test the LLMJudge class directly with mocked OpenAI client."""
    print("\n" + "=" * 60)
    print("TEST 1: LLMJudge Class (Direct)")
    print("=" * 60)

    # Create judge instance
    judge = LLMJudge(
        api_base="http://localhost:8080/v1",
        model="test-model",
    )

    # Mock the OpenAI client
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "yes"

    with patch.object(judge.client.chat.completions, 'create', return_value=mock_completion):
        # Test IE (Information Extraction) prompt
        result = judge.evaluate(
            question="What is my favorite color?",
            expected_answer="Blue",
            model_response="Your favorite color is blue.",
            behavior_type="IE",
        )

        assert result.is_correct is True, f"Expected True, got {result.is_correct}"
        assert result.model == "test-model"
        assert result.latency_ms > 0
        print(f"  IE Test: PASS (is_correct={result.is_correct})")

        # Test with "no" response
        mock_completion.choices[0].message.content = "no"
        result = judge.evaluate(
            question="What is my favorite color?",
            expected_answer="Blue",
            model_response="I don't know your favorite color.",
            behavior_type="IE",
        )

        assert result.is_correct is False, f"Expected False, got {result.is_correct}"
        print(f"  IE Test (incorrect): PASS (is_correct={result.is_correct})")

        # Test TR (Temporal Reasoning) - allows off-by-one errors
        mock_completion.choices[0].message.content = "yes"
        result = judge.evaluate(
            question="How many days ago did I start my new job?",
            expected_answer="10 days",
            model_response="You started 11 days ago.",
            behavior_type="TR",
        )

        assert result.is_correct is True
        print(f"  TR Test: PASS (is_correct={result.is_correct})")

    print("\n  All LLMJudge class tests passed!")
    return True


def test_llm_judge_prompts():
    """Test that correct prompts are generated for each behavior type."""
    print("\n" + "=" * 60)
    print("TEST 2: LLMJudge Prompt Templates")
    print("=" * 60)

    judge = LLMJudge()

    # Check all behavior types have prompts
    behavior_types = ["IE", "MR", "TR", "KU", "ABS"]
    for bt in behavior_types:
        assert bt in judge.PROMPTS, f"Missing prompt for {bt}"
        print(f"  {bt}: Prompt exists ({len(judge.PROMPTS[bt])} chars)")

    # Check TR has the off-by-one tolerance text
    assert "off-by-one" in judge.PROMPTS["TR"].lower()
    print("  TR prompt contains 'off-by-one' tolerance: PASS")

    # Check KU has the knowledge update text
    assert "updated answer" in judge.PROMPTS["KU"].lower()
    print("  KU prompt contains 'updated answer' guidance: PASS")

    # Check ABS is for unanswerable questions
    assert "unanswerable" in judge.PROMPTS["ABS"].lower()
    print("  ABS prompt mentions 'unanswerable': PASS")

    print("\n  All prompt template tests passed!")
    return True


def test_health_check():
    """Test the health check functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Health Check")
    print("=" * 60)

    judge = LLMJudge()

    # Test failure case
    with patch.object(judge.client.chat.completions, 'create', side_effect=Exception("Connection refused")):
        result = judge.health_check()
        assert result is False, "Health check should fail when server is down"
        print("  Health check (failure): PASS")

    # Test success case
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "yes"

    with patch.object(judge.client.chat.completions, 'create', return_value=mock_completion):
        result = judge.health_check()
        assert result is True, "Health check should pass when server is up"
        print("  Health check (success): PASS")

    print("\n  All health check tests passed!")
    return True


def test_evaluator_with_llm_judge():
    """Test the Evaluator class with LLM-as-Judge integration."""
    print("\n" + "=" * 60)
    print("TEST 4: Evaluator + LLM-as-Judge Integration")
    print("=" * 60)

    # Create mock orchestrator
    mock_orchestrator = MagicMock()

    # Setup mock responses for each query
    responses = [
        "Your favorite color is blue.",  # Correct IE
        "You enjoy painting and hiking.",  # Correct MR
        "You started your job 10 days ago.",  # Correct TR
    ]
    mock_orchestrator.chat = MagicMock(side_effect=responses)
    mock_orchestrator.last_trace.return_value = MagicMock(
        ops_applied=[],
        latency_ms=100.0,
        feedback=None
    )
    mock_orchestrator.ltm_snapshot.return_value = []
    mock_orchestrator.stm_stats.return_value = MagicMock(total_tokens=500)

    # Create mock judge
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.evaluate.side_effect = [
        JudgeResult(is_correct=True, raw_response="yes", latency_ms=50.0, model="test-judge"),
        JudgeResult(is_correct=True, raw_response="yes", latency_ms=55.0, model="test-judge"),
        JudgeResult(is_correct=True, raw_response="yes", latency_ms=48.0, model="test-judge"),
    ]

    # Create evaluator with LLM judge
    evaluator = Evaluator(
        orchestrator=mock_orchestrator,
        llm_judge=mock_judge,
        use_llm_judge=True,
    )

    # Run evaluation
    queries = [
        {"query_id": "test_001_ie", "query_text": "What is my favorite color?"},
        {"query_id": "test_002_mr", "query_text": "What are my hobbies?"},
        {"query_id": "test_003_tr", "query_text": "When did I start my job?"},
    ]

    results = evaluator.evaluate_questions(queries, FAKE_DATASET)

    # Verify results
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    print(f"  Evaluated {len(results)} questions: PASS")

    # Check all used LLM judge
    for i, result in enumerate(results):
        assert result.validation_method == "llm_judge", f"Query {i} should use llm_judge"
        assert result.judge_result is not None, f"Query {i} should have judge_result"
        assert result.is_correct is True, f"Query {i} should be correct"
        print(f"  Query {i+1} ({result.behavior_type}): correct={result.is_correct}, method={result.validation_method}")

    # Verify judge was called correctly
    assert mock_judge.evaluate.call_count == 3, f"Judge should be called 3 times, was {mock_judge.evaluate.call_count}"
    print(f"  Judge called {mock_judge.evaluate.call_count} times: PASS")

    # Test metrics calculation
    summary = calculate_metrics(queries, results, [])
    assert summary.total_queries == 3
    assert summary.correct == 3
    assert summary.accuracy == 1.0
    assert summary.llm_judge_queries == 3
    assert summary.heuristic_queries == 0
    print(f"  Metrics: accuracy={summary.accuracy:.0%}, judge_queries={summary.llm_judge_queries}")

    print("\n  All evaluator integration tests passed!")
    return True


def test_evaluator_fallback_to_heuristic():
    """Test that evaluator falls back to heuristic when LLM judge fails."""
    print("\n" + "=" * 60)
    print("TEST 5: Fallback to Heuristic on Judge Failure")
    print("=" * 60)

    # Create mock orchestrator
    mock_orchestrator = MagicMock()
    mock_orchestrator.chat.return_value = "Your favorite color is blue."
    mock_orchestrator.last_trace.return_value = MagicMock(
        ops_applied=[],
        latency_ms=100.0,
        feedback=None
    )
    mock_orchestrator.ltm_snapshot.return_value = []
    mock_orchestrator.stm_stats.return_value = MagicMock(total_tokens=500)

    # Create mock judge that fails
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.evaluate.side_effect = Exception("Judge server error")

    # Create evaluator
    evaluator = Evaluator(
        orchestrator=mock_orchestrator,
        llm_judge=mock_judge,
        use_llm_judge=True,
    )

    # Run evaluation
    queries = [{"query_id": "test_001_ie", "query_text": "What is my favorite color?"}]
    results = evaluator.evaluate_questions(queries, FAKE_DATASET)

    # Should fall back to heuristic
    assert len(results) == 1
    assert results[0].validation_method == "heuristic"
    assert results[0].is_correct is True  # Heuristic matches "blue" in response
    print(f"  Fallback to heuristic: PASS (method={results[0].validation_method})")
    print(f"  Heuristic correctly matched answer: {results[0].is_correct}")

    print("\n  Fallback test passed!")
    return True


def test_end_to_end_with_mock_dataset():
    """Test loading and processing the fake dataset end-to-end."""
    print("\n" + "=" * 60)
    print("TEST 6: End-to-End with Mock Dataset")
    print("=" * 60)

    # Create temp dataset file
    dataset_path = create_test_dataset_file()
    print(f"  Created test dataset: {dataset_path}")

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    assert len(loaded_data) == 3
    print(f"  Loaded {len(loaded_data)} questions: PASS")

    # Verify schema
    for item in loaded_data:
        assert "question_id" in item
        assert "question_type" in item
        assert "question" in item
        assert "answer" in item
        assert "haystack_sessions" in item
    print("  Schema validation: PASS")

    # Verify behavior type mapping
    from evaluation.evaluators import Evaluator
    assert Evaluator._map_behavior("single-session-user") == "IE"
    assert Evaluator._map_behavior("multi-session") == "MR"
    assert Evaluator._map_behavior("temporal-reasoning") == "TR"
    print("  Behavior type mapping: PASS")

    # Cleanup
    import shutil
    shutil.rmtree(dataset_path.parent)
    print(f"  Cleanup: PASS")

    print("\n  End-to-end test passed!")
    return True


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("LLM-as-JUDGE INTEGRATION TESTS")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    tests = [
        ("LLMJudge Class", test_llm_judge_class),
        ("Prompt Templates", test_llm_judge_prompts),
        ("Health Check", test_health_check),
        ("Evaluator Integration", test_evaluator_with_llm_judge),
        ("Fallback to Heuristic", test_evaluator_fallback_to_heuristic),
        ("End-to-End", test_end_to_end_with_mock_dataset),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Finished: {datetime.now().isoformat()}")

    if failed == 0:
        print("\n  All tests passed!")
        return 0
    else:
        print(f"\n  {failed} test(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
