"""
evaluation/evaluate_session.py
──────────────────────────────
Evaluate E2E session results using LLM-as-Judge.

This script loads a session.jsonl file created by run_e2e_longmemeval.py
and evaluates the agent responses against expected answers.

Usage:
    python -m evaluation.evaluate_session --session evaluation/logs/e2e_20260323_161131/session.jsonl
    python -m evaluation.evaluate_session --session <path> --output results.json
    python -m evaluation.evaluate_session --session <path> --judge-model gpt-4

Output:
    JSON file with evaluation metrics and per-question judgments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Import the existing LLMJudge
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.llm_judge import LLMJudge, JudgeResult

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    """Single interaction record from session.jsonl."""
    interaction_id: int
    timestamp: str
    question_id: str
    question_type: str
    user_input: str
    expected_answer: str
    agent_response: str
    latency_ms: float
    stm_tokens: int
    stm_utilization: float
    ltm_entries: int
    tool_calls: list[dict]
    memory_ops: list[dict]
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> SessionRecord:
        return cls(**d)


@dataclass
class EvaluationResult:
    """Evaluation result for a single question."""
    question_id: str
    question_type: str
    question: str
    expected_answer: str
    agent_response: str
    judgment: str  # "correct", "incorrect", "abstained", "error"
    confidence: float  # 0.0 - 1.0
    reasoning: str
    evaluation_method: str  # "exact_match", "llm_judge", etc.
    latency_ms: float
    stm_tokens: int
    ltm_entries: int


@dataclass
class SessionMetrics:
    """Aggregated metrics for the session."""
    total_questions: int
    answered_questions: int
    correct: int
    incorrect: int
    abstained: int
    errors: int
    accuracy: float
    avg_latency_ms: float
    avg_stm_tokens: float
    total_ltm_entries: int


class SessionLoader:
    """Load and parse session.jsonl files."""

    def __init__(self, session_path: Path):
        self.session_path = session_path
        self.metadata_path = session_path.with_suffix(".metadata.json")
        self.records: list[SessionRecord] = []
        self.metadata: Optional[dict] = None

    def load(self) -> list[SessionRecord]:
        """Load all records from session.jsonl."""
        if not self.session_path.exists():
            raise FileNotFoundError(f"Session file not found: {self.session_path}")

        logger.info(f"Loading session from {self.session_path}")

        with open(self.session_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = SessionRecord.from_dict(data)
                    self.records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")
                except TypeError as e:
                    logger.warning(f"Skipping invalid record on line {line_num}: {e}")

        # Load metadata if available
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        logger.info(f"Loaded {len(self.records)} records")
        return self.records

    def get_question_records(self) -> list[SessionRecord]:
        """Get only the question evaluation records (not session turns)."""
        return [r for r in self.records if not r.question_type.startswith("session_turn_")]

    def get_session_turn_records(self) -> list[SessionRecord]:
        """Get only the session turn records (haystack conversations)."""
        return [r for r in self.records if r.question_type.startswith("session_turn_")]


class SessionEvaluator:
    """Evaluate a complete session."""

    def __init__(
        self,
        judge: Optional[LLMJudge] = None,
        use_llm_judge: bool = True,
        default_behavior_type: str = "IE",
    ):
        self.judge = judge
        self.use_llm_judge = use_llm_judge
        self.default_behavior_type = default_behavior_type
        self.results: list[EvaluationResult] = []

    def evaluate_session(self, records: list[SessionRecord]) -> SessionMetrics:
        """Evaluate all question records in the session."""
        question_records = [r for r in records if not r.question_type.startswith("session_turn_")]

        logger.info(f"Evaluating {len(question_records)} questions...")

        for record in question_records:
            result = self._evaluate_record(record)
            self.results.append(result)

        return self._compute_metrics()

    def _evaluate_record(self, record: SessionRecord) -> EvaluationResult:
        """Evaluate a single record."""
        # Handle errors
        if record.error:
            return EvaluationResult(
                question_id=record.question_id,
                question_type=record.question_type,
                question=record.user_input,
                expected_answer=record.expected_answer,
                agent_response=record.agent_response,
                judgment="error",
                confidence=0.0,
                reasoning=f"Error during generation: {record.error}",
                evaluation_method="error",
                latency_ms=record.latency_ms,
                stm_tokens=record.stm_tokens,
                ltm_entries=record.ltm_entries,
            )

        # Use LLM-as-Judge if enabled and judge is available
        if self.use_llm_judge and self.judge and record.expected_answer:
            try:
                judge_result = self.judge.evaluate(
                    question=record.user_input,
                    expected_answer=record.expected_answer,
                    model_response=record.agent_response,
                    behavior_type=self._map_question_type(record.question_type),
                )
                judgment = "correct" if judge_result.is_correct else "incorrect"
                confidence = 1.0 if judge_result.is_correct else 0.0
                reasoning = judge_result.raw_response[:500]  # Truncate for storage
                method = "llm_judge"
                judge_latency = judge_result.latency_ms
            except Exception as e:
                logger.warning(f"LLM judge failed for {record.question_id}: {e}")
                # Fall back to exact match
                judgment, confidence, reasoning = self._exact_match_evaluate(record)
                method = "exact_match_fallback"
                judge_latency = 0
        else:
            # Simple exact match fallback
            judgment, confidence, reasoning = self._exact_match_evaluate(record)
            method = "exact_match"
            judge_latency = 0

        return EvaluationResult(
            question_id=record.question_id,
            question_type=record.question_type,
            question=record.user_input,
            expected_answer=record.expected_answer,
            agent_response=record.agent_response,
            judgment=judgment,
            confidence=confidence,
            reasoning=reasoning,
            evaluation_method=method,
            latency_ms=record.latency_ms + judge_latency,
            stm_tokens=record.stm_tokens,
            ltm_entries=record.ltm_entries,
        )

    def _map_question_type(self, question_type: str) -> str:
        """Map question type to behavior type for LLM judge.

        LongMemEval question types:
        - IE: Information Extraction
        - MR: Multi-hop Reasoning
        - TR: Temporal Reasoning
        - KU: Knowledge Update
        - ABS: Abstain/Unanswerable
        """
        # Map common question type patterns
        type_mapping = {
            "IE": "IE",
            "MR": "MR",
            "TR": "TR",
            "KU": "KU",
            "ABS": "ABS",
            "abstain": "ABS",
            "temporal": "TR",
            "multi_hop": "MR",
            "multi-hop": "MR",
        }

        question_upper = question_type.upper()
        for key, value in type_mapping.items():
            if key.upper() in question_upper:
                return value

        return self.default_behavior_type

    def _exact_match_evaluate(self, record: SessionRecord) -> tuple[str, float, str]:
        """Evaluate using exact match heuristic."""
        if not record.agent_response or record.agent_response.strip() == "":
            return "abstained", 0.0, "Empty response"

        if not record.expected_answer:
            return "abstained", 0.0, "No expected answer provided"

        # Simple containment check
        expected_lower = record.expected_answer.lower()
        response_lower = record.agent_response.lower()

        if expected_lower in response_lower or response_lower in expected_lower:
            return "correct", 1.0, "Exact match (answer contained in response)"

        # Check for partial word matches
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())
        if expected_words and response_words:
            overlap = len(expected_words & response_words) / len(expected_words)
            if overlap > 0.8:
                return "correct", overlap, f"High word overlap ({overlap:.0%})"

        return "incorrect", 0.0, "Answer not found in response"

    def _compute_metrics(self) -> SessionMetrics:
        """Compute aggregated metrics from results."""
        if not self.results:
            return SessionMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0)

        answered = [r for r in self.results if r.judgment != "error"]
        correct = sum(1 for r in self.results if r.judgment == "correct")
        incorrect = sum(1 for r in self.results if r.judgment == "incorrect")
        abstained = sum(1 for r in self.results if r.judgment == "abstained")
        errors = sum(1 for r in self.results if r.judgment == "error")

        accuracy = correct / len(answered) if answered else 0.0
        avg_latency = sum(r.latency_ms for r in self.results) / len(self.results)
        avg_stm = sum(r.stm_tokens for r in self.results) / len(self.results)
        max_ltm = max(r.ltm_entries for r in self.results) if self.results else 0

        return SessionMetrics(
            total_questions=len(self.results),
            answered_questions=len(answered),
            correct=correct,
            incorrect=incorrect,
            abstained=abstained,
            errors=errors,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            avg_stm_tokens=avg_stm,
            total_ltm_entries=max_ltm,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate E2E session results using LLM-as-Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a session
    python -m evaluation.evaluate_session --session evaluation/logs/e2e_20260323_161131/session.jsonl

    # Save results to file
    python -m evaluation.evaluate_session --session <path> --output results.json

    # Use specific judge model (OpenAI-compatible endpoint)
    python -m evaluation.evaluate_session --session <path> --judge-api-base http://localhost:8080/v1 --judge-model Qwen3.5-9B

    # Use exact match only (no LLM judge)
    python -m evaluation.evaluate_session --session <path> --no-llm-judge
""",
    )

    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to session.jsonl file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results (default: print to stdout)",
    )
    parser.add_argument(
        "--judge-api-base",
        type=str,
        default="http://localhost:8080/v1",
        help="API base URL for judge LLM (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default="EMPTY",
        help="API key for judge LLM (default: EMPTY for local servers)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Qwen3.5-9B-UD-Q4_K_XL.gguf",
        help="Model name for the judge (default: Qwen3.5-9B-UD-Q4_K_XL.gguf)",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=120.0,
        help="Timeout for judge API calls in seconds (default: 120)",
    )
    parser.add_argument(
        "--judge-use-json",
        action="store_true",
        help="Use JSON response format for judge (structured output)",
    )
    parser.add_argument(
        "--behavior-type",
        type=str,
        default="IE",
        choices=["IE", "MR", "TR", "KU", "ABS"],
        help="Default behavior type for evaluation (default: IE)",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Use exact match only, skip LLM-as-Judge",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate session file
    if not args.session.exists():
        logger.error(f"Session file not found: {args.session}")
        return 1

    # Load session
    try:
        loader = SessionLoader(args.session)
        records = loader.load()
    except Exception as e:
        logger.error(f"Failed to load session: {e}")
        return 1

    if not records:
        logger.error("No records found in session file")
        return 1

    # Get session stats
    question_records = loader.get_question_records()
    turn_records = loader.get_session_turn_records()

    logger.info(f"Session stats:")
    logger.info(f"  Total records: {len(records)}")
    logger.info(f"  Session turns: {len(turn_records)}")
    logger.info(f"  Questions: {len(question_records)}")

    # Initialize judge if requested
    judge = None
    if not args.no_llm_judge:
        try:
            logger.info(f"Initializing LLM judge at {args.judge_api_base}...")
            judge = LLMJudge(
                api_base=args.judge_api_base,
                api_key=args.judge_api_key,
                model=args.judge_model,
                timeout=args.judge_timeout,
                use_json=args.judge_use_json,
                temperature=0.0,
            )
            # Health check
            if judge.health_check():
                logger.info("Judge health check passed")
            else:
                logger.warning("Judge health check failed - will use fallback")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM judge: {e}")
            logger.warning("Falling back to exact match evaluation")
            judge = None

    # Evaluate
    evaluator = SessionEvaluator(
        judge=judge,
        use_llm_judge=(judge is not None and not args.no_llm_judge),
        default_behavior_type=args.behavior_type,
    )
    metrics = evaluator.evaluate_session(records)

    # Build output
    output = {
        "session_file": str(args.session),
        "evaluated_at": datetime.now().isoformat(),
        "judge_config": {
            "api_base": args.judge_api_base if not args.no_llm_judge else None,
            "model": args.judge_model if not args.no_llm_judge else None,
            "used_llm_judge": judge is not None and not args.no_llm_judge,
            "default_behavior_type": args.behavior_type,
        },
        "metrics": asdict(metrics),
        "results": [asdict(r) for r in evaluator.results],
    }

    # Output results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions:     {metrics.total_questions}")
    print(f"Answered:            {metrics.answered_questions}")
    print(f"Correct:             {metrics.correct} ({metrics.accuracy:.1%})")
    print(f"Incorrect:           {metrics.incorrect}")
    print(f"Abstained:           {metrics.abstained}")
    print(f"Errors:              {metrics.errors}")
    print(f"Avg latency:         {metrics.avg_latency_ms:.0f}ms")
    print(f"Avg STM tokens:      {metrics.avg_stm_tokens:.0f}")
    print(f"Max LTM entries:     {metrics.total_ltm_entries}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
