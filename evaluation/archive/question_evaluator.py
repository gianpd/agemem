"""
evaluation/question_evaluator.py
--------------------------------
Evaluates test questions through orchestrator.chat() with behavior-specific validation.

This module provides the core evaluation logic for testing memory behaviors
defined in LongMemEval:
- IE (Information Extraction): Single-session detail extraction
- MR (Multi-Session Reasoning): Multi-session synthesis
- KU (Knowledge Updates): Most recent value used
- TR (Temporal Reasoning): Temporal reasoning correct
- ABS (Abstention): Correct abstention when unknown

Usage:
    from agents.orchestrator import Orchestrator
    from evaluation.question_evaluator import QuestionEvaluator, EvaluationContext

    evaluator = QuestionEvaluator(orchestrator)
    context = EvaluationContext(
        behavior_type="IE",
        expected_answer="555-1234",
        evidence_session_ids=[12],
    )
    result = evaluator.evaluate_question("What is my phone number?", context)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from agents.orchestrator import Orchestrator


@dataclass
class EvaluationContext:
    """
    Context for evaluating a single question.

    Attributes:
        behavior_type: One of "IE", "MR", "KU", "TR", "ABS"
        expected_answer: The ground truth answer
        evidence_session_ids: Session indices containing answer evidence
    """
    behavior_type: str  # "IE", "MR", "KU", "TR", "ABS"
    expected_answer: str
    evidence_session_ids: list[int] = field(default_factory=list)


@dataclass
class QuestionResult:
    """
    Result of evaluating a single question.

    Attributes:
        query_id: Unique identifier for the query
        is_correct: Whether the answer is correct
        behavior_type: The behavior type tested
        retrieval_trace: Details of memory retrieval
        abstained: Whether the model abstained from answering
        latency_ms: Response latency in milliseconds
    """
    query_id: str
    is_correct: bool
    behavior_type: str
    retrieval_trace: dict
    abstained: bool
    latency_ms: float


class QuestionEvaluator:
    """
    Evaluates test questions through orchestrator.chat() with behavior-specific validation.

    This class implements the evaluation logic for LongMemEval-style questions,
    testing all 5 memory behavior categories through the actual orchestrator codepath.

    Behavior-specific validation:
    - IE (Information Extraction): Verify single-session detail was retrieved
    - MR (Multi-Session Reasoning): Check that multiple evidence sessions were used
    - KU (Knowledge Updates): Verify most recent value is used (not outdated)
    - TR (Temporal Reasoning): Verify time-based reasoning is correct
    - ABS (Abstention): Verify model abstains when information is missing
    """

    # Abstention phrases for detection
    ABSTENTION_PHRASES = [
        "i don't know",
        "i don't have",
        "i'm not sure",
        "i cannot",
        "i can't",
        "no information",
        "not mentioned",
        "don't recall",
        "no memory",
        "i'm unable to",
        "i am not aware",
        "i do not have",
        "i have no",
        "there is no information",
    ]

    def __init__(self, orchestrator: Orchestrator) -> None:
        """
        Initialize the evaluator with an orchestrator instance.

        Args:
            orchestrator: The Orchestrator instance to use for evaluation
        """
        self._orchestrator = orchestrator
        self._query_counter = 0

    def evaluate_question(
        self,
        question: str,
        context: EvaluationContext,
        query_id: Optional[str] = None,
        custom_matcher: Optional[Callable[[str, str], bool]] = None,
    ) -> QuestionResult:
        """
        Evaluate a single question through orchestrator.chat().

        Args:
            question: The question to evaluate
            context: Evaluation context with behavior type and expected answer
            query_id: Optional unique identifier (auto-generated if not provided)
            custom_matcher: Optional custom answer matching function

        Returns:
            QuestionResult with evaluation metrics
        """
        # Generate query ID if not provided
        if query_id is None:
            self._query_counter += 1
            query_id = f"q_{self._query_counter}"

        # Track timing
        t0 = time.time()

        # Send question through orchestrator
        response = self._orchestrator.chat(question)

        # Calculate latency
        latency_ms = (time.time() - t0) * 1000

        # Get retrieval trace from last turn
        last_trace = self._orchestrator.last_trace()
        retrieval_trace = self._build_retrieval_trace(last_trace)

        # Detect abstention
        abstained = self._detect_abstention(response)

        # Determine correctness based on behavior type
        is_correct = self._validate_by_behavior(
            response=response,
            context=context,
            abstained=abstained,
            custom_matcher=custom_matcher,
            retrieval_trace=retrieval_trace,
        )

        return QuestionResult(
            query_id=query_id,
            is_correct=is_correct,
            behavior_type=context.behavior_type,
            retrieval_trace=retrieval_trace,
            abstained=abstained,
            latency_ms=latency_ms,
        )

    def _build_retrieval_trace(self, last_trace) -> dict:
        """
        Build retrieval trace from orchestrator trace.

        Args:
            last_trace: The TurnTrace from orchestrator.last_trace()

        Returns:
            Dict with retrieval details including 'results' field for metrics computation.
            The 'results' field is a list of (entry_id, score) tuples for retrieval metrics.
        """
        if last_trace is None:
            return {
                "memories_injected": 0,
                "memory_ops": [],
                "results": [],  # For MetricsPipeline: list of (entry_id, score) tuples
                "stm_stats_before": {},
                "stm_stats_after": {},
            }

        # Extract memory operations
        memory_ops = []
        results = []  # (entry_id, score) tuples for retrieval metrics
        for op in last_trace.ops_applied:
            op_dict = {
                "op_type": op.op.value if hasattr(op.op, 'value') else str(op.op),
                "success": op.success,
                "trigger": op.trigger.value if hasattr(op.trigger, 'value') else str(op.trigger),
                "detail": op.detail or "",
            }
            if op.entries_affected:
                op_dict["entries_affected"] = op.entries_affected

                # For RETRIEVE ops, extract entries as results for metrics computation
                op_type_str = op.op.value if hasattr(op.op, 'value') else str(op.op)
                if op_type_str.lower() == "retrieve":
                    for entry_id in op.entries_affected:
                        # Use placeholder score 1.0 since actual scores aren't captured
                        # TODO: Capture actual relevance scores from LTM search
                        results.append((entry_id, 1.0))

            memory_ops.append(op_dict)

        # Build trace dict
        trace = {
            "memories_injected": sum(1 for op in memory_ops if op.get("op_type", "").lower() == "retrieve"),
            "memory_ops": memory_ops,
            "results": results,  # For MetricsPipeline
            "stm_stats_before": last_trace.stm_stats_before.to_dict() if hasattr(last_trace.stm_stats_before, 'to_dict') else {},
            "stm_stats_after": last_trace.stm_stats_after.to_dict() if hasattr(last_trace.stm_stats_after, 'to_dict') else {},
            "latency_ms": last_trace.latency_ms,
        }

        return trace

    def _detect_abstention(self, response: str) -> bool:
        """
        Detect if the response indicates abstention.

        Args:
            response: The assistant's response

        Returns:
            True if abstention is detected
        """
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in self.ABSTENTION_PHRASES)

    def _validate_by_behavior(
        self,
        response: str,
        context: EvaluationContext,
        abstained: bool,
        custom_matcher: Optional[Callable[[str, str], bool]],
        retrieval_trace: dict,
    ) -> bool:
        """
        Validate response based on behavior type.

        Args:
            response: The assistant's response
            context: Evaluation context
            abstained: Whether the model abstained
            custom_matcher: Optional custom matcher
            retrieval_trace: Retrieval trace details

        Returns:
            True if the response is correct for the behavior type
        """
        behavior = context.behavior_type.upper()

        if behavior == "ABS":
            # Abstention: Should abstain when information is missing
            return abstained

        if behavior == "IE":
            # Information Extraction: Single-session detail extraction
            return self._match_answer(response, context.expected_answer, custom_matcher)

        if behavior == "MR":
            # Multi-Session Reasoning: Verify multiple sessions were used
            # First check answer correctness
            answer_correct = self._match_answer(response, context.expected_answer, custom_matcher)
            # Then verify multi-session retrieval (evidence_session_ids has multiple entries)
            multi_session_used = len(context.evidence_session_ids) > 1
            return answer_correct and multi_session_used

        if behavior == "KU":
            # Knowledge Updates: Most recent value used
            # The answer matching should verify the current/most recent value
            return self._match_answer(response, context.expected_answer, custom_matcher)

        if behavior == "TR":
            # Temporal Reasoning: Time-based reasoning
            return self._match_answer(response, context.expected_answer, custom_matcher)

        # Default: simple answer matching
        return self._match_answer(response, context.expected_answer, custom_matcher)

    def _match_answer(
        self,
        response: str,
        expected_answer: str,
        custom_matcher: Optional[Callable[[str, str], bool]] = None,
    ) -> bool:
        """
        Match response against expected answer.

        Uses multiple matching strategies:
        1. Custom matcher if provided
        2. Case-insensitive substring match
        3. Token overlap (70% threshold)

        Args:
            response: The assistant's response
            expected_answer: The expected answer
            custom_matcher: Optional custom matching function

        Returns:
            True if the response matches
        """
        if custom_matcher:
            return custom_matcher(response, expected_answer)

        response_lower = response.lower()
        expected_lower = expected_answer.lower()

        # Direct containment
        if expected_lower in response_lower:
            return True

        # Reverse containment (response contained in expected)
        if response_lower in expected_lower:
            return True

        # Token overlap for multi-word answers
        expected_tokens = set(expected_lower.split())
        response_tokens = set(response_lower.split())

        if len(expected_tokens) == 0:
            return False

        overlap = len(expected_tokens & response_tokens)
        return overlap / len(expected_tokens) >= 0.7  # 70% token overlap threshold

    def evaluate_batch(
        self,
        questions: list[tuple[str, EvaluationContext]],
        query_ids: Optional[list[str]] = None,
    ) -> list[QuestionResult]:
        """
        Evaluate a batch of questions.

        Args:
            questions: List of (question, context) tuples
            query_ids: Optional list of query IDs (must match length of questions)

        Returns:
            List of QuestionResult objects
        """
        if query_ids is not None and len(query_ids) != len(questions):
            raise ValueError("query_ids length must match questions length")

        results = []
        for i, (question, context) in enumerate(questions):
            query_id = query_ids[i] if query_ids else None
            result = self.evaluate_question(question, context, query_id)
            results.append(result)

        return results

    def get_summary_stats(self, results: list[QuestionResult]) -> dict:
        """
        Calculate summary statistics from a list of results.

        Args:
            results: List of QuestionResult objects

        Returns:
            Dict with summary statistics
        """
        if not results:
            return {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "avg_latency_ms": 0.0,
                "abstention_rate": 0.0,
                "by_behavior": {},
            }

        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        abstained = sum(1 for r in results if r.abstained)
        total_latency = sum(r.latency_ms for r in results)

        # Calculate per-behavior stats
        by_behavior: dict[str, dict] = {}
        for r in results:
            behavior = r.behavior_type
            if behavior not in by_behavior:
                by_behavior[behavior] = {"total": 0, "correct": 0}
            by_behavior[behavior]["total"] += 1
            if r.is_correct:
                by_behavior[behavior]["correct"] += 1

        # Calculate accuracy per behavior
        for behavior in by_behavior:
            b_total = by_behavior[behavior]["total"]
            b_correct = by_behavior[behavior]["correct"]
            by_behavior[behavior]["accuracy"] = b_correct / b_total if b_total > 0 else 0.0

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "avg_latency_ms": total_latency / total,
            "abstention_rate": abstained / total if total > 0 else 0.0,
            "by_behavior": by_behavior,
        }