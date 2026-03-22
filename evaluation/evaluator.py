"""
evaluation/evaluator.py
----------------------
Consolidated evaluation logic for AgeMem benchmarking.

Combines core Evaluator class with query evaluation logic previously
duplicated in batch_runner.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Literal

from agents.orchestrator import Orchestrator
from evaluation.llm_judge import LLMJudge, JudgeResult


@dataclass
class SessionReplayResult:
    """Result of replaying sessions through the orchestrator."""
    session_id: str
    turns_processed: int
    ltm_entries_added: int
    stm_tokens_at_end: int
    learning_scores: list[float] = field(default_factory=list)


@dataclass
class EvaluationContext:
    """Context for evaluating a single question."""
    behavior_type: str  # "IE", "MR", "KU", "TR", "ABS"
    expected_answer: str
    question: str
    evidence_session_ids: list[int] = field(default_factory=list)


@dataclass
class QuestionResult:
    """Result of evaluating a single question."""
    query_id: str
    is_correct: bool
    behavior_type: str
    retrieval_trace: dict
    abstained: bool
    latency_ms: float
    judge_result: Optional[JudgeResult] = None
    validation_method: Literal["heuristic", "llm_judge"] = "heuristic"


class Evaluator:
    """Core evaluator for AgeMem benchmarking with heuristic and LLM-as-Judge support."""

    ABSTENTION_PHRASES = [
        "i don't know", "i don't have", "i'm not sure", "i cannot",
        "i can't", "no information", "not mentioned", "don't recall",
        "no memory", "i'm unable to", "i am not aware", "i do not have",
        "i have no", "there is no information",
    ]

    BEHAVIOR_MAP = {
        "single-session-user": "IE", "single-session-assistant": "IE",
        "preference": "IE", "single_hop": "IE", "implicit_preference_v2": "IE",
        "assistant_previnfo": "IE", "multi-session": "MR",
        "multi_session_synthesis": "MR", "two_hop": "MR", "aggregation": "MR",
        "comparison": "MR", "knowledge-update": "KU", "knowledge": "KU",
        "knowledge_update": "KU", "temporal-reasoning": "TR", "temporal": "TR",
        "temp_reasoning_implicit": "TR", "temp_reasoning_explicit": "TR",
        "time-reference": "TR", "date-filtering": "TR",
        "abstention": "ABS", "unknown": "ABS",
    }

    def __init__(
        self,
        orchestrator: Orchestrator,
        llm_judge: Optional[LLMJudge] = None,
        use_llm_judge: bool = True,
    ) -> None:
        self._orchestrator = orchestrator
        self._llm_judge = llm_judge
        self._use_llm_judge = use_llm_judge and llm_judge is not None

    def replay_sessions(
        self, sessions: list[list[dict]], behavior_type: str = "IE"
    ) -> list[SessionReplayResult]:
        """Replay sessions through the orchestrator."""
        results = []
        for i, session in enumerate(sessions):
            result = self._replay_single(session, f"{behavior_type}_{i}")
            results.append(result)
        return results

    def evaluate_questions(
        self, queries: list[dict], raw_data: list[dict]
    ) -> list[QuestionResult]:
        """Evaluate questions through the orchestrator."""
        instance_map = {
            inst.get("question_id", ""): inst
            for inst in raw_data if inst.get("question_id")
        }
        results = []
        for query in queries:
            query_id = query.get("query_id", "")
            instance = instance_map.get(query_id, {})
            results.append(self.evaluate_query(query, instance))
        return results

    def evaluate_query(
        self, query: dict, instance: Optional[dict] = None
    ) -> QuestionResult:
        """
        Evaluate a single query. Unified entry point consolidating
        logic previously duplicated in batch_runner._evaluate_query.
        """
        query_text = query.get("query_text", "")
        query_id = query.get("query_id", "")
        instance = instance or {}

        context = EvaluationContext(
            behavior_type=self._map_behavior(
                instance.get("question_type", "retrieval"), query_id
            ),
            expected_answer=instance.get("answer", ""),
            question=instance.get("question", ""),
            evidence_session_ids=instance.get("answer_session_ids", []),
        )
        return self._evaluate_single(query_text, query_id, context)

    def _replay_single(
        self, session: list[dict], session_id: str
    ) -> SessionReplayResult:
        """Replay a single session."""
        turns_processed = 0
        learning_scores = []
        ltm_before = len(self._orchestrator.ltm_snapshot())

        for turn in session:
            if turn.get('role') != 'user':
                continue
            content = turn.get('content', '')
            if not content:
                continue
            self._orchestrator.chat(content)
            turns_processed += 1
            trace = self._orchestrator.last_trace()
            if trace and trace.feedback:
                learning_scores.append(trace.feedback.score)

        return SessionReplayResult(
            session_id=session_id,
            turns_processed=turns_processed,
            ltm_entries_added=len(self._orchestrator.ltm_snapshot()) - ltm_before,
            stm_tokens_at_end=self._orchestrator.stm_stats().total_tokens,
            learning_scores=learning_scores,
        )

    def _evaluate_single(
        self, query_text: str, query_id: str, context: EvaluationContext
    ) -> QuestionResult:
        """Evaluate a single question with LLM-as-Judge or heuristic."""
        t0 = time.time()
        response = self._orchestrator.chat(query_text)
        generation_latency_ms = (time.time() - t0) * 1000

        retrieval_trace = self._build_trace(self._orchestrator.last_trace())
        abstained = self._detect_abstention(response)

        judge_result = None
        validation_method = "heuristic"

        if self._use_llm_judge and self._llm_judge:
            try:
                judge_result = self._llm_judge.evaluate(
                    question=context.question,
                    expected_answer=context.expected_answer,
                    model_response=response,
                    behavior_type=context.behavior_type,
                )
                is_correct = judge_result.is_correct
                validation_method = "llm_judge"
            except Exception as e:
                print(f"[WARN] LLM judge failed for {query_id}: {e}")
                is_correct = self._validate(response, context, abstained)
        else:
            is_correct = self._validate(response, context, abstained)

        return QuestionResult(
            query_id=query_id,
            is_correct=is_correct,
            behavior_type=context.behavior_type,
            retrieval_trace=retrieval_trace,
            abstained=abstained,
            latency_ms=generation_latency_ms + (judge_result.latency_ms if judge_result else 0),
            judge_result=judge_result,
            validation_method=validation_method,
        )

    def _build_trace(self, last_trace) -> dict:
        """Build retrieval trace from orchestrator trace."""
        if last_trace is None:
            return {"results": [], "latency_ms": 0.0}
        results = []
        for op in last_trace.ops_applied:
            op_type = op.op.value if hasattr(op.op, 'value') else str(op.op)
            if op_type.lower() == "retrieve" and op.entries_affected:
                for entry_id in op.entries_affected:
                    results.append((entry_id, 1.0))
        return {
            "results": results,
            "latency_ms": getattr(last_trace, 'latency_ms', 0.0),
        }

    def _detect_abstention(self, response: str) -> bool:
        """Detect if response indicates abstention."""
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in self.ABSTENTION_PHRASES)

    def _validate(
        self, response: str, context: EvaluationContext, abstained: bool
    ) -> bool:
        """Validate response based on behavior type (heuristic)."""
        if context.behavior_type.upper() == "ABS":
            return abstained
        return self._match_answer(response, context.expected_answer)

    def _match_answer(self, response: str, expected: str) -> bool:
        """Match response against expected answer (heuristic)."""
        if not expected:
            return False
        response_lower = response.lower()
        expected_lower = expected.lower()
        if expected_lower in response_lower:
            return True
        expected_tokens = set(expected_lower.split())
        response_tokens = set(response_lower.split())
        if not expected_tokens:
            return False
        return len(expected_tokens & response_tokens) / len(expected_tokens) >= 0.7

    @staticmethod
    def _map_behavior(question_type: str, question_id: str = "") -> str:
        """Map LongMemEval question type to behavior code."""
        if question_id and "_abs" in question_id:
            return "ABS"
        key = question_type.lower().replace("_", "-")
        return Evaluator.BEHAVIOR_MAP.get(key, "IE")