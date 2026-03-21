"""
evaluation/llm_judge.py
───────────────────────
LLM-as-Judge client for AgeMem evaluation.

Integrates with llama.cpp server or any OpenAI-compatible endpoint
to evaluate response correctness using task-specific prompts from
LongMemEval methodology.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import backoff
    HAVE_BACKOFF = True
except ImportError:
    HAVE_BACKOFF = False

try:
    import openai
    from openai import OpenAI
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False
    OpenAI = object  # type: ignore


@dataclass
class JudgeResult:
    """Result of LLM judgment."""
    is_correct: bool
    raw_response: str
    latency_ms: float
    model: str


class LLMJudge:
    """
    LLM-as-Judge for evaluating AgeMem responses.

    Supports local llama.cpp server via OpenAI-compatible API.
    """

    # Task-specific prompts from LongMemEval
    PROMPTS = {
        "IE": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",

        "MR": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",

        "TR": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",

        "KU": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",

        "ABS": """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {answer}

Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only.""",
    }

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        api_key: str = "EMPTY",
        model: str = "qwen3.5-9b",
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> None:
        """
        Initialize LLM-as-Judge.

        Args:
            api_base: OpenAI-compatible API endpoint (llama.cpp server)
            api_key: API key (use "EMPTY" for local servers)
            model: Model name for the judge
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
        """
        if not HAVE_OPENAI:
            raise ImportError("openai package is required for LLMJudge. Install with: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def _call_judge(self, prompt: str) -> tuple[str, float]:
        """
        Call the judge model with optional retry logic.

        Returns:
            Tuple of (response_text, latency_ms)
        """
        t0 = time.time()

        # Apply backoff decorator if available
        if HAVE_BACKOFF:
            @backoff.on_exception(
                backoff.expo,
                (openai.RateLimitError, openai.APIError),
                max_tries=3,
            )
            def _call():
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            completion = _call()
        else:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        latency_ms = (time.time() - t0) * 1000
        response = completion.choices[0].message.content.strip()
        return response, latency_ms

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        model_response: str,
        behavior_type: str,
    ) -> JudgeResult:
        """
        Evaluate a response using task-specific criteria.

        Args:
            question: The original question
            expected_answer: The expected/correct answer
            model_response: The response from the model being evaluated
            behavior_type: One of IE, MR, TR, KU, ABS

        Returns:
            JudgeResult with correctness determination
        """
        # Get appropriate prompt template
        prompt_template = self.PROMPTS.get(behavior_type, self.PROMPTS["IE"])

        # Format prompt
        prompt = prompt_template.format(
            question=question,
            answer=expected_answer,
            response=model_response,
        )

        # Call judge
        raw_response, latency_ms = self._call_judge(prompt)

        # Parse yes/no response
        is_correct = "yes" in raw_response.lower()

        return JudgeResult(
            is_correct=is_correct,
            raw_response=raw_response,
            latency_ms=latency_ms,
            model=self.model,
        )

    def health_check(self) -> bool:
        """Check if judge server is accessible."""
        try:
            # Simple health check
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say yes"}],
                max_tokens=5,
                temperature=0,
            )
            return True
        except Exception:
            return False
