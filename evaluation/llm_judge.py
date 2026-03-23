"""
evaluation/llm_judge.py
───────────────────────
LLM-as-Judge client for AgeMem evaluation.

Integrates with llama.cpp server or any OpenAI-compatible endpoint
to evaluate response correctness using task-specific prompts from
LongMemEval methodology.
"""

from __future__ import annotations

import json
import os
import time
import threading
from dataclasses import dataclass
from typing import Optional
from functools import wraps

import httpx

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


class RateLimiter:
    """Thread-safe rate limiter with adaptive backoff."""

    def __init__(self, min_interval: float = 0.5, backoff_factor: float = 2.0, max_backoff: float = 60.0):
        self.min_interval = min_interval
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._current_backoff = 0.0
        self._consecutive_429s = 0

    def wait_before_request(self) -> None:
        """Wait appropriate time before making a request."""
        with self._lock:
            now = time.time()
            wait_time = max(self._current_backoff, self.min_interval - (now - self._last_request_time))
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_request_time = time.time()

    def on_success(self) -> None:
        """Reset backoff on successful request."""
        with self._lock:
            self._consecutive_429s = 0
            self._current_backoff = 0.0

    def on_rate_limit(self, retry_after: Optional[float] = None) -> None:
        """Increase backoff on rate limit."""
        with self._lock:
            self._consecutive_429s += 1
            # Use retry_after if provided, otherwise exponential backoff
            if retry_after and retry_after > 0:
                self._current_backoff = min(retry_after, self.max_backoff)
            else:
                self._current_backoff = min(
                    self.min_interval * (self.backoff_factor ** self._consecutive_429s),
                    self.max_backoff
                )


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

Is the model response correct? Answer with ONLY a single word: yes or no. Do not include any thinking, reasoning, or explanation.""",

        "MR": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer with ONLY a single word: yes or no. Do not include any thinking, reasoning, or explanation.""",

        "TR": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer with ONLY a single word: yes or no. Do not include any thinking, reasoning, or explanation.""",

        "KU": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer with ONLY a single word: yes or no. Do not include any thinking, reasoning, or explanation.""",

        "ABS": """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {answer}

Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer with ONLY a single word: yes or no. Do not include any thinking, reasoning, or explanation.""",
    }

    # JSON-format prompts for structured output
    PROMPTS_JSON = {
        "IE": """Evaluate if the model response contains the correct answer.

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Return a JSON object with a single boolean field "correct" (true if the response contains the correct answer or is equivalent, false otherwise).""",

        "MR": """Evaluate if the model response contains the correct answer.

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Return a JSON object with a single boolean field "correct" (true if the response contains the correct answer or is equivalent, false otherwise).""",

        "TR": """Evaluate if the model response contains the correct answer.

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Return a JSON object with a single boolean field "correct" (true if the response contains the correct answer, false otherwise). Note: Do not penalize off-by-one errors for time calculations.""",

        "KU": """Evaluate if the model response contains the correct updated answer.

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Return a JSON object with a single boolean field "correct" (true if the response contains the updated correct answer, false otherwise).""",

        "ABS": """Evaluate if the model correctly identifies the question as unanswerable.

Question: {question}
Explanation: {answer}
Model Response: {response}

Return a JSON object with a single boolean field "correct" (true if the model correctly identifies the question as unanswerable, false otherwise).""",
    }

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        temperature: float = 0.0,
        max_tokens: int = 250,
        timeout: float = 120.0,
        use_json: bool = False,
        min_request_interval: float = 0.5,
    ) -> None:
        """
        Initialize LLM-as-Judge.

        Args:
            api_base: OpenAI-compatible API endpoint (llama.cpp server)
            api_key: API key (use "EMPTY" for local servers)
            model: Model name for the judge
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds (default: 120.0)
            use_json: Use JSON response format for structured output
            min_request_interval: Minimum seconds between requests (default: 0.5)
        """
        if not HAVE_OPENAI:
            raise ImportError("openai package is required for LLMJudge. Install with: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.use_json = use_json

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

        # Rate limiter for API calls
        self._rate_limiter = RateLimiter(min_interval=min_request_interval)

    def _call_judge(self, prompt: str) -> tuple[str, float]:
        """
        Call the judge model with rate limiting and retry logic.

        Returns:
            Tuple of (response_text, latency_ms)
        """
        t0 = time.time()

        # Build API call parameters
        call_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": 1,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add JSON response format if enabled
        if self.use_json:
            call_params["response_format"] = {"type": "json_object"}

        def _extract_retry_after(error: Exception) -> Optional[float]:
            """Extract retry-after from error response."""
            try:
                if hasattr(error, 'response') and error.response:
                    headers = getattr(error.response, 'headers', {})
                    if headers:
                        retry_after = headers.get('retry-after')
                        if retry_after:
                            return float(retry_after)
                # Check for x-ratelimit-reset header (OpenRouter style)
                if hasattr(error, 'response') and error.response:
                    headers = getattr(error.response, 'headers', {})
                    reset = headers.get('x-ratelimit-reset')
                    if reset:
                        return float(reset)
            except (ValueError, TypeError, AttributeError):
                pass
            return None

        max_attempts = 8
        last_error = None

        for attempt in range(max_attempts):
            # Wait before request (rate limiting)
            self._rate_limiter.wait_before_request()

            try:
                completion = self.client.chat.completions.create(**call_params)
                self._rate_limiter.on_success()

                latency_ms = (time.time() - t0) * 1000
                response = completion.choices[0].message.content.strip()
                return response, latency_ms

            except openai.RateLimitError as e:
                retry_after = _extract_retry_after(e)
                self._rate_limiter.on_rate_limit(retry_after)
                last_error = e

                # Wait with exponential backoff + jitter
                wait_time = self._rate_limiter._current_backoff
                jitter = wait_time * 0.1 * (0.5 - time.time() % 1)  # Simple jitter
                actual_wait = wait_time + abs(jitter)

                import logging
                logging.getLogger(__name__).warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_attempts}), "
                    f"waiting {actual_wait:.1f}s before retry"
                )
                time.sleep(actual_wait)
                continue

            except openai.APIError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    wait_time = 2 ** (attempt + 1)  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                raise

        # All retries exhausted
        raise last_error or RuntimeError("Max retries exceeded")

    def _parse_judge_response(self, raw_response: str) -> bool:
        """
        Parse judge response to extract yes/no answer.
        Handles JSON format, chain-of-thought models, and text responses.

        Args:
            raw_response: Raw text from judge model

        Returns:
            True if answer is yes/correct, False otherwise
        """
        import re

        # First try JSON parsing if enabled
        if self.use_json:
            # Try parsing entire response as JSON first
            try:
                data = json.loads(raw_response)
                if "correct" in data:
                    return bool(data["correct"])
                if "is_correct" in data:
                    return bool(data["is_correct"])
                if "answer" in data:
                    return str(data["answer"]).lower() in ("yes", "true", "1")
            except json.JSONDecodeError:
                pass

            # Try extracting JSON from response (handles thinking + JSON)
            json_pattern = r'\{[^{}]*"correct"\s*:\s*(true|false)[^{}]*\}'
            match = re.search(json_pattern, raw_response, re.IGNORECASE)
            if match:
                try:
                    # Find the full JSON object
                    start = raw_response.find('{')
                    if start != -1:
                        # Find matching closing brace
                        depth = 0
                        for i, c in enumerate(raw_response[start:]):
                            if c == '{':
                                depth += 1
                            elif c == '}':
                                depth -= 1
                                if depth == 0:
                                    json_str = raw_response[start:start+i+1]
                                    data = json.loads(json_str)
                                    if "correct" in data:
                                        return bool(data["correct"])
                                    break
                except (json.JSONDecodeError, ValueError):
                    pass

        response_lower = raw_response.lower()

        # Look for "yes" or "no" at start/end of response or after common delimiters
        # Also handles Qwen-style thinking tags (Unicode U+200B zero-width space variants)
        patterns = [
            r'^\s*(yes|no)[\s\.,;!?]*$',  # standalone yes/no
            r'[\n\r]\s*(yes|no)\s*$',  # yes/no at end of line
            r'answer[\s]*[:\-]?\s*(yes|no)',  # "answer: yes/no"
            r'^(yes|no)\s*[\.,;]',  # yes/no at start
            r'[\.,;]\s*(yes|no)\s*$',  # yes/no at end after punctuation
            r'\*?\s*(yes|no)\s*$',  # after thinking close tag or similar markers
            r'(?:thinking|thought).*?(yes|no)',  # after thinking section
            r'(?:conclusion|conclude).*?(yes|no)',  # after conclusion
            r'(?:therefore|thus|so).*?(yes|no)',  # after reasoning words
        ]

        for pattern in patterns:
            match = re.search(pattern, response_lower, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1) == "yes"

        # Fallback: simple substring search (original behavior)
        return "yes" in response_lower

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
        if self.use_json:
            prompt_template = self.PROMPTS_JSON.get(behavior_type, self.PROMPTS_JSON["IE"])
        else:
            prompt_template = self.PROMPTS.get(behavior_type, self.PROMPTS["IE"])

        # Format prompt
        prompt = prompt_template.format(
            question=question,
            answer=expected_answer,
            response=model_response,
        )

        # Call judge
        raw_response, latency_ms = self._call_judge(prompt)

        # Parse yes/no response (handle chain-of-thought models)
        is_correct = self._parse_judge_response(raw_response)

        return JudgeResult(
            is_correct=is_correct,
            raw_response=raw_response,
            latency_ms=latency_ms,
            model=self.model,
        )

    def health_check(self) -> bool:
        """Check if judge server is accessible."""
        try:
            # Simple health check with explicit timeout
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say yes"}],
                max_tokens=5,
                temperature=0,
                timeout=30.0,
            )
            return True
        except Exception:
            return False
