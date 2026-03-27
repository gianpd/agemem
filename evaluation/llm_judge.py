"""
LLM Judge for evaluating model answers against ground-truth.

Targets a local llama.cpp server (OpenAI-compatible /v1/chat/completions).
Handles Qwen3-style <think>...</think> blocks in responses before score parsing.

Usage:
    judge = LLMJudge(api_base="http://localhost:8080/v1", model="qwen3-8b")
    result = judge.evaluate(
        question="Who was the first US president?",
        expected_answer="George Washington",
        model_response="The first president was George Washington.",
        behavior_type="HOTPOT_J",
    )
    print(result.score, result.parse_status)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ParseStatus(str, Enum):
    OK            = "ok"
    PARSE_FAILURE = "parse_failure"
    HTTP_ERROR    = "http_error"
    TIMEOUT       = "timeout"
    EXCEPTION     = "exception"


@dataclass
class JudgeResult:
    """Full provenance for a single judge call."""
    score: float                    # 0.0–1.0; may be 0.0 on failure
    parse_status: ParseStatus
    raw_response: str               # full text returned by the model
    think_block: str                # <think>…</think> content, empty if absent
    score_text: str                 # the substring that was parsed into score
    latency_ms: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.parse_status == ParseStatus.OK


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are an expert judge evaluating the correctness of answers to questions.

Given the following information:
- Question: {question}
- Ground-truth Answer: {ground_truth}
- Agent's Answer: {agent_answer}

Please evaluate the generated answer on a scale of 0.0 to 1.0:
- 1.0: Perfect match or equivalent correct answer
- 0.8-0.9: Mostly correct with minor differences
- 0.6-0.7: Partially correct or close approximation
- 0.4-0.5: Some correct elements but significant errors
- 0.2-0.3: Mostly incorrect with few correct elements
- 0.0-0.1: Completely incorrect or irrelevant

Respond with only a number between 0.0 and 1.0 (e.g., "0.85")\
"""


# ---------------------------------------------------------------------------
# Think-block stripping
# ---------------------------------------------------------------------------

# Matches <think>…</think> including multiline content.
# Non-greedy so nested or multiple blocks are handled independently.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def strip_think_block(text: str) -> tuple[str, str]:
    """
    Remove all <think>…</think> blocks from *text*.

    Returns:
        (clean_text, think_content)
        think_content is the concatenated inner text of all think blocks,
        empty string if none were present.
    """
    think_parts: list[str] = []

    def _capture(m: re.Match) -> str:
        think_parts.append(m.group(1).strip())
        return ""

    clean = _THINK_RE.sub(_capture, text).strip()
    return clean, "\n\n".join(think_parts)


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

# Ordered from most-specific to least-specific.
_SCORE_PATTERNS: list[re.Pattern] = [
    # Explicit marker:  FINAL_SCORE: 0.85
    re.compile(r"FINAL_SCORE\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    # Bare decimal first (0.85, .9)
    re.compile(r"\b(0?\.[0-9]+|1\.0|0\.0)\b"),
    # Integer 0 or 1
    re.compile(r"\b([01])\b"),
]


def parse_score(text: str) -> tuple[float, str, ParseStatus]:
    """
    Extract a 0.0–1.0 score from *text* (think-block already stripped).

    Returns:
        (score, matched_text, status)
    """
    for pattern in _SCORE_PATTERNS:
        matches = pattern.findall(text)
        # Take the *last* match — chain-of-thought models put the answer last
        if matches:
            raw = matches[-1]
            try:
                v = float(raw)
                if 0.0 <= v <= 1.0:
                    return v, raw, ParseStatus.OK
            except ValueError:
                continue

    return 0.0, "", ParseStatus.PARSE_FAILURE


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Calls a local llama.cpp OpenAI-compatible server to score agent answers.

    Parameters
    ----------
    api_base:
        Base URL of the llama.cpp server, e.g. "http://localhost:8080/v1".
    model:
        Model name string forwarded to the API (llama.cpp ignores it but
        some proxies use it for routing).
    api_key:
        Passed as Bearer token; use "EMPTY" for keyless local servers.
    temperature:
        Sampling temperature. 0.0 gives deterministic greedy output.
    max_tokens:
        Upper bound on response length. 512 is enough for score-only replies;
        increase to ~3000 if the model emits <think> blocks.
    timeout_s:
        HTTP request timeout in seconds.
    retries:
        Number of retry attempts on transient HTTP/network errors.
    retry_delay_s:
        Seconds to wait between retries (simple fixed backoff).
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        model: str = "local-model",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 120.0,
        retries: int = 2,
        retry_delay_s: float = 2.0,
    ) -> None:
        self.endpoint = api_base.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_delay_s = retry_delay_s

        logger.info(
            "LLMJudge initialised: endpoint=%s model=%s temperature=%s max_tokens=%s",
            self.endpoint, self.model, self.temperature, self.max_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        model_response: str,
        behavior_type: str = "HOTPOT_J",   # reserved for future routing
    ) -> JudgeResult:
        """
        Score *model_response* against *expected_answer* for *question*.

        Always returns a JudgeResult; never raises. Check result.ok to
        distinguish genuine scores from failures.
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=expected_answer,
            agent_answer=model_response,
        )

        start = time.perf_counter()
        raw, status, error = self._call_with_retry(prompt)
        latency_ms = (time.perf_counter() - start) * 1000

        if status != ParseStatus.OK:
            # HTTP/timeout/exception — no score available
            return JudgeResult(
                score=0.0,
                parse_status=status,
                raw_response=raw,
                think_block="",
                score_text="",
                latency_ms=latency_ms,
                error=error,
            )

        # Strip Qwen think block before score extraction
        clean, think = strip_think_block(raw)

        score, score_text, parse_status = parse_score(clean)

        if parse_status != ParseStatus.OK:
            logger.warning(
                "Score parse failed for behavior_type=%s | clean_response=%r",
                behavior_type, clean[:300],
            )

        logger.debug(
            "Judge result: score=%.2f status=%s think_len=%d score_text=%r latency=%.0fms",
            score, parse_status.value, len(think), score_text, latency_ms,
        )

        return JudgeResult(
            score=score,
            parse_status=parse_status,
            raw_response=raw,
            think_block=think,
            score_text=score_text,
            latency_ms=latency_ms,
            error=error,
        )

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    def _call_with_retry(self, prompt: str) -> tuple[str, ParseStatus, Optional[str]]:
        """
        POST to the llama.cpp endpoint with fixed-backoff retries.

        Returns:
            (response_text, status, error_message)
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error: Optional[str] = None

        for attempt in range(self.retries + 1):
            if attempt > 0:
                logger.info("Retrying judge call (attempt %d/%d)…", attempt + 1, self.retries + 1)
                time.sleep(self.retry_delay_s)

            try:
                resp = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()

            except requests.Timeout as exc:
                last_error = f"Timeout after {self.timeout_s}s: {exc}"
                logger.warning(last_error)
                continue  # retry

            except requests.HTTPError as exc:
                last_error = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                logger.warning("Judge HTTP error: %s", last_error)
                # 4xx errors are not retryable
                if exc.response.status_code < 500:
                    return "", ParseStatus.HTTP_ERROR, last_error
                continue  # retry 5xx

            except requests.RequestException as exc:
                last_error = f"Request error: {exc}"
                logger.warning(last_error)
                continue

            except Exception as exc:
                last_error = f"Unexpected error: {exc}"
                logger.exception(last_error)
                return "", ParseStatus.EXCEPTION, last_error

            # Success — extract text
            try:
                text = resp.json()["choices"][0]["message"]["content"]
                return text, ParseStatus.OK, None
            except (KeyError, IndexError, ValueError) as exc:
                last_error = f"Unexpected response shape: {exc} | body={resp.text[:300]}"
                logger.warning(last_error)
                return resp.text, ParseStatus.PARSE_FAILURE, last_error

        # All retries exhausted
        final_status = ParseStatus.TIMEOUT if "Timeout" in (last_error or "") else ParseStatus.HTTP_ERROR
        return "", final_status, last_error


# ---------------------------------------------------------------------------
# Smoke-test  (python llm_judge.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    # --- Unit tests for pure functions (no server required) ---

    print("── strip_think_block ──────────────────────────────────────")
    cases = [
        "<think>This is my reasoning.</think>0.85",
        "<think>\nStep 1: analyse\nStep 2: conclude\n</think>\n0.9",
        "No think block here. 0.75",
        "<THINK>case-insensitive</THINK>1.0",
        "<think>first</think> some text <think>second</think>0.5",
    ]
    for c in cases:
        clean, think = strip_think_block(c)
        print(f"  input   : {c!r}")
        print(f"  clean   : {clean!r}")
        print(f"  think   : {think!r}")
        print()

    print("── parse_score ────────────────────────────────────────────")
    score_cases = [
        ("0.85", 0.85, ParseStatus.OK),
        ("FINAL_SCORE: 0.9", 0.9, ParseStatus.OK),
        ("The answer is 0.75 out of 1.0", 1.0, ParseStatus.OK),  # last match wins
        ("1", 1.0, ParseStatus.OK),
        ("0", 0.0, ParseStatus.OK),
        ("no number here", 0.0, ParseStatus.PARSE_FAILURE),
    ]
    all_passed = True
    for text, expected_score, expected_status in score_cases:
        score, matched, status = parse_score(text)
        ok = (score == expected_score) and (status == expected_status)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {text!r:40s} → score={score} matched={matched!r} status={status.value}")
        if not ok:
            print(f"      expected score={expected_score} status={expected_status.value}")
            all_passed = False
    print()

    if not all_passed:
        print("Some unit tests FAILED.")
        sys.exit(1)
    print("All unit tests passed.\n")

    # --- Live server test (skipped if server not reachable) ---
    api_base = "http://localhost:8080/v1"
    print(f"── Live server test ({api_base}) ──────────────────────────")
    try:
        ping = requests.get(api_base.replace("/v1", "/health"), timeout=2.0)
        server_up = ping.ok
    except Exception:
        server_up = False

    if not server_up:
        print("  Server not reachable — skipping live test.")
    else:
        judge = LLMJudge(api_base=api_base)
        result = judge.evaluate(
            question="What is the capital of France?",
            expected_answer="Paris",
            model_response="The capital of France is Paris.",
        )
        print(f"  score       : {result.score}")
        print(f"  parse_status: {result.parse_status.value}")
        print(f"  score_text  : {result.score_text!r}")
        print(f"  think_block : {result.think_block[:120]!r}")
        print(f"  latency_ms  : {result.latency_ms:.0f}")
        print(f"  error       : {result.error}")