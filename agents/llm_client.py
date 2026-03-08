"""
agents/llm_client.py
─────────────────────
Thin wrapper around any OpenAI-compatible HTTP endpoint.

Assumptions
───────────
* The caller supplies a pre-configured `openai.OpenAI` (or compatible) client.
* This module contains zero business logic — just sends messages, returns text.
* Structured JSON responses are requested via response_format when needed.

Why a wrapper and not direct openai calls everywhere?
──────────────────────────────────────────────────────
Centralises retry logic, logging, and token-budget enforcement.  Makes it
easy to swap the backend (local Ollama, Azure OpenAI, Anthropic via compat
layer) without touching agent code.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional


class ToolCallResponse(Exception):
    """Raised when the LLM returns a tool call instead of text."""
    
    def __init__(self, tool_call):
        self.tool_call = tool_call
        super().__init__(f"Tool call: {tool_call.function.name}")


class LLMClient:

    def __init__(self, client: Any, default_model: str, default_temperature: float = 0.2) -> None:
        """
        Parameters
        ----------
        client : openai.OpenAI or any compatible object with
                 client.chat.completions.create(**kwargs)
        default_model : str
        default_temperature : float
        """
        self._client = client
        self._model = default_model
        self._temperature = default_temperature
        self._total_calls = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    # ── Core call ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        retries: int = 2,
        tools: Optional[list[dict]] = None,
    ) -> str:
        """
        Send a chat completion request.

        Returns the assistant text content as a plain string.
        Raises RuntimeError after exhausting retries.
        
        If tools are provided and the model returns a tool call, raises
        ToolCallResponse with the tool call details.
        """
        kwargs: dict = {
            "model": model or self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                self._total_calls += 1
                usage = getattr(response, "usage", None)
                if usage:
                    self._total_tokens_in  += getattr(usage, "prompt_tokens",     0)
                    self._total_tokens_out += getattr(usage, "completion_tokens",  0)
                
                message = response.choices[0].message

                # Check for tool calls
                tool_calls = getattr(message, "tool_calls", None)
                if isinstance(tool_calls, list) and len(tool_calls) > 0:
                    raise ToolCallResponse(tool_calls[0])
                
                return message.content or ""
            except Exception as exc:
                # Don't retry on tool calls - let them propagate
                if isinstance(exc, ToolCallResponse):
                    raise
                last_exc = exc
                if attempt < retries:
                    time.sleep(1.5 ** attempt)

        raise RuntimeError(f"LLM call failed after {retries+1} attempts: {last_exc}")

    def chat_json(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 512,
    ) -> dict:
        """
        Convenience: request JSON mode and parse the result.
        Falls back to string parsing if the model does not support json_mode.
        """
        raw = self.chat(messages, model=model, max_tokens=max_tokens, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Strip markdown fences if present
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM returned non-JSON: {raw!r}") from e

    # ── Stats ─────────────────────────────────────────────────────────────────

    def usage_stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
        }
