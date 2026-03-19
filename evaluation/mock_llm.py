"""
evaluation/mock_llm.py
----------------------

Stateful mock LLM client for deterministic evaluation testing.

Supports multiple strategies:
- template: Pattern-match queries to responses
- record_replay: Replay pre-recorded real LLM responses
- state_machine: Track conversation state for contextual responses
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from agents.llm_client import LLMClient


class MockStrategy(Enum):
    """Available mock response strategies."""
    TEMPLATE = "template"
    RECORD_REPLAY = "record_replay"
    STATE_MACHINE = "state_machine"


class StatefulMockLLM(LLMClient):
    """
    Stateful mock LLM client for deterministic evaluation testing.

    This mock is fully compatible with the real LLMClient interface and
    supports multiple strategies for generating responses.

    Strategies
    ----------
    template (default):
        Pattern-match user queries against registered templates.
        Returns the first matching response or a default.

    record_replay:
        Replay pre-recorded responses based on query hashes.
        Useful for reproducing exact LLM behavior from recordings.

    state_machine:
        Track conversation state and return context-aware responses.
        Useful for multi-turn conversations where responses depend on
        previous interactions.

    Examples
    --------
    >>> mock = StatefulMockLLM(strategy="template")
    >>> mock.add_response_template("weather", "The weather is sunny.")
    >>> mock.chat([{"role": "user", "content": "What's the weather?"}])
    'The weather is sunny.'

    >>> # State machine mode
    >>> mock = StatefulMockLLM(strategy="state_machine")
    >>> mock.add_state_transition("greeting", "hello", "asked_name", "Hi! What's your name?")
    >>> mock.set_initial_state("greeting")
    >>> mock.chat([{"role": "user", "content": "hello"}])
    "Hi! What's your name?"
    """

    def __init__(
        self,
        strategy: str = "template",
        default_response: str = "I don't have specific information about that.",
    ):
        """
        Initialize the stateful mock LLM.

        Parameters
        ----------
        strategy : str
            Response strategy: "template", "record_replay", or "state_machine"
        default_response : str
            Response when no match found
        """
        # Don't call LLMClient.__init__ - we have our own mock state
        self._model = "mock"
        self._temperature = 0.2
        self._strategy = MockStrategy(strategy)
        self._default_response = default_response

        # Template strategy state
        self._response_templates: dict[str, str] = {}

        # Record-replay strategy state
        self._recorded_responses: dict[str, str] = {}

        # State machine strategy state
        self._state_transitions: dict[str, dict[str, tuple[str, str]]] = {}
        self._current_state: Optional[str] = None

        # Common tracking
        self._call_history: list[dict] = []
        self._total_calls = 0

    # ── Template Strategy ─────────────────────────────────────────────────────

    def add_response_template(self, pattern: str, response: str) -> None:
        """
        Add a response template for pattern matching.

        Parameters
        ----------
        pattern : str
            Substring pattern to match in user messages (case-insensitive)
        response : str
            Response to return when pattern matches
        """
        self._response_templates[pattern.lower()] = response

    # ── Record-Replay Strategy ────────────────────────────────────────────────

    def add_recorded_response(self, query_hash: str, response: str) -> None:
        """
        Add a pre-recorded response for replay.

        Parameters
        ----------
        query_hash : str
            Hash of the query (e.g., MD5 or SHA256 of the user message)
        response : str
            Pre-recorded LLM response
        """
        self._recorded_responses[query_hash] = response

    def compute_query_hash(self, messages: list[dict]) -> str:
        """
        Compute a hash for a list of messages.

        Useful for creating consistent keys for record_replay mode.

        Parameters
        ----------
        messages : list[dict]
            Chat messages to hash

        Returns
        -------
        str
            MD5 hash of the message content
        """
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    # ── State Machine Strategy ───────────────────────────────────────────────

    def set_initial_state(self, state: str) -> None:
        """
        Set the initial state for state machine mode.

        Parameters
        ----------
        state : str
            Initial state name
        """
        self._current_state = state

    def add_state_transition(
        self,
        from_state: str,
        trigger: str,
        to_state: str,
        response: str,
    ) -> None:
        """
        Add a state transition rule.

        Parameters
        ----------
        from_state : str
            Current state name
        trigger : str
            Trigger pattern to match (case-insensitive substring)
        to_state : str
            Next state after transition
        response : str
            Response to return during this transition
        """
        if from_state not in self._state_transitions:
            self._state_transitions[from_state] = {}
        self._state_transitions[from_state][trigger.lower()] = (to_state, response)

    # ── Core Chat Method ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        retries: int = 2,
        tools: Optional[list] = None,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> str:
        """
        Mock chat that returns configured response based on strategy.

        Parameters
        ----------
        messages : list[dict]
            Chat messages (extracts last user message for matching)
        model : str, optional
            Ignored in mock
        max_tokens : int
            Ignored in mock
        temperature : float, optional
            Ignored in mock
        json_mode : bool
            Ignored in mock (returns plain text)
        retries : int
            Ignored in mock (always succeeds)
        tools : list, optional
            Ignored in mock
        timeout : float
            Ignored in mock

        Returns
        -------
        str
            Mock response based on strategy
        """
        # Extract last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        timestamp = datetime.now().isoformat()

        # Get response based on strategy
        response = self._get_response(user_message, messages)

        # Record call history
        self._call_history.append({
            "timestamp": timestamp,
            "user_message": user_message,
            "response": response,
            "strategy": self._strategy.value,
            "state": self._current_state,
        })
        self._total_calls += 1

        return response

    def _get_response(self, user_message: str, messages: list[dict]) -> str:
        """
        Get response based on current strategy.

        Parameters
        ----------
        user_message : str
            Last user message
        messages : list[dict]
            Full message history

        Returns
        -------
        str
            Response from appropriate strategy
        """
        if self._strategy == MockStrategy.TEMPLATE:
            return self._template_response(user_message)
        elif self._strategy == MockStrategy.RECORD_REPLAY:
            return self._record_replay_response(messages)
        elif self._strategy == MockStrategy.STATE_MACHINE:
            return self._state_machine_response(user_message)
        else:
            return self._default_response

    def _template_response(self, user_message: str) -> str:
        """Find first matching template response."""
        user_lower = user_message.lower()
        for pattern, response in self._response_templates.items():
            if pattern in user_lower:
                return response
        return self._default_response

    def _record_replay_response(self, messages: list[dict]) -> str:
        """Look up response by query hash."""
        query_hash = self.compute_query_hash(messages)
        return self._recorded_responses.get(query_hash, self._default_response)

    def _state_machine_response(self, user_message: str) -> str:
        """Get response based on current state and trigger."""
        if self._current_state is None:
            return self._default_response

        transitions = self._state_transitions.get(self._current_state, {})
        user_lower = user_message.lower()

        for trigger, (to_state, response) in transitions.items():
            if trigger in user_lower:
                self._current_state = to_state
                return response

        return self._default_response

    # ── History & Stats ───────────────────────────────────────────────────────

    def get_call_history(self) -> list[dict]:
        """
        Get history of calls made to mock LLM.

        Returns
        -------
        list[dict]
            List of call records with timestamp, user_message, response,
            strategy, and state (for state_machine mode)
        """
        return self._call_history.copy()

    def clear_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()

    def usage_stats(self) -> dict:
        """
        Get mock usage statistics.

        Returns
        -------
        dict
            Statistics including total calls
        """
        return {
            "total_calls": self._total_calls,
            "strategy": self._strategy.value,
            "current_state": self._current_state,
            "template_count": len(self._response_templates),
            "recorded_count": len(self._recorded_responses),
            "state_count": len(self._state_transitions),
        }

    # ── Utility Methods ────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Reset state machine to initial state (no effect on other strategies)."""
        self._current_state = None

    def load_templates_from_dict(self, templates: dict[str, str]) -> None:
        """
        Load multiple response templates from a dictionary.

        Parameters
        ----------
        templates : dict[str, str]
            Dictionary mapping patterns to responses
        """
        for pattern, response in templates.items():
            self.add_response_template(pattern, response)

    def load_recorded_from_dict(self, recordings: dict[str, str]) -> None:
        """
        Load multiple recorded responses from a dictionary.

        Parameters
        ----------
        recordings : dict[str, str]
            Dictionary mapping query hashes to responses
        """
        for query_hash, response in recordings.items():
            self.add_recorded_response(query_hash, response)

    def load_state_machine_from_dict(self, config: dict) -> None:
        """
        Load state machine configuration from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary with 'initial_state' and 'transitions' keys.
            Transitions is a list of (from_state, trigger, to_state, response).
        """
        if "initial_state" in config:
            self.set_initial_state(config["initial_state"])

        for transition in config.get("transitions", []):
            from_state, trigger, to_state, response = transition
            self.add_state_transition(from_state, trigger, to_state, response)