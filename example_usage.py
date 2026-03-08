"""
example_usage.py
─────────────────
How to wire up AgeMem-hybrid with a real OpenAI-compatible client.

Usage:
    OPENAI_API_KEY=sk-... python example_usage.py
"""

from __future__ import annotations

import os
import sys
from dotenv import load_dotenv
load_dotenv()
API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("BASE_URL")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# ── 1. Import the system ──────────────────────────────────────────────────────
from core.config import AgememConfig
from agents.llm_client import LLMClient
from agents.orchestrator import Orchestrator


def build_orchestrator(api_key: str, base_url: str | None = None) -> Orchestrator:
    """
    Wire up AgeMem-hybrid with any OpenAI-compatible API.

    For local models (e.g. Ollama):
        build_orchestrator(api_key="ollama", base_url="http://localhost:11434/v1")

    For OpenAI:
        build_orchestrator(api_key=os.environ["OPENAI_API_KEY"])

    For Azure OpenAI:
        build_orchestrator(
            api_key=os.environ["AZURE_OPENAI_KEY"],
            base_url="https://<resource>.openai.azure.com/openai/deployments/<deployment>"
        )
    """
    # Import openai here so the rest of the codebase stays library-free
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    cfg = AgememConfig(
        DEFAULT_MODEL="MiniMax-M2.5",
        MEMORY_AGENT_MODEL="kimi-2.5",
        STM_TOKEN_LIMIT=6_000,
        STM_WARNING_THRESHOLD=0.75,
        STM_CRITICAL_THRESHOLD=0.90,
        LTM_PROMOTE_THRESHOLD=0.65,
        LEARNING_SCORE_PROMPT_EVERY_N=3,
        TRIGGER_EVERY_N_TURNS=10,
    )

    llm = LLMClient(client, default_model=cfg.DEFAULT_MODEL)
    return Orchestrator(llm=llm, config=cfg)


def demo_session(orch: Orchestrator) -> None:
    """Simulate a multi-turn session and print diagnostics."""
    turns = [
        "My name is Alice and I'm working on a Python data pipeline project.",
        "The pipeline reads from Kafka topics and writes to BigQuery.",
        "We're using Apache Beam with the Dataflow runner.",
        "Can you explain the difference between bounded and unbounded PCollections?",
        "What are some common pitfalls when using side inputs in Beam?",
        "Let's talk about something else. What's the weather like on Mars?",
        "Back to my project — what's the best way to handle late data in streaming?",
        "How should I handle schema evolution in my Beam pipeline?",
        "What is the capital of France?",
        "Summarise everything you know about my project so far.",
    ]

    print("=" * 60)
    print("AgeMem-Hybrid Demo Session")
    print("=" * 60)

    for i, user_input in enumerate(turns):
        print(f"\n[Turn {i+1}] USER: {user_input}")
        response = orch.chat(user_input)
        print(f"[Turn {i+1}] ASSISTANT: {response[:200]}{'...' if len(response) > 200 else ''}")

        trace = orch.last_trace()
        stats = trace.stm_stats_after
        print(
            f"  STM: {stats.total_tokens} tokens "
            f"({stats.utilisation_ratio:.0%} of limit), "
            f"{stats.message_count} messages"
        )
        if trace.ops_applied:
            for op in trace.ops_applied:
                if op.success:
                    print(f"  OP [{op.trigger.value}] {op.op.value}: {op.detail}")
        if trace.feedback:
            print(f"  LEARNING SCORE: {trace.feedback.score:.2f} — {trace.feedback.rationale}")

    print("\n" + "=" * 60)
    print(f"LTM Store: {len(orch.ltm_snapshot())} entries")
    for entry in orch.ltm_snapshot():
        print(f"  [{entry['entry_id']}] score={entry['learning_score']:.2f}: {entry['content'][:80]}")


if __name__ == "__main__":
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        print("Set OPENAI_API_KEY to run the demo.")
        sys.exit(1)
    orch = build_orchestrator(api_key=api_key, base_url=BASE_URL)
    demo_session(orch)
