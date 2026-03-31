"""
Diagnostic script: simulates the memory retrieval flow to expose
why irrelevant memories get injected with relevance_score=1.0 and is_pinned=True.

Reproduces the scenario where user asks "where i live?" but gets
pizza preferences, language preferences, and doc refs injected as
pinned system messages.

Usage:
    python scripts/diagnose_memory_retrieval.py
"""

import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional


# ── Minimal type stubs (no imports needed from agemem) ──────────────────────

@dataclass
class MemoryEntry:
    content: str
    entry_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    learning_score: float = 0.0
    similarity_score: float = 0.0
    tags: list[str] = field(default_factory=list)
    source_turn: int = 0

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = hashlib.sha1(
                f"{self.content}{self.created_at}".encode()
            ).hexdigest()[:12]


@dataclass
class ContextMessage:
    role: str
    content: Optional[str] = None
    turn_index: int = 0
    token_estimate: int = 0
    relevance_score: float = 1.0
    is_pinned: bool = False
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class TokenCounter:
    def count(self, text: str) -> int:
        words = len(text.split())
        return max(1, int(words * 0.75)) + 4

    def count_messages(self, messages: list[ContextMessage]) -> int:
        return sum(self.count(m.content or "") for m in messages)


# ── Reproduction of STMContext.retrieve() — the buggy path ──────────────────

def stm_retrieve_buggy(messages: list[ContextMessage], entries: list[MemoryEntry], turn_index: int):
    """
    Exact reproduction of STMContext.retrieve() from stm_context.py:165-197.
    This is the CURRENT (buggy) behaviour.
    """
    existing_ids = {
        m.content.split("]")[0].lstrip("[MEMORY:")
        for m in messages
        if m.role == "system" and "[MEMORY:" in (m.content or "")
    }

    injected = []
    tc = TokenCounter()
    for entry in entries:
        tag = f"[MEMORY:{entry.entry_id}]"
        if entry.entry_id in existing_ids:
            continue

        msg = ContextMessage(
            role="system",
            content=f"{tag} {entry.content}",
            turn_index=turn_index,
            token_estimate=tc.count(entry.content),
            relevance_score=entry.learning_score,  # BUG: uses learning_score, not semantic similarity
            is_pinned=True,                         # BUG: ALL entries pinned regardless of relevance
        )
        messages.append(msg)
        injected.append(entry.entry_id)

    return injected


def stm_retrieve_fixed(messages: list[ContextMessage], entries: list[MemoryEntry], turn_index: int, semantic_scores: dict[str, float]):
    """
    Actual fix (matching stm_context.py): only pin entries with similarity >= 0.75.
    Low-similarity entries are added but not pinned (evictable).
    """
    tc = TokenCounter()
    PIN_THRESHOLD = 0.75

    injected = []
    for entry in entries:
        tag = f"[MEMORY:{entry.entry_id}]"
        sim = semantic_scores.get(entry.entry_id, 0.0)
        entry.similarity_score = sim

        should_pin = sim >= PIN_THRESHOLD

        msg = ContextMessage(
            role="system",
            content=f"{tag} {entry.content}",
            turn_index=turn_index,
            token_estimate=tc.count(entry.content),
            relevance_score=sim,       # FIX: use actual semantic similarity
            is_pinned=should_pin,       # FIX: only pin highly relevant entries
        )
        messages.append(msg)
        injected.append(entry.entry_id)

    return injected


# ── Simulated LTM entries (mimicking real stored memories) ──────────────────

def create_test_ltm():
    """Create LTM entries that match the observed scenario."""
    return [
        MemoryEntry(
            content="i like pizza with peperoni and njua",
            learning_score=1.0,   # High learning score (agent "learned" this well)
            tags=["food", "preference"],
            source_turn=3,
        ),
        MemoryEntry(
            content="User preference: Prefers responses in French, concise, and no bullet points.",
            learning_score=1.0,
            tags=["preference", "language"],
            source_turn=5,
        ),
        MemoryEntry(
            content="docs/agemem_hotpotqa_evaluation_instructions.md",
            learning_score=1.0,
            tags=["document", "reference"],
            source_turn=6,
        ),
        MemoryEntry(
            content="User lives in Roma, VIA ELEONORA DUSE 53",
            learning_score=0.9,
            tags=["address", "personal"],
            source_turn=2,
        ),
        MemoryEntry(
            content="User works as a software engineer, mainly Python",
            learning_score=0.8,
            tags=["work", "skill"],
            source_turn=4,
        ),
    ]


# ── Simulated semantic search results ──────────────────────────────────────

def simulate_semantic_search(query: str, ltm_entries: list[MemoryEntry]) -> list[tuple[MemoryEntry, float]]:
    """
    Simulate what semantic search would return for the query.
    Returns (entry, semantic_similarity) pairs.

    For "where i live?", the address entry should be most similar,
    and pizza preference should be least similar.
    """
    # Simulated cosine similarity scores (what embedding-based search would produce)
    if "live" in query.lower() or "where" in query.lower() or "address" in query.lower():
        simulated_scores = {
            "User lives in Roma, VIA ELEONORA DUSE 53": 0.82,
            "User works as a software engineer, mainly Python": 0.45,
            "User preference: Prefers responses in French, concise, and no bullet points.": 0.30,
            "i like pizza with peperoni and njua": 0.12,
            "docs/agemem_hotpotqa_evaluation_instructions.md": 0.08,
        }
    else:
        # Default: all same score
        simulated_scores = {e.content: 0.5 for e in ltm_entries}

    results = []
    for entry in ltm_entries:
        score = simulated_scores.get(entry.content, 0.5)
        results.append((entry, score))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def simulate_token_overlap_search(query: str, ltm_entries: list[MemoryEntry]) -> list[tuple[MemoryEntry, float]]:
    """
    Simulate token overlap search (the fallback when semantic search is off).
    Uses Jaccard-like overlap.
    """
    STOPWORDS = {"the", "a", "an", "is", "in", "on", "at", "to", "of",
                 "and", "or", "but", "for", "with", "was", "be", "are", "i"}

    def tokenize(text):
        tokens = set()
        for w in text.split():
            w = w.lower().strip(".,!?;:\"'()")
            if w not in STOPWORDS and (len(w) > 2 or w.isdigit()):
                tokens.add(w)
        return tokens

    query_tokens = tokenize(query)
    results = []

    for entry in ltm_entries:
        content_tokens = tokenize(entry.content)
        if not query_tokens or not content_tokens:
            overlap = 0.0
        else:
            intersection = query_tokens & content_tokens
            union = query_tokens | content_tokens
            overlap = len(intersection) / len(union)

        # Apply recency + learning score boost (same formula as LTMStore)
        now = time.time()
        age_days = (now - entry.updated_at) / 86400
        recency = 2.718 ** (-age_days / 7)
        score = 0.5 * overlap + 0.3 * entry.learning_score + 0.2 * recency

        results.append((entry, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ── Formatting helpers ─────────────────────────────────────────────────────

def format_message(msg: ContextMessage) -> dict:
    return {
        "role": msg.role,
        "content": msg.content,
        "turn_index": msg.turn_index,
        "token_estimate": msg.token_estimate,
        "relevance_score": msg.relevance_score,
        "is_pinned": msg.is_pinned,
        "tool_call_id": msg.tool_call_id,
        "tool_calls": msg.tool_calls,
    }


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_messages(messages: list[ContextMessage], label: str):
    print(f"  [{label}] Messages in STM context:")
    for i, m in enumerate(messages):
        pin_marker = " [PINNED]" if m.is_pinned else ""
        print(f"    [{i}] role={m.role} | relevance={m.relevance_score:.2f} | "
              f"tokens={m.token_estimate}{pin_marker}")
        content_preview = (m.content or "")[:80]
        print(f"        content: {content_preview}...")
    print()


# ── Main simulation ────────────────────────────────────────────────────────

def run_simulation():
    print_section("MEMORY RETRIEVAL DIAGNOSTIC SIMULATION")

    # ── Step 1: Setup LTM with test entries ──
    print_section("STEP 1: LTM Entries (stored memories)")
    ltm_entries = create_test_ltm()
    for entry in ltm_entries:
        print(f"  [{entry.entry_id}] learning_score={entry.learning_score:.1f} | "
              f"tags={entry.tags}")
        print(f"    content: {entry.content}")
    print()

    # ── Step 2: User sends query ──
    user_query = "where i live ?"
    print_section(f"STEP 2: User Query = '{user_query}'")

    # ── Step 3: Semantic search ──
    print_section("STEP 3: Semantic Search Results (what the retriever returns)")
    search_results = simulate_semantic_search(user_query, ltm_entries)
    for entry, similarity in search_results:
        print(f"  [{entry.entry_id}] semantic_similarity={similarity:.2f} | "
              f"learning_score={entry.learning_score:.1f}")
        print(f"    content: {entry.content}")
    print()

    # ── Step 4: BUGGY PATH — what the current code does ──
    print_section("STEP 4: BUGGY PATH — STMContext.retrieve() (current behaviour)")
    print("  The retrieve() method in stm_context.py:183-186 does:")
    print("    - is_pinned = True          (ALL entries pinned, regardless of relevance)")
    print("    - relevance_score = learning_score  (NOT semantic similarity)")
    print()

    buggy_messages: list[ContextMessage] = [
        ContextMessage(role="system", content="You are AgeMem...", is_pinned=True),
    ]
    # Simulate: retriever returns top entries (even low-relevance ones from fallback)
    # In the real scenario, context-aware retrieval returns 0 results (min_similarity=0.65)
    # then falls back to query-only search which uses token overlap + learning_score boost
    # The learning_score boost pushes irrelevant entries into the results
    buggy_retrieved = [entry for entry, _ in search_results[:5]]  # top 5
    injected = stm_retrieve_buggy(buggy_messages, buggy_retrieved, turn_index=7)

    print(f"  Injected {len(injected)} entries into STM:")
    print_messages(buggy_messages, "BUGGY")

    print("  >>> BUG IDENTIFIED:")
    print("  - Pizza preference (semantic_sim=0.12) is PINNED with relevance_score=1.0")
    print("  - Language preference (semantic_sim=0.30) is PINNED with relevance_score=1.0")
    print("  - Doc reference (semantic_sim=0.08) is PINNED with relevance_score=1.0")
    print("  - These will NEVER be evicted because is_pinned=True")
    print()

    # ── Step 5: Root cause analysis ──
    print_section("STEP 5: ROOT CAUSE ANALYSIS")
    print("""
  The problem has TWO interacting bugs:

  BUG A: STMContext.retrieve() uses learning_score as relevance_score
  ─────────────────────────────────────────────────────────────────
  File: memory/stm_context.py, line 186
  Code: relevance_score=entry.learning_score

  The MemoryEntry.learning_score reflects how well the agent "learned"
  this fact (0.0-1.0), NOT how relevant it is to the current query.
  A pizza preference with learning_score=1.0 gets relevance_score=1.0
  even when the query is about addresses.

  BUG B: STMContext.retrieve() pins ALL retrieved entries
  ─────────────────────────────────────────────────────────────────
  File: memory/stm_context.py, line 185
  Code: is_pinned=True

  Every retrieved memory becomes permanently pinned. The filter()
  and force_fit() methods skip pinned messages, so these irrelevant
  memories accumulate in the context window forever.

  INTERACTION: The context-aware retriever has a min_similarity_threshold
  of 0.65, so irrelevant entries SHOULD be filtered. But when the
  threshold filters everything out, the fallback path
  (CONTEXT_FALLBACK_TO_QUERY_ONLY=True) calls ltm_store.search() which
  uses token overlap + learning_score boost. High learning_score entries
  like "i like pizza" get boosted into the results despite zero
  semantic relevance, then they're pinned forever.
""")

    # ── Step 6: Simulate the fallback path (token overlap) ──
    print_section("STEP 6: Fallback Path — Token Overlap Search (no embeddings)")
    overlap_results = simulate_token_overlap_search(user_query, ltm_entries)
    print("  Token overlap scores for 'where i live ?':")
    for entry, score in overlap_results:
        print(f"    [{entry.entry_id}] score={score:.3f} | "
              f"learning_score={entry.learning_score:.1f}")
        print(f"      content: {entry.content}")
    print()
    print("  Notice: entries with high learning_score rank high despite")
    print("  zero token overlap, because the scoring formula is:")
    print("    score = 0.5 * overlap + 0.3 * learning_score + 0.2 * recency")
    print("  A learning_score of 1.0 gives a baseline of 0.3 + 0.2 = 0.5")
    print("  even with ZERO token overlap!")
    print()

    # ── Step 7: Show the exact message list that reaches the LLM ──
    print_section("STEP 7: Final message list sent to LLM (reproducing observed output)")
    final_messages = buggy_messages.copy()
    final_messages.append(ContextMessage(
        role="user", content="where i live ?", turn_index=7, relevance_score=1.0
    ))

    print("  Messages in the LLM call:")
    print()
    for m in final_messages:
        role_tag = m.role.upper()
        pin_tag = " [PINNED]" if m.is_pinned else ""
        print(f'  {{')
        print(f'    "role": "{m.role}",')
        content = (m.content or "").replace('"', '\\"')
        print(f'    "content": "{content}",')
        print(f'    "turn_index": {m.turn_index},')
        print(f'    "relevance_score": {m.relevance_score},')
        print(f'    "is_pinned": {m.is_pinned}')
        print(f'  }},')
    print()

    # ── Step 8: Proposed fix ──
    print_section("STEP 8: PROPOSED FIX — Fixed retrieve()")
    print("  Changes to STMContext.retrieve() in stm_context.py:")
    print()
    print("  1. Accept semantic similarity scores from the retriever")
    print("  2. Use semantic similarity as relevance_score (not learning_score)")
    print("  3. Only pin entries with similarity >= PIN_THRESHOLD (e.g., 0.7)")
    print("  4. Non-pinned entries can be evicted by filter() when context fills")
    print()

    fixed_messages: list[ContextMessage] = [
        ContextMessage(role="system", content="You are AgeMem...", is_pinned=True),
    ]
    semantic_scores = {e.entry_id: s for e, s in search_results}
    fixed_retrieved = [entry for entry, _ in search_results[:5]]

    # Only retrieve entries above a minimum semantic threshold
    filtered_retrieved = [
        e for e in fixed_retrieved
        if semantic_scores.get(e.entry_id, 0) >= 0.4  # At least some relevance
    ]
    print(f"  Filtered to {len(filtered_retrieved)}/{len(fixed_retrieved)} entries "
          f"with semantic_sim >= 0.4")
    print()

    stm_retrieve_fixed(fixed_messages, filtered_retrieved, turn_index=7,
                       semantic_scores=semantic_scores)
    fixed_messages.append(ContextMessage(
        role="user", content="where i live ?", turn_index=7, relevance_score=1.0
    ))

    print("  Fixed messages in the LLM call:")
    print()
    for m in fixed_messages:
        print(f'  {{')
        print(f'    "role": "{m.role}",')
        content = (m.content or "").replace('"', '\\"')
        print(f'    "content": "{content}",')
        print(f'    "relevance_score": {m.relevance_score:.2f},')
        print(f'    "is_pinned": {m.is_pinned}')
        print(f'  }},')
    print()
    print("  Result: Only the address entry is pinned (semantic_sim=0.82).")
    print("  Other entries are present but evictable if context fills up.")
    print()

    # ── Summary ──
    print_section("SUMMARY")
    print("""
  The observed behaviour (irrelevant memories pinned with score 1.0)
  is caused by two bugs in STMContext.retrieve():

  1. relevance_score = entry.learning_score   (line 186)
     Should be: the semantic similarity score from retrieval

  2. is_pinned = True                         (line 185)
     Should be: conditional on actual relevance to current query

  Fix location: memory/stm_context.py, lines 180-190

  The fix requires passing semantic similarity scores through the
  retrieval chain so STMContext.retrieve() can make informed
  decisions about pinning and scoring.
""")


if __name__ == "__main__":
    run_simulation()
