# LTM Not Triggering: Debug Analysis

## Problem Statement

After 21+ turns of conversation with substantial content (SOA research discussion, tool usage instructions), the LTM remains at **0 entries**:

```
[STM ██░░░░░░░░ 22% ~1295tok | LTM 0 entries | turn 17]
...
[STM █████░░░░░ 50% ~3004tok | LTM 0 entries | turn 21]
```

Expected behavior:
- Learning scorer should collect at turns 3, 6, 9, 12, 15, 18, 21 (every 3 turns)
- MemoryAgent should review at turns 10, 20 (every 10 turns)
- LTM should have entries from high-learning-score moments

---

## Root Cause Analysis

### 1. Silent Failures in LearningScorer (CRITICAL)

**File:** `agents/learning_scorer.py:94-95`

```python
except Exception:
    return None
```

**Problem:** The LearningScorer catches ALL exceptions and returns `None` silently. If:
- The LLM returns invalid JSON
- The model doesn't follow the JSON format
- There's a parsing error
- The `chat_json()` method fails

**No error is logged, no feedback is visible.** The orchestrator continues as if no learning happened.

### 2. Code Quality Bug: Unreachable Code

**File:** `agents/learning_scorer.py:94-106`

The file has duplicate exception handling with unreachable code:

```python
except Exception:
    return None
    score = max(0.0, min(1.0, float(raw.get("score", 0.0))))  # UNREACHABLE
    return LearningFeedback(...)                               # UNREACHABLE
except Exception as e:                                         # UNREACHABLE
    print(f"[DEBUG] LearningScorer.collect() failed: {e}", flush=True)
    return None
    return None                                                # UNREACHABLE
```

This doesn't cause the bug but indicates the file has corruption/duplication issues.

### 3. High Thresholds + Conservative Agent Behavior

**Configuration from `main.py`:**

```python
LEARNING_SCORE_PROMPT_EVERY_N=3,    # Collect every 3 turns
LTM_PROMOTE_THRESHOLD=0.65,          # Need score >= 0.65 for LTM
```

**The problem:** The LearningScorer prompt (`_LEARNING_PROMPT`) explicitly tells the agent:

```
Scoring guide:
  1.0  — Highly novel, specific, reusable fact
  0.7  — Useful context likely needed later
  0.4  — Potentially relevant but uncertain
  0.1  — Routine exchange, no new persistent knowledge
  0.0  — Pure procedural step

Be honest and calibrated. Do not inflate scores.
```

**The conversation content was procedural instructions:**
- "SOA necessaria per lavori di restauro..." (initial query)
- "procedi. Ricerca le info dal corpus..." (tool usage instructions)
- "non devi mai leggere il documento per intero..." (procedural guidance)
- "Voglio che aggiungi questa hint nella memoria a lungo termine" (meta-instruction)

**The agent likely scored these interactions as 0.1-0.4** because:
- They're procedural/tool instructions, not "novel facts"
- The prompt explicitly says "Do not inflate scores"
- No personal information, preferences, or project details were shared

### 4. Memory Agent Confidence Threshold

**File:** `agents/memory_agent.py:91`

```
Keep confidence >= 0.7 for ADD operations. Lower-confidence items should be omitted.
```

**File:** `agents/orchestrator.py:545`

```python
if ltm_op.confidence < 0.6:
    continue  # Skip low-confidence operations
```

The MemoryAgent may run at turn 20 but decide:
- Content doesn't meet the 0.7 confidence threshold for ADD
- Nothing novel to store
- No SUMMARY needed (requires "6+ consecutive exchanges about the same topic")

### 5. Missing Diagnostics

**File:** `main.py:148-151` diagnostics only show:

```python
for op in trace.ops_applied:
    if op.success:
        trigger_name = op.trigger.value
        print(f"  [MEM] {op.op.value.upper()} triggered by {trigger_name}")
```

**What's NOT shown:**
- Whether LearningScorer was invoked
- What learning score was returned
- Whether MemoryAgent was triggered
- MemoryAgent's rationale
- Why operations were rejected

---

## Evidence from Logs

### Turn 17-21 Analysis

| Turn | STM Tokens | Events | Expected LTM Activity |
|------|------------|--------|----------------------|
| 17 | 1295 (22%) | Research plan | None (no trigger) |
| 18 | 1348 (22%) | Tool calls | **Learning score SHOULD collect** |
| 19 | 2751 (46%) | Tool calls | None |
| 20 | 2969 (49%) | Web search, tool instructions | **MemoryAgent SHOULD review** |
| 21 | 3004 (50%) | Web search | **Learning score SHOULD collect** |

**Only visible MEM operations:** `RETRIEVE` from tool calls (not LTM ADD/UPDATE/DELETE)

---

## Why LTM is Empty

### Scenario 1: LearningScorer is Failing Silently (Most Likely)

The `chat_json()` call fails (invalid JSON from model), returns None, and is silently ignored.

**To verify:** Add debug logging before line 94:
```python
except Exception as e:
    print(f"[DEBUG] LearningScorer failed: {e}", flush=True)
    return None
```

### Scenario 2: Agent Returns Low Scores (Also Likely)

The agent correctly assesses that procedural instructions aren't worth memorizing:
- Score 0.1-0.4 for tool usage guidance
- Score 0.0-0.2 for general research queries
- No user preferences, identity, or project facts shared

### Scenario 3: MemoryAgent Runs but Declines to ADD

At turn 20, MemoryAgent likely reviewed and decided:
- No entries meet the confidence >= 0.7 threshold
- No 6+ consecutive exchanges on same topic (SUMMARY not needed)
- Content is procedural, not factual

---

## Fixes Required

### Fix 1: Add Debug Logging to LearningScorer

```python
def collect(self, ...):
    probe_messages = list(context_messages) + [
        {"role": "user", "content": _LEARNING_PROMPT}
    ]
    print(f"[DEBUG] LearningScorer: Collecting at turn {turn_index}...", flush=True)
    try:
        raw = self._llm.chat_json(messages=probe_messages, max_tokens=200)
        score = max(0.0, min(1.0, float(raw.get("score", 0.0))))
        print(f"[DEBUG] LearningScorer: Got score={score}, content='{raw.get('affected_content', '')[:50]}...'", flush=True)
        return LearningFeedback(...)
    except Exception as e:
        print(f"[DEBUG] LearningScorer FAILED: {e}", flush=True)
        return None
```

### Fix 2: Show Learning Feedback in Diagnostics

In `main.py`, add to `print_diagnostics()`:

```python
if trace.feedback:
    print(f"  [LEARN] score={trace.feedback.score:.2f}: {trace.feedback.rationale}")
if trace.memory_agent_rationale:
    print(f"  [AGENT] {trace.memory_agent_rationale}")
```

### Fix 3: Lower Thresholds for Testing

In `main.py`, temporarily adjust:

```python
cfg = AgememConfig(
    LTM_PROMOTE_THRESHOLD=0.40,  # Lower from 0.65 for testing
    LEARNING_SCORE_PROMPT_EVERY_N=1,  # Collect every turn for debugging
    TRIGGER_EVERY_N_TURNS=5,  # More frequent reviews
)
```

### Fix 4: Fix Unreachable Code in LearningScorer

Clean up `learning_scorer.py` lines 94-106:

```python
except Exception as e:
    print(f"[DEBUG] LearningScorer.collect() failed: {e}", flush=True)
    return None
```

---

## Test Scenario to Verify LTM Works

Run this conversation and verify LTM captures the information:

```
You: Mi chiamo Marco e lavoro come ingegnere edile. Sto preparando una gara d'appalto.
[Wait for learning score at next multiple of 3]

You: Il mio progetto riguarda il restauro di un ponte storico a Firenze.
[Wait for learning score]

You: Ho bisogno di sapere i requisiti SOA per partecipare.
[Wait for learning score]

/memory
[Should show entries about "Marco", "ingegnere edile", "ponte storico Firenze"]
```

If LTM is still empty after sharing personal/project information, the LearningScorer is failing silently.

---

## Summary

| Issue | Severity | Location |
|-------|----------|----------|
| Silent failures in LearningScorer | **CRITICAL** | `learning_scorer.py:94` |
| Unreachable code | Minor | `learning_scorer.py:96-106` |
| Missing diagnostics | Medium | `main.py:148` |
| High thresholds + procedural content | Expected Behavior | Config + Conversation |

**Recommendation:** Add debug logging first to confirm which scenario is happening. If LearningScorer is failing on JSON parsing, fix the LLM client or prompt. If scores are just low, the system is working as designed.
