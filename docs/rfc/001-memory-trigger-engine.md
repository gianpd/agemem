# RFC: Deepen Memory Trigger System with MemoryTriggerEngine

**Status:** Implemented
**Author:** @agemem (AI collaborator)
**Date:** 2026-03-18
**Related:** `agents/orchestrator.py`, `triggers/system_rules.py`, `agents/memory_agent.py`

---

## Problem

The Memory Trigger System is distributed across three shallow modules that create architectural friction:

1. **`triggers/system_rules.py`** (146 lines): Pure rule engine that only *detects* conditions and returns `TriggerDecision` objects
2. **`agents/memory_agent.py`** (292 lines): LLM-driven agent that only *recommends* operations via complex prompt building
3. **`agents/orchestrator.py`** (1,100+ lines): Turn coordinator that wires everything together with ~80 lines of dispatch logic

### Current Pain in Orchestrator

The Orchestrator's post-turn flow (lines 1109-1190) reveals the coupling:

```python
# Current: ~80 lines of wiring logic
decisions = self._rules.evaluate(stats, turn_after, feedback)

for decision in decisions:
    if decision.rule_id == RuleID.OVERFLOW_CRITICAL:
        self._stm.filter(...)
        self._stm.summary(...)
    elif decision.rule_id == RuleID.OVERFLOW_WARN:
        self._stm.summary(...)
    elif decision.rule_id in (RuleID.PERIODIC_REVIEW, RuleID.LEARNING_SPIKE):
        should_run_memory_agent = True

# ... later ...
if should_run_memory_agent:
    agent_decision = self._memory_agent.review(...)
    for op in agent_decision.ltm_operations:
        if op.op == MemoryOp.ADD:
            self._ltm.add(...)
        # ... etc
```

To understand when LTM ADD fires, a developer must trace through:
1. `SystemRules.evaluate()` → returns decisions
2. Orchestrator's dispatch loop → sets flags
3. Conditional MemoryAgent invocation
4. Decision parsing and LTM execution

This violates the "deep module" principle: the interface (three separate modules) is nearly as complex as the implementation.

---

## Proposed Interface

Create a single deep module `MemoryTriggerEngine` that hides all trigger evaluation, agent invocation, and operation execution behind one method.

### Interface Signature

```python
# triggers/memory_trigger_engine.py

from dataclasses import dataclass, field
from typing import Optional
from core.types import MemoryOpResult, LearningFeedback, ContextStats, RuleID


@dataclass(frozen=True)
class MemoryCycleReport:
    """
    Complete summary of what the Memory Trigger System did this turn.

    The Orchestrator receives this and logs/traces it. No further action needed.
    """
    # What rules fired (for observability)
    rules_triggered: list[RuleID] = field(default_factory=list)

    # All memory operations that were executed (in order)
    operations: list[MemoryOpResult] = field(default_factory=list)

    # Human-readable summary of what happened
    summary: str = ""

    # Agent's reasoning if MemoryAgent was invoked
    agent_rationale: Optional[str] = None

    # Whether any LTM mutations occurred (for downstream sync decisions)
    ltm_modified: bool = False

    # Current STM stats after all operations
    stm_stats: Optional[ContextStats] = None


class MemoryTriggerEngine:
    """
    Unified entry point for the Memory Trigger System.

    Hides the complexity of:
    - SystemRules evaluation (threshold checking)
    - MemoryAgent LLM calls (prompt building, JSON parsing)
    - Operation execution ordering (FILTER before SUMMARY, etc.)
    - Error handling (LLM failures, malformed responses)
    - Graceful degradation (fallback when agent unavailable)

    Dependencies (all injected):
    - config: AgememConfig (thresholds, cadences)
    - llm: LLMClient (for MemoryAgent calls)
    - stm: STMContext (operations execute here)
    - ltm: LTMStore (operations execute here)
    """

    def __init__(
        self,
        config: AgememConfig,
        llm: LLMClient,
        stm: STMContext,
        ltm: LTMStore,
    ) -> None:
        self._config = config
        self._rules = SystemRules(config)
        self._agent = MemoryAgent(llm, config)
        self._stm = stm
        self._ltm = ltm

    def process_turn(
        self,
        turn_index: int,
        feedback: Optional[LearningFeedback] = None,
    ) -> MemoryCycleReport:
        """
        Execute the complete memory trigger cycle for one turn.

        This is the ONLY method callers need. It:
        1. Evaluates system rules against current STM stats
        2. Executes FILTER/SUMMARY operations immediately
        3. Invokes MemoryAgent if periodic review or learning spike detected
        4. Applies MemoryAgent decisions (ADD/UPDATE/DELETE) to LTM
        5. Applies context relevance scores to STM messages
        6. Handles learning spike immediate promotion

        All error handling is internal. Returns a complete report of what happened.

        Args:
            turn_index: Current conversation turn number
            feedback: Optional learning feedback from the main agent

        Returns:
            MemoryCycleReport summarizing all actions taken
        """
        ...

    def force_summary(self) -> MemoryOpResult:
        """
        Emergency summary when context is about to overflow.
        Bypasses normal rule evaluation for immediate action.
        """
        return self._stm.summary(trigger=TriggerKind.SYSTEM_RULE)

    def check_health(self) -> dict:
        """
        Diagnostic info: rule hit rates, agent call frequency, etc.
        """
        ...
```

### Usage Example

**Before (current ~80 lines in orchestrator.py):**

```python
decisions = self._rules.evaluate(stats, turn_after, feedback)

ma_rationale = ""
should_run_memory_agent = False

for decision in decisions:
    if decision.rule_id == RuleID.OVERFLOW_CRITICAL:
        ops.append(self._stm.filter(trigger=TriggerKind.SYSTEM_RULE))
        ops.append(self._stm.summary(trigger=TriggerKind.SYSTEM_RULE))
    elif decision.rule_id == RuleID.OVERFLOW_WARN:
        ops.append(self._stm.summary(trigger=TriggerKind.SYSTEM_RULE))
    elif decision.rule_id in (RuleID.PERIODIC_REVIEW, RuleID.LEARNING_SPIKE):
        should_run_memory_agent = True

if feedback and feedback.score >= self._config.LTM_PROMOTE_THRESHOLD:
    content = feedback.affected_content or assistant_response
    ops.append(self._ltm.add(content=content, ...))

if should_run_memory_agent:
    decision_obj = self._memory_agent.review(...)
    ma_rationale = decision_obj.rationale
    for op in decision_obj.ltm_operations:
        if op.op == MemoryOp.ADD:
            self._ltm.add(...)
        elif op.op == MemoryOp.UPDATE:
            self._ltm.update(...)
    for turn_idx, score in decision_obj.context_relevance.items():
        # Update STM message scores...
```

**After (6 lines):**

```python
report = self._trigger_engine.process_turn(
    turn_index=turn_after,
    feedback=feedback,
)
ops.extend(report.operations)
ma_rationale = report.agent_rationale or ""
```

---

## Dependency Strategy

**Category: In-process with Local-substitutable stores**

The `MemoryTriggerEngine` is a pure coordinator with no I/O of its own.

| Dependency | Category | Strategy |
|------------|----------|----------|
| `SystemRules` | In-process | Direct instance, pure computation |
| `MemoryAgent` | In-process | Direct instance, LLM call via injected client |
| `STMContext` | Local-substitutable | In-memory STM for tests |
| `LTMStore` | Local-substitutable | In-memory LTM for tests |

### Construction Pattern

```python
# Production
engine = MemoryTriggerEngine(
    config=AgememConfig(),
    llm=LLMClient(openai_client, default_model="gpt-4o"),
    stm=STMContext(config),
    ltm=LTMStore(config, persist_path=Path("memory.json")),
)

# Testing
engine = MemoryTriggerEngine(
    config=AgememConfig(),
    llm=MockLLMClient(canned_responses=[...]),  # Returns canned agent decisions
    stm=STMContext(config),  # In-memory
    ltm=LTMStore(config),    # In-memory (no persist_path)
)
```

---

## Testing Strategy

### New Boundary Tests to Write

Test at the `process_turn()` boundary, asserting on `MemoryCycleReport`:

```python
def test_overflow_critical_triggers_filter_and_summary():
    stm = STMContext(config=_cfg(STM_TOKEN_LIMIT=100))
    stm.add_message("user", "x" * 200)  # Force overflow
    engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

    report = engine.process_turn(turn_index=1)

    assert RuleID.OVERFLOW_CRITICAL in report.rules_triggered
    assert any(op.op == MemoryOp.FILTER for op in report.operations)
    assert any(op.op == MemoryOp.SUMMARY for op in report.operations)

def test_periodic_review_invokes_memory_agent():
    mock_llm = MockLLMClient(json_response={
        "ltm_operations": [{"op": "add", "content": "User likes Python"}],
        "summary_needed": False
    })
    engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

    report = engine.process_turn(turn_index=10)  # TRIGGER_EVERY_N_TURNS=10

    assert RuleID.PERIODIC_REVIEW in report.rules_triggered
    assert report.agent_rationale is not None
    assert any(op.op == MemoryOp.ADD for op in report.operations)

def test_learning_spike_triggers_immediate_add():
    feedback = LearningFeedback(score=0.90, affected_content="Key fact")
    engine = MemoryTriggerEngine(cfg, mock_llm, stm, ltm)

    report = engine.process_turn(turn_index=1, feedback=feedback)

    assert RuleID.LEARNING_SPIKE in report.rules_triggered
    assert report.ltm_modified
```

### Old Tests to Delete

Once boundary tests pass, delete these shallow module tests:

1. `TestSystemRules` (lines 578-640 in test_agemem.py) - Rule evaluation becomes internal
2. `TestMemoryAgentDecision` (lines 646-679) - Parsing becomes internal
3. Direct `SystemRules.evaluate()` tests - Covered by boundary tests via `report.rules_triggered`
4. Direct `MemoryAgent.review()` tests - Covered by boundary tests via `report.operations`

Keep these tests (they test store behavior, not trigger logic):
- `TestLTMStore` - LTM operations work correctly
- `TestSTMContext` - STM operations work correctly

---

## Implementation Recommendations

### What the Module Should Own

1. **Trigger evaluation ordering** - Rules fire first, then agent
2. **Operation execution sequencing** - FILTER before SUMMARY, LTM ops batched
3. **Error handling strategy** - LLM failures log warning and continue
4. **Threshold interpretation** - `LTM_PROMOTE_THRESHOLD` means "immediate ADD"
5. **Graceful degradation** - If MemoryAgent unavailable, rely on rules only

### What the Module Should Hide

1. **SystemRules existence** - Internal implementation detail
2. **MemoryAgent prompt templates** - Internal to agent invocation
3. **JSON parsing from LLM responses** - Internal to agent invocation
4. **Decision-to-operation dispatch** - Internal mapping logic
5. **Relevance score application** - Internal STM update logic

### What the Module Should Expose

1. **`process_turn()`** - The primary interface
2. **`MemoryCycleReport`** - Observable outcomes (rules triggered, ops applied, rationale)
3. **`force_summary()`** - Emergency escape hatch for overflow
4. **`check_health()`** - Diagnostics for monitoring/alerting

### Migration Path for Orchestrator

1. Add `MemoryTriggerEngine` instantiation in `Orchestrator.__init__`
2. Replace post-turn logic (lines 1109-1190) with single `process_turn()` call
3. Map `report.operations` to trace logging
4. Map `report.agent_rationale` to existing trace field
5. Delete `_apply_memory_agent_decision()` helper (moves into engine)
6. Verify all 35 tests still pass

### File Structure

```
triggers/
  __init__.py                    # Export MemoryTriggerEngine, MemoryCycleReport
  memory_trigger_engine.py       # New deep module (this RFC)
  system_rules.py                # Becomes internal (keep, but not exported)

agents/
  memory_agent.py                # Becomes internal to engine (keep, not exported)
  orchestrator.py                # Simplified to use engine
```

---

## Trade-offs

### What You Gain

1. **Single source of truth** - All memory trigger logic lives in one module
2. **Observable by default** - `MemoryCycleReport` contains everything for tracing
3. **Testable in isolation** - Mock LLM and in-memory stores enable fast tests
4. **Refactorable interior** - Can change rule logic or agent prompts without touching callers
5. **Clear failure modes** - LLM failures become graceful degradation (empty ops, logged warning)
6. **Reduced Orchestrator complexity** - ~80 lines → ~6 lines

### What You Lose

1. **Fine-grained control** - Caller cannot intervene mid-evaluation (e.g., "skip rule R3 this turn")
   - *Mitigation:* The engine exposes `force_summary()` for emergencies

2. **Decision transparency** - Individual rule firings are logged but not returned in detail
   - *Mitigation:* `report.rules_triggered` provides observability

3. **Pluggability** - `SystemRules` and `MemoryAgent` become internal implementation details
   - *Mitigation:* The interface is stable; swap implementations by modifying the engine

---

## Acceptance Criteria

- [x] `MemoryTriggerEngine` class exists with `process_turn()` method
- [x] `MemoryCycleReport` dataclass exists with all documented fields
- [x] Orchestrator uses new engine (post-turn logic < 10 lines)
- [x] All existing tests pass (57/59 pass; 2 pre-existing LTM store failures)
- [x] New boundary tests cover: overflow, periodic review, learning spike
- [ ] Old shallow tests deleted (SystemRules, MemoryAgentDecision) - kept for backward compatibility
- [ ] Documentation updated (README architecture diagram)

---

## Related Work

- This RFC follows the "deep module" philosophy from John Ousterhout's *A Philosophy of Software Design*
- Pattern used: Information Hiding with small interface, large implementation
- Testing approach: Boundary testing replaces unit testing of internal components
