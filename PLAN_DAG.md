# PLAN_DAG.md — AgeMem-Hybrid LTM Verification

## 1. Project Summary

AgeMem-Hybrid is an inference-only memory management system for LLM agents that implements Long-Term Memory (LTM) and Short-Term Memory (STM) without RL training. The system uses deterministic System Rules (R1-R4) to trigger memory operations, a Learning Scorer for agent self-assessment, and a Memory Agent for qualitative decisions. "Verified" means all LTM rules are tested against real LLM behavior, documented bugs are fixed, and regression tests achieve ≥80% coverage.

## 2. LTM Rules Inventory

| ID | Rule Description | File + Line Range | Testable Condition |
|----|------------------|-------------------|-------------------|
| LTM-01 | **R1 OVERFLOW_WARN**: When STM utilisation ≥ WARNING_THRESHOLD, trigger SUMMARY | `triggers/system_rules.py:94-104` | `stats.utilisation_ratio >= config.STM_WARNING_THRESHOLD` → returns decision with `rule_id=OVERFLOW_WARN` |
| LTM-02 | **R2 OVERFLOW_CRITICAL**: When STM utilisation ≥ CRITICAL_THRESHOLD, force FILTER + SUMMARY | `triggers/system_rules.py:82-93` | `stats.utilisation_ratio >= config.STM_CRITICAL_THRESHOLD` → returns decision with `rule_id=OVERFLOW_CRITICAL`, priority=100 |
| LTM-03 | **R3 PERIODIC_REVIEW**: Every N turns, invoke MemoryAgent review | `triggers/system_rules.py:108-125` | `turn_index % N == 0` AND `turn_index != last_review_turn` → returns decision with `rule_id=PERIODIC_REVIEW` |
| LTM-04 | **R4 LEARNING_SPIKE**: When learning_score ≥ IMMEDIATE_THRESHOLD, immediate LTM candidacy | `triggers/system_rules.py:127-145` | `feedback.score >= config.LEARNING_SCORE_THRESHOLD_IMMEDIATE` → returns decision with `rule_id=LEARNING_SPIKE`, priority=90 |
| LTM-05 | **Learning Score Collection**: Collect agent self-assessment every N turns | `agents/learning_scorer.py:collect()` | `turn_index % LEARNING_SCORE_PROMPT_EVERY_N == 0` → returns `LearningFeedback` with score in [0,1] |
| LTM-06 | **LTM ADD on Threshold**: Auto-promote content to LTM when score ≥ PROMOTE_THRESHOLD | `agents/orchestrator.py:465-481` | `feedback.score >= config.LTM_PROMOTE_THRESHOLD` → `ltm.add()` called with `TriggerKind.LEARNING_SCORE` |
| LTM-07 | **LTM Duplicate Detection**: Update existing entry instead of ADD if similar exists | `memory/ltm_store.py:66-74` | `add()` with similar content and `learning_score >= LTM_UPDATE_THRESHOLD` → returns `op=UPDATE` not `op=ADD` |
| LTM-08 | **MemoryAgent Confidence Gate**: Only apply ops with confidence ≥ 0.6 | `agents/orchestrator.py:574-575` | `ltm_op.confidence < 0.6` → operation skipped |
| LTM-09 | **LTM Entry Pruning**: Remove lowest-scored entries when exceeding MAX_ENTRIES | `memory/ltm_store.py:218-228` | When `len(entries) > LTM_MAX_ENTRIES`, lowest `learning_score` entries are deleted |
| LTM-10 | **LTM Search/Retrieve**: Inject top-k relevant LTM entries into STM per turn | `agents/orchestrator.py:327-330` | Every `chat()` call → `ltm.search(user_input, top_k=3)` results injected via `stm.retrieve()` |
| LTM-11 | **Double Overflow Guard**: force_fit called both pre-turn AND post-response | `agents/orchestrator.py:323-324, 430-431` | After every user input AND after assistant response → `force_fit()` returns list of ops |
| LTM-12 | **No Silent Failures**: LearningScorer errors are logged, not swallowed | `agents/learning_scorer.py:94-106` | Any exception in `collect()` logs to stderr and returns `None` (not silent) |

## 3. Bug Risk Assessment

| Rule ID | Risk | Reasoning |
|---------|------|-----------|
| LTM-01 | LOW | Straightforward threshold comparison, already unit-tested (T13) |
| LTM-02 | LOW | Same as LTM-01, simple threshold logic |
| LTM-03 | MEDIUM | `_last_review_turn` state can get stale; edge case if evaluate() called multiple times per turn |
| LTM-04 | LOW | Simple float comparison, but relies on LearningScorer working correctly |
| LTM-05 | **HIGH** | `learning_scorer.py` has known unreachable code (lines 94-106) and silent failure risk per `docs/LTM_DEBUG_ANALYSIS.md` |
| LTM-06 | MEDIUM | Depends on LTM-05 working; empty `affected_content` fallback logic exists but may have edge cases |
| LTM-07 | LOW | Duplicate detection is deterministic word-comparison, well-tested (T03) |
| LTM-08 | LOW | Simple float comparison in orchestrator |
| LTM-09 | LOW | Pruning is deterministic sort-and-drop, tested (T05) |
| LTM-10 | LOW | Called every turn, no conditional logic |
| LTM-11 | LOW | Double call is explicit in orchestrator, tested (T20) |
| LTM-12 | **HIGH** | Currently silent failures hide bugs; needs logging verification |

## 4. DAG: Task Nodes

```
NODE 00
  name: Environment Sanity Check
  depends_on: [NONE]
  owner: MAIN_AGENT
  skill: NONE
  inputs: [.venv existence, Python version, import paths]
  outputs: [sanity_report.json]
  success_criteria: |
    - Python 3.11+ detected
    - `python -c "from core.types import *; from memory.ltm_store import LTMStore"` succeeds
    - All imports in test_agemem.py resolve
  estimated_turns: 2

NODE 01
  name: Interactive LTM Probe
  depends_on: [NODE 00]
  owner: MAIN_AGENT
  skill: NONE
  inputs: [main.py, agents/orchestrator.py, mock LLM client]
  outputs: [probe_report.md, observed_behaviors.json, bug_list.md]
  success_criteria: |
    - Agent runs 5+ simulated turns with mock LLM
    - Records actual LearningFeedback scores returned
    - Records actual MemoryAgent decisions
    - Documents any divergence from expected LTM rules
    - Bug list contains at least one confirmed bug or "NO BUGS FOUND"
  estimated_turns: 8

NODE 02
  name: Bug Identification & Fix
  depends_on: [NODE 01]
  owner: MAIN_AGENT
  skill: NONE
  inputs: [bug_list.md, source files]
  outputs: [git diff with fixes, fix_summary.md]
  success_criteria: |
    - Each HIGH/MEDIUM risk bug from assessment is either fixed or documented as "WONTFIX" with rationale
    - All changes pass `python -m py_compile` on modified files
    - No syntax errors or obvious regressions introduced
  estimated_turns: 5

NODE 03
  name: Regression Verification
  depends_on: [NODE 02]
  owner: MAIN_AGENT
  skill: NONE
  inputs: [fixed codebase, probe_report.md from NODE 01]
  outputs: [regression_report.md, updated_observed_behaviors.json]
  success_criteria: |
    - Re-run the same probe from NODE 01
    - Each fixed bug no longer reproduces
    - No new bugs introduced (regression count = 0)
    - LTM entries are created when expected (verified by checking ltm_snapshot())
  estimated_turns: 4

NODE 04
  name: Unit Test Generation
  depends_on: [NODE 02]
  owner: SUBAGENT:test-writer
  skill: NONE
  inputs: [fixed codebase, LTM rules inventory]
  outputs: [test_ltm_rules.py with tests for LTM-01 through LTM-12]
  success_criteria: |
    - Minimum 12 test cases, one per LTM rule
    - Each test has docstring referencing Rule ID
    - Tests use mocked LLM (no network calls)
    - All tests can be run with `python -m pytest test_ltm_rules.py -v`
  estimated_turns: 6

NODE 05
  name: Unit Test Execution & Coverage Gate
  depends_on: [NODE 04, NODE 03]
  owner: MAIN_AGENT
  skill: NONE
  inputs: [test_ltm_rules.py, existing test_agemem.py]
  outputs: [test_results.json, coverage_report.txt]
  success_criteria: |
    - All tests pass (exit code 0)
    - Combined coverage of ltm_store.py, system_rules.py, learning_scorer.py ≥ 80%
    - Coverage report shows line-by-line hit/miss
  estimated_turns: 3

NODE 06
  name: Progress Audit & Final Report
  depends_on: [NODE 05]
  owner: MAIN_AGENT
  skill: NONE
  inputs: [All previous node outputs]
  outputs: [progress.md, final_report.md]
  success_criteria: |
    - progress.md follows template format exactly
    - Each LTM rule has status: PASS / FAIL / PARTIAL
    - Git commit hash of verified state recorded
    - Known limitations documented
  estimated_turns: 3

NODE 07
  name: LTM Rule Cross-Reference Audit
  depends_on: [NODE 05]
  owner: SUBAGENT:reviewer
  skill: NONE
  inputs: [PLAN_DAG.md LTM rules inventory, test_ltm_rules.py, source code]
  outputs: [audit_report.md]
  success_criteria: |
    - Every LTM-01 through LTM-12 rule has at least one test case covering it
    - Every test case maps to at least one rule
    - Any gaps between rules and tests are documented
  estimated_turns: 4
```

## 5. Execution Order

```
                    ┌─────────────────────────────────────┐
                    │           NODE 00                   │
                    │    Environment Sanity Check         │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │           NODE 01                   │
                    │     Interactive LTM Probe           │
                    │   ⏸ HUMAN CHECKPOINT: bug review    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
        ┌─────────────────────┐      ┌─────────────────────┐
        │      NODE 02        │      │      NODE 04        │
        │ Bug Identification  │      │  Unit Test Gen      │
        │      & Fix          │      │  (test-writer)      │
        └──────────┬──────────┘      └──────────┬──────────┘
                   │                            │
                   ▼                            │
        ┌─────────────────────┐                 │
        │      NODE 03        │                 │
        │Regression Verification│               │
        └──────────┬──────────┘                 │
                   │                            │
                   └────────────┬───────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │      NODE 05        │
                    │  Test Execution &   │
                    │   Coverage Gate     │
                    │ ⏸ HUMAN CHECKPOINT │
                    └──────────┬──────────┘
                               │
                  ┌────────────┼────────────┐
                  │            │            │
                  ▼            │            ▼
        ┌─────────────────┐   │   ┌─────────────────┐
        │    NODE 07      │   │   │    NODE 06      │
        │  Cross-Ref Audit│   │   │  Final Report   │
        │   (reviewer)    │   │   │                 │
        └─────────────────┘   │   └─────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    🎯 COMPLETE      │
                    └─────────────────────┘
```

## 6. Rollback Rules

| Node | Failure Condition | Rollback Action |
|------|-------------------|-----------------|
| NODE 00 | Environment check fails | Halt execution, print setup instructions for fixing environment |
| NODE 01 | Cannot run probe | Skip to NODE 04 (unit test generation) with assumption that bugs exist per debug analysis |
| NODE 02 | Fix introduces syntax error | `git checkout -- <file>` to revert, try alternative fix approach |
| NODE 03 | Regression shows new bugs | If regression count > 0, return to NODE 02 with new bug list; max 3 iterations |
| NODE 04 | Test generation fails | MAIN_AGENT takes over test writing; SUBAGENT failure documented |
| NODE 05 | Coverage < 80% | Return to NODE 04 with coverage gap analysis; request additional tests |
| NODE 07 | Audit finds uncovered rules | Return to NODE 04 to add missing tests |
| Any | Human requests abort | Commit current progress to branch `ltm-verification-partial`, halt |

## 7. Human Checkpoints

### Checkpoint 1: After NODE 01 (Bug List Review)
- **Trigger**: NODE 01 completes with bug_list.md
- **Human Action Required**: Review bug_list.md
- **Options**:
  - `APPROVE` → Proceed to NODE 02 and NODE 04
  - `ADD <description>` → Add additional bugs to investigate
  - `SKIP NODE 02` → Skip bug fixes, proceed only to test generation
  - `ABORT` → Halt execution, commit current state

### Checkpoint 2: After NODE 05 (Test Coverage Review)
- **Trigger**: NODE 05 completes with coverage_report.txt
- **Human Action Required**: Review coverage results
- **Options**:
  - `APPROVE` → Proceed to NODE 06 and NODE 07
  - `RAISE <threshold>` → Require higher coverage (e.g., "RAISE 90" for 90%)
  - `RETURN 04` → Send back to test writer with specific requests
  - `ABORT` → Halt execution

### Checkpoint 3: Final Approval (After NODE 06)
- **Trigger**: NODE 06 completes with progress.md and final_report.md
- **Human Action Required**: Final review
- **Options**:
  - `APPROVE` → Commit PLAN_DAG.md changes, tag as verified
  - `REQUEST <changes>` → Iterate on final report
  - `ABORT` → Discard changes, document reason

---

## Appendix: Reference Files

| File | Purpose |
|------|---------|
| `triggers/system_rules.py` | R1-R4 rule implementations |
| `agents/learning_scorer.py` | LTM-05, LTM-12 (HIGH RISK) |
| `agents/orchestrator.py` | LTM-06, LTM-08, LTM-10, LTM-11 |
| `memory/ltm_store.py` | LTM-07, LTM-09 |
| `docs/LTM_DEBUG_ANALYSIS.md` | Known bug documentation |
| `test_agemem.py` | Existing tests T01-T21 |

## Appendix: Verification Commands

```bash
# Run existing tests
python -m unittest test_agemem -v

# Run new LTM rule tests (after NODE 04)
python -m pytest test_ltm_rules.py -v --cov=memory --cov=triggers --cov=agents

# Quick smoke test
python -c "from agents.orchestrator import Orchestrator; print('OK')"
```
