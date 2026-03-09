# Autonomous Execution Mechanism

> This document explains **how the agent system operates end-to-end** once initiated —
> the wiring between progress.md, init.sh, skills, subagents, and the DAG. It is both
> a reference for the human operator and a document the agent can read to self-orient.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HUMAN OPERATOR                               │
│  Prompt 1 → reviews DAG → APPROVE                               │
│  Prompt 2 → monitors checkpoints → APPROVE-BUGS / APPROVE-TESTS │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MAIN AGENT SESSION                               │
│                                                                  │
│  bash .claude/init.sh  ←──── runs at session start/resume       │
│         │                                                        │
│         ▼                                                        │
│  reads progress.md ──→ finds next [  ] or [~] node              │
│         │                                                        │
│         ▼                                                        │
│  PLAN_DAG.md ──────→ reads node spec (inputs/outputs/criteria)   │
│         │                                                        │
│         ├── code fix node? ──→ invokes /refactor-with-tests      │
│         │                              (skill)                   │
│         ├── test node?  ──→ delegates to SUBAGENT:test-writer    │
│         │                              (agent)                   │
│         └── other node? ──→ executes directly                    │
│                                                                  │
│  After each node:                                                │
│    updates progress.md → git commit → moves to next node        │
│                                                                  │
│  At checkpoint nodes:                                            │
│    prints checkpoint message → STOPS → waits for human          │
└─────────────────────────────────────────────────────────────────┘
                         │
           ┌─────────────┼──────────────┐
           ▼             ▼              ▼
  .claude/skills/   .claude/agents/   progress.md
  refactor-with-    test-writer.md    (audit log)
  tests/SKILL.md
```

---

## The State Machine

Each node follows this exact lifecycle. The main agent enforces it without exception.

```
PENDING [ ] → IN_PROGRESS [~] → COMPLETE [x]
                    │
                    └──→ BLOCKED [!] → (rollback per PLAN_DAG.md)
```

| Status | Meaning | What happens next |
|---|---|---|
| `[ ]` | Not started | Next session picks it up |
| `[~]` | In progress | Current session is working it |
| `[x]` | Complete | Move to next node |
| `[!]` | Blocked | Apply rollback rule, append blocker to progress.md |

A node is never marked `[x]` until its `success_criteria` from PLAN_DAG.md is met exactly.

---

## How progress.md Enables Autonomy

`progress.md` is the **single source of truth** for all sessions. It eliminates the
need for human coordination between sessions by encoding all state in git.

Every node completion appends a structured log entry:

```markdown
### NODE 02 — Bug Fix | COMPLETE | 2026-03-09T14:32:00Z
**Agent:** MAIN_AGENT
**Actions taken:**
- Read LTM storage logic in memory_manager.py
- Identified off-by-one error in message counter (line 87)
- Invoked /refactor-with-tests memory_manager.py "LTM-01: counter resets at 9 not 10"

**Findings:**
- BUG-1: `if count >= 9` should be `if count >= 10` (LTM-01)
- BUG-2: importance flag never checked after initial classification (LTM-02)

**Artifacts produced:**
- memory_manager.py (fixed)
- commit: abc1234

**Tests passed:** Deferred to NODE 04
**Blockers:** NONE
**Next node to run:** NODE 03
```

When a new session starts, `bash .claude/init.sh` prints the last few log entries
and the first uncompleted node — the agent has full context in under 30 seconds.

---

## How Skills Are Invoked

The `/refactor-with-tests` skill is invoked by the main agent whenever NODE 02
(or any fix node) is active. The invocation pattern is:

```
/refactor-with-tests <filepath> "<exact problem statement from Bug Registry>"
```

Example:
```
/refactor-with-tests src/memory_manager.py "BUG-1: LTM-01 counter triggers at 9
messages instead of 10. count variable initialized to 1 instead of 0 at line 87."
```

The skill then:
1. Reads the file and the problem statement
2. Makes a minimal, targeted fix
3. Verifies the fix by running main.py
4. Writes targeted unit tests
5. Runs pytest
6. Commits everything with a structured message
7. Returns a structured report that the main agent appends to progress.md

**Why a skill instead of inline instructions?**
Skills are versioned, reusable across projects, and apply consistently regardless
of which session runs them. The behavior is the same whether it's the first run
or a resume after an interruption three days later.

---

## How Subagents Are Delegated

When NODE 04 begins, the main agent delegates to `test-writer` with a full context
package. The delegation prompt the main agent sends:

```
Subagent test-writer:

BUG-N: <copy exact entry from Bug Registry in progress.md>
LTM Rule: LTM-XX — <rule description>
Fixed file: <path>
Test file target: tests/test_ltm_<rule_id>.py

Write tests per your SKILL.md instructions.
Run pytest before returning.
Return your structured summary when done.
```

The subagent runs in its own context with only the tools it needs (Read, Write, Bash(pytest*)).
It cannot modify production code. This isolation prevents test-writing from
accidentally introducing regressions.

The main agent receives the subagent's structured summary and appends it to progress.md.

---

## Human Checkpoint Protocol

There are exactly two hard checkpoints. The main agent CANNOT continue past them without
explicit human approval. This is enforced by the agent printing a stop message and
entering an await state — it will not execute any tools until the human responds.

### Checkpoint 1: After NODE 01 (Bug Discovery)
**Human reviews:** The complete list of LTM rules found to be failing and the
raw interaction transcripts from talking to the live agent.

**Human decision:**
- `APPROVE-BUGS` — confirms the bug list is accurate, agent proceeds to fix
- Corrections — human amends the bug list, agent updates Bug Registry, re-presents

**Why this matters:** Without human validation, an agent might fix the wrong thing
or misidentify a feature as a bug. The human confirms intent here.

### Checkpoint 2: After NODE 05 (Test Coverage)
**Human reviews:** pytest output, coverage report, and that all LTM rules
have at least one passing test.

**Human decision:**
- `APPROVE-TESTS` — confirms test quality, agent proceeds to Final Report
- Corrections — agent adds missing tests or re-runs failing ones

**Why this matters:** Tests written by an agent can be tautological (testing the
implementation rather than the spec). Human review catches this.

---

## Resilience: What Happens When Things Go Wrong

| Failure scenario | Detection | Recovery |
|---|---|---|
| Session interrupted mid-node | Node marked `[~]` in progress.md | Next session resumes from `[~]` node |
| main.py crashes during probe | Bash exit code non-zero | Agent logs blocker `[!]`, applies NODE 01 rollback rule from PLAN_DAG.md |
| Fix breaks existing tests | pytest exits non-zero | /refactor-with-tests skill reverts change, logs to Bug Registry |
| Subagent produces failing tests | pytest exits non-zero | Main agent re-delegates with additional context |
| DAG node has no clear success path | Agent cannot meet success_criteria | Node marked `[!]`, human checkpoint triggered |

---

## File Inventory After Full Setup

```
repo-root/
├── PLAN_DAG.md                         ← approved by human in Prompt 1
├── progress.md                         ← live audit log, updated every node
├── .claude/
│   ├── init.sh                         ← run at every session start
│   ├── skills/
│   │   └── refactor-with-tests/
│   │       └── SKILL.md                ← invoked for all code fixes
│   └── agents/
│       └── test-writer.md              ← delegated for all test writing
└── tests/
    └── test_ltm_<rule_id>.py           ← written by test-writer subagent
```

---

## Reusing This System for Other Projects

This entire mechanism is project-agnostic. To adapt it:

1. **Replace the LTM probe** (NODE 01) with whatever interactive verification
   your project requires — API calls, UI walkthroughs, integration tests.

2. **Update the LTM Rules Inventory section** in PLAN_DAG.md with your project's
   invariants or acceptance criteria.

3. **Keep the skeleton unchanged:** init.sh, progress.md structure, skill invocation
   pattern, subagent delegation pattern, and checkpoint protocol are universal.

4. **Adjust node count:** Add or remove nodes in PLAN_DAG.md. The state machine
   and progress.md format handle any number of nodes without modification.

The system works for: API integration testing, database migration verification,
performance regression hunting, security audit + fix cycles, and any workflow
that alternates between "discover problems" and "fix + verify" phases.