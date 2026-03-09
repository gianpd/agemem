# AgeMem-Hybrid LTM Verification — Strategic Plan

> A production-grade, reusable Claude Code workflow for interactive verification,
> bug-fixing, and test generation. Designed for the AgeMem-Hybrid project and
> generalizable to any codebase.

---

## What This Is

A **three-prompt, two-checkpoint** system that takes a codebase from
"LTM behavior is unknown" to "LTM rules verified, bugs fixed, tests committed"
with minimal human time and full auditability.

---

## Artifacts in This Package

| File | Purpose | When Used |
|---|---|---|
| `01_PLANNING_PROMPT.md` | Prompt 1 — Planning Agent builds the DAG | Before any code work |
| `02_EXECUTION_BOOTSTRAP_PROMPT.md` | Prompt 2 — Bootstraps infra + runs the DAG | After human approves DAG |
| `03_AUTONOMOUS_EXECUTION_MECHANISM.md` | Reference — explains how all pieces wire together | Ongoing reference |

---

## How to Run This (End to End)

### Step 1 — Run the Planning Session
```
Paste 01_PLANNING_PROMPT.md into a new Claude Code session.
Claude reads the codebase and produces PLAN_DAG.md.
Review PLAN_DAG.md. Type APPROVE when satisfied.
```

### Step 2 — Run the Execution Session
```
Paste 02_EXECUTION_BOOTSTRAP_PROMPT.md into a new Claude Code session.
Claude creates init.sh, progress.md, skills, and subagents, then begins the DAG.
You can now disengage.
```

### Step 3 — Respond to Checkpoints
```
Claude will stop twice and ask for your review:
  - APPROVE-BUGS: after the live LTM probe, to confirm the bug list
  - APPROVE-TESTS: after tests are written, to confirm coverage

Type the approval token to continue. Optionally provide corrections first.
```

### Step 4 — Done
```
NODE 06 (Final Report) writes a summary to progress.md.
All LTM rules are verified, bugs are fixed, tests are committed.
```

---

## Human Time Required

| Activity | Estimated time |
|---|---|
| Review PLAN_DAG.md (Prompt 1) | 10–15 min |
| APPROVE-BUGS checkpoint | 5 min |
| APPROVE-TESTS checkpoint | 5 min |
| **Total** | **~25 min** |

Everything else runs autonomously.

---

## System Design Principles

**One source of truth:** `progress.md` holds all state. Any session can resume from it.

**Minimal human surface:** Humans only touch decision points where machine judgment is insufficient (confirming bug intent, approving test quality).

**Skill isolation:** Code fixes happen inside `/refactor-with-tests`. Test writing happens inside the `test-writer` subagent. The main agent orchestrates; specialists execute.

**Auditability by construction:** Every node completion appends a structured log. The git history mirrors the DAG. There is no invisible work.

**Resilience:** Interrupted sessions resume via `bash .claude/init.sh` + progress.md. No state lives only in a session's context window.