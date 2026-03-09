---
name: refactor-with-tests
description: >
  Use when a module needs fixing AND tests written for the fix.
  Invoke as: /refactor-with-tests <filepath> "<problem statement>"
tools: Read, Write, Bash(python*), Bash(pytest*), Bash(git*)
model: claude-sonnet-4-6
---

# Skill: refactor-with-tests

## When Invoked
You receive: a filepath and a problem statement describing a confirmed bug.

## Procedure

### Step 1: Understand the Bug
- Read the file thoroughly
- Re-read the problem statement
- Identify the exact lines causing the failure

### Step 2: Fix — Minimal Surface Area
- Change only what is necessary to fix the stated problem
- Do not refactor unrelated logic
- Add inline comments explaining WHY each change was made

### Step 3: Verify the Fix Works
- Run the system-level probe manually: `uv run main.py` (non-interactively if possible)
- Confirm the previously-failing LTM rule now triggers correctly

### Step 4: Write Targeted Unit Tests
The tests MUST:
- Target the EXACT bug that was fixed (not generic coverage)
- Include a comment block at the top: "# Tests for BUG-N: <title>"
- Contain one test that WOULD HAVE CAUGHT the bug before the fix
- Contain one test that verifies the fix is correct
- Contain edge-case tests for boundary conditions of the LTM rule
- Use pytest; mock external LLM calls with `unittest.mock.patch`

### Step 5: Run Tests
```bash
pytest tests/ -v --tb=short --cov=. --cov-report=term-missing
```
- All new tests must pass
- No existing tests may be broken

### Step 6: Commit
```bash
git add <changed files> <test files>
git commit -m "fix(<module>): <what was wrong>

BUG-N: <title>
Root cause: <one sentence>
Tests added: <test file name>"
```

### Step 7: Report
Return a structured summary:
- Bug fixed: BUG-N
- Files changed: list
- Tests written: list with test names
- Coverage delta: before → after
- All tests green: Y/N
