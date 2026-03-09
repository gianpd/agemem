# Prompt 1 — Planning Session (Human-in-the-Loop DAG Design)

> **When to use:** Fire this prompt as the **very first Claude Code session** for the AgeMem-Hybrid LTM verification project.  
> **Session type:** Interactive (human reviews and approves before proceeding).  
> **Expected output:** A validated `PLAN_DAG.md` committed to the repo root.

---

## Prompt Text (paste verbatim into Claude Code)

```
You are the Planning Agent for the AgeMem-Hybrid LTM verification project.

## Your Single Objective
Produce a complete, validated PLAN_DAG.md that will guide all future autonomous
agent sessions. Do NOT write any code yet. Do NOT start progress.md yet.
This session ends when a human approves the DAG.

## Context — Read First
1. Read the entire repository structure with `find . -type f | head -80`
2. Read main.py
3. Read any file that contains "LTM", "ltm", "memory" in its name or path
4. Read any existing README or docs

## What You Must Produce: PLAN_DAG.md

The file must contain these exact sections:

### 1. Project Summary (3–5 lines)
What AgeMem-Hybrid is, what LTM rules exist, what "verified" means.

### 2. LTM Rules Inventory
A numbered list of every LTM rule you found in the codebase.
For each rule, write:
- Rule ID (e.g., LTM-01)
- Rule description (one sentence)
- File + line range where it is implemented
- Testable condition (what must be true for this rule to pass)

### 3. Bug Risk Assessment
For each LTM rule, rate the implementation risk: LOW / MEDIUM / HIGH.
Explain your reasoning in one sentence.

### 4. DAG: Task Nodes
Define every task node. Use this exact format for each node:

```
NODE <ID>
  name: <short name>
  depends_on: [<NODE IDs> or NONE]
  owner: <MAIN_AGENT | SUBAGENT:test-writer | SUBAGENT:reviewer>
  skill: <skill name or NONE>
  inputs: <files or artifacts this node consumes>
  outputs: <files or artifacts this node produces>
  success_criteria: <exact, machine-checkable condition>
  estimated_turns: <integer>
```

Required nodes (you may add more if justified):
- NODE 00: Environment Sanity Check
- NODE 01: Interactive LTM Probe (agent talks to main.py, observes real behavior)
- NODE 02: Bug Identification & Fix
- NODE 03: Regression Verification (re-run the probe after fix)
- NODE 04: Unit Test Generation (SUBAGENT:test-writer)
- NODE 05: Unit Test Execution & Coverage Gate
- NODE 06: Progress Audit & Final Report

### 5. Execution Order
A plain ASCII DAG diagram showing node dependencies.
Example:
  NODE00 → NODE01 → NODE02 → NODE03
                                ↓
                            NODE04 → NODE05 → NODE06

### 6. Rollback Rules
What to do if any node fails. Be specific per node.

### 7. Human Checkpoints
Mark every node where a human must approve before proceeding.
Minimum required checkpoints: after NODE01 (bug list review) and after NODE05 (test coverage review).

## After Writing PLAN_DAG.md

1. Print a summary table of all nodes with their success criteria.
2. Print: "⏸ HUMAN CHECKPOINT: Please review PLAN_DAG.md.
   Reply APPROVE to continue, or reply with corrections."
3. STOP. Do not proceed until the human replies.

If the human replies with corrections:
- Apply them directly to PLAN_DAG.md
- Reprint the summary table
- Ask for approval again

If the human replies APPROVE:
- Run: git add PLAN_DAG.md && git commit -m "plan: initial DAG for LTM verification"
- Print: "✅ DAG committed. Run Prompt 2 (EXECUTION_BOOTSTRAP_PROMPT.md) to begin autonomous execution."
- End the session.
```

---

## What the Human Reviews Before Approving

Check the following before typing APPROVE:

| Check | What to verify |
|---|---|
| LTM Rules complete | Every rule in the codebase appears in the inventory |
| Testable criteria | Each success criterion is specific and binary (pass/fail) |
| Correct owners | Nodes that write tests are assigned to SUBAGENT:test-writer |
| Realistic estimates | `estimated_turns` are not all set to 1 |
| No scope creep | No node rewrites unrelated code |
| Checkpoints present | At least 2 human checkpoints are marked |