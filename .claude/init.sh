#!/usr/bin/env bash
# AgeMem-Hybrid LTM Verification — Session Initializer
# Run this at the start of EVERY new agent session: bash .claude/init.sh

set -euo pipefail

echo "=== AgeMem-Hybrid Session Init ==="
echo "Date: $(date -u)"
echo ""

# 1. Show git status
echo "--- Git Status ---"
git log --oneline -8
echo ""

# 2. Show current DAG progress
echo "--- DAG Progress ---"
grep -E "^\- \[" progress.md | head -30 || echo "(progress.md not found)"
echo ""

# 3. Show the NEXT uncompleted node
echo "--- Next Node ---"
grep -m1 "^\- \[ \]" progress.md || echo "ALL NODES COMPLETE"
echo ""

# 4. Environment checks
echo "--- Environment ---"
python --version 2>&1 || echo "WARNING: python not found"
pip show anthropic 2>/dev/null | grep Version || echo "WARNING: anthropic SDK not installed"
echo ""

echo "=== Init complete. Proceed with the next unchecked node. ==="
