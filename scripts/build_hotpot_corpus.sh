#!/bin/bash
# Build HotpotQA corpus from validation split with gliner NER
# Usage: ./scripts/build_hotpot_corpus.sh [--gold-only]
#
# This uses gliner for entity extraction which makes corpus tools work.
# Estimated time: ~14 hours for full corpus (~66k docs)

set -e

# Configuration
SPLIT="validation"
SETTING="distractor"
CORPUS_DIR="corpus"
LOG_FILE="logs/hotpot_corpus_build.log"
LABELS="generic"
BATCH_SIZE=50

# Check for gold-only flag
GOLD_ONLY=""
if [ "$1" == "--gold-only" ]; then
    GOLD_ONLY="--gold-only"
    echo "Mode: GOLD ONLY (smaller corpus, ~13k gold paragraphs)"
fi

# Create directories
mkdir -p logs

echo "========================================"
echo "HotpotQA Corpus Builder (with GLiNER)"
echo "========================================"
echo "Split: ${SPLIT}"
echo "Setting: ${SETTING}"
echo "Corpus: ${CORPUS_DIR}"
echo "Labels: ${LABELS}"
echo "Log: ${LOG_FILE}"
echo ""
echo "Estimated time: ~14 hours for full corpus"
echo "========================================"

# Run corpus builder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

# Use venv python
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [ ! -f "${PYTHON}" ]; then
    echo "ERROR: Python not found at ${PYTHON}"
    echo "Please ensure .venv is set up"
    exit 1
fi

echo "Using Python: ${PYTHON}"
echo ""

"${PYTHON}" evaluation/hotpot_corpus_fast.py \
    --split "${SPLIT}" \
    --setting "${SETTING}" \
    --corpus-dir "${CORPUS_DIR}" \
    --labels "${LABELS}" \
    --batch-size "${BATCH_SIZE}" \
    ${GOLD_ONLY} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================"
echo "Corpus build complete!"
echo "Corpus: ${CORPUS_DIR}"
echo "Log: ${LOG_FILE}"
echo "========================================"

# Show stats
echo ""
echo "Corpus statistics:"
echo "  Total documents: $(ls -1 ${CORPUS_DIR}/*.md 2>/dev/null | wc -l || echo 'N/A')"
echo "  HotpotQA documents: $(grep -l 'source: hotpotqa' ${CORPUS_DIR}/*.md 2>/dev/null | wc -l || echo 'N/A')"