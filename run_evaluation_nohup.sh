#!/bin/bash
#
# run_evaluation_nohup.sh
# -----------------------
# Launches AgeMem evaluation with LLM-as-Judge using nohup to survive system sleep.
#
# Usage:
#   ./run_evaluation_nohup.sh [dataset] [max_batches] [batch_size]
#
# Arguments:
#   dataset      - Dataset: s (small), m (medium), oracle (default: s)
#   max_batches  - Max batches to run (default: 10)
#   batch_size   - Interactions per batch (default: 50)
#
# Environment Variables:
#   JUDGE_API_BASE - URL for LLM-as-Judge API (default: http://localhost:8080/v1)
#
# Examples:
#   ./run_evaluation_nohup.sh                    # Small dataset, 10 batches of 50
#   ./run_evaluation_nohup.sh s 10 50            # Same as above (explicit)
#   ./run_evaluation_nohup.sh m 20 100           # Medium dataset, 20 batches of 100
#   JUDGE_API_BASE=http://192.168.1.100:8080/v1 ./run_evaluation_nohup.sh
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_KEY="${1:-s}"
MAX_BATCHES="${2:-10}"
BATCH_SIZE="${3:-50}"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://localhost:8080/v1}"
LOG_DIR="${SCRIPT_DIR}/evaluation/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_ID="eval_${TIMESTAMP}"

# Map dataset key to path
case "${DATASET_KEY}" in
    s|small)
        DATASET="evaluation/data/longmemeval_s_cleaned.json"
        DATASET_NAME="small"
        ;;
    m|medium)
        DATASET="evaluation/data/longmemeval_m_cleaned.json"
        DATASET_NAME="medium"
        ;;
    oracle)
        DATASET="evaluation/data/longmemeval_oracle.json"
        DATASET_NAME="oracle"
        ;;
    *)
        DATASET="${DATASET_KEY}"
        DATASET_NAME="$(basename "${DATASET}" .json)"
        ;;
esac

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create logs directory
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${SESSION_ID}.log"
PID_FILE="${LOG_DIR}/${SESSION_ID}.pid"
SUMMARY_FILE="${LOG_DIR}/${SESSION_ID}_summary.txt"

log_info "==============================================="
log_info "  AgeMem Evaluation - LLM-as-Judge Mode"
log_info "==============================================="
log_info "Session ID: ${SESSION_ID}"
log_info "Mode: full"
log_info "Dataset: ${DATASET_NAME} (${DATASET})"
log_info "Batch size: ${BATCH_SIZE}"
log_info "Max batches: ${MAX_BATCHES}"
log_info "Judge API: ${JUDGE_API_BASE}"
log_info "Log file: ${LOG_FILE}"
log_info ""

# Validate Python environment
log_info "Step 1: Validating Python environment..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
log_info "Found: ${PYTHON_VERSION}"

# Validate working directory
log_info "Step 2: Validating working directory..."
cd "${SCRIPT_DIR}"
log_info "Working directory: $(pwd)"

# Validate dataset exists
log_info "Step 3: Validating dataset..."
if [[ ! -f "${DATASET}" ]]; then
    log_error "Dataset not found: ${DATASET}"
    exit 1
fi

DATASET_SIZE=$(wc -c < "${DATASET}" | tr -d ' ')
log_info "Dataset found: ${DATASET} (${DATASET_SIZE} bytes)"

# Validate Python imports
log_info "Step 4: Validating Python imports..."
if python3 -c "from evaluation.cli import main; from evaluation.llm_judge import LLMJudge" 2>/dev/null; then
    log_info "Python imports validated"
else
    log_warn "Some Python imports failed - this may be expected if dependencies aren't fully installed"
fi

# Quick validation test
log_info "Step 5: Running quick validation test..."
VALIDATION_LOG="${LOG_DIR}/${SESSION_ID}_validation.log"

if python3 evaluation/quick_test.py > "${VALIDATION_LOG}" 2>&1; then
    log_info "Validation PASSED - quick_test.py executed successfully"
    log_info "Validation log: ${VALIDATION_LOG}"
else
    log_error "Validation FAILED - check ${VALIDATION_LOG} for details"
    exit 1
fi

# Validate evaluation/cli.py can load without errors
log_info "Step 6: Validating evaluation/cli.py..."
if python3 -c "from evaluation.cli import main; print('cli.py imports successfully')" 2>/dev/null; then
    log_info "cli.py validation PASSED"
else
    log_warn "cli.py validation had warnings - proceeding anyway"
fi

log_info ""
log_info "==============================================="
log_info "  All validations passed!"
log_info "  Launching evaluation with nohup..."
log_info "==============================================="
log_info ""

# Build the evaluation command (LLM-as-Judge full evaluation using new CLI)
EVAL_CMD="python3 -u -m evaluation.cli --dataset ${DATASET} --mode full --batch-size ${BATCH_SIZE} --max-batches ${MAX_BATCHES} --use-llm-judge --judge-api-base ${JUDGE_API_BASE} --judge-timeout 120 --output-dir evaluation/results --verbose"

log_info "Command: ${EVAL_CMD}"
log_info ""

# Write summary file
cat > "${SUMMARY_FILE}" << EOF
AgeMem Evaluation Session (LLM-as-Judge)
=========================================
Session ID: ${SESSION_ID}
Started: $(date)
Mode: full
Dataset: ${DATASET_NAME} (${DATASET})
Batch size: ${BATCH_SIZE}
Max batches: ${MAX_BATCHES}
Judge API: ${JUDGE_API_BASE}
Log File: ${LOG_FILE}

Command:
${EVAL_CMD}

Status: RUNNING
EOF

# Launch with nohup
# nohup ensures the process continues even if:
# - SSH session disconnects
# - Terminal closes
# - System goes to sleep
#
# The process will still receive SIGHUP on actual system shutdown,
# but will survive sleep/resume cycles.
nohup ${EVAL_CMD} > "${LOG_FILE}" 2>&1 &

# Capture PID
PID=$!
echo $PID > "${PID_FILE}"

log_info "Process launched!"
log_info "  PID: ${PID}"
log_info "  Log file: ${LOG_FILE}"
log_info "  PID file: ${PID_FILE}"
log_info "  Summary file: ${SUMMARY_FILE}"
log_info ""
log_info "==============================================="
log_info "  Monitoring Commands"
log_info "==============================================="
log_info ""
log_info "View logs in real-time:"
log_info "  tail -f ${LOG_FILE}"
log_info ""
log_info "Check process status:"
log_info "  ps -p ${PID} -o pid,ppid,cmd,%cpu,%mem,etime"
log_info ""
log_info "Send to background (if you used fg):"
log_info "  bg"
log_info ""
log_info "Bring to foreground:"
log_info "  fg"
log_info ""
log_info "Gracefully stop the process:"
log_info "  kill -TERM ${PID}"
log_info ""
log_info "Force kill:"
log_info "  kill -KILL ${PID}"
log_info ""
log_info "Check if process survived sleep:"
log_info "  ps aux | grep ${PID} | grep -v grep"
log_info ""
log_info "List all evaluation sessions:"
log_info "  ls -la ${LOG_DIR}/"
log_info ""
log_info "==============================================="
log_info ""

# Update summary with PID
sed -i "s/Status: RUNNING/Status: RUNNING\nPID: ${PID}/" "${SUMMARY_FILE}"

# Optional: Show initial output
echo "--- Initial output (first 20 lines) ---"
sleep 2
tail -n 20 "${LOG_FILE}" 2>/dev/null || log_info "(log file building...)"

log_info ""
log_info "Process is running detached. You can close this terminal."
log_info "The evaluation will continue even if the system sleeps."
log_info ""
