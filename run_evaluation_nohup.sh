#!/bin/bash
#
# run_evaluation_nohup.sh
# -----------------------
# Launches AgeMem E2E LongMemEval evaluation using nohup to survive system sleep.
#
# This script runs the full end-to-end evaluation that replays complete conversation
# sessions and logs every interaction to session.jsonl for later LLM-as-Judge evaluation.
#
# Usage:
#   ./run_evaluation_nohup.sh [dataset] [limit] [target_messages] [resume_session]
#
# Arguments:
#   dataset          - Dataset: s (small), m (medium) (default: s)
#   limit            - Number of instances to process (default: 1, ~550 messages each)
#   target_messages  - Target total message count (0 = all, default: 0)
#   resume_session   - Path to existing session.jsonl to resume from (optional)
#
# Examples:
#   ./run_evaluation_nohup.sh                    # Small dataset, 1 instance (~550 messages)
#   ./run_evaluation_nohup.sh s 1                # Same as above (explicit)
#   ./run_evaluation_nohup.sh s 5                # Small dataset, 5 instances (~2750 messages)
#   ./run_evaluation_nohup.sh m 1                # Medium dataset, 1 instance
#   ./run_evaluation_nohup.sh s 5 0              # Small dataset, 5 instances, all messages
#   ./run_evaluation_nohup.sh s 5 0 evaluation/logs/e2e_20260323_120000/session.jsonl  # Resume
#
# Post-Evaluation:
#   After completion, use the evaluation script to score responses:
#     python3 -m evaluation.evaluate_session --session <session_dir>/session.jsonl
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_KEY="${1:-s}"
LIMIT="${2:-1}"
TARGET_MESSAGES="${3:-0}"
RESUME_SESSION="${4:-}"
LOG_DIR="${SCRIPT_DIR}/evaluation/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_ID="e2e_${TIMESTAMP}"

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
    *)
        DATASET="${DATASET_KEY}"
        DATASET_NAME="$(basename "${DATASET}" .json)"
        ;;
esac

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_highlight() {
    echo -e "${BLUE}[NOTE]${NC} $1"
}

# Create logs directory and session output directory
mkdir -p "${LOG_DIR}"
SESSION_OUTPUT_DIR="${LOG_DIR}/${SESSION_ID}"
mkdir -p "${SESSION_OUTPUT_DIR}"

LOG_FILE="${SESSION_OUTPUT_DIR}/run.log"
PID_FILE="${SESSION_OUTPUT_DIR}/run.pid"
SUMMARY_FILE="${SESSION_OUTPUT_DIR}/summary.txt"

log_info "==============================================="
log_info "  AgeMem E2E LongMemEval Evaluation"
log_info "==============================================="
log_info "Session ID: ${SESSION_ID}"
log_info "Dataset: ${DATASET_NAME} (${DATASET})"
log_info "Instances: ${LIMIT}"
log_info "Target messages: ${TARGET_MESSAGES} (0=all)"
if [[ -n "${RESUME_SESSION}" ]]; then
    log_info "Resume from: ${RESUME_SESSION}"
fi
log_info "Session output: ${SESSION_OUTPUT_DIR}"
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
INSTANCE_COUNT=$(python3 -c "import json; print(len(json.load(open('${DATASET}'))))" 2>/dev/null || echo "?")
log_info "Dataset found: ${DATASET}"
log_info "  Size: ${DATASET_SIZE} bytes"
log_info "  Total instances: ${INSTANCE_COUNT}"

# Validate e2e module imports
log_info "Step 4: Validating Python imports..."
if python3 -c "from evaluation.run_e2e_longmemeval import main; print('e2e module imports successfully')" 2>/dev/null; then
    log_info "E2E module validated"
else
    log_warn "E2E module import had warnings - proceeding anyway"
fi

log_info ""
log_info "==============================================="
log_info "  All validations passed!"
log_info "  Launching E2E evaluation with nohup..."
log_info "==============================================="
log_info ""

# Build the evaluation command
# --output-dir is set to the session directory so session.jsonl is saved there
EVAL_CMD="python3 -u -m evaluation.run_e2e_longmemeval --dataset ${DATASET} --limit ${LIMIT} --target-messages ${TARGET_MESSAGES} --output-dir ${SESSION_OUTPUT_DIR} --verbose"

# Add resume flag if specified
RESUME_FLAG=""
if [[ -n "${RESUME_SESSION}" ]]; then
    if [[ ! -f "${RESUME_SESSION}" ]]; then
        log_error "Resume session file not found: ${RESUME_SESSION}"
        exit 1
    fi
    RESUME_FLAG="--resume ${RESUME_SESSION}"
    EVAL_CMD="${EVAL_CMD} ${RESUME_FLAG}"
fi

log_info "Command: ${EVAL_CMD}"
log_info ""
log_highlight "Output files that will be created:"
log_highlight "  Session log:    ${SESSION_OUTPUT_DIR}/session.jsonl"
log_highlight "  Metadata:       ${SESSION_OUTPUT_DIR}/session.metadata.json"
log_highlight "  Run log:        ${LOG_FILE}"
log_highlight ""
log_highlight "After completion, evaluate with:"
log_highlight "  python3 -m evaluation.evaluate_session --session ${SESSION_OUTPUT_DIR}/session.jsonl"
log_info ""

# Write summary file
RESUME_LINE=""
if [[ -n "${RESUME_SESSION}" ]]; then
    RESUME_LINE="Resume from: ${RESUME_SESSION}"
fi

cat > "${SUMMARY_FILE}" << EOF
AgeMem E2E LongMemEval Session
==============================
Session ID: ${SESSION_ID}
Started: $(date)
Dataset: ${DATASET_NAME} (${DATASET})
Instances: ${LIMIT}
Target messages: ${TARGET_MESSAGES}
${RESUME_LINE}
Output directory: ${SESSION_OUTPUT_DIR}

Command:
${EVAL_CMD}

Output Files:
  session.jsonl       - Complete interaction log (one JSON per line)
  session.metadata.json - Session metadata (status, counts, config)
  run.log             - Console output from the run
  summary.txt         - This file

Post-Evaluation Commands:
  # View session stats
  python3 -c "import json; d=json.load(open('${SESSION_OUTPUT_DIR}/session.metadata.json')); print(json.dumps(d, indent=2))"

  # Run LLM-as-Judge evaluation (when implemented)
  python3 -m evaluation.evaluate_session --session ${SESSION_OUTPUT_DIR}/session.jsonl --output ${SESSION_OUTPUT_DIR}/evaluation_results.json

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
log_info "  Session directory: ${SESSION_OUTPUT_DIR}"
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
log_info "Check session progress:"
log_info "  python3 -c \"import json; d=json.load(open('${SESSION_OUTPUT_DIR}/session.metadata.json')); print(f\"Status: {d['status']}, Completed: {d['completed_interactions']}/{d['total_interactions']}\")\""
log_info ""
log_info "Count completed interactions:"
log_info "  wc -l ${SESSION_OUTPUT_DIR}/session.jsonl"
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
log_info "List all E2E sessions:"
log_info "  ls -la ${LOG_DIR}/ | grep e2e_"
log_info ""
log_info "Evaluate completed session:"
log_info "  python3 -m evaluation.evaluate_session --session ${SESSION_OUTPUT_DIR}/session.jsonl"
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
