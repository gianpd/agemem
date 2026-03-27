#!/usr/bin/env bash
# =============================================================================
# run_hotpotqa_resume.sh
#
# Resumes HotpotQA evaluation from a previous log using resume_hotpotqa.py.
# Uses tmux to survive terminal close, SSH disconnect, and system sleep.
#
# Usage:
#   chmod +x run_hotpotqa_resume.sh
#   ./run_hotpotqa_resume.sh /path/to/logfile.log
#   ./run_hotpotqa_resume.sh evaluation/logs/hotpotqa_20260326_120905.log
#
#   # With custom judge model (e.g., Gemini via OpenRouter)
#   JUDGE_MODEL=google/gemini-3.1-flash-lite-preview ./run_hotpotqa_resume.sh logfile.log
#
#   # Retry failed samples
#   RETRY_FAILED=true ./run_hotpotqa_resume.sh logfile.log
#
# Output:
#   evaluation/results/hotpot_resume_<timestamp>.json
#   evaluation/results/hotpot_resume_<timestamp>.md
#   evaluation/logs/hotpot_resume_<timestamp>.log
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env vars)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${PROJECT_ROOT}/evaluation/results"
LOGS_DIR="${PROJECT_ROOT}/evaluation/logs"
PERSIST_DIR="${PROJECT_ROOT}/evaluation/persist/resume_${TIMESTAMP}"

RESUME_SCRIPT="${PROJECT_ROOT}/evaluation/resume_hotpotqa.py"

# Required: log file to resume from
LOG_FILE="${1:-}"

# Optional overrides
MODE="${MODE:-corpus}"
SPLIT="${SPLIT:-validation}"
SETTING="${SETTING:-distractor}"
CORPUS_DIR="${CORPUS_DIR:-${PROJECT_ROOT}/corpus}"
LIMIT="${LIMIT:-0}"
SKIP_INGEST="${SKIP_INGEST:-true}"  # Default to true for resume (corpus already ingested)
RETRY_FAILED="${RETRY_FAILED:-false}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-3.1-flash-lite-preview}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"

TMUX_SESSION="hotpot_resume_${TIMESTAMP}"

JSON_OUTPUT="${RESULTS_DIR}/hotpot_resume_${TIMESTAMP}.json"
MD_OUTPUT="${RESULTS_DIR}/hotpot_resume_${TIMESTAMP}.md"
LOG_OUTPUT="${LOGS_DIR}/hotpot_resume_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" &>/dev/null || die "'$1' not found in PATH"
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

preflight() {
    log "── Preflight checks ──────────────────────────────────────────"

    # Check required args
    [[ -n "${LOG_FILE}" ]] || die "Usage: $0 <log_file>"
    [[ -f "${LOG_FILE}" ]] || die "Log file not found: ${LOG_FILE}"

    # Check required tools
    require_cmd tmux
    require_cmd python3

    # Check script exists
    [[ -f "${RESUME_SCRIPT}" ]] || die "Resume script not found: ${RESUME_SCRIPT}"

    # Check corpus if needed
    if [[ "${MODE}" == "corpus" && "${SKIP_INGEST}" != "true" ]]; then
        [[ -d "${CORPUS_DIR}" ]] || die "Corpus directory not found: ${CORPUS_DIR}"
    fi

    # Check OpenRouter API key if using OpenRouter
    if [[ "${JUDGE_BASE_URL}" == *"openrouter"* ]]; then
        if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
            # Try to load from .env
            if [[ -f "${PROJECT_ROOT}/.env" ]]; then
                log "  Loading OPENROUTER_API_KEY from .env"
                export $(grep -E '^OPENROUTER_API_KEY=' "${PROJECT_ROOT}/.env" | xargs) 2>/dev/null || true
            fi
            [[ -n "${OPENROUTER_API_KEY:-}" ]] || die "OPENROUTER_API_KEY not set (required for OpenRouter judge)"
        fi
    fi

    log "  log file      : ${LOG_FILE}"
    log "  mode          : ${MODE}"
    log "  split         : ${SPLIT} / ${SETTING}"
    log "  judge model   : ${JUDGE_MODEL}"
    log "  judge url     : ${JUDGE_BASE_URL}"
    log "  skip ingest   : ${SKIP_INGEST}"
    log "  retry failed  : ${RETRY_FAILED}"
    log "  limit         : ${LIMIT:-all remaining}"
    log "  tmux session  : ${TMUX_SESSION}"
    log "  json output   : ${JSON_OUTPUT}"
    log "  log output    : ${LOG_OUTPUT}"
}

# ---------------------------------------------------------------------------
# Generate markdown summary from JSON results
# ---------------------------------------------------------------------------

generate_md_report() {
    local json_path="$1"
    local md_path="$2"

    log "Generating markdown report → ${md_path}"

    # Extract summary from checkpoint data in JSON
    local total successful failed coverage mean_score
    total=$(jq -r '. | length' "${json_path}" 2>/dev/null || echo "N/A")

    cat > "${md_path}" <<EOF
# HotpotQA Resume Evaluation Report

**Run timestamp:** ${TIMESTAMP}
**Resumed from:** ${LOG_FILE}
**Completed at:** $(date '+%Y-%m-%d %H:%M:%S')

---

## Configuration

| Parameter     | Value                    |
|---------------|--------------------------|
| Mode          | ${MODE}                  |
| Split         | ${SPLIT} / ${SETTING}    |
| Judge model   | ${JUDGE_MODEL}           |
| Judge URL     | ${JUDGE_BASE_URL}        |
| Skip ingest   | ${SKIP_INGEST}           |
| Retry failed  | ${RETRY_FAILED}          |

---

## Results Summary

| Metric                   | Value        |
|--------------------------|--------------|
| Total evaluated          | ${total}     |

---

## Files

| File | Path |
|------|------|
| Raw JSON results | \`${JSON_OUTPUT}\` |
| Run log          | \`${LOG_OUTPUT}\`    |

---

*Generated by run_hotpotqa_resume.sh*
EOF

    log "Markdown report written."
}

# ---------------------------------------------------------------------------
# Build tmux command
# ---------------------------------------------------------------------------

build_command() {
    local cmd=""

    # Change to project directory
    cmd+="cd '${PROJECT_ROOT}'"

    # Activate venv if exists
    if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
        cmd+=" && source .venv/bin/activate"
    fi

    # Export env vars
    cmd+=" && export JUDGE_BASE_URL='${JUDGE_BASE_URL}'"
    cmd+=" && export JUDGE_BASE_MODEL='${JUDGE_MODEL}'"
    if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
        cmd+=" && export OPENROUTER_API_KEY='${OPENROUTER_API_KEY}'"
    fi

    # Build Python command using explicit venv Python
    VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
    if [[ ! -f "${VENV_PYTHON}" ]]; then
        die "Venv Python not found: ${VENV_PYTHON}. Please create venv first."
    fi

    cmd+=" && '${VENV_PYTHON}' '${RESUME_SCRIPT}'"
    cmd+=" --log '${LOG_FILE}'"
    cmd+=" --mode ${MODE}"
    cmd+=" --split ${SPLIT}"
    cmd+=" --setting ${SETTING}"
    cmd+=" --output '${JSON_OUTPUT}'"
    cmd+=" --persist-dir '${PERSIST_DIR}'"

    if [[ "${MODE}" == "corpus" ]]; then
        cmd+=" --corpus-dir '${CORPUS_DIR}'"
    fi

    if [[ "${SKIP_INGEST}" == "true" ]]; then
        cmd+=" --skip-ingest"
    fi

    if [[ "${RETRY_FAILED}" == "true" ]]; then
        cmd+=" --retry-failed"
    fi

    if [[ "${LIMIT}" -gt 0 ]]; then
        cmd+=" --limit ${LIMIT}"
    fi

    if [[ -n "${JUDGE_MODEL}" ]]; then
        cmd+=" --judge-model '${JUDGE_MODEL}'"
    fi

    # Tee output to log file
    cmd+=" 2>&1 | tee '${LOG_OUTPUT}'"

    echo "${cmd}"
}

# ---------------------------------------------------------------------------
# Run in tmux
# ---------------------------------------------------------------------------

run_in_tmux() {
    local cmd="$1"

    log "── Creating tmux session ─────────────────────────────────────"
    log "  Session : ${TMUX_SESSION}"
    log "  Command : python3 ${RESUME_SCRIPT} ..."

    # Create new tmux session (detached)
    tmux new-session -d -s "${TMUX_SESSION}" -n "eval"

    # Send the command to the session
    tmux send-keys -t "${TMUX_SESSION}:eval" "${cmd}" Enter

    log "  Session created and command sent"
    log ""
    log "── To attach and watch progress: ──────────────────────────────"
    log "  tmux attach -t ${TMUX_SESSION}"
    log ""
    log "── To detach (keep running): ──────────────────────────────────"
    log "  Press Ctrl+B then D"
    log ""
    log "── To check status: ───────────────────────────────────────────"
    log "  tmux ls"
    log "  tmux capture-pane -t ${TMUX_SESSION} -p"
    log ""
    log "── To kill session: ───────────────────────────────────────────"
    log "  tmux kill-session -t ${TMUX_SESSION}"
}

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

main() {
    mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}" "${PERSIST_DIR}"

    preflight

    # Build and run
    local cmd
    cmd=$(build_command)

    run_in_tmux "${cmd}"

    log ""
    log "── Resume started in tmux ─────────────────────────────────────"
    log "  JSON output : ${JSON_OUTPUT}"
    log "  Log output  : ${LOG_OUTPUT}"
    log "  tmux session: ${TMUX_SESSION}"
}

main "$@"
