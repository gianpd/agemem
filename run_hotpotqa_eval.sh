#!/usr/bin/env bash
# =============================================================================
# run_hotpotqa_eval.sh
#
# Runs HotpotQA full-validation evaluation in corpus mode.
# Survives terminal close, SSH disconnect, and system sleep via nohup + disown.
#
# Usage:
#   chmod +x run_hotpotqa_eval.sh
#   ./run_hotpotqa_eval.sh                        # default corpus = ./corpus
#   CORPUS_DIR=/data/my_corpus ./run_hotpotqa_eval.sh
#   SKIP_INGEST=true ./run_hotpotqa_eval.sh       # use pre-ingested corpus docs
#   ./run_hotpotqa_eval.sh --dry-run              # print config and exit
#
# Output:
#   evaluation/results/hotpotqa_<timestamp>.json  (raw results)
#   evaluation/results/hotpotqa_<timestamp>.md    (human-readable summary)
#   evaluation/logs/hotpotqa_<timestamp>.log      (full stdout+stderr)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration  (override via env vars)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CORPUS_DIR="${CORPUS_DIR:-${PROJECT_ROOT}/corpus}"
RESULTS_DIR="${PROJECT_ROOT}/evaluation/results"
LOGS_DIR="${PROJECT_ROOT}/evaluation/logs"
PERSIST_DIR="${PROJECT_ROOT}/evaluation/persist/${TIMESTAMP}"

JSON_OUTPUT="${RESULTS_DIR}/hotpotqa_${TIMESTAMP}.json"
MD_OUTPUT="${RESULTS_DIR}/hotpotqa_${TIMESTAMP}.md"
LOG_FILE="${LOGS_DIR}/hotpotqa_${TIMESTAMP}.log"
PID_FILE="${LOGS_DIR}/hotpotqa_${TIMESTAMP}.pid"

# Resolve uv absolute path before sudo strips PATH
PYTHON=(uv run python)
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/run_hotpotqa.py"

SPLIT="validation"
SETTING="distractor"
MODE="corpus"
LIMIT="${LIMIT:-0}"   # 0 = all samples; set LIMIT=50 for a quick smoke-test
SKIP_INGEST="${SKIP_INGEST:-false}"   # set to "true" to use pre-ingested corpus docs

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

    [[ -f "${EVAL_SCRIPT}" ]]  || die "Eval script not found: ${EVAL_SCRIPT}"
    [[ -d "${CORPUS_DIR}" ]]   || die "Corpus directory not found: ${CORPUS_DIR}"

    CORPUS_COUNT=$(find "${CORPUS_DIR}" -maxdepth 1 -name "*.md" | wc -l)
    [[ "${CORPUS_COUNT}" -gt 0 ]] || die "Corpus directory is empty (no .md files): ${CORPUS_DIR}"

    log "  eval script : ${EVAL_SCRIPT}"
    log "  corpus dir  : ${CORPUS_DIR} (${CORPUS_COUNT} documents)"
    log "  split       : ${SPLIT} / ${SETTING}"
    log "  limit       : ${LIMIT:-all samples}"
    log "  skip ingest : ${SKIP_INGEST}"
    log "  json output : ${JSON_OUTPUT}"
    log "  md output   : ${MD_OUTPUT}"
    log "  log file    : ${LOG_FILE}"
}

# ---------------------------------------------------------------------------
# Generate markdown summary from JSON results
# ---------------------------------------------------------------------------

generate_md_report() {
    local json_path="$1"
    local md_path="$2"

    log "Generating markdown report → ${md_path}"

    # Use jq to extract summary fields; fall back gracefully if jq is missing
    local total scored mean_score parse_failures exceptions
    total=$(jq -r '.summary.total          // "N/A"' "${json_path}")
    scored=$(jq -r '.summary.scored         // "N/A"' "${json_path}")
    mean_score=$(jq -r '.summary.mean_j_score  // "N/A"' "${json_path}" | awk '{printf "%.4f", $1}')
    parse_failures=$(jq -r '.summary.parse_failures // "N/A"' "${json_path}")
    exceptions=$(jq -r '.summary.exceptions     // "N/A"' "${json_path}")

    local mode judge_model split setting limit
    mode=$(jq -r '.config.mode        // "N/A"' "${json_path}")
    judge_model=$(jq -r '.config.judge_model // "N/A"' "${json_path}")
    split=$(jq -r '.config.split       // "N/A"' "${json_path}")
    setting=$(jq -r '.config.setting     // "N/A"' "${json_path}")
    limit=$(jq -r '.config.limit       // "N/A"' "${json_path}")

    cat > "${md_path}" <<EOF
# HotpotQA Evaluation Report

**Run timestamp:** ${TIMESTAMP}
**Completed at:** $(date '+%Y-%m-%d %H:%M:%S')

---

## Configuration

| Parameter     | Value                    |
|---------------|--------------------------|
| Mode          | ${mode}                  |
| Split         | ${split} / ${setting}    |
| Limit         | ${limit} (0 = all)       |
| Judge model   | ${judge_model}           |
| Corpus dir    | ${CORPUS_DIR}            |
| Corpus docs   | ${CORPUS_COUNT}          |

---

## Results Summary

| Metric                   | Value        |
|--------------------------|--------------|
| Total samples            | ${total}     |
| Scored (OK)              | ${scored}    |
| Parse failures           | ${parse_failures} |
| Exceptions               | ${exceptions} |
| **Mean J-score (scored)**| **${mean_score}** |

### Paper targets (AgeMem arXiv:2601.01885v1)

| Model         | J-score |
|---------------|---------|
| AgeMem-noRL   | 54.49   |
| AgeMem (RL)   | 55.49   |

---

## Files

| File | Path |
|------|------|
| Raw JSON results | \`${JSON_OUTPUT}\` |
| Run log          | \`${LOG_FILE}\`    |

---

## Per-sample breakdown (top 20 by J-score)

\`\`\`
$(jq -r '
  .results
  | sort_by(-.j_score)
  | .[0:20]
  | .[]
  | "\(.j_score)\t\(.sample_id)\t\(.expected_answer)\t\(.predicted_answer[0:60])"
' "${json_path}" 2>/dev/null || echo "Could not extract per-sample data")
\`\`\`

---

*Generated by run_hotpotqa_eval.sh*
EOF

    log "Markdown report written."
}

# ---------------------------------------------------------------------------
# Core eval runner  (called inside nohup subprocess)
# ---------------------------------------------------------------------------

run_eval() {
    log "── Starting evaluation ───────────────────────────────────────"
    log "  PID: $$"
    echo $$ > "${PID_FILE}"

    # Prevent the OS from suspending this process group during sleep/idle.
    # systemd-inhibit requires D-Bus access; fall back silently if denied.
    local inhibit_cmd=""
    if command -v systemd-inhibit &>/dev/null; then
        if systemd-inhibit --what=sleep:idle --who=hotpotqa_eval --why=evaluation true 2>/dev/null; then
            inhibit_cmd="systemd-inhibit --what=sleep:idle --who=hotpotqa_eval --why=evaluation"
            log "  systemd-inhibit available — sleep inhibit active"
        else
            log "  systemd-inhibit access denied — continuing without sleep inhibit (nohup handles disconnect)"
        fi
    else
        log "  systemd-inhibit not found — nohup handles disconnect, but system sleep may pause I/O"
    fi

    local limit_arg=""
    [[ "${LIMIT}" -gt 0 ]] && limit_arg="--limit ${LIMIT}"

    local skip_ingest_arg=""
    [[ "${SKIP_INGEST}" == "true" ]] && skip_ingest_arg="--skip-ingest"

    # shellcheck disable=SC2086
    ${inhibit_cmd} "${PYTHON[@]}" "${EVAL_SCRIPT}" \
        --split    "${SPLIT}"       \
        --setting  "${SETTING}"     \
        --mode     "${MODE}"        \
        --corpus-dir "${CORPUS_DIR}" \
        --persist-dir "${PERSIST_DIR}" \
        --output   "${JSON_OUTPUT}" \
        ${limit_arg} \
        ${skip_ingest_arg}

    local exit_code=$?

    if [[ ${exit_code} -ne 0 ]]; then
        log "ERROR: eval script exited with code ${exit_code}"
        echo "## ERROR" >> "${MD_OUTPUT}" 2>/dev/null || true
        rm -f "${PID_FILE}"
        exit ${exit_code}
    fi

    log "── Eval complete — generating report ─────────────────────────"
    generate_md_report "${JSON_OUTPUT}" "${MD_OUTPUT}"

    rm -f "${PID_FILE}"
    log "── Done ──────────────────────────────────────────────────────"
    log "  JSON : ${JSON_OUTPUT}"
    log "  MD   : ${MD_OUTPUT}"
    log "  Log  : ${LOG_FILE}"
}

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

main() {
    # Dry-run: print config and exit
    if [[ "${1:-}" == "--dry-run" ]]; then
        preflight
        log "Dry-run complete — no evaluation started."
        exit 0
    fi

    mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}" "${PERSIST_DIR}"

    preflight

    log "── Launching in background (nohup) ───────────────────────────"
    log "  Follow live progress with:"
    log "    tail -f ${LOG_FILE}"
    log "  Check if still running with:"
    log "    cat ${PID_FILE} | xargs ps -p"

    # Export everything the subprocess needs
    export CORPUS_DIR RESULTS_DIR LOGS_DIR PERSIST_DIR
    export JSON_OUTPUT MD_OUTPUT LOG_FILE PID_FILE
    export TIMESTAMP SPLIT SETTING MODE LIMIT CORPUS_COUNT SKIP_INGEST
    export PYTHON EVAL_SCRIPT PROJECT_ROOT

    # Re-invoke this script with a sentinel flag so the subprocess runs
    # run_eval() instead of spawning another background process.
    nohup bash "${BASH_SOURCE[0]}" --_run_eval >> "${LOG_FILE}" 2>&1 &
    BGPID=$!
    disown "${BGPID}"   # detach from shell's job table — survives terminal close

    log "  Background PID : ${BGPID}"
    log "  PID file       : ${PID_FILE}"
    echo "${BGPID}" > "${PID_FILE}"
}

# Sentinel: called by the nohup subprocess
if [[ "${1:-}" == "--_run_eval" ]]; then
    run_eval
else
    main "$@"
fi