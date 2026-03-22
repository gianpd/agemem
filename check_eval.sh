#!/bin/bash
# Check AgeMem evaluation progress

set -e

EVAL_DIR="evaluation"
RESULTS_DIR="$EVAL_DIR/results"
LOGS_DIR="$EVAL_DIR/logs"

echo "============================================"
echo "AgeMem Evaluation Status"
echo "============================================"
echo ""

# Check if eval process is running
echo "## Process Status"
PID=$(ps aux | grep -E "evaluation.cli|evaluation.runner" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null || echo "unknown")
    echo "Running: PID $PID (elapsed: $ELAPSED)"
else
    echo "No evaluation process running"
fi
echo ""

# Find latest checkpoint
echo "## Latest Checkpoint"
LATEST_CHECKPOINT=$(ls -t "$RESULTS_DIR"/*_checkpoint.json 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "File: $(basename "$LATEST_CHECKPOINT")"
    echo ""
    # Parse checkpoint JSON
    python3 -c "
import json
with open('$LATEST_CHECKPOINT') as f:
    data = json.load(f)
print(f\"Session: {data.get('session_id', 'N/A')}\")
print(f\"Status: {data.get('status', 'N/A')}\")
prog = data.get('progress', {})
print(f\"Batches completed: {prog.get('completed_batches', 0)}\")
print(f\"Interactions completed: {prog.get('completed_interactions', 0)} / {prog.get('total_interactions', 0)}\")
metrics = data.get('aggregated_metrics', {})
if metrics:
    print(f\"\nMetrics:\")
    print(f\"  Queries: {metrics.get('total_queries', 0)}\")
    print(f\"  Correct: {metrics.get('correct', 0)}\")
    print(f\"  Accuracy: {metrics.get('accuracy', 0):.2%}\")
    print(f\"  Abstained: {metrics.get('abstained', 0)}\")
    print(f\"  LLM Judge queries: {metrics.get('llm_judge_queries', 0)}\")
    print(f\"  Avg latency: {metrics.get('avg_latency_ms', 0):.0f}ms\")
"
else
    echo "No checkpoint found"
fi
echo ""

# Check for partial metrics
PARTIAL_METRICS=$(ls -t "$RESULTS_DIR"/*_partial_metrics.json 2>/dev/null | head -1)
if [ -n "$PARTIAL_METRICS" ]; then
    echo "## Partial Metrics"
    python3 -c "
import json
with open('$PARTIAL_METRICS') as f:
    data = json.load(f)
metrics = data.get('metrics', {})
print(f\"  Queries: {metrics.get('total_queries', 0)}\")
print(f\"  Correct: {metrics.get('correct', 0)}\")
print(f\"  Accuracy: {metrics.get('accuracy', 0):.2%}\")
print(f\"  Abstained: {metrics.get('abstained', 0)}\")
print(f\"  LLM Judge queries: {metrics.get('llm_judge_queries', 0)}\")
"
    echo ""
fi

# Count batch result files
echo "## Batch Results"
BATCH_COUNT=$(ls "$RESULTS_DIR"/*_batch_*.jsonl 2>/dev/null | wc -l)
echo "Batch files: $BATCH_COUNT"
echo ""

# Latest log activity
echo "## Recent Log Activity"
LATEST_LOG=$(ls -t "$LOGS_DIR"/*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Log: $(basename "$LATEST_LOG")"
    echo ""
    # Show last few non-debug lines
    grep -v "^\[DEBUG\]" "$LATEST_LOG" 2>/dev/null | tail -10 || tail -10 "$LATEST_LOG"
else
    echo "No log files found"
fi
echo ""

# LLM Judge status
echo "## LLM-as-Judge Status"
if curl -s --connect-timeout 2 "http://localhost:8080/v1/models" > /dev/null 2>&1; then
    MODEL=$(curl -s "http://localhost:8080/v1/models" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
    echo "Server: UP (model: $MODEL)"
else
    echo "Server: DOWN or unreachable"
fi

echo ""
echo "============================================"