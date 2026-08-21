#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -S

# Single-submission validation gate: run the full 16-condition short debug on
# the current source, then spend the remaining wall time measuring production
# speed for the calibration shard (full/score_transformer/MountainCar, seed 10)
# via the segmented production path.  The job's exit code reflects only the
# debug gate and real pilot crashes; a pilot stopped by the wall-time budget is
# reported as a successful measurement because its resume state stays valid.

set -euo pipefail

PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

if [[ -n "${PJM_BULKNUM:-}" ]]; then
  echo "Bulk submission is disabled; submit this script as one normal job." >&2
  exit 2
fi

CAMPAIGN_ID="${SB_POMDP_CAMPAIGN_ID:-}"
if (( ${#CAMPAIGN_ID} > 100 )) || \
   [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]*[A-Za-z0-9_-])?$ ]]; then
  echo "SB_POMDP_CAMPAIGN_ID must contain one safe ASCII identifier (<=100 chars)." >&2
  exit 2
fi

TOTAL_SECONDS=7200
TEARDOWN_MARGIN=360
MINIMUM_PILOT_SECONDS=600
PILOT_SHARD_INDEX=11
PILOT_SEGMENT_UPDATES="${SB_POMDP_PILOT_SEGMENT_UPDATES:-10}"
if [[ ! "$PILOT_SEGMENT_UPDATES" =~ ^[1-9][0-9]*$ ]]; then
  echo "SB_POMDP_PILOT_SEGMENT_UPDATES must be a positive integer." >&2
  exit 2
fi
CONFIG_PATH="${SB_POMDP_CONFIG:-config/production.json}"
if [[ ! "$CONFIG_PATH" =~ ^config/[A-Za-z0-9_.-]+\.json$ ]] || [[ ! -f "$CONFIG_PATH" ]]; then
  echo "SB_POMDP_CONFIG must name an existing config/*.json file: $CONFIG_PATH" >&2
  exit 2
fi
export SB_POMDP_CONFIG="$CONFIG_PATH"
echo "config: $CONFIG_PATH"
START_EPOCH=$(date +%s)

# Background GPU sampler: one CSV row per GPU every 30 s for the whole job.
# PJM statistics do not record GPU utilization, so this is the only record of
# whether the training actually keeps the GPUs busy and how much memory it uses.
GPU_LOG_DIR="$PROJECT_ROOT/logs/gpu-usage"
GPU_LOG="$GPU_LOG_DIR/${CAMPAIGN_ID}.csv"
GPU_SAMPLER_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  mkdir -p "$GPU_LOG_DIR"
  ( while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader
      sleep 30
    done >>"$GPU_LOG" 2>/dev/null ) &
  GPU_SAMPLER_PID=$!
  echo "GPU sampler running (pid $GPU_SAMPLER_PID) -> $GPU_LOG"
else
  echo "nvidia-smi unavailable; GPU usage will not be recorded." >&2
fi

report_gpu_usage() {
  if [[ -n "$GPU_SAMPLER_PID" ]]; then
    kill "$GPU_SAMPLER_PID" 2>/dev/null || true
    wait "$GPU_SAMPLER_PID" 2>/dev/null || true
  fi
  if [[ -s "$GPU_LOG" ]]; then
    echo "=== GPU usage summary (per GPU: samples, mean/max util %, max mem MiB) ==="
    awk -F', ' '
      { gsub(/ %| MiB/, "");
        idx=$2; n[idx]++; util[idx]+=$3;
        if ($3+0 > maxu[idx]) maxu[idx]=$3+0;
        if ($5+0 > maxm[idx]) maxm[idx]=$5+0;
        total=$6 }
      END { for (i in n)
        printf "GPU %s: samples=%d mean_util=%.1f%% max_util=%d%% max_mem=%d/%d MiB\n",
               i, n[i], util[i]/n[i], maxu[i], maxm[i], total }
    ' "$GPU_LOG" | sort
    echo "full samples: $GPU_LOG"
  fi
}
trap report_gpu_usage EXIT

echo "=== phase 1: short debug (16 shards) campaign=$CAMPAIGN_ID ==="
if ! SB_POMDP_CAMPAIGN_ID="$CAMPAIGN_ID" bash jobs/genkai_debug_short.sh; then
  echo "Debug gate failed; pilot skipped." >&2
  exit 1
fi

ELAPSED=$(( $(date +%s) - START_EPOCH ))
REMAINING=$(( TOTAL_SECONDS - ELAPSED - TEARDOWN_MARGIN ))
PILOT_ID="${CAMPAIGN_ID}-pilot"
echo "=== phase 2: pilot shard ${PILOT_SHARD_INDEX} campaign=$PILOT_ID budget=${REMAINING}s ==="
if (( REMAINING < MINIMUM_PILOT_SECONDS )); then
  echo "Less than ${MINIMUM_PILOT_SECONDS}s remain after the debug phase; pilot skipped."
  echo "Debug gate passed; resubmit with a fresh campaign id if a pilot is still needed."
  exit 0
fi

IFS=',' read -r -a AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PILOT_GPU="${AVAILABLE_GPUS[0]}"

PILOT_STATUS=0
CUDA_VISIBLE_DEVICES="$PILOT_GPU" \
SB_POMDP_SHARD_INDEX="$PILOT_SHARD_INDEX" \
SB_POMDP_CAMPAIGN_ID="$PILOT_ID" \
SB_POMDP_SEGMENT_UPDATES="$PILOT_SEGMENT_UPDATES" \
timeout --signal=TERM --kill-after=60s "${REMAINING}s" \
  bash jobs/genkai_production.sh || PILOT_STATUS=$?

PILOT_ROOT="results/campaigns/$PILOT_ID"
COMMITTED_UPDATES="unknown"
METRICS_FILE=$(find "$PILOT_ROOT" -name metrics.csv -path "*seed_*" 2>/dev/null | head -n 1 || true)
if [[ -n "$METRICS_FILE" && -f "$METRICS_FILE" ]]; then
  DATA_ROWS=$(( $(wc -l <"$METRICS_FILE") - 1 ))
  (( DATA_ROWS < 0 )) && DATA_ROWS=0
  COMMITTED_UPDATES="$DATA_ROWS"
fi

echo "=== pilot summary ==="
echo "pilot exit code: $PILOT_STATUS (0=segment complete, 124/137=wall-time stop)"
echo "metrics rows committed (>= updates durably finished): $COMMITTED_UPDATES"
echo "pilot artifacts: $PILOT_ROOT"

if [[ "$PILOT_STATUS" -eq 0 ]]; then
  echo "Pilot segment of $PILOT_SEGMENT_UPDATES updates completed inside the window."
  exit 0
fi
if [[ "$PILOT_STATUS" -eq 124 || "$PILOT_STATUS" -eq 137 ]]; then
  echo "Pilot stopped by the wall-time budget; committed updates above are the measurement."
  echo "The shard remains resumable with the same campaign id, shard index, and source."
  exit 0
fi
echo "Pilot failed with exit $PILOT_STATUS before the wall-time budget; inspect the log above." >&2
exit 1
