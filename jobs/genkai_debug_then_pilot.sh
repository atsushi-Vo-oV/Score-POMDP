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
START_EPOCH=$(date +%s)

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
