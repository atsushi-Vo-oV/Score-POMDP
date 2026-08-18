#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -S

set -euo pipefail

PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

if [[ -n "${PJM_BULKNUM:-}" ]]; then
  echo "Bulk submission is disabled; submit this script as one normal job." >&2
  exit 2
fi

CAMPAIGN_ID="${SB_POMDP_CAMPAIGN_ID:-}"
if (( ${#CAMPAIGN_ID} > 128 )) || \
   [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]*[A-Za-z0-9_-])?$ ]]; then
  echo "SB_POMDP_CAMPAIGN_ID must contain one safe ASCII identifier." >&2
  exit 2
fi

# Run the four matched MountainCar debug conditions sequentially on one GPU.
# A failed condition is recorded, but does not prevent the remaining conditions
# from running. Each condition keeps its own campaign shard/result root and log.
SHARDS=(3 11 7 15)
DESCRIPTIONS=(
  "full / score_transformer"
  "tbptt_1 / score_transformer"
  "full / gru"
  "tbptt_1 / gru"
)
LOG_DIR="$PROJECT_ROOT/logs/genkai-debug/$CAMPAIGN_ID"
mkdir -p "$LOG_DIR"
FAILED_SHARDS=()

echo "Combined debug campaign: $CAMPAIGN_ID"
for index in "${!SHARDS[@]}"; do
  shard="${SHARDS[$index]}"
  log_file="$LOG_DIR/shard$(printf '%03d' "$shard").log"
  echo "=== running debug shard ${shard}: ${DESCRIPTIONS[$index]} / MountainCar / seed 0 ==="
  if SB_POMDP_SHARD_INDEX="$shard" \
     SB_POMDP_CAMPAIGN_ID="$CAMPAIGN_ID" \
     bash jobs/genkai_debug.sh >"$log_file" 2>&1; then
    echo "=== debug shard ${shard}: completed; log=${log_file} ==="
  else
    FAILED_SHARDS+=("$shard")
    echo "=== debug shard ${shard}: failed; log=${log_file}; continuing ===" >&2
    tail -n 40 "$log_file" >&2 || true
  fi
done

if (( ${#FAILED_SHARDS[@]} > 0 )); then
  echo "Combined debug job failed for shard(s): ${FAILED_SHARDS[*]}" >&2
  exit 1
fi

echo "Combined debug job completed all shards: ${SHARDS[*]}"
