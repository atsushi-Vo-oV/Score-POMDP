#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

set -euo pipefail

PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16
EXPECTED_PRODUCTION_SHARDS=80

CONFIG_PATH="${SB_POMDP_CONFIG:-config/production.json}"
if [[ ! "$CONFIG_PATH" =~ ^config/[A-Za-z0-9_.-]+\.json$ ]] || [[ ! -f "$CONFIG_PATH" ]]; then
  echo "SB_POMDP_CONFIG must name an existing config/*.json file: $CONFIG_PATH" >&2
  exit 2
fi

if [[ -n "${PJM_BULKNUM:-}" ]]; then
  echo "Bulk submission is disabled; use the shell campaign launcher." >&2
  exit 2
fi
SHARD_INDEX="${SB_POMDP_SHARD_INDEX:-}"
if [[ ! "$SHARD_INDEX" =~ ^[1-9][0-9]{0,2}$ ]]; then
  echo "SB_POMDP_SHARD_INDEX must be an integer." >&2
  exit 2
fi
if (( 10#$SHARD_INDEX < 1 || 10#$SHARD_INDEX > EXPECTED_PRODUCTION_SHARDS )); then
  echo "SB_POMDP_SHARD_INDEX must be in 1-${EXPECTED_PRODUCTION_SHARDS}." >&2
  exit 2
fi

CAMPAIGN_ID="${SB_POMDP_CAMPAIGN_ID:-}"
if (( ${#CAMPAIGN_ID} > 128 )) || \
   [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]*[A-Za-z0-9_-])?$ ]]; then
  echo "SB_POMDP_CAMPAIGN_ID must contain one safe ASCII identifier." >&2
  exit 2
fi
SHARD_TAG=$(printf '%03d' "$((10#$SHARD_INDEX))")
SEGMENT_UPDATES="${SB_POMDP_SEGMENT_UPDATES:-}"

# One independent gradient-mode/method/task/seed shard per normal job.
# Independent result roots avoid concurrent writes to one manifest or CSV.
COMMON_ARGUMENTS=(
  --config "$CONFIG_PATH"
  --bulk-index "$SHARD_INDEX"
  --campaign-id "$CAMPAIGN_ID"
  --override "experiment.output_dir=results/campaigns/$CAMPAIGN_ID"
  --label "genkai-${CAMPAIGN_ID}-shard${SHARD_TAG}"
)

if [[ -n "$SEGMENT_UPDATES" ]]; then
  if [[ ! "$SEGMENT_UPDATES" =~ ^[1-9][0-9]*$ ]]; then
    echo "SB_POMDP_SEGMENT_UPDATES must be a positive integer." >&2
    exit 2
  fi
  # Only full/score_transformer (indices 1--20) requires segmented resume.
  if (( 10#$SHARD_INDEX > 20 )); then
    echo "Segmented production is restricted to full/score_transformer shards 1--20." >&2
    exit 2
  fi
  LOCK_DIR="logs/genkai-locks/$CAMPAIGN_ID"
  LOCK_FILE="$LOCK_DIR/shard${SHARD_TAG}.lock"
  mkdir -p "$LOCK_DIR"
  # flock stays authoritative, but it only excludes when the shared filesystem
  # honours advisory locks.  Keep an owner stamp as best-effort double
  # protection: read it before opening the descriptor, refresh it after
  # acquiring, and clear it on a clean exit so a leftover stamp really means
  # the previous holder died.  Stamp format: host pid epoch iso8601.
  LOCK_FRESH_SECONDS=$((168 * 3600))
  THIS_HOST=$(hostname)
  PREVIOUS_HOST=""
  PREVIOUS_PID=""
  PREVIOUS_EPOCH=""
  if [[ -s "$LOCK_FILE" ]]; then
    read -r PREVIOUS_HOST PREVIOUS_PID PREVIOUS_EPOCH _ <"$LOCK_FILE" || true
  fi
  NOW_EPOCH=$(date -u +%s)
  if [[ -n "$PREVIOUS_HOST" && "$PREVIOUS_HOST" != "$THIS_HOST" ]] \
     && [[ "$PREVIOUS_EPOCH" =~ ^[0-9]+$ ]] \
     && (( NOW_EPOCH - 10#$PREVIOUS_EPOCH < LOCK_FRESH_SECONDS )); then
    echo "WARNING: $LOCK_FILE still names host $PREVIOUS_HOST pid $PREVIOUS_PID from" \
         "$(( (NOW_EPOCH - 10#$PREVIOUS_EPOCH) / 60 )) minute(s) ago." >&2
    echo "WARNING: that segment did not exit cleanly. If it is still running, this job" \
         "would corrupt campaign $CAMPAIGN_ID shard $SHARD_TAG; confirm with pjstat -E." >&2
  fi
  # Open read-write so a refused acquisition never truncates the holder's stamp.
  exec 9<>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "Another segment is already running for campaign $CAMPAIGN_ID shard $SHARD_TAG." >&2
    exit 75
  fi
  trap ': >"$LOCK_FILE"' EXIT
  printf '%s %s %s %s\n' \
    "$THIS_HOST" "$$" "$(date -u +%s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$LOCK_FILE"
  python3.11 -m sb_pomdp.compare \
    "${COMMON_ARGUMENTS[@]}" \
    --segment-updates "$SEGMENT_UPDATES" \
    --run-dir "results/campaigns/$CAMPAIGN_ID/shard${SHARD_TAG}" \
    --resume
else
  python3.11 -m sb_pomdp.compare "${COMMON_ARGUMENTS[@]}"
fi
