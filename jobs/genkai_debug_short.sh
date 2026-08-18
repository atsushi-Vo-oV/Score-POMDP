#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:30:00
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

for command_name in flock timeout; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command_name" >&2
    exit 2
  fi
done

IFS=',' read -r -a AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#AVAILABLE_GPUS[@]} < 4 )); then
  echo "Short debug requires one B node with four visible GPUs." >&2
  exit 2
fi
GPUS=("${AVAILABLE_GPUS[@]:0:4}")
for gpu in "${GPUS[@]}"; do
  if [[ -z "$gpu" ]]; then
    echo "CUDA_VISIBLE_DEVICES contains an empty GPU identifier." >&2
    exit 2
  fi
done

# First cover all four model/gradient paths on the slowest task, then let each
# free GPU claim another shard.  All 16 active DEBUG conditions are attempted.
SHARDS=(3 11 7 15 4 2 1 12 10 9 8 6 5 16 14 13)
LOG_DIR="$PROJECT_ROOT/logs/genkai-debug/$CAMPAIGN_ID"
SCHEDULER_DIR="$LOG_DIR/short-scheduler"
mkdir -p "$LOG_DIR"
if ! mkdir "$SCHEDULER_DIR"; then
  echo "Short-debug scheduler state already exists: $SCHEDULER_DIR" >&2
  exit 2
fi
QUEUE_FILE="$SCHEDULER_DIR/queue"
QUEUE_LOCK="$SCHEDULER_DIR/queue.lock"
printf '%s\n' "${SHARDS[@]}" >"$QUEUE_FILE"
: >"$QUEUE_LOCK"

WATCHDOG_SECONDS=1680
START_EPOCH=$(date +%s)

claim_shard() {
  local worker_index="$1"
  local temporary_queue="$SCHEDULER_DIR/queue.worker${worker_index}.tmp"
  (
    flock -x 9
    local shard=""
    if IFS= read -r shard <"$QUEUE_FILE" && [[ -n "$shard" ]]; then
      tail -n +2 "$QUEUE_FILE" >"$temporary_queue"
      mv "$temporary_queue" "$QUEUE_FILE"
      printf '%s\n' "$shard"
    fi
  ) 9>"$QUEUE_LOCK"
}

run_worker() {
  local worker_index="$1"
  local gpu="$2"
  local worker_failed=0
  local shard elapsed remaining log_file status_file exit_code

  while true; do
    shard=$(claim_shard "$worker_index")
    if [[ -z "$shard" ]]; then
      break
    fi
    log_file="$LOG_DIR/shard$(printf '%03d' "$shard").log"
    status_file="$SCHEDULER_DIR/shard$(printf '%03d' "$shard").status"
    elapsed=$(( $(date +%s) - START_EPOCH ))
    remaining=$(( WATCHDOG_SECONDS - elapsed ))
    if (( remaining <= 0 )); then
      printf '%s\n' "not-started: global 28-minute watchdog expired" >"$status_file"
      worker_failed=1
      continue
    fi

    echo "=== worker ${worker_index}: starting shard ${shard} on GPU ${gpu}; ${remaining}s remain ==="
    if CUDA_VISIBLE_DEVICES="$gpu" \
       SB_POMDP_SHORT_DEBUG=1 \
       SB_POMDP_SHARD_INDEX="$shard" \
       SB_POMDP_CAMPAIGN_ID="$CAMPAIGN_ID" \
       timeout --signal=TERM --kill-after=15s "${remaining}s" \
       bash jobs/genkai_debug.sh >"$log_file" 2>&1; then
      printf '%s\n' completed >"$status_file"
      echo "=== worker ${worker_index}: shard ${shard} completed; log=${log_file} ==="
    else
      exit_code=$?
      printf 'failed: exit=%s\n' "$exit_code" >"$status_file"
      echo "=== worker ${worker_index}: shard ${shard} failed with exit ${exit_code}; log=${log_file} ===" >&2
      worker_failed=1
    fi
  done
  return "$worker_failed"
}

echo "Short debug campaign: $CAMPAIGN_ID"
echo "Attempting 16 shards on GPUs: ${GPUS[*]}"
WORKER_PIDS=()
for worker_index in "${!GPUS[@]}"; do
  run_worker "$worker_index" "${GPUS[$worker_index]}" &
  WORKER_PIDS+=("$!")
done

JOB_FAILED=0
for worker_pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$worker_pid"; then
    JOB_FAILED=1
  fi
done

FAILED_SHARDS=()
for shard in "${SHARDS[@]}"; do
  status_file="$SCHEDULER_DIR/shard$(printf '%03d' "$shard").status"
  if [[ ! -f "$status_file" ]] || [[ "$(<"$status_file")" != "completed" ]]; then
    FAILED_SHARDS+=("$shard")
  fi
done

if (( ${#FAILED_SHARDS[@]} > 0 )); then
  echo "Short debug failed or timed out for shard(s): ${FAILED_SHARDS[*]}" >&2
  exit 1
fi
if (( JOB_FAILED != 0 )); then
  echo "A short-debug worker failed without a shard status." >&2
  exit 1
fi

echo "Short debug completed all 16 shards."
