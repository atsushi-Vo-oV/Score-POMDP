#!/usr/bin/env bash
# Called in the selected tree, with the per-job SB_POMDP_* exports already set.
set -euo pipefail
exec 2>&1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
if [[ "${SB_POMDP_SKIP_MODULES:-1}" != 1 ]]; then
  module purge
  module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
fi
for key in SB_POMDP_CONFIG SB_POMDP_CAMPAIGN_ID SB_POMDP_SHARD_INDEX SB_POMDP_BULK_INDEX SB_POMDP_SEGMENT_UPDATES; do
  [[ -n "${!key:-}" ]] || { echo "Missing $key"; exit 2; }
done
[[ "$SB_POMDP_CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
for value in "$SB_POMDP_SHARD_INDEX" "$SB_POMDP_BULK_INDEX" "$SB_POMDP_SEGMENT_UPDATES"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid positive index/update count: $value"; exit 2; }
done
args=(--config "$SB_POMDP_CONFIG" --campaign-id "$SB_POMDP_CAMPAIGN_ID"
      --bulk-index "$SB_POMDP_BULK_INDEX" --shard-index "$SB_POMDP_SHARD_INDEX"
      --resume --segment-updates "$SB_POMDP_SEGMENT_UPDATES")
# A dedicated plain-text overrides file: one dotted KEY=JSON per line, no shell evaluation.
# The tree's .sb_overrides is deliberately NOT sourced (see README).
if [[ -n "${SB_POMDP_POPX_OVERRIDES:-}" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    args+=(--override "$line")
  done < "$SB_POMDP_POPX_OVERRIDES"
fi
# Execution-only belief acceleration (same switches as the main trainer's .sb_overrides):
# SB_POMDP_POPX_FAST=1 selects the analytic score with CUDA Graphs, otherwise autograd.
if [[ "${SB_POMDP_POPX_FAST:-0}" == 1 ]]; then
  args+=(--override model.score_implementation=analytic --override model.belief_cuda_graphs=true)
else
  args+=(--override model.score_implementation=autograd --override model.belief_cuda_graphs=false)
fi
args+=(--override model.belief_compile=off)
mkdir -p "logs/popx-locks/$SB_POMDP_CAMPAIGN_ID"
command -v flock >/dev/null || { echo 'flock is required for shard locking (Linux util-linux)'; exit 2; }
exec 9>"logs/popx-locks/$SB_POMDP_CAMPAIGN_ID/shard$SB_POMDP_SHARD_INDEX.lock"
flock -n 9 || { echo 'Another process owns this population shard'; exit 75; }
exec "${SB_POMDP_PYTHON:-python3.11}" -m sb_pomdp.popexplore "${args[@]}"
