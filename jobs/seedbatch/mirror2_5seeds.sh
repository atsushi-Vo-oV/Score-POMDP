#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S
# Example only. On math, launch this from its normal run_shard.sh environment
# with SB_POMDP_SKIP_MODULES=1 if CUDA/PyTorch are already configured.
set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"
if [[ "${SB_POMDP_SKIP_MODULES:-0}" != 1 ]]; then
  module purge
  module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
CONFIG_PATH="${SB_POMDP_CONFIG:-config/trysvgdlight_mirror2_tf-svgd.json}"
CAMPAIGN_ID="${SB_POMDP_CAMPAIGN_ID:-seedbatch-mirror2}"
SEED_TEXT="${SB_POMDP_SEEDS:-10 11 12 13 14}"
SEGMENT_UPDATES="${SB_POMDP_SEGMENT_UPDATES:-25}"
if [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]*[A-Za-z0-9_-])?$ ]]; then
  echo 'SB_POMDP_CAMPAIGN_ID must be a safe campaign identifier' >&2; exit 2
fi
if [[ ! "$SEGMENT_UPDATES" =~ ^[1-9][0-9]*$ ]]; then
  echo 'SB_POMDP_SEGMENT_UPDATES must be positive' >&2; exit 2
fi
read -r -a SEEDS <<< "$SEED_TEXT"
if (( ${#SEEDS[@]} == 0 )); then
  echo 'SB_POMDP_SEEDS must contain integers' >&2; exit 2
fi
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed" >&2; exit 2
  fi
done
mkdir -p "logs/genkai-locks/$CAMPAIGN_ID"
# Campaign-wide lock also excludes differently ordered/overlapping seed sets.
exec 9>"logs/genkai-locks/$CAMPAIGN_ID/seedbatch.lock"
flock -n 9 || { echo 'Another seedbatch segment owns this campaign' >&2; exit 75; }
"${SB_POMDP_PYTHON:-python3.11}" -m sb_pomdp.seedbatch \
  --config "$CONFIG_PATH" --campaign-id "$CAMPAIGN_ID" --seeds "${SEEDS[@]}" \
  --override experiment.device=cuda --resume --segment-updates "$SEGMENT_UPDATES"
