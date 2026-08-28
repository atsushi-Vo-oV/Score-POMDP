#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for the symlog critic-target transform (new trainer code): one
# production PPO update plus one evaluation on the P3O-identical light_dark
# task for score_transformer and gru, on a MIG slice. Prints value_loss so
# the scale fix is visible in the log. Fails fast without CUDA.
# Submit from the repository root: pjsub jobs/ld2d_p3o_symlog/debug_symlog.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$(nproc)"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for METHOD in score_transformer gru; do
  echo "=== debug symlog arm: $METHOD ==="
  python3.11 -m sb_pomdp.compare \
    --config config/ld2d_p3o_symlog.json \
    --gradient-modes full \
    --methods "$METHOD" \
    --tasks light_dark \
    --seeds 0 \
    --override "experiment.seeds=[0]" \
    --override experiment.eval_episodes=1 \
    --override ppo.num_envs=4 \
    --override ppo.epochs=1 \
    --max-updates 1 \
    --label "symlog-debug-$STAMP-$METHOD"
  for f in results/*symlog-debug-$STAMP-$METHOD*/full/*/light_dark/seed_000/metrics.csv; do
    echo "value_loss (symlog units): $(tail -1 "$f" | cut -d, -f4)"
  done
done
echo "symlog debug completed all arms."
