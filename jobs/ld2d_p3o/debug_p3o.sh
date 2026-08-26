#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for the P3O-style light_dark variant: the environment gained new
# code paths (noise_gain, terminal_cost, process_noise_std), so one production
# PPO update plus one evaluation runs on GPU for a score method and a
# baseline before production. Fails fast without CUDA.
# Submit from the repository root: pjsub jobs/ld2d_p3o/debug_p3o.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for METHOD in score_transformer gru; do
  echo "=== debug p3o arm: $METHOD ==="
  python3.11 -m sb_pomdp.compare \
    --config config/ld2d_p3o.json \
    --gradient-modes full \
    --methods "$METHOD" \
    --tasks light_dark \
    --seeds 0 \
    --override "experiment.seeds=[0]" \
    --override experiment.eval_episodes=1 \
    --override ppo.num_envs=4 \
    --override ppo.epochs=1 \
    --max-updates 1 \
    --label "p3o-debug-$STAMP-$METHOD"
done
echo "p3o debug completed all arms."
