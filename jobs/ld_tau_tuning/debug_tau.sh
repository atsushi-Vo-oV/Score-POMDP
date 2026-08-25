#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for the Langevin-temperature study: one production-model PPO
# update plus one evaluation episode on light_dark (seed 0) for each new
# mechanism - fixed temperature, learned temperature schedule, and warm-start
# initialisation. Fails fast without CUDA.
# Submit from the repository root: pjsub jobs/ld_tau_tuning/debug_tau.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
run_arm () {
  ARM="$1"; shift
  echo "=== debug tau arm: $ARM ==="
  python3.11 -m sb_pomdp.compare \
    --config config/ld_tau_tuning.json \
    --gradient-modes full \
    --methods score_transformer \
    --tasks light_dark \
    --seeds 0 \
    --override "experiment.seeds=[0]" \
    --override experiment.eval_episodes=1 \
    --override ppo.num_envs=4 \
    --override ppo.epochs=1 \
    --max-updates 1 \
    --label "tau-debug-$STAMP-$ARM" \
    "$@"
}

run_arm tau010 --override model.langevin_temperature=0.1
run_arm tau_learn --override model.langevin_temperature_learnable=true
run_arm warm_tau010 \
  --override model.langevin_warm_start=true \
  --override model.langevin_temperature=0.1
echo "tau debug completed all arms."
