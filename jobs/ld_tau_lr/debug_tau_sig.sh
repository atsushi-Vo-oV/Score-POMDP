#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for the sigmoid schedule bound: one production-model PPO update
# plus one evaluation episode on light_dark with both Langevin schedules
# learned, model.langevin_schedule_bound=sigmoid, and the 100x schedule
# learning rate. Fails fast without CUDA. MIG slice.
# Submit from the repository root: pjsub jobs/ld_tau_lr/debug_tau_sig.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
echo "=== debug tau-sig arm: ld_lr100_sig ==="
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
  --override model.langevin_step_size_learnable=true \
  --override model.langevin_temperature_learnable=true \
  --override model.langevin_schedule_bound=sigmoid \
  --override ppo.langevin_schedule_lr_multiplier=100 \
  --max-updates 1 \
  --label "tausig-debug-$STAMP-ld_lr100_sig"
echo "tau-sig debug completed."
