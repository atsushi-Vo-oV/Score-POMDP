#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for the schedule-learning-rate study: one production-model PPO
# update plus one evaluation episode with both Langevin schedules learned and
# ppo.langevin_schedule_lr_multiplier=100 (two optimizer param groups), on
# light_dark and on masked_cartpole. Fails fast without CUDA. MIG slice.
# Submit from the repository root: pjsub jobs/ld_tau_lr/debug_tau_lr.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
run_arm () {
  ARM="$1"; CONFIG="$2"; TASK="$3"; shift 3
  echo "=== debug tau-lr arm: $ARM ==="
  python3.11 -m sb_pomdp.compare \
    --config "$CONFIG" \
    --gradient-modes full \
    --methods score_transformer \
    --tasks "$TASK" \
    --seeds 0 \
    --override "experiment.seeds=[0]" \
    --override experiment.eval_episodes=1 \
    --override ppo.num_envs=4 \
    --override ppo.epochs=1 \
    --override model.langevin_step_size_learnable=true \
    --override model.langevin_temperature_learnable=true \
    --override ppo.langevin_schedule_lr_multiplier=100 \
    --max-updates 1 \
    --label "taulr-debug-$STAMP-$ARM" \
    "$@"
}

run_arm ld_lr100 config/ld_tau_tuning.json light_dark
run_arm cp_lr100 config/two_task_full_v2.json masked_cartpole
echo "tau-lr debug completed all arms."
