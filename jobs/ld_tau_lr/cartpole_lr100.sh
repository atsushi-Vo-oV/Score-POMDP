#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# schedule-lr study arm 'cartpole_lr100': learned per-step log-alpha and
# log-tau schedules at 100x the policy learning rate on masked_cartpole,
# where the ULA noise floor sqrt(2 alpha tau) = 0.2 (normalised units) is
# far above the 0.05-0.1 one-step observation difference that velocity
# estimation needs. full/score_transformer/masked_cartpole seed 10, v2
# protocol, capped at 50 updates (direct compare launch has no resume; the
# score full-mode CartPole rate is ~10-18 min/update). The fixed-schedule
# reference is the v2 score/masked_cartpole run at update 50 (greedy 67.2,
# stochastic 38.6); there is no x1 learned-schedule CartPole control.
# Submit from the repository root AFTER debug_tau_lr.sh has passed:
#   pjsub jobs/ld_tau_lr/cartpole_lr100.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/two_task_full_v2.json   --gradient-modes full   --methods score_transformer   --tasks masked_cartpole   --seeds 10   --max-updates 50   --override model.langevin_step_size_learnable=true   --override model.langevin_temperature_learnable=true   --override ppo.langevin_schedule_lr_multiplier=100   --label "cp-tau_lr100"
