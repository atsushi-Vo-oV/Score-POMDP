#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# schedule-lr study arm 'tau_lr100': the tau_learn arm of the ld-tau study
# (learned per-step log-alpha and log-tau schedules) with the schedule
# parameters trained at 100x the policy learning rate
# (ppo.langevin_schedule_lr_multiplier=100, i.e. 2.5e-2 in log space per
# Adam step - enough to move alpha by a factor of e^3.7 within ~150 steps if
# the gradient sign is consistent). full/score_transformer/light_dark seed
# 10, 100 updates, otherwise the exact tau-study protocol; the x1 control is
# the completed tau_learn arm. MIG slice (~10 min/update, ~17 h).
# Submit from the repository root AFTER debug_tau_lr.sh has passed:
#   pjsub jobs/ld_tau_lr/tau_lr100.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/ld_tau_tuning.json   --gradient-modes full   --methods score_transformer   --tasks light_dark   --seeds 10   --override model.langevin_temperature_learnable=true   --override ppo.langevin_schedule_lr_multiplier=100   --label "ld-tau_lr100"
