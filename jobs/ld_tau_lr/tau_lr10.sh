#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# schedule-lr study arm 'tau_lr10': the tau_learn arm of the ld-tau study
# (learned per-step log-alpha and log-tau schedules) with the schedule
# parameters trained at 10x the policy learning rate
# (ppo.langevin_schedule_lr_multiplier=10). full/score_transformer/light_dark
# seed 10, 100 updates, otherwise the exact tau-study protocol; the x1
# control is the completed tau_learn arm (alpha 0.0200-0.0204, tau
# 0.968-0.991 after 100 updates). MIG slice (~10 min/update, ~17 h).
# Submit from the repository root AFTER debug_tau_lr.sh has passed:
#   pjsub jobs/ld_tau_lr/tau_lr10.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/ld_tau_tuning.json   --gradient-modes full   --methods score_transformer   --tasks light_dark   --seeds 10   --override model.langevin_temperature_learnable=true   --override ppo.langevin_schedule_lr_multiplier=10   --label "ld-tau_lr10"
