#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# schedule-lr study arm 'tau_lr100_sig': identical to tau_lr100 (learned
# per-step log-alpha / log-tau schedules at 100x the policy learning rate,
# full/score_transformer/light_dark seed 10, 100 updates, tau-study protocol)
# except that the schedules are kept inside their ranges by a sigmoid
# reparameterisation (model.langevin_schedule_bound=sigmoid) instead of a
# hard clamp. In tau_lr100 the step-size schedule was pushed past the 0.5
# ceiling by update 50 and the clamp's zero gradient froze it there for the
# rest of the run (raw log values 0.65-0.76); the sigmoid bound keeps a live
# gradient everywhere so the schedule can still move once it nears a bound.
# Ranges are unchanged (alpha in [1e-4, 0.5], tau in [1e-3, 4]). MIG slice
# (~11 min/update, ~19 h).
# Submit from the repository root AFTER debug_tau_sig.sh has passed:
#   pjsub jobs/ld_tau_lr/tau_lr100_sig.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/ld_tau_tuning.json   --gradient-modes full   --methods score_transformer   --tasks light_dark   --seeds 10   --override model.langevin_temperature_learnable=true   --override model.langevin_schedule_bound=sigmoid   --override ppo.langevin_schedule_lr_multiplier=100   --label "ld-tau_lr100_sig"
