#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# ld-tau study arm 'tau003': fixed tau=0.03.
# full/score_transformer/light_dark seed 10, 100 updates, otherwise the exact
# two-task-full-v2 protocol. The tau=1.0 reference is the completed v2
# score/light_dark run (compare at update 100).
# Submit from the repository root AFTER debug_tau.sh has passed:
#   pjsub jobs/ld_tau_tuning/tau003.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16

python3.11 -m sb_pomdp.compare   --config config/ld_tau_tuning.json   --gradient-modes full   --methods score_transformer   --tasks light_dark   --seeds 10   --override model.langevin_temperature=0.03   --label "ld-tau003"
