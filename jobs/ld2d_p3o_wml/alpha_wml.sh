#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# P3O-style trainer arm: full/score_alpha on the P3O-constant light_dark task
# with ppo.algorithm=p3o_wml (see score_wml.sh for the mechanism). PPO
# reference: score_alpha -84.6 final / -41 best early checkpoint.
# Submit from the repository root AFTER debug_wml.sh has passed:
#   pjsub jobs/ld2d_p3o_wml/alpha_wml.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/ld2d_p3o_wml.json   --gradient-modes full   --methods score_alpha   --tasks light_dark   --seeds 10   --label "wml-score_alpha"
