#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Belief-supervision arm: full/score_alpha (see score_aux.sh). score_alpha has
# no score-vector channel at all, so a living belief is its only possible
# information path - the cleanest test of the auxiliary loss.
# Submit from the repository root AFTER debug_aux.sh has passed:
#   pjsub jobs/belief_aux/alpha_aux.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/ld2d_p3o_wml_demo_aux.json   --gradient-modes full   --methods score_alpha   --tasks light_dark   --seeds 10   --label "aux-score_alpha"
