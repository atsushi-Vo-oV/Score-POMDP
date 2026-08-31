#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Demonstration-seeded WML arm: full/score_transformer on the P3O-constant
# light_dark task. 8 of 64 population slots are driven by the scripted
# passive-Kalman homing controller (reference return about -15.5, far above
# every learned policy on this task); their actions enter the exp(eta*R)
# weighted maximum likelihood through forward-noised diffusion chains, so the
# policy first imitates observation-exploiting behaviour and the belief is
# trained on that behaviour distribution. Plain-WML reference (no demos):
# score_transformer -220 final, oscillating. ~2 h measured per 200 updates.
# Submit from the repository root AFTER debug_demo.sh has passed:
#   pjsub jobs/ld2d_p3o_wml/score_demo.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/ld2d_p3o_wml_demo.json   --gradient-modes full   --methods score_transformer   --tasks light_dark   --seeds 10   --label "wmldemo-score_transformer"
