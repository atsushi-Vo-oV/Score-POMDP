#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Belief-supervision arm: full/score_transformer, demo-seeded WML on the
# P3O-constant light_dark task PLUS the predictive-sufficiency auxiliary loss
# (model.observation_prediction_coef=1.0): particles + action must predict
# the next observation through a per-particle mixture density whose input is
# particle positions only - the score-vector shortcut cannot satisfy it, and
# the gradient reaches the energy networks through the replayed ULA chain.
# State is never observed; the signal is the agent's own (o, a) stream.
# References without the aux loss: stochastic peak -15.4 @100 but belief dead
# (condition R2 <= 0.1, particle R2 = 0, cloud std 1.0). ~2 h per 200 updates.
# Submit from the repository root AFTER debug_aux.sh has passed:
#   pjsub jobs/belief_aux/score_aux.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/ld2d_p3o_wml_demo_aux.json   --gradient-modes full   --methods score_transformer   --tasks light_dark   --seeds 10   --label "aux-score_transformer"
