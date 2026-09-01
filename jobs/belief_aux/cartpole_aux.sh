#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Belief-supervision arm: full/score_transformer on masked_cartpole with plain
# PPO (v2 protocol) plus the predictive-sufficiency loss, capped at 100
# updates (direct launch, no resume). Predicting the next (position, angle)
# from particles requires velocity, which no reward-driven variant ever
# encoded (condition R2(velocity) ~ 0.03 vs gru 0.73); if the auxiliary loss
# works, the particle cloud must sharpen and the probe R2 must rise even if
# the return stays at the memoryless ceiling (~40-70).
# Submit from the repository root AFTER debug_aux.sh has passed:
#   pjsub jobs/belief_aux/cartpole_aux.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/two_task_full_v2.json   --gradient-modes full   --methods score_transformer   --tasks masked_cartpole   --seeds 10   --max-updates 100   --override model.observation_prediction_coef=1.0   --override "model.observation_predictor_hidden=[64,64]"   --label "aux-cartpole"
