#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Belief-supervision arm on bonus3 LD2D (see ld2d_aux_score.sh) for
# full/score_alpha - no score-vector channel, so a living belief is its only
# information path. PPO, 100 updates cap.
# Submit from the repository root: pjsub jobs/belief_aux/ld2d_aux_alpha.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/ld2d_refs.json   --gradient-modes full   --methods score_alpha   --tasks light_dark   --seeds 10   --max-updates 100   --override model.observation_prediction_coef=1.0   --override "model.observation_predictor_hidden=[64,64]"   --label "aux-ld2d-score_alpha"
