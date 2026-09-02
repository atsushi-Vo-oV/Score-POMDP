#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Belief-supervision arm on a task where the signal exists: bonus3 LD2D
# (noise_gain 0.5 -> observation sigma ~2 at the start, ~0.7 near x1=4) with
# plain PPO (v2 protocol, config/ld2d_refs.json) plus the predictive-
# sufficiency loss. On the P3O-constant task the next observation carries
# only ~0.02 nats/step of state information (sigma ~7-11), so the first LD
# arms could not test the mechanism; here the budget is ~0.2-2 nats/step.
# full/score_transformer seed 10, capped at 100 updates (no resume).
# Submit from the repository root: pjsub jobs/belief_aux/ld2d_aux_score.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/ld2d_refs.json   --gradient-modes full   --methods score_transformer   --tasks light_dark   --seeds 10   --max-updates 100   --override model.observation_prediction_coef=1.0   --override "model.observation_predictor_hidden=[64,64]"   --label "aux-ld2d-score_transformer"
