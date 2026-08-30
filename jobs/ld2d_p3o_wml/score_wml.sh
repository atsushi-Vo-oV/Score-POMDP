#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# P3O-style trainer arm: full/score_transformer on the P3O-constant light_dark
# task with ppo.algorithm=p3o_wml - a 64-slot population per 30-step episode,
# exp(eta * R) weights over the population (eta 0.05, resample interval 30 =
# pure episodic tilt), and critic-free weighted maximum likelihood in place of
# the clipped surrogate (symlog critic still trained as a diagnostic).
# PPO references on this task: symlog gru -28.5, score_transformer -29.0 (final
# checkpoint, oscillating). Direct compare launch, no resume: ~19 h estimated
# for 200 updates on a MIG slice.
# Submit from the repository root AFTER debug_wml.sh has passed:
#   pjsub jobs/ld2d_p3o_wml/score_wml.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

python3.11 -m sb_pomdp.compare   --config config/ld2d_p3o_wml.json   --gradient-modes full   --methods score_transformer   --tasks light_dark   --seeds 10   --label "wml-score_transformer"
