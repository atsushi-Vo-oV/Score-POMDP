#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# ld-tau study arm 'alpha_learn': config defaults only - fixed tau=1.0 with the
# learned per-step drift schedule alpha_l (init 0.02). Isolates the effect of
# learning alpha alone against the completed v2 score/light_dark reference,
# whose alpha was fixed.
# full/score_transformer/light_dark seed 10, 100 updates, otherwise the exact
# two-task-full-v2 protocol.
# Submit from the repository root AFTER debug_tau.sh has passed:
#   pjsub jobs/ld_tau_tuning/alpha_learn.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16

python3.11 -m sb_pomdp.compare \
  --config config/ld_tau_tuning.json \
  --gradient-modes full \
  --methods score_transformer \
  --tasks light_dark \
  --seeds 10 \
  --label "ld-alpha_learn"
