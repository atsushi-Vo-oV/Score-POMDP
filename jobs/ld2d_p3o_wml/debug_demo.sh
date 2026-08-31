#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for demonstration-seeded WML (ppo.p3o_demo_slots=8): one update
# per score-family method with 8 of 16 slots driven by the scripted
# passive-Kalman controller (forward-noised chains as the imitation targets).
# Submit from the repository root: pjsub jobs/ld2d_p3o_wml/debug_demo.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for METHOD in score_transformer score_alpha score_deepsets; do
  echo "=== debug wml-demo arm: $METHOD ==="
  python3.11 -m sb_pomdp.compare \
    --config config/ld2d_p3o_wml_demo.json \
    --gradient-modes full \
    --methods "$METHOD" \
    --tasks light_dark \
    --seeds 0 \
    --override "experiment.seeds=[0]" \
    --override experiment.eval_episodes=1 \
    --override ppo.num_envs=16 \
    --override ppo.minibatch_size=480 \
    --override ppo.sequence_microbatch_size=120 \
    --max-updates 1 \
    --label "wmldemo-debug-$STAMP-$METHOD"
done
echo "wml demo debug completed all arms."
