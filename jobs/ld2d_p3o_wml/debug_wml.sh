#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for the P3O-style tilted weighted-maximum-likelihood trainer
# (ppo.algorithm=p3o_wml): one update per score-family method on the
# P3O-constant light_dark task, with a small resample interval so the
# mid-episode resampling code path (environment cloning, builder lineage
# rewrite, belief-history permutation) actually executes on the GPU. MIG.
# Submit from the repository root: pjsub jobs/ld2d_p3o_wml/debug_wml.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for METHOD in score_transformer score_alpha score_deepsets; do
  echo "=== debug wml arm: $METHOD ==="
  python3.11 -m sb_pomdp.compare \
    --config config/ld2d_p3o_wml.json \
    --gradient-modes full \
    --methods "$METHOD" \
    --tasks light_dark \
    --seeds 0 \
    --override "experiment.seeds=[0]" \
    --override experiment.eval_episodes=1 \
    --override ppo.num_envs=16 \
    --override ppo.minibatch_size=480 \
    --override ppo.sequence_microbatch_size=120 \
    --override ppo.p3o_resample_interval=5 \
    --max-updates 1 \
    --label "wml-debug-$STAMP-$METHOD"
done
echo "wml debug completed all arms."
