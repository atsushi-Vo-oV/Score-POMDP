#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for the Mountain Hike task: one production-model update plus one
# evaluation episode for every method in the campaign (seed 0). MIG slice.
# Submit from the repository root: pjsub jobs/mountain_hike/debug_hike.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for METHOD in score_transformer score_alpha score_deepsets gru rnn particle_filter; do
  echo "=== debug hike arm: $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config config/mountain_hike.json     --gradient-modes full     --methods "$METHOD"     --tasks mountain_hike     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "hike-debug-$STAMP-$METHOD"
done
echo "hike debug completed all arms."
