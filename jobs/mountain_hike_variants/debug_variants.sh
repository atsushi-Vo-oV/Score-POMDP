#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -j
#PJM -S

# DEBUG gate for the Mountain Hike variant campaigns: one update per arm for
# the parameter-matched KAN variant (6 methods), the parameter-matched alpha
# (1) and the 64-particle variant (4). Logs seconds per PPO update so the KAN
# and K=64 wall-time can be judged before the 200/100-update runs.
# Submit from the repository root: pjsub jobs/mountain_hike_variants/debug_variants.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
run_arm () {
  CONFIG="$1"; METHOD="$2"; TAG="$3"
  echo "=== debug variants arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks mountain_hike     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "variants-debug-$STAMP-$TAG-$METHOD"
}
for METHOD in score_transformer score_alpha score_deepsets gru rnn particle_filter; do
  run_arm config/mountain_hike_kan.json "$METHOD" kan
done
run_arm config/mountain_hike_alphamatch.json score_alpha alphamatch
for METHOD in score_transformer score_alpha score_deepsets particle_filter; do
  run_arm config/mountain_hike_k64.json "$METHOD" k64
done
echo "variants debug completed all arms."
