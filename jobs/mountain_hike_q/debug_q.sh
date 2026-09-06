#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -j
#PJM -S

# DEBUG gate for the action-value critic study: one update per arm (4 envs, 1 epoch).
# Submit from the repository root: pjsub jobs/mountain_hike_q/debug_q.sh

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
  echo "=== debug q arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks mountain_hike     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "q-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/mountain_hike_v2q.json "score_transformer" v2q
run_arm config/mountain_hike_v2q.json "score_alpha" v2q
run_arm config/mountain_hike_v2q.json "score_deepsets" v2q
run_arm config/mountain_hike_v2q_mix05.json "score_transformer" v2q_mix05
run_arm config/mountain_hike_q_gru.json "gru" q_gru
echo "q debug completed all arms."
