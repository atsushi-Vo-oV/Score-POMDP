#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=04:00:00
#PJM -j
#PJM -S

# DEBUG gate for the score-v2 Mountain Hike study: one FULL-SIZE update (16 envs, 4 epochs) per lite arm, because the L=16 arms
# only ran out of MIG memory at production size, not in the 4-env DEBUG.
# Submit from the repository root: pjsub jobs/mountain_hike_v2/debug_v2_lite.sh

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
  echo "=== debug v2 arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks mountain_hike     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --max-updates 1     --label "v2-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/mountain_hike_v2g_liteL16.json "score_transformer" v2g_liteL16
run_arm config/mountain_hike_v2g_liteL16.json "score_alpha" v2g_liteL16
run_arm config/mountain_hike_v2g_liteK32.json "score_transformer" v2g_liteK32
run_arm config/mountain_hike_v2g_liteK32.json "score_alpha" v2g_liteK32
echo "v2 lite debug completed all arms."
