#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -S

# DEBUG gate for core6: one update (4 envs, 1 epoch) of the three decoupled-noise transformer arms on
# mirror / mirror2 / tps.  Submit from the repository root: pjsub jobs/core6/debug_core6.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
run_arm () {
  CONFIG="$1"; METHOD="$2"; TASK="$3"; TAG="$4"
  echo "=== debug core6 arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks "$TASK"     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "core6-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/core6_mirror_b02.json score_transformer mountain_hike mirror-b02
run_arm config/core6_mirror_blearn.json score_transformer mountain_hike mirror-blearn
run_arm config/core6_mirror_b02wd.json score_transformer mountain_hike mirror-b02wd
run_arm config/core6_mirror2_b02.json score_transformer mountain_hike mirror2-b02
run_arm config/core6_mirror2_blearn.json score_transformer mountain_hike mirror2-blearn
run_arm config/core6_mirror2_b02wd.json score_transformer mountain_hike mirror2-b02wd
run_arm config/core6_tps_b02.json score_transformer mountain_hike tps-b02
run_arm config/core6_tps_blearn.json score_transformer mountain_hike tps-blearn
run_arm config/core6_tps_b02wd.json score_transformer mountain_hike tps-b02wd
echo "core6 debug completed all arms."
