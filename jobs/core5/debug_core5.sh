#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=04:00:00
#PJM -j
#PJM -S

# DEBUG gate for core5: one update (4 envs, 1 epoch) of both new transformer arms (alpha 0.2,
# learned alpha) on all six tasks, plus the DPFRL rerun config of core4 Light-Dark 2D.
# Submit from the repository root: pjsub jobs/core5/debug_core5.sh

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
  echo "=== debug core5 arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks "$TASK"     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "core5-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/core5_mh_a02.json score_transformer mountain_hike mh-a02
run_arm config/core5_mh_alearn.json score_transformer mountain_hike mh-alearn
run_arm config/core5_ld2d_a02.json score_transformer light_dark ld2d-a02
run_arm config/core5_ld2d_alearn.json score_transformer light_dark ld2d-alearn
run_arm config/core5_pendbal_a02.json score_transformer masked_pendulum pendbal-a02
run_arm config/core5_pendbal_alearn.json score_transformer masked_pendulum pendbal-alearn
run_arm config/core5_mirror_a02.json score_transformer mountain_hike mirror-a02
run_arm config/core5_mirror_alearn.json score_transformer mountain_hike mirror-alearn
run_arm config/core5_mirror2_a02.json score_transformer mountain_hike mirror2-a02
run_arm config/core5_mirror2_alearn.json score_transformer mountain_hike mirror2-alearn
run_arm config/core5_tps_a02.json score_transformer mountain_hike tps-a02
run_arm config/core5_tps_alearn.json score_transformer mountain_hike tps-alearn
run_arm config/core4_ld2d.json particle_filter light_dark ld2d-pf
echo "core5 debug completed all arms."
