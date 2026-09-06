#!/bin/bash
#PJM -L rscgrp=a-batch
#PJM -L vnode-core=30
#PJM -L elapse=02:00:00
#PJM -j
#PJM -S

# CPU timing probe on node group A: two production-size PPO updates (16 envs x
# 128 steps, 4 epochs) of the v2 score_transformer arm and of the gru baseline,
# on 30 cores, to compare seconds/update against the MIG slice (~10 min for
# score arms, ~1 min for gru) before moving small experiments to a-batch.
# Submit from the repository root: pjsub jobs/mountain_hike_v2/timing_cpu.sh
set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=30
export MKL_NUM_THREADS=30

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
run_arm () {
  CONFIG="$1"; METHOD="$2"; TAG="$3"
  echo "=== cpu timing arm: $TAG / $METHOD ($(nproc) cores visible, OMP $OMP_NUM_THREADS) ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks mountain_hike     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.device=cpu     --override experiment.eval_episodes=2     --override experiment.eval_interval_updates=2     --max-updates 2     --label "cpu-timing-$STAMP-$TAG-$METHOD"
}
run_arm config/mountain_hike.json gru gru
run_arm config/mountain_hike_v2.json score_transformer v2
run_arm config/mountain_hike_v2.json score_alpha v2
echo "cpu timing completed all arms."
