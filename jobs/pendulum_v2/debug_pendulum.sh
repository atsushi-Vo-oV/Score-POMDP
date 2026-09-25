#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -S

# DEBUG gate for the masked-pendulum grid: one update per arm (4 envs, 1 epoch) through
# every trainer path (gru, DPFRL pf, score v1 transformer, v2 alpha, v4 deepsets, v5 transformer).
# Submit from the repository root: pjsub jobs/pendulum_v2/debug_pendulum.sh

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
  echo "=== debug pendulum arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks masked_pendulum     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "pendulum-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/pendulum_v1.json gru v1
run_arm config/pendulum_v1.json particle_filter v1
run_arm config/pendulum_v1.json score_transformer v1
run_arm config/pendulum_v2.json score_alpha v2
run_arm config/pendulum_v4.json score_deepsets v4
run_arm config/pendulum_v5.json score_transformer v5
echo "pendulum debug completed all arms."
