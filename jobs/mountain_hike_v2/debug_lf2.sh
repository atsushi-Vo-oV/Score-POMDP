#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=01:30:00
#PJM -j
#PJM -S

# DEBUG gate for the FLOPs-matched score arms of designs v1, v3-v6 (K=4, L=2, energy [32,32]):
# one update per arm (4 envs, 1 epoch) on each task.
# Submit from the repository root: pjsub jobs/mountain_hike_v2/debug_lf2.sh

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
  echo "=== debug lf arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks "$TASK"     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "lf-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/mountain_hike_v1lf.json score_transformer mountain_hike mh-v1
run_arm config/mountain_hike_v4lf.json score_deepsets mountain_hike mh-v4
run_arm config/ld2d_v6lf.json score_alpha light_dark ld-v6
run_arm config/pendulum_v5lf.json score_transformer masked_pendulum pd-v5
run_arm config/pendulum_v3lf.json score_alpha masked_pendulum pd-v3
echo "lf2 debug completed all arms."
