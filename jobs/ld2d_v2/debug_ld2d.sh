#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=01:30:00
#PJM -j
#PJM -S

# DEBUG gate for the Light-Dark 2D design-grid study: one update (4 envs, 1 epoch)
# through gru, the dpfrl PF, and score v1..v4.
# Submit from the repository root: pjsub jobs/ld2d_v2/debug_ld2d.sh

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
  echo "=== debug ld2d arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks light_dark     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "ld2d-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/ld2d_v1.json gru v1
run_arm config/ld2d_v1.json particle_filter v1
run_arm config/ld2d_v1.json score_transformer v1
run_arm config/ld2d_v2.json score_transformer v2
run_arm config/ld2d_v3.json score_alpha v3
run_arm config/ld2d_v4.json score_deepsets v4
echo "ld2d debug completed all arms."
