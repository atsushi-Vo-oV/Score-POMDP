#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -S

# DEBUG gate for the FLOPs-matched mirror-world grid: one update per arm (4 envs, 1 epoch)
# through every trainer path (gru, rnn, DPFRL pf, score v1lf transformer, v2lf alpha,
# v4lf deepsets, v5lf transformer, v6lf deepsets).
# Submit from the repository root: pjsub jobs/mirror/debug_mirror_lf.sh

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
  echo "=== debug mirror lf arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks mountain_hike     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "mirrorlf-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/mirror_v1lf.json gru v1lf
run_arm config/mirror_v1lf.json rnn v1lf
run_arm config/mirror_v1lf.json particle_filter v1lf
run_arm config/mirror_v1lf.json score_transformer v1lf
run_arm config/mirror_v2lf.json score_alpha v2lf
run_arm config/mirror_v4lf.json score_deepsets v4lf
run_arm config/mirror_v5lf.json score_transformer v5lf
run_arm config/mirror_v6lf.json score_deepsets v6lf
echo "mirror lf debug completed all arms."
