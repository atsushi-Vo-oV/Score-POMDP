#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -S

# DEBUG gate for the score-v2 Mountain Hike study: one update per arm (4 envs, 1 epoch) for the 15 non-lite arms.
# Submit from the repository root: pjsub jobs/mountain_hike_v2/debug_v2.sh

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
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks mountain_hike     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "v2-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/mountain_hike_v2.json "score_transformer" v2
run_arm config/mountain_hike_v2.json "score_alpha" v2
run_arm config/mountain_hike_v2.json "score_deepsets" v2
run_arm config/mountain_hike_v2g.json "score_transformer" v2g
run_arm config/mountain_hike_v2g.json "score_alpha" v2g
run_arm config/mountain_hike_v2g.json "score_deepsets" v2g
run_arm config/mountain_hike_v2g_a01.json "score_transformer" v2g_a01
run_arm config/mountain_hike_v2g_a03.json "score_transformer" v2g_a03
run_arm config/mountain_hike_v2g_t03.json "score_transformer" v2g_t03
run_arm config/mountain_hike_v2g_t01.json "score_transformer" v2g_t01
run_arm config/mountain_hike_v2g_lr1e3.json "score_transformer" v2g_lr1e3
run_arm config/mountain_hike_v2g_ent03.json "score_transformer" v2g_ent03
run_arm config/mountain_hike_v2g_rw10.json "score_transformer" v2g_rw10
run_arm config/mountain_hike_v2g_rw01.json "score_transformer" v2g_rw01
run_arm config/mountain_hike_v2g_ns.json "score_transformer" v2g_ns
echo "v2 debug completed all arms."
