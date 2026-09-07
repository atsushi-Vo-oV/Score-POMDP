#!/bin/bash
#PJM -L rscgrp=a-batch
#PJM -L vnode-core=30
#PJM -L elapse=03:00:00
#PJM -j
#PJM -S

# CPU DEBUG gate for the teleport study: one update (4 envs, 1 epoch) of the new
# environment option through every trainer path (score v1/v2, gru, dpfrl PF).
# Submit from the repository root: pjsub jobs/mountain_hike_tp/debug_tps_cpu.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
eval "${SB_POMDP_CPU_ENV:-source ~/cpu-torch/bin/activate}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=30

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
run_arm () {
  CONFIG="$1"; METHOD="$2"; TAG="$3"
  echo "=== debug tps arm: $TAG / $METHOD ==="
  python3 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks mountain_hike     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.device=cpu     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "tps-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/mountain_hike_tps_v1.json gru v1
run_arm config/mountain_hike_tps_v1.json particle_filter v1
run_arm config/mountain_hike_tps_v1.json score_transformer v1
run_arm config/mountain_hike_tps_v2.json score_transformer v2
echo "tps debug completed all arms."
