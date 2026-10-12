#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -S

# DEBUG gate for the core4 group: one update per arm (4 envs, 1 epoch) for the transformer k16f on
# all three tasks plus gru / rnn / pf on one task each.
# Submit from the repository root: pjsub jobs/core4/debug_core4.sh

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
  echo "=== debug core4 arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks "$TASK"     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "core4-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/core4_mh.json score_transformer mountain_hike mh
run_arm config/core4_ld2d.json score_transformer light_dark ld2d
run_arm config/core4_pendbal.json score_transformer masked_pendulum pendbal
run_arm config/core4_mh.json gru mountain_hike mh
run_arm config/core4_ld2d.json rnn light_dark ld2d
run_arm config/core4_pendbal.json particle_filter masked_pendulum pendbal
echo "core4 debug completed all arms."
