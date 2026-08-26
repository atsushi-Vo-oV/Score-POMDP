#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for the alpha-vector comparison: one production-model PPO update
# plus one evaluation episode on both tasks (seed 0) for each new arm -
# score_alpha with the cold belief, score_alpha with the warm belief, and
# score_transformer with the warm belief. Fails fast without CUDA.
# Submit from the repository root: pjsub jobs/two_task_alpha/debug_alpha.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
run_arm () {
  ARM="$1"; CFG="$2"; METHOD="$3"
  for TASK in masked_cartpole light_dark; do
    echo "=== debug alpha arm: $ARM / $TASK ==="
    python3.11 -m sb_pomdp.compare \
      --config "config/$CFG" \
      --gradient-modes full \
      --methods "$METHOD" \
      --tasks "$TASK" \
      --seeds 0 \
      --override "experiment.seeds=[0]" \
      --override experiment.eval_episodes=1 \
      --override ppo.num_envs=4 \
      --override ppo.epochs=1 \
      --max-updates 1 \
      --label "alpha-debug-$STAMP-$ARM-$TASK"
  done
}

run_arm cold_alpha two_task_alpha_cold.json score_alpha
run_arm warm_alpha two_task_alpha_warm.json score_alpha
run_arm warm_transformer two_task_alpha_warm.json score_transformer
echo "alpha debug completed all arms."
