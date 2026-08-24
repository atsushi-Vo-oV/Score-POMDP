#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for the particle_filter baseline: one production-model PPO update
# plus one evaluation episode for full/particle_filter on masked_cartpole and
# light_dark (seed 0), sequentially on one GPU. Fails fast without CUDA.
# Submit from the repository root: pjsub jobs/two_task_full_pf/debug_pf.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for TASK in masked_cartpole light_dark; do
  echo "=== debug full/particle_filter/$TASK ==="
  python3.11 -m sb_pomdp.compare     --config config/two_task_full_pf.json     --gradient-modes full     --methods particle_filter     --tasks "$TASK"     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "pf-debug-$STAMP-$TASK"
done
echo "particle_filter debug completed both tasks."
