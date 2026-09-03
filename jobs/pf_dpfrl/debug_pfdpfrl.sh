#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for the DPFRL-faithful particle filter: one update + one evaluation
# episode for the parameter-matched and paper-size configurations.
# Submit from the repository root: pjsub jobs/pf_dpfrl/debug_pfdpfrl.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for CONFIG in config/mountain_hike_pfdpfrl.json config/mountain_hike_pfdpfrl_paper.json; do
  echo "=== debug pfdpfrl arm: $CONFIG ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods particle_filter     --tasks mountain_hike     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --max-updates 1     --label "pfdpfrl-debug-$STAMP"
done
echo "pfdpfrl debug completed all arms."
