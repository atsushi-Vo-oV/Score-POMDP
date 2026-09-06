#!/bin/bash
#PJM -L rscgrp=a-batch
#PJM -L vnode-core=30
#PJM -L elapse=72:00:00
#PJM -j
#PJM -S

# ablation control: warm start ONLY, transformer, 100 updates
# CPU (node group A) copy of ab_warm_score.sh: same campaign id, config with experiment.device=cpu,
# 30 cores. Never run together with the MIG copy (same campaign directory).
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_ab_warm_cpu.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="mountain-hike-ab-warm-v1"
export SB_POMDP_SEGMENT_UPDATES=100
export SB_POMDP_THREADS=30
export SB_POMDP_CPU_ENV="${SB_POMDP_CPU_ENV:-source ~/cpu-torch/bin/activate}"
exec bash jobs/genkai_production_cpu.sh
