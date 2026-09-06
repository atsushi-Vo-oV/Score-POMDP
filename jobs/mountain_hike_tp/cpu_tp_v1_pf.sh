#!/bin/bash
#PJM -L rscgrp=a-batch
#PJM -L vnode-core=30
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike TELEPORT (p=0.03) / v1 baseline / particle_filter / seed 10, 100 updates, on node group A (CPU).
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_tp_v1_cpu.json"
export SB_POMDP_SHARD_INDEX=6
export SB_POMDP_CAMPAIGN_ID="mountain-hike-tp-v1-v1"
export SB_POMDP_SEGMENT_UPDATES=100
export SB_POMDP_THREADS=30
export SB_POMDP_CPU_ENV="${SB_POMDP_CPU_ENV:-source ~/cpu-torch/bin/activate}"
exec bash jobs/genkai_production_cpu.sh
