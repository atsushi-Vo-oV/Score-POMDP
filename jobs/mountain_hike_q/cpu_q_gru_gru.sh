#!/bin/bash
#PJM -L rscgrp=a-batch
#PJM -L vnode-core=30
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# gru baseline with the action-value critic, 200 updates
# CPU (node group A) copy of q_gru_gru.sh: same campaign id, config with experiment.device=cpu,
# 30 cores. Never run together with the MIG copy (same campaign directory).
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_q_gru_cpu.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="mountain-hike-q-gru-v1"
export SB_POMDP_SEGMENT_UPDATES=200
export SB_POMDP_THREADS=30
export SB_POMDP_CPU_ENV="${SB_POMDP_CPU_ENV:-source ~/cpu-torch/bin/activate}"
exec bash jobs/genkai_production_cpu.sh
