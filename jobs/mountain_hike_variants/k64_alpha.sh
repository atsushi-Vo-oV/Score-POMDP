#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / 64 particles (K=16 -> 64) / score_alpha / seed 10, 100 updates
# (ULA and set-encoder cost scale with K). Does resolution/coverage of the
# particle cloud change anything? alpha uses the parameter-matched trunk.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_k64.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="mountain-hike-k64-v1"
export SB_POMDP_SEGMENT_UPDATES=100
exec bash jobs/genkai_production.sh
