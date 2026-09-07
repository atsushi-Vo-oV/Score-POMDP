#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=36:00:00
#PJM -j
#PJM -S

# Mountain Hike TELEPORT (p=0.03) + ABS-SYMMETRIC observation (o=|s|+noise) / v3 / score_alpha / seed 10, 100 updates.
# Kidnapped-robot variant: the belief must be rebuilt after a jump; warm-start
# chains move slowly, particle-filter resampling does not. Resubmit to resume.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_tps_v3.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="mountain-hike-tps-v3-v1"
export SB_POMDP_SEGMENT_UPDATES=100
exec bash jobs/genkai_production.sh
