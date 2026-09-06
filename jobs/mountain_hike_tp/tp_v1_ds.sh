#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=36:00:00
#PJM -j
#PJM -S

# Mountain Hike TELEPORT (p=0.03 uniform relocation per step) / v1 / score_deepsets / seed 10, 100 updates.
# Kidnapped-robot variant: the belief must be rebuilt after a jump; warm-start
# chains move slowly, particle-filter resampling does not. Resubmit to resume.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_tp_v1.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="mountain-hike-tp-v1-v1"
export SB_POMDP_SEGMENT_UPDATES=100
exec bash jobs/genkai_production.sh
