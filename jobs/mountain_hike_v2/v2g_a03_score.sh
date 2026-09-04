#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / score v2 study / v2g_a03 / score_transformer / seed 10, 100 updates.
# v2g, ULA step 0.02 -> 0.3
# Segmented: on a wall-time stop resubmit this same file to resume from latest.pt.
# Submit from the repository root AFTER debug_v2.sh (and debug_v2_lite.sh for
# the lite arms) has passed: pjsub jobs/mountain_hike_v2/v2g_a03_score.sh
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_v2g_a03.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v2g-a03-v1"
export SB_POMDP_SEGMENT_UPDATES=100
exec bash jobs/genkai_production.sh
