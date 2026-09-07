#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / score v2 study / v4 / score_alpha / seed 11, 200 updates.
# v4 = inverse of v3: warm start + observation anchor WITHOUT auxiliary losses (no g)
# Segmented: on a wall-time stop resubmit this same file to resume from latest.pt.
# Submit from the repository root AFTER debug_v2.sh (and debug_v2_lite.sh for
# the lite arms) has passed: pjsub jobs/mountain_hike_v2/v4_alpha_s11.sh
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_v4_seeds.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v4-seeds-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
