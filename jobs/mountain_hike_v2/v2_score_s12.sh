#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / score v2 study / v2 / score_transformer / seed 12, 200 updates.
# warm start + observation anchor + reward/obs aux (no drift g)
# Segmented: on a wall-time stop resubmit this same file to resume from latest.pt.
# Submit from the repository root AFTER debug_v2.sh (and debug_v2_lite.sh for
# the lite arms) has passed: pjsub jobs/mountain_hike_v2/v2_score_s12.sh
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_v2_seeds.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v2-seeds-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
