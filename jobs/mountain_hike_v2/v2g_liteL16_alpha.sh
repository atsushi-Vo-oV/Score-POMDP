#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / score v2 study / v2g_liteL16 / score_alpha / seed 10, 100 updates.
# v2g, small networks (energy [16,16], d_model 32, policy [48,48]) with L 4 -> 16
# Segmented: on a wall-time stop resubmit this same file to resume from latest.pt.
# Submit from the repository root AFTER debug_v2.sh (and debug_v2_lite.sh for
# the lite arms) has passed: pjsub jobs/mountain_hike_v2/v2g_liteL16_alpha.sh
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_v2g_liteL16.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v2g-liteL16-v1"
export SB_POMDP_SEGMENT_UPDATES=100
exec bash jobs/genkai_production.sh
