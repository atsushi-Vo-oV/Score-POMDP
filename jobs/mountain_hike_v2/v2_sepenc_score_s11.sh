#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=60:00:00
#PJM -j
#PJM -S

# Mountain Hike / v2 + separate critic encoder (model.value_encoder=separate) / score_transformer / seed 11, 200 updates.
# The critic reads the particles through its own set encoder, so the value loss
# never shapes the policy's belief representation (v2 shares one encoder).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_v2_sepenc.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v2-sepenc-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
