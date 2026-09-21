#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S

# Light-Dark 2D (P3O-symlog task) / v2 + separate critic encoder (model.value_encoder=separate) / score_deepsets / seed 13, 200 updates.
# The critic reads the particles through its own set encoder, so the value loss
# never shapes the policy's belief representation (v2 shares one encoder).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/ld2d_v2_sepenc.json"
export SB_POMDP_SHARD_INDEX=14
export SB_POMDP_CAMPAIGN_ID="ld2d-v2-sepenc-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
