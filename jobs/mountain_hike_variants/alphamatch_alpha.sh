#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / score_alpha with the trunk widened to match score_transformer's
# parameter count (alpha_feedforward_dim 704, 16 pieces: 75,269 vs 75,286).
# Tests whether alpha's earlier edge came from having fewer parameters.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_alphamatch.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="mountain-hike-alphamatch-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
