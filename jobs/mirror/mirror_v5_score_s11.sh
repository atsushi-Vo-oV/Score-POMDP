#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S

# Mirror-world Mountain Hike / v5 / score_transformer / seed 11, 200 updates. v2 with the set-level mixture-density aux predictor.
# Task: body frame = world frame reflected by an unknown m in {+-1}^2 (fixed per episode);
# observation m*s + N(0, 3^2) plus a signpost m*exp(-|s|^2/8) + N(0, 0.2^2) readable only
# near the origin; action a moves the world state by m*a; reward = DVRL ridge 1 only
# (world y = 5, best toward x = 8); start = one of the four mirror corners; 100 steps.
# The world-position belief is four-modal until the signpost is read.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/mirror_v5.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="mirror-v5-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
