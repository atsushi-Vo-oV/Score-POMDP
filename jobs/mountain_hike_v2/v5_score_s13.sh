#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=60:00:00
#PJM -j
#PJM -S

# Mountain Hike / v5 / score_transformer / seed 13, 200 updates.
# v5 = v2 (warm start + observation anchor + aux) with the set-level mixture-density aux predictor. The aux losses read the whole particle set through a DeepSets
# network that outputs a 4-component Gaussian mixture (model.aux_predictor_kind=set_mixture).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_v5.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v5-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
