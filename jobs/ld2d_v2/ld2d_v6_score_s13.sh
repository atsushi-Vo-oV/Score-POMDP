#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S

# Light-Dark 2D (P3O-symlog task) / v6 / score_transformer / seed 13, 200 updates.
# v6 = v3 (cold chain + aux) with the set-level mixture-density aux predictor. The aux losses read the whole particle set through a DeepSets
# network that outputs a 4-component Gaussian mixture (model.aux_predictor_kind=set_mixture).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/ld2d_v6.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="ld2d-v6-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
