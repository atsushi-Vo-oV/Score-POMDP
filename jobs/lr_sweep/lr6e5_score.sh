#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / learning-rate sweep 6.25e-5 (x1/4) / score_transformer / seed 10.
# The trapped arms (score_transformer, pf at -445) move at the same KL as the
# learning gru (0.002-0.003), so this tests both directions: faster escape
# (x4) vs. slower early drift that may avoid walking out of the box (x1/4).
# Segmented: resubmit to resume.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_lr6e5.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="mountain-hike-lr6e5-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
