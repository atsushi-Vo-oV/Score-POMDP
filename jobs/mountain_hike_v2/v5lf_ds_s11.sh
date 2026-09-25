#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=36:00:00
#PJM -j
#PJM -S

# Mountain Hike / v5 FLOPs-matched ("v5lf") / score_deepsets / seed 11, 200 updates.
# Same design as v5 but with K=4 particles, L=2 ULA steps
# and energy MLP [32,32]: 0.68-0.72 MFLOP per environment step at inference, between
# the GRU (0.46) and the DPFRL particle filter (0.87); v2 proper costs 14.7-16.2 MFLOP.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_v5lf.json"
export SB_POMDP_SHARD_INDEX=12
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v5lf-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
