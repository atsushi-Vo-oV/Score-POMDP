#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S

# Light-Dark 2D (P3O-symlog task: light 5.0, initial_std 1.0, no goal bonus) / v3 / score_alpha / seed 11, 200 updates.
# Same score design grid as Mountain Hike (v1 cold, v2 warm+anchor+aux, v3 cold+aux, v4 warm+anchor).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/ld2d_v3.json"
export SB_POMDP_SHARD_INDEX=7
export SB_POMDP_CAMPAIGN_ID="ld2d-v3-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
