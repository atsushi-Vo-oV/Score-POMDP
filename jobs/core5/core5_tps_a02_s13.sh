#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=72:00:00
#PJM -j
#PJM -S

# core5 / teleport + |s| Hike (p=0.03) / score transformer k16f, alpha 0.2 fixed / seed 13, 100 updates.
# Same as the core4 transformer (K=16, L=12, energy [8], d_model 32, ff 64, tau 1, v2 belief)
# except for the ULA step size.  Estimated ~31 min/update -> ~52 h in total.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/core5_tps_a02.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="core5-tps-a02-v1"
export SB_POMDP_SEGMENT_UPDATES=100
exec bash jobs/genkai_production.sh
