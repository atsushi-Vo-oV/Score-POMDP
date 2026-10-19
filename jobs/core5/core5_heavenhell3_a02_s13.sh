#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=120:00:00
#PJM -j
#PJM -S

# core5 / Heaven-Hell n=3 (horizon 100) / score transformer k16f, alpha 0.2 fixed / seed 13, 200 updates.
# Same as the core4 transformer (K=16, L=12, energy [8], d_model 32, ff 64, tau 1, v2 belief)
# except for the ULA step size.  Estimated ~32 min/update -> ~107 h in total.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/core5_heavenhell3_a02.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="core5-heavenhell3-a02-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
