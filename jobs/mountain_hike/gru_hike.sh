#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike (Igl et al. 2018, DVRL) / full / gru / seed 10.
# Faithful port of the reference DeathValleyEnv with the mountainHike.yaml
# constants (paper coordinates: start N((-8.5,-8.5), I), |a| <= 0.5, transition
# std 0.25, observation std 3.0 = the paper's hardest noise level, 75 steps,
# terrain reward minus 0.01|a|, box penalty -6). Protocol as ld2d-refs-v1
# (200 updates, lr 2.5e-4, entropy 0.01, clip 10) with symlog critic targets
# (per-step rewards range about -6 .. -0.3). Segmented: on a wall-time stop
# resubmit this same file to resume from latest.pt.
# Submit from the repository root AFTER debug_hike.sh has passed:
#   pjsub jobs/mountain_hike/gru_hike.sh
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike.json"
export SB_POMDP_SHARD_INDEX=10
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
