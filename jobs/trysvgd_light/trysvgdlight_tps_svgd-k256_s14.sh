#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / Hike |s| observation + teleport 0.03 / score transformer SVGD with the direct score, K = 256 particles / seed 14.
# k256: num_particles 16 -> 256, everything else as the plain light arm (compute not matched: pair terms and attention grow with K^2).
# 213 updates = 98 random-action prediction + 100 PPO + 15 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_tps_tf-svgd-k256.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-tps-tf-svgd-k256-v1"
export SB_POMDP_SEGMENT_UPDATES=213
exec bash jobs/genkai_production.sh
