#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-alpha / Hike |s| observation + teleport 0.03 / alpha-pool encoder + alpha_lse critic + diffusion actor, direct score, SVGD / seed 13.
# 213 updates = 98 random-action prediction + 100 PPO + 15 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdalpha_tps_direct-svgd.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="trysvgdalpha-tps-direct-svgd-v2"
export SB_POMDP_SEGMENT_UPDATES=213
exec bash jobs/genkai_production.sh
