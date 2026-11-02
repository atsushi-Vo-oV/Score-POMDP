#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-alpha / disk tag, range observations (noisy distance to every ghost, no view cone) / alpha-pool encoder + alpha_lse critic + diffusion actor, direct score, SVGD / seed 14.
# 1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdalpha_tagr_direct-svgd.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgdalpha-tagr-direct-svgd-v2"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
