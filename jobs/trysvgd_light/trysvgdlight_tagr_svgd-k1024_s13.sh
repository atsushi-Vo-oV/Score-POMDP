#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / disk tag, range observations (noisy distance to every ghost, no view cone) / score transformer SVGD with the direct score, K = 1024 particles / seed 13.
# k1024: num_particles 16 -> 1024, everything else as the plain light arm (compute not matched: pair terms and attention grow with K^2).
# 1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_tagr_tf-svgd-k1024.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-tagr-tf-svgd-k1024-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
