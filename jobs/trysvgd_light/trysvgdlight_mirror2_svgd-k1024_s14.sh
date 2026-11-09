#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / mirror-world Hike (w=4, outside -5) / score transformer SVGD with the direct score, K = 1024 particles / seed 14.
# k1024: num_particles 16 -> 1024, everything else as the plain light arm (compute not matched: pair terms and attention grow with K^2).
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_mirror2_tf-svgd-k1024.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-mirror2-tf-svgd-k1024-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
