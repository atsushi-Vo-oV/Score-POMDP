#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / mirror-world Hike (w=4, outside -5) / score transformer SVGD with the direct score / seed 12.
# logbound: direct score squashed by x / log(e + x^2) (unbounded, sublinear) instead of 10 tanh(x / 10); no -x restoring term at t = 0.
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_mirror2_tf-svgd-logbound.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-mirror2-tf-svgd-logbound-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
