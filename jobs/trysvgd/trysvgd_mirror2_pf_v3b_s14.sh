#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd / mirror-world Hike (w=4, outside -5) / particle_filter v3 (bilinear observation-compatibility head, FLOPs-matched) / seed 14.
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_mirror2_pf_v3b.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgd-mirror2-pf-v3b"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
