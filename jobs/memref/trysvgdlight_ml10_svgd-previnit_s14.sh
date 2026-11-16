#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# memref (experiment 29) / ml10: bsuite memory length N = 10 (a bit shown at the first step decides the reward of the last action) / seed 14.
# light TF-SVGD, v1 default, chain started at the previous particles
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_ml10_tf-svgd-prev-init.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-ml10-tf-svgd-prev-init-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
