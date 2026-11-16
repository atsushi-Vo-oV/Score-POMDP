#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# memref (experiment 29) / ml30: bsuite memory length N = 30 (a bit shown at the first step decides the reward of the last action) / seed 13.
# light TF-SVGD, v2 default
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_ml30_tf-svgd.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-ml30-tf-svgd-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
