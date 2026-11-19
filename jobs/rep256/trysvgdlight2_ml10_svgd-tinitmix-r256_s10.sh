#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# enlarged, representation-matched comparison (experiment 31) / bsuite memory length, N = 10 / seed 10.
# light TF-SVGD, v2 default, transition network started as a pull to the previous particles near the query.
# Every arm carries 256 numbers per step (16 particles x 16 dimensions, or a GRU state of 256) at about 12 MFLOP per acting step.
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_ml10_tf-svgd-tinitmix-r256.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-ml10-tf-svgd-tinitmix-r256-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
