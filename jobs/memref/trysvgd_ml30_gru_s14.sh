#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# memref (experiment 29) / ml30: bsuite memory length N = 30 (a bit shown at the first step decides the reward of the last action) / seed 14.
# GRU baseline (FLOPs-matched to the light TF-SVGD)
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_ml30_gru.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgd-ml30-gru-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
