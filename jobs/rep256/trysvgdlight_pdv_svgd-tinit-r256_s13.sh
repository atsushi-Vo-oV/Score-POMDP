#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# enlarged, representation-matched comparison (experiment 31) / pendulum without velocity / seed 13.
# light TF-SVGD, v1 default, transition network started as a pull to the previous cloud.
# Every arm carries 256 numbers per step (16 particles x 16 dimensions, or a GRU state of 256) at about 12 MFLOP per acting step.
# 1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_pdv_tf-svgd-tinit-r256.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-pdv-tf-svgd-tinit-r256-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
