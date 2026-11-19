#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# transition initialisation (experiment 30) / bsuite memory length, N = 10 / light TF-SVGD, v2 default / seed 12.
# The transition network starts as 4 * (previous particle - x) per pair with uniform attention, and the initial network starts as 4 * (B o - x) (pair units);
# nothing is added to the score and the weights stay trainable.
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_ml10_tf-svgd-tinitq.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-ml10-tf-svgd-tinitq-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
