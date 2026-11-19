#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# transition initialisation (experiment 30) / pendulum without velocity / light TF-SVGD, v2 default / seed 14.
# The transition network starts as 4 * (previous particle - x) per pair with attention logit -4 * sum_d z_d tanh(z_d / 2), every query is pulled to the previous particles near it (pair units);
# nothing is added to the score and the weights stay trainable.
# 1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_pdv_tf-svgd-tinitmix.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-pdv-tf-svgd-tinitmix-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
