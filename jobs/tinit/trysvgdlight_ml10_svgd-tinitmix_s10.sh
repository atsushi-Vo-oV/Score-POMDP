#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# transition initialisation (experiment 30) / bsuite memory length, N = 10 / light TF-SVGD, v1 default / seed 10.
# The transition network starts as 4 * (previous particle - x) per pair with attention logit -4 * sum_d z_d tanh(z_d / 2), every query is pulled to the previous particles near it (pair units);
# nothing is added to the score and the weights stay trainable.
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_ml10_tf-svgd-tinitmix.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-ml10-tf-svgd-tinitmix-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
