#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# memref (experiment 29) / pdv: pendulum swing-up without the angular velocity (POPGym StatelessPendulumEasy / SB3 PendulumNoVel-v1) / seed 12.
# light TF-SVGD, v1 default, chain started at the previous particles
# 1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_pdv_tf-svgd-prev-init.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-pdv-tf-svgd-prev-init-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
