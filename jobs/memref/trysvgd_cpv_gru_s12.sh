#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# memref (experiment 29) / cpv: cart-pole without velocities (positions only; SB3 CartPoleNoVel-v1 / POPGym StatelessCartPole) / seed 12.
# GRU baseline (FLOPs-matched to the light TF-SVGD)
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_cpv_gru.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="trysvgd-cpv-gru-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
