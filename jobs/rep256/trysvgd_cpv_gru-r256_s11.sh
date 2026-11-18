#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# enlarged, representation-matched comparison (experiment 31) / cart-pole without velocities / seed 11.
# GRU baseline, hidden state 256, encoder FLOPs-matched to the enlarged light.
# Every arm carries 256 numbers per step (16 particles x 16 dimensions, or a GRU state of 256) at about 12 MFLOP per acting step.
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_cpv_gru-r256.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgd-cpv-gru-r256-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
