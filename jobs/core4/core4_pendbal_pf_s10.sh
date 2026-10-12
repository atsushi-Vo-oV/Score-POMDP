#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# core4 / Balancing pendulum (fixed PPO settings) / particle_filter / seed 10, 200 updates.
# core4 = gru, rnn, DPFRL particle filter and the score transformer (v2 belief: warm start +
# observation anchor + aux losses) at a matched inference budget: K=16 particles, L=1 Langevin
# step, energy MLP [12,12], d_model 32, ff 64 -> 0.81 MFLOP/step (gru 0.46, PF 0.87).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/core4_pendbal.json"
export SB_POMDP_SHARD_INDEX=16
export SB_POMDP_CAMPAIGN_ID="core4-pendbal-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
