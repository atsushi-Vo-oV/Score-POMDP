#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# memref (experiment 29) / pdvn: pendulum swing-up without the angular velocity, observation noise 0.1 clipped to [-1, 1] (POPGym NoisyStatelessPendulumEasy) / seed 12.
# memoryless control: MLP on the current observation alone
# 1000 PPO updates (no prediction phases).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_pdvn_mlp.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="trysvgd-pdvn-mlp-v1"
export SB_POMDP_SEGMENT_UPDATES=1000
exec bash jobs/genkai_production.sh
