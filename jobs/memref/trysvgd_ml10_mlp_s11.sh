#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# memref (experiment 29) / ml10: bsuite memory length N = 10 (a bit shown at the first step decides the reward of the last action) / seed 11.
# memoryless control: MLP on the current observation alone
# 200 PPO updates (no prediction phases).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_ml10_mlp.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgd-ml10-mlp-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
