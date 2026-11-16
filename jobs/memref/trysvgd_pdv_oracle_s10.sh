#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# memref (experiment 29) / pdv: pendulum swing-up without the angular velocity (POPGym StatelessPendulumEasy / SB3 PendulumNoVel-v1) / seed 10.
# full-state control: MLP on the true simulator state
# 1000 PPO updates (no prediction phases).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_pdv_oracle.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="trysvgd-pdv-oracle-v1"
export SB_POMDP_SEGMENT_UPDATES=1000
exec bash jobs/genkai_production.sh
