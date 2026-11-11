#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / Car-Flag (green flag at +1 or -1, side revealed only within 0.2 of the priest at 0.5) / score transformer SVGD with the direct score, long budget / seed 13.
# long: ppo.total_steps 409600 -> 4096000 (2498 updates = 98 random-action prediction + 2000 PPO + 400 interleaved prediction); everything else as the 333-update arm.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_carflag_tf-svgd-long.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-carflag-tf-svgd-long-v1"
export SB_POMDP_SEGMENT_UPDATES=2498
exec bash jobs/genkai_production.sh
