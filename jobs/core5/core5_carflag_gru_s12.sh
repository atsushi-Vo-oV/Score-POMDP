#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# core5 / Car-Flag (horizon 160) / gru / seed 12, 200 updates.
# core4 method set (gru, rnn, DPFRL PF, score transformer k16f alpha 0.02) on a new task.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/core5_carflag.json"
export SB_POMDP_SHARD_INDEX=8
export SB_POMDP_CAMPAIGN_ID="core5-carflag-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
