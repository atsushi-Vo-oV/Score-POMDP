#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# observation-leak comparison set / disk tag, range observations (noisy distance to every ghost, no view cone) / seed 12.
# GRU baseline at the energy-TF FLOPs budget, heads read the projected hidden state only (control)
# Same seeds, schedule and FLOPs budget as the other arms of jobs/trysvgd_obs (see README.md).
# 1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_tagr_gru.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="trysvgd-tagr-gru-ctl-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
