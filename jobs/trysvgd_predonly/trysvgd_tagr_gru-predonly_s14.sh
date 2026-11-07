#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# prediction-only representation set / disk tag, range observations (noisy distance to every ghost, no view cone) / seed 14.
# GRU baseline at the energy-TF FLOPs budget; recurrent state trained by the next-observation loss only (AIS-style), PPO trains the heads through a stop-gradient
# Control = the matching *-ctl-v1 arm of jobs/trysvgd_obs (same seeds, schedule and FLOPs budget; see README.md).
# 1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction; the next-observation loss is also applied on every PPO rollout.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_tagr_gru-predonly.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgd-tagr-gru-predonly-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
