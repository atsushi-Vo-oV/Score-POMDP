#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# prediction-only representation set / mountain hike with the two-axis mirror (hidden m, signpost at the origin) / seed 11.
# GRU baseline at the energy-TF FLOPs budget; recurrent state trained by the next-observation loss only (AIS-style), PPO trains the heads through a stop-gradient
# Control = the matching *-ctl-v1 arm of jobs/trysvgd_obs (same seeds, schedule and FLOPs budget; see README.md).
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction; the next-observation loss is also applied on every PPO rollout.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_mirror2_gru-predonly.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgd-mirror2-gru-predonly-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
