#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# observation-leak comparison set / mountain hike with the two-axis mirror (hidden m, signpost at the origin) / seed 11.
# GRU baseline as the control, but heads read [condition, o_t] (model.policy_observation_input = true)
# Same seeds, schedule and FLOPs budget as the other arms of jobs/trysvgd_obs (see README.md).
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_mirror2_gru-obs.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgd-mirror2-gru-obs-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
