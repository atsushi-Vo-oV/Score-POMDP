#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# observation-leak comparison set / disk tag, range observations (noisy distance to every ghost, no view cone) / seed 10.
# DPFRL particle filter v3 (mlp observation function, K = 16) at the energy-TF FLOPs budget, heads read the condition only (control)
# Same seeds, schedule and FLOPs budget as the other arms of jobs/trysvgd_obs (see README.md).
# 1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_tagr_pf_v3.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="trysvgd-tagr-pf-v3-ctl-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
