#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / PF-RNN-DPFRL-faithful particle filter / seed 12.
# Stochastic PF-GRU transitions (reparameterised, learned scale), discriminative
# log-weights, soft resampling every step (alpha 0.9 = DPFRL's Mountain Hike
# value), mean-particle + 8 MGF features; exogenous noise stored in the input
# stream so replay/resume stay exact. Widths chosen to match the deterministic
# baseline's parameter count (76,080 vs 75,952): pf_hidden [108,192], K=16, D=16.
# Segmented: resubmit to resume.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_pfdpfrl.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="mountain-hike-pfdpfrl-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
