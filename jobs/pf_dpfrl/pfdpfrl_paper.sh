#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / DPFRL-faithful particle filter at the paper's sizes
# (K=30 particles, latent dim 128, alpha 0.9, 8 MGF features; ~212k params, so
# not parameter-matched). Reference point for the mechanism at its published
# scale. Segmented: resubmit to resume.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_pfdpfrl_paper.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="mountain-hike-pfdpfrl-paper-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
