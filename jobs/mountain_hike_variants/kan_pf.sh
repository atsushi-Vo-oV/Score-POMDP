#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / KAN variant / particle_filter / seed 10: all learned networks are
# parameter-matched KANs (energies for score models; heads and trunks for
# everyone). score_alpha uses the parameter-matched trunk (ff 704, 16 pieces).
# Otherwise identical to mountain-hike-v1. Segmented (resubmit to resume).
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_kan.json"
export SB_POMDP_SHARD_INDEX=16
export SB_POMDP_CAMPAIGN_ID="mountain-hike-kan-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
