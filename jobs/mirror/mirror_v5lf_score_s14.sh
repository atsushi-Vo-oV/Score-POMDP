#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=36:00:00
#PJM -j
#PJM -S

# Mirror-world Mountain Hike / v5lf (FLOPs-matched) / score_transformer / seed 14, 200 updates. v2 with the set-level mixture-density aux predictor.
# Score arms run the lf chain: K=4 particles, L=2 ULA steps, energy MLP [32,32]
# (0.68-0.72 MFLOP per environment step, between gru 0.46 and DPFRL PF 0.87).
# Task: body frame = world frame reflected by an unknown m in {+-1}^2 (fixed per episode);
# observation m*s + N(0, 3^2) plus a signpost m*exp(-|s|^2/8) + N(0, 0.2^2) readable only
# near the origin; action a moves the world state by m*a; reward = DVRL ridge 1 only;
# start = one of the four mirror corners; 100 steps. Four-modal belief until the signpost.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/mirror_v5lf.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="mirror-v5lf-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
