#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# core4mm / mirror-world Hike (ridge 1, corners, w=2, outside -1.5) / gru / seed 12, 200 updates.
# core4 = gru, rnn, DPFRL particle filter and the score transformer (v2 belief: warm start +
# observation anchor + aux losses) at a matched inference budget: K=16 particles, L=12 Langevin
# steps, energy MLP [8] (one SiLU hidden layer), d_model 32, ff 64 -> 1.11 MFLOP/step
# (gru 0.46, PF 0.87).  Measured on core4: ~32 min/update at horizon 75, so horizon 100 needs
# ~143 h; the elapse above is sized for one segment.  Resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/core4_mirror.json"
export SB_POMDP_SHARD_INDEX=8
export SB_POMDP_CAMPAIGN_ID="core4-mirror-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
