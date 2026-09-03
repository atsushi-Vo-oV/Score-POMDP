#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / KAN heads + trunk (MLP energies) / score_deepsets / seed 10.
# KAN energy networks cost ~9x per PPO update on the GPU (DEBUG: 456 s vs 49 s),
# which puts 200 updates out of reach; this arm applies the parameter-matched
# KAN to the policy/value heads and the diffusion denoiser only - the same
# components the KAN baselines (gru/rnn/pf) replace - so the function-class
# comparison stays fair. alpha uses the parameter-matched trunk. Segmented.
# Submit from the repository root: pjsub jobs/mountain_hike_variants/kanht_ds.sh
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_kanht.json"
export SB_POMDP_SHARD_INDEX=7
export SB_POMDP_CAMPAIGN_ID="mountain-hike-kanht-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
