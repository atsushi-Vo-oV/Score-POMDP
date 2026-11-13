#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / Hike |s| observation + teleport 0.03 / score transformer SVGD with the directly parameterised score / seed 11.
# uniform_init_u0: chain start U(-sqrt(3), sqrt(3)) per latent coordinate at every step; at t = 0 the observation anchor is
# applied to that uniform noise instead of N(0, I) (model.langevin_initial_noise=uniform).
# -JOINT: one network maps (o, x, a, attention summary of the previous particles) to the score;
# otherwise the observation term and the attention transition term are separate networks, summed.
# As try-svgd (no warm start, learned alpha, prediction phases, diffusion PPO policy) except that the
# networks output the score itself instead of a scalar energy differentiated by autograd.
# 213 updates.  Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_tps_tf-svgd-uniform-init-u0.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-tps-tf-svgd-uniform-init-u0-v1"
export SB_POMDP_SEGMENT_UPDATES=213
exec bash jobs/genkai_production.sh
