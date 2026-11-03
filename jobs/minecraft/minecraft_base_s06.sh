#!/bin/bash
# Minecraft walk / GRU / RNN / particle-filter baselines (VAE latent 32) / seed 12.  Math server only (MineRL in ~/workspace/agent/minecraft; 4 Minecraft
# instances per training process, Xvfb).  Not a Genkai job.  Segmented: resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/minecraft_base.json"
export SB_POMDP_SHARD_INDEX=6
export SB_POMDP_CAMPAIGN_ID="minecraft-base-v1"
export SB_POMDP_SEGMENT_UPDATES=400
exec bash jobs/genkai_production.sh
