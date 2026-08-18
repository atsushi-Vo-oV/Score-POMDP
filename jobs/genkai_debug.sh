#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -S

set -euo pipefail

PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SB_POMDP_OMP_NUM_THREADS:-16}"
EXPECTED_DEBUG_SHARDS=16

if [[ -n "${PJM_BULKNUM:-}" ]]; then
  echo "Bulk submission is disabled; submit this script with one explicit shard." >&2
  exit 2
fi
SHARD_INDEX="${SB_POMDP_SHARD_INDEX:-}"
if [[ ! "$SHARD_INDEX" =~ ^[1-9][0-9]{0,2}$ ]] || \
   (( 10#$SHARD_INDEX < 1 || 10#$SHARD_INDEX > EXPECTED_DEBUG_SHARDS )); then
  echo "SB_POMDP_SHARD_INDEX must be an integer in 1-${EXPECTED_DEBUG_SHARDS}." >&2
  exit 2
fi

CAMPAIGN_ID="${SB_POMDP_CAMPAIGN_ID:-}"
if (( ${#CAMPAIGN_ID} > 128 )) || \
   [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]*[A-Za-z0-9_-])?$ ]]; then
  echo "SB_POMDP_CAMPAIGN_ID must contain one safe ASCII identifier." >&2
  exit 2
fi

# Two methods across all four tasks, repeated for full and one-step TBPTT.
PAIRS=(
  "score_transformer masked_cartpole"
  "score_transformer masked_pendulum"
  "score_transformer masked_mountain_car_continuous"
  "score_transformer light_dark"
  "gru masked_cartpole"
  "gru masked_pendulum"
  "gru masked_mountain_car_continuous"
  "gru light_dark"
)
ZERO_BASED=$((10#$SHARD_INDEX - 1))
PAIR_INDEX=$((ZERO_BASED % ${#PAIRS[@]}))
MODE_INDEX=$((ZERO_BASED / ${#PAIRS[@]}))
MODES=("full" "tbptt_1")
GRADIENT_MODE="${MODES[$MODE_INDEX]}"
read -r METHOD TASK <<< "${PAIRS[$PAIR_INDEX]}"
SHARD_TAG=$(printf '%03d' "$((10#$SHARD_INDEX))")

SHORT_DEBUG="${SB_POMDP_SHORT_DEBUG:-0}"
if [[ "$SHORT_DEBUG" != "0" && "$SHORT_DEBUG" != "1" ]]; then
  echo "SB_POMDP_SHORT_DEBUG must be either 0 or 1." >&2
  exit 2
fi
DEBUG_OVERRIDES=(--override experiment.eval_episodes=1)
if [[ "$SHORT_DEBUG" == "1" ]]; then
  # Preserve the production model, K/L, T=128, logical minibatch 512, and
  # sequence microbatch 128.  Reduce only independent environments and PPO
  # replay epochs for the bounded short-debug profile.
  DEBUG_OVERRIDES+=(--override ppo.num_envs=4 --override ppo.epochs=1)
fi

# One PPO update and one evaluation episode for the selected condition.  DEBUG
# runs at validation seed 0, which config/production.json (seeds 10--14) does not
# declare, so the seed is stated explicitly; --seeds is validated against the
# configured list.  The resolved config is unaffected because the comparison
# driver always rewrites experiment.seeds with the selected seed.
python3.11 -m sb_pomdp.compare \
  --config config/production.json \
  --gradient-modes "$GRADIENT_MODE" \
  --methods "$METHOD" \
  --tasks "$TASK" \
  --seeds 0 \
  --override "experiment.seeds=[0]" \
  --max-updates 1 \
  --campaign-id "$CAMPAIGN_ID" \
  --campaign-shard-index "$SHARD_INDEX" \
  --override "experiment.output_dir=results/campaigns/$CAMPAIGN_ID" \
  "${DEBUG_OVERRIDES[@]}" \
  --label "genkai-${CAMPAIGN_ID}-shard${SHARD_TAG}"
