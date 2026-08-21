#!/bin/bash
# Launcher for the 48-shard pre-experiment campaign (config/pre_experiment.json:
# 2 gradient modes x 2 methods x 4 tasks x 3 seeds). Every shard fits well
# inside one 168-hour job at the measured pre-experiment speed, so all shards
# are submitted as independent normal jobs -- no PJM step chains, no segmented
# resume. Nothing is submitted unless the operator runs this script.
#
# usage: bash jobs/submit_pre_campaign.sh [--dry-run] [--seed-offset 0|1|2] <campaign-id>
#   --dry-run      print the pjsub commands without submitting
#   --seed-offset  submit only one of the three seeds (0->10, 1->11, 2->12);
#                  omit to submit all 48 shards

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="config/pre_experiment.json"
TOTAL_SHARDS=48
SEEDS_PER_CELL=3

DRY_RUN=0
SEED_OFFSET=""
CAMPAIGN_ID=""
while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --seed-offset)
      if (( $# < 2 )) || [[ ! "$2" =~ ^[0-2]$ ]]; then
        echo "--seed-offset requires 0, 1, or 2." >&2
        exit 2
      fi
      SEED_OFFSET="$2"; shift 2 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)
      if [[ -n "$CAMPAIGN_ID" ]]; then
        echo "Exactly one campaign id is expected." >&2
        exit 2
      fi
      CAMPAIGN_ID="$1"; shift ;;
  esac
done

if (( ${#CAMPAIGN_ID} > 128 )) || \
   [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]*[A-Za-z0-9_-])?$ ]]; then
  echo "campaign id must contain one safe ASCII identifier." >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing $CONFIG_PATH" >&2
  exit 2
fi

# Bulk-index order is mode -> method -> task -> seed (matching sb_pomdp.compare):
# full/score 1-12, full/gru 13-24, tbptt_1/score 25-36, tbptt_1/gru 37-48, and
# within each mode/method block task-major with SEEDS_PER_CELL consecutive seeds.
SHARDS=()
for (( index = 1; index <= TOTAL_SHARDS; index++ )); do
  if [[ -n "$SEED_OFFSET" ]] && (( (index - 1) % SEEDS_PER_CELL != SEED_OFFSET )); then
    continue
  fi
  SHARDS+=("$index")
done

echo "campaign=$CAMPAIGN_ID config=$CONFIG_PATH shards=${#SHARDS[@]}" >&2
for index in "${SHARDS[@]}"; do
  command=(pjsub -x "SB_POMDP_CONFIG=${CONFIG_PATH},SB_POMDP_SHARD_INDEX=${index},SB_POMDP_CAMPAIGN_ID=${CAMPAIGN_ID}" jobs/genkai_production.sh)
  if (( DRY_RUN )); then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
done

if (( DRY_RUN )); then
  echo "Dry run only; nothing was submitted." >&2
else
  echo "Submitted ${#SHARDS[@]} independent shard job(s) for campaign $CAMPAIGN_ID." >&2
fi
