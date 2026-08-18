#!/bin/bash

set -euo pipefail

DRY_RUN=0
SEED_OFFSET=0
CAMPAIGN_ID=""

usage() {
  cat <<'EOF'
Usage: bash jobs/submit_production_campaign.sh [--dry-run] [--seed-offset 0..4] [campaign-id]

Submits one complete 16-condition production seed. Independent shards are
scheduled in parallel. The four full/score_transformer shards are each a
25-step PJM chain of 10-update resumable segments.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --seed-offset)
      if (( $# < 2 )); then
        usage >&2
        exit 2
      fi
      SEED_OFFSET="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$CAMPAIGN_ID" ]]; then
        echo "Only one campaign ID may be supplied." >&2
        exit 2
      fi
      CAMPAIGN_ID="$1"
      shift
      ;;
  esac
done

if [[ ! "$SEED_OFFSET" =~ ^[0-4]$ ]]; then
  echo "--seed-offset must be one integer in 0..4 (production seeds 10..14)." >&2
  exit 2
fi
if [[ -z "$CAMPAIGN_ID" ]]; then
  CAMPAIGN_ID="prod-two-method-seed$((10 + SEED_OFFSET))-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if (( ${#CAMPAIGN_ID} > 128 )) || \
   [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]*[A-Za-z0-9_-])?$ ]]; then
  echo "campaign-id must contain one safe ASCII identifier." >&2
  exit 2
fi

if (( ! DRY_RUN )) && ! command -v pjsub >/dev/null 2>&1; then
  echo "pjsub is not available; run this script on a Genkai login node." >&2
  exit 127
fi

run_command() {
  if (( DRY_RUN )); then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# Matrix order is mode -> method -> task -> seed. These bases select seed 10;
# adding SEED_OFFSET selects the same 16 conditions for seeds 11--14.
FULL_SCORE_BASES=(1 6 11 16)
ONE_SHOT_BASES=(21 26 31 36 41 46 51 56 61 66 71 76)
SEGMENT_UPDATES=10
SEGMENT_COUNT=25

echo "Production campaign: $CAMPAIGN_ID"
echo "Production seed: $((10 + SEED_OFFSET))"

for base in "${FULL_SCORE_BASES[@]}"; do
  shard=$((base + SEED_OFFSET))
  chain_arguments=(
    --shard-index "$shard"
    --segment-updates "$SEGMENT_UPDATES"
    --segments "$SEGMENT_COUNT"
    "$CAMPAIGN_ID"
  )
  if (( DRY_RUN )); then
    chain_arguments=(--dry-run "${chain_arguments[@]}")
  fi
  bash jobs/submit_production_chain.sh "${chain_arguments[@]}"
done

for base in "${ONE_SHOT_BASES[@]}"; do
  shard=$((base + SEED_OFFSET))
  exported="SB_POMDP_SHARD_INDEX=$shard,SB_POMDP_CAMPAIGN_ID=$CAMPAIGN_ID"
  run_command pjsub -x "$exported" jobs/genkai_production.sh
done

echo "Submitted all 16 conditions for campaign $CAMPAIGN_ID"
