#!/bin/bash

set -euo pipefail

DRY_RUN=0
SHARD_INDEX=""
SEGMENT_UPDATES=""
SEGMENT_COUNT=""
CAMPAIGN_ID=""
JOB_SCRIPT="jobs/genkai_production.sh"

usage() {
  cat <<'EOF'
Usage: bash jobs/submit_production_chain.sh [--dry-run] --shard-index N --segment-updates N --segments N campaign-id

Registers one resumable production shard as separate PJM step subjobs. Genkai's
site pjsub wrapper accepts one script_file per invocation, so later subjobs are
attached to the first submission with jid=<step-job-id>.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --shard-index)
      SHARD_INDEX="${2:-}"
      shift 2
      ;;
    --segment-updates)
      SEGMENT_UPDATES="${2:-}"
      shift 2
      ;;
    --segments)
      SEGMENT_COUNT="${2:-}"
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

if [[ ! "$SHARD_INDEX" =~ ^[1-9][0-9]*$ ]] || (( 10#$SHARD_INDEX > 20 )); then
  echo "--shard-index must select a full/score_transformer shard in 1..20." >&2
  exit 2
fi
if [[ ! "$SEGMENT_UPDATES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--segment-updates must be a positive integer." >&2
  exit 2
fi
if [[ ! "$SEGMENT_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "--segments must be a positive integer." >&2
  exit 2
fi
if (( ${#CAMPAIGN_ID} > 128 )) || \
   [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]*[A-Za-z0-9_-])?$ ]]; then
  echo "campaign-id must contain one safe ASCII identifier." >&2
  exit 2
fi
if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Run this command from the project root; missing $JOB_SCRIPT." >&2
  exit 2
fi
if (( ! DRY_RUN )) && ! command -v pjsub >/dev/null 2>&1; then
  echo "pjsub is not available; run this script on a Genkai login node." >&2
  exit 127
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

exported="SB_POMDP_SHARD_INDEX=$SHARD_INDEX,SB_POMDP_CAMPAIGN_ID=$CAMPAIGN_ID,SB_POMDP_SEGMENT_UPDATES=$SEGMENT_UPDATES"
first_sparam="sn=0"

if (( DRY_RUN )); then
  print_command pjsub -z jid --step -x "$exported" --sparam "$first_sparam" "$JOB_SCRIPT"
  for ((step = 1; step < SEGMENT_COUNT; step++)); do
    previous_step=$((step - 1))
    print_command \
      pjsub --step \
      -x "$exported" \
      --sparam "jid=STEP_JOB_ID,sn=$step,sd=ec!=0:all:$previous_step" \
      "$JOB_SCRIPT"
  done
  exit 0
fi

first_subjob_id=$(
  pjsub -z jid --step -x "$exported" --sparam "$first_sparam" "$JOB_SCRIPT"
)
if [[ ! "$first_subjob_id" =~ ^([0-9]+)_0$ ]]; then
  echo "pjsub returned an unexpected first subjob ID: $first_subjob_id" >&2
  exit 1
fi
step_job_id="${BASH_REMATCH[1]}"
echo "Registering later segments under step job $step_job_id." >&2

for ((step = 1; step < SEGMENT_COUNT; step++)); do
  previous_step=$((step - 1))
  pjsub --step \
    -x "$exported" \
    --sparam "jid=$step_job_id,sn=$step,sd=ec!=0:all:$previous_step" \
    "$JOB_SCRIPT"
done

echo "Registered $SEGMENT_COUNT segments for shard $SHARD_INDEX as step job $step_job_id."
