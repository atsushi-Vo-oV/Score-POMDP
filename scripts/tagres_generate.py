"""Generate the slow elliptic tag game (tagres, experiment 25): configs and job files.

tagres = tagre (elliptic range observations, 64 environments, 64 particles, 2420 updates) with the ghost pursuit speed
lowered from 0.8-1.0 to 0.5-0.7 of the player's top speed, exactly as tagrs does for tagr (experiment 24).
Arms: light TF-SVGD v2 default, light TF-SVGD v1 default, GRU, PF v3.  Run from the repository root:
    python scripts/tagres_generate.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLOW = {"ghost_speed_min": 0.5, "ghost_speed_max": 0.7}
V2 = {"langevin_observation_anchor": False, "langevin_proposal": "uniform", "langevin_initial_noise": "uniform",
      "direct_score_bound": None, "initial_score_prior": 0.0}
# name of the new config: (source config, label, campaign, job stem, description, v2)
ARMS = {
    "trysvgdlight2_tagres_tf-svgd": ("trysvgdlight_tagre_tf-svgd", "trysvgdlight2-tagres-tf-svgd", "trysvgdlight2-tagres-tf-svgd-env64-k64-v1",
                                     "trysvgdlight2_tagres_svgd", "light TF-SVGD, v2 default", True),
    "trysvgdlight_tagres_tf-svgd": ("trysvgdlight_tagre_tf-svgd", "trysvgdlight-tagres-tf-svgd", "trysvgdlight-tagres-tf-svgd-env64-k64-v1",
                                    "trysvgdlight_tagres_svgd", "light TF-SVGD, v1 default", False),
    "trysvgd_tagres_gru": ("trysvgd_tagre_gru", "trysvgd-tagres-gru", "trysvgd-tagres-gru-env64-k64-v1", "trysvgd_tagres_gru", "GRU baseline", False),
    "trysvgd_tagres_pf_v3": ("trysvgd_tagre_pf_v3", "trysvgd-tagres-pf-v3", "trysvgd-tagres-pf-v3-env64-k64-v1", "trysvgd_tagres_pf_v3",
                             "particle filter v3 baseline", False),
}
JOB = """#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# tagres (slow elliptic tag) / {description} / seed {seed}: disk tag with elliptic range observations as tagre, but the ghosts
# pursue at 0.5-0.7 of the player's top speed instead of 0.8-1.0 (experiment 25).
# 2420 updates = 25 random-action prediction + 2000 PPO + 395 interleaved prediction (64 environments, 64 particles).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/{config}.json"
export SB_POMDP_SHARD_INDEX={shard}
export SB_POMDP_CAMPAIGN_ID="{campaign}"
export SB_POMDP_SEGMENT_UPDATES=2420
exec bash jobs/genkai_production.sh
"""


def main() -> None:
    (ROOT / "jobs" / "tagres").mkdir(parents=True, exist_ok=True)
    for name, (source, label, campaign, stem, description, v2) in ARMS.items():
        cfg = json.loads((ROOT / "config" / f"{source}.json").read_text(encoding="utf-8"))
        cfg["experiment"]["label"] = label
        cfg["environment"]["tasks"]["disk_tag"].update(SLOW)
        if v2:
            cfg["model"].update(V2)
        (ROOT / "config" / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        for seed in range(10, 15):
            text = JOB.format(description=description, seed=seed, config=name, shard=seed - 9, campaign=campaign)
            (ROOT / "jobs" / "tagres" / f"{stem}_s{seed}.sh").write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(ARMS)} configs, {5 * len(ARMS)} jobs")


if __name__ == "__main__":
    main()
