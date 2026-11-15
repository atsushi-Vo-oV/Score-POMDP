"""Generate the every-step observation anchor arms (experiment 27): configs and job files.

Four arms per task, on both defaults of the light TF-SVGD:
  anchor-all      the observation anchor moves the chain start of every step: x_start = mu(o_t) + sigma(o_t) * noise
  anchor-all-l1   the same with one transport stage instead of 12 (does the anchor alone carry the learning?)
On v1 the only change is langevin_observation_anchor_steps ("initial" -> "all"); on v2 the anchor is switched back on
as well (v2 has none).  Run from the repository root:
    python scripts/anchor_all_generate.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LF = chr(10)
# task: (updates, description of the task for the job header)
TASKS = {
    "mirror2": (333, "mirror-world Hike (w=4, outside -5)"),
    "tps": (213, "Hike |s| observation + teleport 0.03"),
    "carflag": (333, "Car-Flag (green flag at +1 or -1, side revealed only within 0.2 of the priest at 0.5)"),
    "tagrs": (1293, "disk tag with range observations, ghosts at 0.5-0.7 of the player's top speed"),
}
# family prefix: (config prefix, campaign prefix, name of the default, extra model keys)
FAMILIES = {
    "v1": ("trysvgdlight", "trysvgdlight", "v1 default", {"langevin_observation_anchor_steps": "all"}),
    "v2": ("trysvgdlight2", "trysvgdlight2", "v2 default", {"langevin_observation_anchor": True, "langevin_observation_anchor_steps": "all"}),
}
# arm suffix: (job-stem suffix, extra model keys, description)
ARMS = {
    "anchor-all": ("anchorall", {}, "observation anchor at every step, 12 transport stages"),
    "anchor-all-l1": ("anchorall-l1", {"langevin_steps": 1}, "observation anchor at every step, 1 transport stage"),
}
JOB = """#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# every-step observation anchor / {task_text} / light TF-SVGD, {default} / seed {seed}.
# {arm_text}: the chain of every step starts at mu(o_t) + sigma(o_t) * noise (experiment 27).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/{config}.json"
export SB_POMDP_SHARD_INDEX={shard}
export SB_POMDP_CAMPAIGN_ID="{campaign}"
export SB_POMDP_SEGMENT_UPDATES={updates}
exec bash jobs/genkai_production.sh
"""


def assets():
    """(config name, source config, label, campaign, job stem, model keys, updates, header fields) of every arm."""
    for task, (updates, task_text) in TASKS.items():
        for prefix, campaign_prefix, default, family_keys in FAMILIES.values():
            for arm, (stem, arm_keys, arm_text) in ARMS.items():
                yield (f"{prefix}_{task}_tf-svgd-{arm}", f"{prefix}_{task}_tf-svgd", f"{campaign_prefix}-{task}-tf-svgd-{arm}",
                       f"{campaign_prefix}-{task}-tf-svgd-{arm}-v1", f"{prefix}_{task}_svgd-{stem}", {**family_keys, **arm_keys}, updates,
                       {"task_text": task_text, "default": default, "arm_text": arm_text})


def main() -> None:
    (ROOT / "jobs" / "anchor_all").mkdir(parents=True, exist_ok=True)
    count = 0
    for name, source, label, campaign, stem, keys, updates, fields in assets():
        cfg = json.loads((ROOT / "config" / f"{source}.json").read_text(encoding="utf-8"))
        cfg["experiment"]["label"] = label
        cfg["model"].update(keys)
        (ROOT / "config" / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + LF, encoding="utf-8", newline=LF)
        for seed in range(10, 15):
            text = JOB.format(config=name, shard=seed - 9, campaign=campaign, updates=updates, seed=seed, **fields)
            (ROOT / "jobs" / "anchor_all" / f"{stem}_s{seed}.sh").write_text(text, encoding="utf-8", newline=LF)
        count += 1
    print(f"{count} configs, {5 * count} jobs")


if __name__ == "__main__":
    main()
