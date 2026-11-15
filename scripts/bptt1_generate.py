"""Generate the one-step gradient arms (experiment 28): configs and job files.

Three arms per task, on both defaults of the light TF-SVGD:
  bptt1            one-step gradients in time order (model.belief_gradient_mode=tbptt_1), PPO and prediction phases alike
  stored           the same gradients from the stored incoming particles, all (time, environment) rows at once
                   (model.belief_replay=stored), PPO and prediction phases alike
  stored-predfull  stored replay for the PPO update, full backpropagation through time for the prediction phases
Run from the repository root:
    python scripts/bptt1_generate.py
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
FAMILIES = {"trysvgdlight2": "v2 default", "trysvgdlight": "v1 default"}
# arm: (extra model keys, description)
ARMS = {
    "bptt1": ({}, "one-step gradients in time order for the PPO update and the prediction phases"),
    "stored": ({"belief_replay": "stored"}, "one-step gradients from the stored incoming particles, all rows at once, for the PPO update and the prediction phases"),
    "stored-predfull": ({"belief_replay": "stored", "prediction_replay": "full"},
                        "stored one-step replay for the PPO update, full backpropagation through time for the prediction phases"),
}
JOB = """#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# one-step belief gradients / {task_text} / light TF-SVGD, {default} / seed {seed}.
# {arm_text} (experiment 28).
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
        for prefix, default in FAMILIES.items():
            for arm, (arm_keys, arm_text) in ARMS.items():
                yield (f"{prefix}_{task}_tf-svgd-{arm}", f"{prefix}_{task}_tf-svgd", f"{prefix}-{task}-tf-svgd-{arm}",
                       f"{prefix}-{task}-tf-svgd-{arm}-v1", f"{prefix}_{task}_svgd-{arm}", {"belief_gradient_mode": "tbptt_1", **arm_keys},
                       updates, {"task_text": task_text, "default": default, "arm_text": arm_text[0].upper() + arm_text[1:]})


def main() -> None:
    (ROOT / "jobs" / "bptt1").mkdir(parents=True, exist_ok=True)
    count = 0
    for name, source, label, campaign, stem, keys, updates, fields in assets():
        cfg = json.loads((ROOT / "config" / f"{source}.json").read_text(encoding="utf-8"))
        cfg["experiment"]["label"] = label
        cfg["model"].update(keys)
        cfg["comparison"]["belief_gradient_modes"] = ["tbptt_1"]
        (ROOT / "config" / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + LF, encoding="utf-8", newline=LF)
        for seed in range(10, 15):
            text = JOB.format(config=name, shard=seed - 9, campaign=campaign, updates=updates, seed=seed, **fields)
            (ROOT / "jobs" / "bptt1" / f"{stem}_s{seed}.sh").write_text(text, encoding="utf-8", newline=LF)
        count += 1
    print(f"{count} configs, {5 * count} jobs")


if __name__ == "__main__":
    main()
