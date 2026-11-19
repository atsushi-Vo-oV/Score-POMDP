"""Generate the transition-initialisation arms (experiment 30, "v2-initialized" / "v1-initialized"): configs and jobs.

One arm on each default of the light TF-SVGD.  The inference is unchanged; only the starting weights of the transition
network differ: model.transition_persistence_init = 4 makes the transition term start as 4 * (mean of the previous
cloud - x).  Disk tag has 6 latent dimensions and a hidden width of 8, so the exact two-units-per-coordinate form does
not fit; it uses model.transition_persistence_units = "single".

"v2-initialized-mix" / "v1-initialized-mix" (per-particle attention): model.transition_persistence_kernel = 4 also starts
the attention logit of a pair as -4 * sum_d z_d tanh(z_d / 2) (z = previous particle - x; -2 |z|^2 near 0), so the
transition term starts as a pull to the previous particles near the query instead of to the mean of the cloud (it is
not in general the score of a density; see persistence_initialise_).  It needs the paired units, so disk tag has no such arm at this
size (experiment 31 has it at the enlarged size).

"v2-initialized-t0": the uniform-attention start plus model.initial_score_projection_precision = 4, which makes the
initial network start as 4 * (B o - x) (a narrow first cloud around a linear image of the first observation).  v2 only
(the v1 default has its own observation anchor, a bound and a prior at t = 0); it needs paired units in the initial
network as well, so disk tag has no such arm at this size.  Run from the repository root:
    python scripts/tinit_generate.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LF = chr(10)
STRENGTH = 4.0
KERNEL = 4.0
PROJECTION = 4.0
# variant: (suffix of the names, extra model keys, sentence of the job header, families)
VARIANTS = {
    "uniform": ("tinit", {}, "uniform attention", ("v2", "v1")),
    "mixture": ("tinitmix", {"transition_persistence_kernel": KERNEL},
                "attention logit -%g * sum_d z_d tanh(z_d / 2), every query is pulled to the previous particles near it" % KERNEL,
                ("v2", "v1")),
    "narrow": ("tinitq", {"initial_score_projection_precision": PROJECTION},
               "uniform attention, and the initial network starts as %g * (B o - x)" % PROJECTION, ("v2",)),
}
SHORT = "333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction."
LONG = "1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction."
# task: (updates, updates line, description of the task for the job header, extra model keys)
TASKS = {
    "mirror2": (333, SHORT, "mirror-world Hike (w=4, outside -5)", {}),
    "tps": (213, "213 updates.", "Hike |s| observation + teleport 0.03", {}),
    "tagr": (1293, LONG, "disk tag with range observations (6 latent dimensions)", {"transition_persistence_units": "single"}),
    "ml10": (333, SHORT, "bsuite memory length, N = 10", {}),
    "ml30": (333, SHORT, "bsuite memory length, N = 30", {}),
    "cpv": (333, SHORT, "cart-pole without velocities", {}),
    "pdv": (1293, LONG, "pendulum without velocity", {}),
}
# family: (config / campaign prefix, name of the default)
FAMILIES = {"v2": ("trysvgdlight2", "v2 default"), "v1": ("trysvgdlight", "v1 default")}
JOB = """#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# transition initialisation (experiment 30) / {task_text} / light TF-SVGD, {default} / seed {seed}.
# The transition network starts as {strength:g} * (previous particle - x) per pair with {attention} ({units} units);
# nothing is added to the score and the weights stay trainable.
# {updates_line}
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
    for task, (updates, updates_line, task_text, task_keys) in TASKS.items():
        for suffix, variant_keys, attention, families in VARIANTS.values():
            if variant_keys and task_keys.get("transition_persistence_units", "pair") != "pair":
                continue  # the per-particle attention and the initial-network start need the paired units
            for prefix, default in (FAMILIES[family] for family in families):
                keys = {"transition_persistence_init": STRENGTH, **task_keys, **variant_keys}
                yield (f"{prefix}_{task}_tf-svgd-{suffix}", f"{prefix}_{task}_tf-svgd", f"{prefix}-{task}-tf-svgd-{suffix}",
                       f"{prefix}-{task}-tf-svgd-{suffix}-v1", f"{prefix}_{task}_svgd-{suffix}", keys, updates,
                       {"task_text": task_text, "default": default, "updates_line": updates_line, "strength": STRENGTH,
                        "units": keys.get("transition_persistence_units", "pair"), "attention": attention})


def main() -> None:
    (ROOT / "jobs" / "tinit").mkdir(parents=True, exist_ok=True)
    count = 0
    for name, source, label, campaign, stem, keys, updates, fields in assets():
        cfg = json.loads((ROOT / "config" / f"{source}.json").read_text(encoding="utf-8"))
        cfg["experiment"]["label"] = label
        cfg["model"].update(keys)
        (ROOT / "config" / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + LF, encoding="utf-8", newline=LF)
        for seed in range(10, 15):
            text = JOB.format(config=name, shard=seed - 9, campaign=campaign, updates=updates, seed=seed, **fields)
            (ROOT / "jobs" / "tinit" / f"{stem}_s{seed}.sh").write_text(text, encoding="utf-8", newline=LF)
        count += 1
    print(f"{count} configs, {5 * count} jobs")


if __name__ == "__main__":
    main()
