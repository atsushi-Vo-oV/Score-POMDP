"""Generate the enlarged, representation-matched comparison (experiment 31): configs and job files.

Every arm carries 256 numbers from one step to the next and spends about the same acting compute (12 MFLOP per step):
  light TF-SVGD   16 particles x 16 latent dimensions (model.belief_latent_dim 16), score networks of hidden width 32
                  (model.energy_hidden [32]); v2 and v1 defaults, each plain and with the transition-network
                  initialisation of experiment 30 (transition_persistence_init 4, exact paired units: 32 >= 2 x 16)
  particle filter 16 particles x 16 dimensions (as before), widths scaled to the light's FLOPs
  GRU             hidden state 256, encoder widths scaled to the light's FLOPs
WIDTHS holds the measured values (scratchpad flops_rep256.py, the same measurement as the earlier FLOPs matches; the
target is the enlarged light v2 of the task).  Run from the repository root:
    python scripts/rep256_generate.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LF = chr(10)
LARGE = {"belief_latent_dim": 16, "energy_hidden": [32]}
TINIT = {"transition_persistence_init": 4.0}
SHORT = "333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction."
LONG = "1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction."
# task: (updates, updates line, description of the task for the job header)
TASKS = {
    "mirror2": (333, SHORT, "mirror-world Hike (w=4, outside -5)"),
    "tps": (213, "213 updates.", "Hike |s| observation + teleport 0.03"),
    "tagr": (1293, LONG, "disk tag with range observations"),
    "ml10": (333, SHORT, "bsuite memory length, N = 10"),
    "ml30": (333, SHORT, "bsuite memory length, N = 30"),
    "cpv": (333, SHORT, "cart-pole without velocities"),
    "pdv": (1293, LONG, "pendulum without velocity"),
}
# FLOPs-matched widths per task (enlarged light v2: 11.7-12.0 MFLOP per acting step, 43-48 thousand parameters;
# GRU 5.8-5.9 million parameters, particle filter 3.5-3.6 million).
WIDTHS = {
    "mirror2": {"gru_encoder_hidden": [3470, 1335], "pf_hidden": [1347, 2394]},  # light 11.93 MFLOP
    "tps": {"gru_encoder_hidden": [3467, 1334], "pf_hidden": [1345, 2392]},      # light 11.90 MFLOP
    "tagr": {"gru_encoder_hidden": [3475, 1336], "pf_hidden": [1350, 2400]},     # light 11.98 MFLOP
    "ml10": {"gru_encoder_hidden": [3469, 1334], "pf_hidden": [1346, 2393]},     # light 11.80 MFLOP
    "ml30": {"gru_encoder_hidden": [3469, 1334], "pf_hidden": [1346, 2393]},     # light 11.80 MFLOP
    "cpv": {"gru_encoder_hidden": [3467, 1334], "pf_hidden": [1345, 2392]},      # light 11.78 MFLOP
    "pdv": {"gru_encoder_hidden": [3431, 1320], "pf_hidden": [1328, 2362]},      # light 11.67 MFLOP
}
GRU_HIDDEN = 256
# arm: (config name, label, job stem, source config, model keys, method, description)
ARMS = {
    "light2": ("trysvgdlight2_{t}_tf-svgd-r256", "trysvgdlight2-{t}-tf-svgd-r256", "trysvgdlight2_{t}_svgd-r256",
               "trysvgdlight2_{t}_tf-svgd", LARGE, None, "light TF-SVGD, v2 default"),
    "light2-tinit": ("trysvgdlight2_{t}_tf-svgd-tinit-r256", "trysvgdlight2-{t}-tf-svgd-tinit-r256", "trysvgdlight2_{t}_svgd-tinit-r256",
                     "trysvgdlight2_{t}_tf-svgd", {**LARGE, **TINIT}, None, "light TF-SVGD, v2 default, transition network started as a pull to the previous cloud"),
    "light1": ("trysvgdlight_{t}_tf-svgd-r256", "trysvgdlight-{t}-tf-svgd-r256", "trysvgdlight_{t}_svgd-r256",
               "trysvgdlight_{t}_tf-svgd", LARGE, None, "light TF-SVGD, v1 default"),
    "light1-tinit": ("trysvgdlight_{t}_tf-svgd-tinit-r256", "trysvgdlight-{t}-tf-svgd-tinit-r256", "trysvgdlight_{t}_svgd-tinit-r256",
                     "trysvgdlight_{t}_tf-svgd", {**LARGE, **TINIT}, None, "light TF-SVGD, v1 default, transition network started as a pull to the previous cloud"),
    "gru": ("trysvgd_{t}_gru-r256", "trysvgd-{t}-gru-r256", "trysvgd_{t}_gru-r256",
            "trysvgd_{t}_pf_v3", {}, "gru", "GRU baseline, hidden state 256, encoder FLOPs-matched to the enlarged light"),
    "pf": ("trysvgd_{t}_pf_v3-r256", "trysvgd-{t}-pf-v3-r256", "trysvgd_{t}_pf_v3-r256",
           "trysvgd_{t}_pf_v3", {}, "particle_filter", "particle filter v3 baseline, 16 x 16, FLOPs-matched to the enlarged light"),
}
JOB = """#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# enlarged, representation-matched comparison (experiment 31) / {task_text} / seed {seed}.
# {description}.
# Every arm carries 256 numbers per step (16 particles x 16 dimensions, or a GRU state of 256) at about 12 MFLOP per acting step.
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
    """(arm, task, config name, source config, label, campaign, job stem, model keys, method, updates, header fields)."""
    for task, (updates, updates_line, task_text) in TASKS.items():
        for arm, (name, label, stem, source, keys, method, description) in ARMS.items():
            name, label, stem, source = (text.format(t=task) for text in (name, label, stem, source))
            yield (arm, task, name, source, label, f"{label}-v1", stem, keys, method, updates,
                   {"task_text": task_text, "description": description, "updates_line": updates_line})


def main() -> None:
    (ROOT / "jobs" / "rep256").mkdir(parents=True, exist_ok=True)
    count = 0
    for arm, task, name, source, label, campaign, stem, keys, method, updates, fields in assets():
        cfg = json.loads((ROOT / "config" / f"{source}.json").read_text(encoding="utf-8"))
        cfg["experiment"]["label"] = label
        cfg["model"].update(copy.deepcopy(keys))
        if method is not None:
            assert cfg["model"]["num_particles"] == 16 and cfg["comparison"]["pf_particle_dim"] == 16
            cfg["comparison"]["methods"] = [method]
            cfg["comparison"]["gru_encoder_hidden"] = list(WIDTHS[task]["gru_encoder_hidden"])
            cfg["comparison"]["gru_hidden_dim"] = GRU_HIDDEN
            cfg["comparison"]["pf_hidden"] = list(WIDTHS[task]["pf_hidden"])
            if method == "gru":
                cfg["comparison"].pop("pf_observation_function", None)
                cfg["comparison"].pop("pf_observation_hidden", None)
        (ROOT / "config" / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + LF, encoding="utf-8", newline=LF)
        for seed in range(10, 15):
            text = JOB.format(config=name, shard=seed - 9, campaign=campaign, updates=updates, seed=seed, **fields)
            (ROOT / "jobs" / "rep256" / f"{stem}_s{seed}.sh").write_text(text, encoding="utf-8", newline=LF)
        count += 1
    print(f"{count} configs, {5 * count} jobs")


if __name__ == "__main__":
    main()
