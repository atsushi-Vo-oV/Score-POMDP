"""Generate the memory tasks taken from the literature (experiment 29): configs and job files.

Question: on tasks where published recurrent policy-gradient agents use memory and a memoryless agent provably or
empirically fails, does the light score belief learn to use memory under the same PPO pipeline as the GRU?

Tasks (the environment constants follow the cited sources):
  cpv   cart-pole without velocities (SB3 CartPoleNoVel-v1 / POPGym StatelessCartPole), 500 steps, 2 actions
  pdv   pendulum swing-up without the angular velocity (POPGym StatelessPendulumEasy / SB3 PendulumNoVel-v1), 200 steps
  pdvn  pdv with Gaussian observation noise 0.1, clipped to [-1, 1] (POPGym NoisyStatelessPendulumEasy)
  ml10  bsuite memory length N = 10: a bit shown at the first step decides the reward of the 10th action
  ml30  bsuite memory length N = 30
Budget: cpv / ml10 / ml30 use the usual 333 updates (409,600 PPO steps; Pleines et al. train the masked cart-pole for
409,600 steps, SB3 for 100,000, bsuite runs 10,000 episodes = 100,000 / 300,000 steps); pdv / pdvn use the 1293-update
schedule of the tag tasks (2,048,000 PPO steps; SB3's 100,000-200,000 steps do not converge, POPGym uses 15,000,000
steps at a learning rate five times lower).
Arms: light TF-SVGD v2 default, v1 default, v1 with the chain started at the previous particles (prev-init), GRU,
particle filter v3, the memoryless observation MLP and, where the hidden state is more than the cue, the full-state MLP.
GRU and particle filter widths are FLOPs-matched to the light TF-SVGD (v1) of the task; WIDTHS holds the measured
values (scratchpad flops_memref.py; a task without an entry keeps the widths of the source config).
Run from the repository root:
    python scripts/memref_generate.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LF = chr(10)
V2 = {"langevin_observation_anchor": False, "langevin_proposal": "uniform", "langevin_initial_noise": "uniform",
      "direct_score_bound": None, "initial_score_prior": 0.0}
SHORT = (409_600, 333, "333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.")
LONG = (2_048_000, 1293, "1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction.")
# task: (environment name, task block, schedule, description, whether the full-state control is run)
TASKS = {
    "cpv": ("masked_cartpole", {"horizon": 500, "observation_noise_std": 0.0}, SHORT,
            "cart-pole without velocities (positions only; SB3 CartPoleNoVel-v1 / POPGym StatelessCartPole)", True),
    "pdv": ("masked_pendulum", {"horizon": 200, "observation_noise_std": 0.0, "observation_clip": False}, LONG,
            "pendulum swing-up without the angular velocity (POPGym StatelessPendulumEasy / SB3 PendulumNoVel-v1)", True),
    "pdvn": ("masked_pendulum", {"horizon": 200, "observation_noise_std": 0.1, "observation_clip": True}, LONG,
             "pendulum swing-up without the angular velocity, observation noise 0.1 clipped to [-1, 1] (POPGym NoisyStatelessPendulumEasy)",
             True),
    "ml10": ("memory_chain", {"horizon": 10, "observation_noise_std": 0.0}, SHORT,
             "bsuite memory length N = 10 (a bit shown at the first step decides the reward of the last action)", False),
    "ml30": ("memory_chain", {"horizon": 30, "observation_noise_std": 0.0}, SHORT,
             "bsuite memory length N = 30 (a bit shown at the first step decides the reward of the last action)", False),
}
# FLOPs-matched widths per task: {"gru_encoder_hidden": [..], "gru_hidden_dim": .., "pf_hidden": [..]}
_PENDULUM = {"gru_encoder_hidden": [716, 275], "gru_hidden_dim": 132, "pf_hidden": [146, 259]}  # light 0.894 MFLOP per step
_CHAIN = {"gru_encoder_hidden": [743, 286], "gru_hidden_dim": 137, "pf_hidden": [158, 281]}  # light 0.838 MFLOP per step
WIDTHS: dict[str, dict] = {
    "cpv": {"gru_encoder_hidden": [895, 344], "gru_hidden_dim": 165, "pf_hidden": [224, 399]},  # light 1.185 MFLOP per step
    "pdv": _PENDULUM, "pdvn": _PENDULUM, "ml10": _CHAIN, "ml30": _CHAIN,
}
# The feed-forward controls have no prediction phases (the trainer supports them for the recurrent baselines only).
NO_PREDICTION = {"prediction_pretrain_steps": 0, "prediction_interval_updates": 0, "prediction_phase_updates": 0}
PF_V3 = {"pf_observation_function": "mlp", "pf_observation_hidden": 32}
# arm: (config name, label, campaign, job stem, description, source config, model overrides, method, comparison overrides)
ARMS = {
    "light2": ("trysvgdlight2_{t}_tf-svgd", "trysvgdlight2-{t}-tf-svgd", "trysvgdlight2-{t}-tf-svgd-v1", "trysvgdlight2_{t}_svgd",
               "light TF-SVGD, v2 default", "trysvgdlight_mirror2_tf-svgd", V2, None, {}),
    "light1": ("trysvgdlight_{t}_tf-svgd", "trysvgdlight-{t}-tf-svgd", "trysvgdlight-{t}-tf-svgd-v1", "trysvgdlight_{t}_svgd",
               "light TF-SVGD, v1 default", "trysvgdlight_mirror2_tf-svgd", {}, None, {}),
    "prev": ("trysvgdlight_{t}_tf-svgd-prev-init", "trysvgdlight-{t}-tf-svgd", "trysvgdlight-{t}-tf-svgd-prev-init-v1",
             "trysvgdlight_{t}_svgd-previnit", "light TF-SVGD, v1 default, chain started at the previous particles",
             "trysvgdlight_mirror2_tf-svgd", {"langevin_warm_start": True}, None, {}),
    "gru": ("trysvgd_{t}_gru", "trysvgd-{t}-gru", "trysvgd-{t}-gru-v1", "trysvgd_{t}_gru",
            "GRU baseline (FLOPs-matched to the light TF-SVGD)", "trysvgd_mirror2_gru", {}, "gru", {}),
    "pf": ("trysvgd_{t}_pf_v3", "trysvgd-{t}-pf-v3", "trysvgd-{t}-pf-v3-v1", "trysvgd_{t}_pf_v3",
           "particle filter v3 baseline (FLOPs-matched to the light TF-SVGD)", "trysvgd_mirror2_pf_v3", {}, "particle_filter", PF_V3),
    "mlp": ("trysvgd_{t}_mlp", "trysvgd-{t}-mlp", "trysvgd-{t}-mlp-v1", "trysvgd_{t}_mlp",
            "memoryless control: MLP on the current observation alone", "trysvgd_mirror2_gru", NO_PREDICTION, "observation_mlp", {}),
    "oracle": ("trysvgd_{t}_oracle", "trysvgd-{t}-oracle", "trysvgd-{t}-oracle-v1", "trysvgd_{t}_oracle",
               "full-state control: MLP on the true simulator state", "trysvgd_mirror2_gru", NO_PREDICTION, "oracle_state", {}),
}
JOB = """#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# memref (experiment 29) / {task}: {task_description} / seed {seed}.
# {description}
# {updates_line}
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/{config}.json"
export SB_POMDP_SHARD_INDEX={shard}
export SB_POMDP_CAMPAIGN_ID="{campaign}"
export SB_POMDP_SEGMENT_UPDATES={updates}
exec bash jobs/genkai_production.sh
"""


def main() -> None:
    (ROOT / "jobs" / "memref").mkdir(parents=True, exist_ok=True)
    configs = jobs = 0
    for task, (environment, block, (total_steps, updates, updates_line), task_description, oracle) in TASKS.items():
        for arm, (name, label, campaign, stem, description, source, model, method, comparison) in ARMS.items():
            if arm == "oracle" and not oracle:
                continue
            name, label, campaign, stem = (text.format(t=task) for text in (name, label, campaign, stem))
            cfg = json.loads((ROOT / "config" / f"{source}.json").read_text(encoding="utf-8"))
            assert cfg["ppo"]["total_steps"] == 409_600 and cfg["model"]["prediction_pretrain_steps"] == 200_000
            cfg["experiment"]["label"] = label
            cfg["experiment"]["tasks"] = [environment]
            cfg["environment"]["tasks"] = {environment: copy.deepcopy(block)}
            cfg["ppo"]["total_steps"] = total_steps
            cfg["model"].update(model)
            cfg["comparison"].update(copy.deepcopy(WIDTHS.get(task, {})))
            cfg["comparison"].update(comparison)
            if method is not None:
                cfg["comparison"]["methods"] = [method]
            (ROOT / "config" / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + LF, encoding="utf-8", newline=LF)
            configs += 1
            arm_updates, arm_line = updates, updates_line
            if model is NO_PREDICTION:
                arm_updates = total_steps // (cfg["ppo"]["num_envs"] * cfg["ppo"]["rollout_steps"])
                arm_line = f"{arm_updates} PPO updates (no prediction phases)."
            for seed in range(10, 15):
                text = JOB.format(task=task, task_description=task_description, seed=seed, description=description,
                                  updates_line=arm_line, config=name, shard=seed - 9, campaign=campaign, updates=arm_updates)
                (ROOT / "jobs" / "memref" / f"{stem}_s{seed}.sh").write_text(text, encoding="utf-8", newline=LF)
                jobs += 1
    print(f"{len(TASKS)} tasks, {configs} configs, {jobs} jobs")


if __name__ == "__main__":
    main()
