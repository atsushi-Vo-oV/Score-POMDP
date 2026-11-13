"""Twelve production arms: five leave-one-out and six PPO comparisons."""
from copy import deepcopy

FULL = dict(population_size=16, init_std=0.01, update_rule="qd", actor_ppo_fraction=1.0,
            ou_std=0.3, ou_tau=20.0, rnd_beta=0.01, inference_actor="optimizer",
            optimizer_agent={"enabled": True, "adopt": "bc"})
ARMS = {
    "full": FULL,
    "no_f": {**FULL, "optimizer_agent": {"enabled": False}, "inference_actor": "mean"},
    "no_a2": {**FULL, "ou_std": 0.0},
    "no_c": {**FULL, "actor_ppo_fraction": 0.0},
    "no_d": {**FULL, "rnd_beta": 0.0},
    "no_qd": {**FULL, "update_rule": "cem"},
    "cem": dict(population_size=16, init_std=0.01, update_rule="cem", actor_ppo_fraction=0.0),
    "es": dict(population_size=16, init_std=0.01, update_rule="es", actor_ppo_fraction=0.0),
    "pbt": dict(population_size=16, init_std=0.01, update_rule="pbt", actor_ppo_fraction=1.0),
    "a2": dict(ou_std=0.3, ou_tau=20.0),
    "d": dict(rnd_beta=0.01),
    "ppo": {},
}
PRODUCTION_ARMS = tuple(ARMS)


def arm_config(name):
    return deepcopy(ARMS[name])
