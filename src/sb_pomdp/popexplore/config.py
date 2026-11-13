"""Closed population schema layered over the unmodified experiment schema."""
from copy import deepcopy
import json
import math
from pathlib import Path

from ..config import ConfigError, ExperimentConfig, _duplicate_checked_object, _reject_nonstandard_number


OPTIMIZER_DEFAULTS = {
    "enabled": False, "adopt": "bc", "window": 5, "margin": 0.1,
    "confidence": 0.95, "bc_steps": 10, "bc_learning_rate": 0.0003,
    "pull_to_optimizer": 0.0,
}

DEFAULTS = {
    "population_size": 1, "init_std": 0.0, "update_rule": "none",
    "selection_interval": 5, "fitness_episodes": 4, "elite_fraction": 0.25,
    "fitness_from_rollout": False,
    "cem_rate": 0.5, "sigma": 0.01, "sigma_min": 0.001,
    "es_learning_rate": 0.01, "pbt_fraction": 0.25, "mutation_std": 0.005,
    "qd_cell_size": 0.25, "qd_capacity": 64, "qd_parent_fraction": 0.25,
    "actor_ppo_fraction": 1.0, "actor_ppo_scale": 1.0,
    "ou_std": 0.0, "ou_tau": 20.0,
    "rnd_beta": 0.0, "rnd_input": "condition", "rnd_hidden": 64,
    "rnd_features": 32, "rnd_learning_rate": 0.0003, "rnd_clip": 5.0,
    "inference_actor": "mean", "freeze_belief": False, "bc_actor_path": "",
    "group": "custom", "arm": "ppo", "checkpoint_seed": -1,
    "optimizer_agent": OPTIMIZER_DEFAULTS,
}


class PopConfig:
    def __init__(self, data):
        raw = deepcopy(data)
        supplied = raw.pop("popexplore", {})
        if not isinstance(supplied, dict) or set(supplied) - DEFAULTS.keys():
            raise ConfigError("popexplore: unknown keys or non-object configuration")
        self.pop = {**deepcopy(DEFAULTS), **supplied}
        p = self.pop
        agent = supplied.get("optimizer_agent", {})
        if not isinstance(agent, dict) or set(agent) - OPTIMIZER_DEFAULTS.keys():
            raise ConfigError("popexplore.optimizer_agent: unknown keys or non-object")
        p["optimizer_agent"] = {**OPTIMIZER_DEFAULTS, **agent}
        f = p["optimizer_agent"]
        for key, default in OPTIMIZER_DEFAULTS.items():
            value = f[key]
            valid = (isinstance(value, bool) if isinstance(default, bool) else
                     isinstance(value, int) and not isinstance(value, bool) if isinstance(default, int) else
                     isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
                     if isinstance(default, float) else isinstance(value, str))
            if not valid:
                raise ConfigError(f"popexplore.optimizer_agent.{key}: invalid type/value")
        if (f["adopt"] not in {"copy", "bc", "none"} or f["window"] < 2 or f["margin"] < 0
                or not .5 < f["confidence"] < 1 or f["bc_steps"] < 1 or f["bc_learning_rate"] <= 0
                or not 0 <= f["pull_to_optimizer"] <= 1):
            raise ConfigError("popexplore.optimizer_agent: invalid setting")
        if f["enabled"] and "inference_actor" not in supplied:
            p["inference_actor"] = "optimizer"
        for key, default in DEFAULTS.items():
            if key == "optimizer_agent":
                continue
            value = p[key]
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            elif isinstance(default, int):
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(default, float):
                valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            else:
                valid = isinstance(value, str)
            if not valid:
                raise ConfigError(f"popexplore.{key}: invalid type/value")
        for key in ("population_size", "selection_interval", "fitness_episodes", "qd_capacity", "rnd_hidden", "rnd_features"):
            if p[key] < 1:
                raise ConfigError(f"popexplore.{key} must be positive")
        for key in ("init_std", "mutation_std", "ou_std", "rnd_beta", "actor_ppo_scale"):
            if p[key] < 0:
                raise ConfigError(f"popexplore.{key} must be nonnegative")
        for key in ("sigma", "sigma_min", "ou_tau", "qd_cell_size", "rnd_learning_rate", "rnd_clip", "es_learning_rate"):
            if p[key] <= 0:
                raise ConfigError(f"popexplore.{key} must be positive")
        for key in ("elite_fraction", "cem_rate", "pbt_fraction"):
            if not 0 < p[key] <= 1:
                raise ConfigError(f"popexplore.{key} must be in (0,1]")
        for key in ("actor_ppo_fraction", "qd_parent_fraction"):
            if not 0 <= p[key] <= 1:
                raise ConfigError(f"popexplore.{key} must be in [0,1]")
        for key, choices in {"update_rule": {"none", "cem", "es", "pbt", "qd"},
                             "rnd_input": {"condition", "observation"},
                             "inference_actor": {"mean", "best", "optimizer"}}.items():
            if p[key] not in choices:
                raise ConfigError(f"popexplore.{key} must be one of {sorted(choices)}")
        if p["sigma_min"] > p["sigma"]:
            raise ConfigError("popexplore.sigma_min cannot exceed sigma")
        if f["enabled"] and p["population_size"] < 2:
            raise ConfigError("popexplore.optimizer_agent requires population_size >= 2")
        if p["inference_actor"] == "optimizer" and not f["enabled"]:
            raise ConfigError("inference_actor=optimizer requires optimizer_agent.enabled")
        if p["update_rule"] != "none" and p["population_size"] < 2:
            raise ConfigError("popexplore selection requires at least two actors")
        if p["update_rule"] == "pbt" and (p["actor_ppo_fraction"] == 0 or p["actor_ppo_scale"] == 0):
            raise ConfigError("popexplore pbt requires actor PPO learning")
        self._requested_freeze = list(raw.get("model", {}).get("freeze_modules", []))
        self._requested_belief_training = raw.get("model", {}).get("belief_training", "joint")
        if p["freeze_belief"]:
            m = raw.setdefault("model", {})
            prefixes = ({"observation_encoder.", "gru.", "condition_projection."}
                        if raw.get("comparison", {}).get("methods") == ["gru"]
                        else {"belief.", "encoder."})
            if raw.get("comparison", {}).get("gru_hidden_dim") == m.get("d_model"):
                prefixes.discard("condition_projection.")  # identity has no parameters
            if m.get("prediction_pretrain_steps", 0) or (m.get("prediction_interval_updates", 0) and m.get("prediction_phase_updates", 0)):
                prefixes.add("prediction_predictor.")
            m["freeze_modules"] = sorted(set(m.get("freeze_modules", [])) | prefixes)
            m["belief_training"] = "joint"
        self.base = ExperimentConfig(raw)
        self.validate()

    def validate(self):
        r = self.base.to_dict()
        m, p = r["model"], r["ppo"]
        methods = r["comparison"]["methods"]
        if methods not in (["score_transformer"], ["gru"]):
            raise ConfigError("popexplore requires comparison.methods=[score_transformer] or [gru]")
        self.method = methods[0]
        gru = self.method == "gru"
        required = {"continuous_policy_kind": "diffusion", "prediction_head": "condition",
                    "critic_kind": "state"}
        if not gru:
            required.update(score_parameterization="direct", particle_sampler="svgd",
                            encoder_kind="transformer", belief_compile="off")
        for key, value in required.items():
            if m.get(key) != value:
                raise ConfigError(f"popexplore requires model.{key}={value}")
        # score_implementation=analytic and belief_cuda_graphs are execution-only switches inside
        # the shared belief module (not vmapped here), so they are allowed as in the main trainer.
        for key in (("policy_observation_input", "score_curl_monitor") if gru else
                    ("belief_activation_checkpoint", "policy_observation_input", "score_curl_monitor")):
            if m.get(key):
                raise ConfigError(f"popexplore does not support model.{key}")
        if not gru and m.get("dropout", 0) != 0:
            raise ConfigError("popexplore requires dropout=0")
        if m.get("value_encoder") != "shared":
            raise ConfigError("popexplore requires model.value_encoder=shared")
        for key in ("reward_prediction_coef", "observation_prediction_coef", "score_curl_penalty"):
            if m.get(key, 0):
                raise ConfigError(f"popexplore does not support model.{key}")
        if p.get("algorithm") != "ppo" or p.get("self_imitation", {}).get("enabled", False):
            raise ConfigError("popexplore requires PPO without self-imitation")
        if gru and (p.get("weight_decay", 0) or p.get("critic_warmup_updates", 0)):
            raise ConfigError("popexplore GRU does not support weight_decay or critic_warmup_updates")
        if p["num_envs"] % self.pop["population_size"]:
            raise ConfigError("popexplore: num_envs must be a multiple of population_size")
        actor_prefix = "heads.diffusion_policy" if gru else "diffusion_policy"
        if self.pop["population_size"] > 1 and any(
                prefix.startswith(actor_prefix) or actor_prefix.startswith(prefix)
                for prefix in m.get("freeze_modules", [])):
            raise ConfigError("popexplore: freeze population actors with actor_ppo_fraction=0")

    def to_dict(self):
        return {**self.base.to_dict(), "popexplore": deepcopy(self.pop)}

    def environment_for_task(self, task):
        return self.base.environment_for_task(task)

    def with_overrides(self, overrides):
        raw = self.to_dict()
        raw["model"]["freeze_modules"] = self._requested_freeze.copy()
        raw["model"]["belief_training"] = self._requested_belief_training
        base = {k: v for k, v in overrides.items() if not k.startswith("popexplore.")}
        raw.update(ExperimentConfig({k:v for k,v in raw.items() if k != "popexplore"}).with_overrides(base).to_dict())
        for key, value in overrides.items():
            if key.startswith("popexplore."):
                target = raw["popexplore"]
                parts = key.split(".")[1:]
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value
        if (overrides.get("popexplore.optimizer_agent.enabled") is True
                and "popexplore.inference_actor" not in overrides and not self.pop["optimizer_agent"]["enabled"]):
            raw["popexplore"]["inference_actor"] = "optimizer"
        return PopConfig(raw)


def load_config(path, overrides=None):
    with Path(path).open(encoding="utf-8-sig") as f:
        raw = json.load(f, object_pairs_hook=_duplicate_checked_object, parse_constant=_reject_nonstandard_number)
    # Overrides must be applied BEFORE validation (e.g. .sb_overrides acceleration).
    for key, value in (overrides or {}).items():
        target = raw
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return PopConfig(raw)
