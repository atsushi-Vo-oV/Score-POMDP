"""Fail closed for configurations outside the light score-transformer path."""
import hashlib
from pathlib import Path

from ..config import ConfigError


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reject_score_curl(model):
    if model.get("score_curl_penalty", 0.0) > 0 or model.get("score_curl_monitor", False):
        raise ConfigError("seedbatch does not support score_curl_penalty or score_curl_monitor")


def validate(config):
    model, ppo = config["model"], config["ppo"]
    reject_score_curl(model)
    required = {
        "score_parameterization": {"direct"},
        "particle_sampler": {"svgd", "ula"},
        "score_implementation": {"autograd", "analytic"},
        "belief_compile": {"off"}, "belief_cuda_graphs": {False},
        "belief_activation_checkpoint": {False},
        "encoder_kind": {"transformer"}, "continuous_policy_kind": {"diffusion"},
        "prediction_head": {"condition"}, "belief_training": {"joint"},
        "policy_observation_input": {False}, "critic_kind": {"state"},
        "initial_belief": {"branch"}, "langevin_proposal": {"noise", "affine", "uniform"},
        "langevin_initial_noise": {"normal"},
        "langevin_observation_anchor_steps": {"initial"},
        "belief_replay": {"sequence"}, "prediction_replay": {"same"},
        "energy_network_kind": {"mlp"}, "head_network_kind": {"mlp"},
        "trunk_network_kind": {"mlp"}, "policy_head_kind": {"mlp"},
        "reward_prediction_coef": {0.0}, "observation_prediction_coef": {0.0},
    }
    for name, allowed in required.items():
        if model[name] not in allowed:
            raise ConfigError(f"seedbatch does not support model.{name}={model[name]!r}; allowed: {allowed}")
    if ppo["algorithm"] != "ppo":
        raise ConfigError("seedbatch requires ppo.algorithm=ppo")
    if ppo["p3o_demo_slots"]:
        raise ConfigError("seedbatch does not support ppo.p3o_demo_slots")
    steps = int(ppo["rollout_steps"])
    mini = int(ppo["minibatch_size"])
    micro = int(ppo["sequence_microbatch_size"])
    if mini < steps or mini % steps or micro < steps or micro > mini or micro % steps or mini % micro:
        raise ConfigError("seedbatch requires sequence minibatch/microbatch sizes divisible by rollout_steps, and minibatch_size divisible by sequence_microbatch_size")
