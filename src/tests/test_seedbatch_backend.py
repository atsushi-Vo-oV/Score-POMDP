"""Ragged episode replay, inactive parameters and non-finite seed isolation."""
import copy

import pytest
import torch

from sb_pomdp.buffer import BeliefReplayPrefix, RolloutBatch
from sb_pomdp.envs import make_env
from sb_pomdp.ppo import update_ppo
from sb_pomdp.seedbatch.backend import SeedBatchBackend
from sb_pomdp.seedbatch.protocol import SeedStream
from sb_pomdp.train import build_optimizer, model_for_environment, seed_everything
from test_seedbatch_training import tiny_config, one_thread


def synthetic_rollout(model, *, initial_only, nonfinite):
    t, e, k, d = 3, 2, model.belief.num_particles, model.belief.state_dim
    o, a = model.belief.observation_dim, model.belief.action_feature_dim
    l, diffusion = 2, model.diffusion_policy.num_steps
    prefixes = []
    for length in ([0, 0] if initial_only else [2, 3]):
        flags = torch.zeros(length, dtype=torch.bool)
        if length:
            flags[0] = True
        prefixes.append(BeliefReplayPrefix(
            observations=torch.randn(length, o), previous_actions=torch.randn(length, a),
            initial=flags, ula_initial_noise=torch.randn(length, k, d),
            ula_step_noises=torch.randn(length, l, k, d),
        ))
    returns = torch.randn(t, e)
    if nonfinite:
        returns[:, 0] = float("nan")
    return RolloutBatch(
        observations=torch.randn(t, e, o), previous_particles=torch.zeros(t, e, k, d),
        previous_actions=torch.randn(t, e, a), initial=torch.full((t, e), initial_only),
        particles=torch.randn(t, e, k, d), ula_initial_noise=torch.randn(t, e, k, d),
        ula_step_noises=torch.randn(t, e, l, k, d), actions=torch.zeros(t, e, a),
        old_log_prob=torch.randn(t, e, diffusion), old_values=torch.zeros(t, e),
        rewards=torch.randn(t, e), dones=torch.zeros(t, e, dtype=torch.bool),
        advantages=torch.randn(t, e), returns=returns,
        diffusion_chains=torch.randn(t, e, diffusion + 1, a), pre_tanh_actions=None,
        belief_prefixes=tuple(prefixes),
    )


@pytest.mark.parametrize("nonfinite", [False, True])
def test_ragged_prefixes_and_seed_specific_gradient_masks(tmp_path, nonfinite, one_thread):
    config = tiny_config(tmp_path, **{
        "ppo.rollout_steps": 3, "ppo.minibatch_size": 3, "ppo.sequence_microbatch_size": 3,
        "ppo.epochs": 1, "ppo.weight_decay": 0.02, "ppo.total_steps": 6,
    })
    models, references, rollouts, streams, reference_streams = [], [], [], [], []
    for i, seed in enumerate([10, 11]):
        seed_everything(seed, True)
        env = make_env("mountain_hike", config.environment_for_task("mountain_hike"), seed=seed)
        model = model_for_environment(env, config["model"], torch.device("cpu"))
        models.append(model)
        references.append(copy.deepcopy(model))
        rollouts.append(synthetic_rollout(model, initial_only=i == 0, nonfinite=nonfinite and i == 0))
        streams.append(SeedStream())
        reference_streams.append(SeedStream())
    optimizers = [build_optimizer(model, config["ppo"]) for model in models]
    ref_optimizers = [build_optimizer(model, config["ppo"]) for model in references]
    backend = SeedBatchBackend(models, optimizers, config["model"], config["ppo"])
    metrics = backend.update(rollouts, streams, "ppo")
    assert backend.batched_calls == 2  # two sequence minibatches, not one call per seed
    for i in range(2):
        expected = reference_streams[i].run(lambda: update_ppo(
            references[i], ref_optimizers[i], rollouts[i], config["model"], config["ppo"]
        ))
        for name, value in models[i].state_dict().items():
            torch.testing.assert_close(value, references[i].state_dict()[name], rtol=3e-4, atol=3e-5)
        assert optimizers[i].state_dict()["state"].keys() == ref_optimizers[i].state_dict()["state"].keys()
        for key, value in optimizers[i].state_dict()["state"].items():
            assert torch.equal(value["step"], ref_optimizers[i].state_dict()["state"][key]["step"])
        for name in metrics[i].__dataclass_fields__:
            assert getattr(metrics[i], name) == pytest.approx(getattr(expected, name), rel=3e-4, abs=3e-4)
