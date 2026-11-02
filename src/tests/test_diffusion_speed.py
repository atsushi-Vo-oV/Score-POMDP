import copy

import pytest
import torch

from sb_pomdp.policies import DiffusionPolicy


def reference(policy, condition, chain):
    distributions = [
        policy.transition_distribution(condition, chain[:, i], step)
        for i, step in enumerate(reversed(range(policy.num_steps)))
    ]
    return (
        torch.stack([dist.log_prob(chain[:, i + 1]) for i, dist in enumerate(distributions)], 1),
        torch.stack([dist.entropy() for dist in distributions], 1),
    )


@pytest.mark.parametrize("min_std", [0.001, 0.05, 0.2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sampled", [False, True])
def test_batched_chain_matches_reference_and_gradients(min_std, dtype, sampled):
    torch.manual_seed(17)
    policy = DiffusionPolicy(
        4, [-1, -2], [1, 2], [32, 32], num_steps=5, beta_start=0.01, beta_end=0.1, min_std=min_std
    ).to(dtype)
    old = copy.deepcopy(policy)
    condition = torch.randn(7, 4, dtype=dtype, requires_grad=True)
    if sampled:
        with torch.no_grad():
            _, chain, _ = policy.sample(condition, deterministic=False)
        chain = chain.detach().requires_grad_()
    else:
        chain = torch.randn(7, 6, 2, dtype=dtype, requires_grad=True)
    old_condition = condition.detach().clone().requires_grad_()
    old_chain = chain.detach().clone().requires_grad_()
    rng = torch.random.get_rng_state().clone()
    actual, entropy = policy.log_prob_chain(condition, chain)
    expected, old_entropy = reference(old, old_condition, old_chain)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(entropy, old_entropy, atol=0, rtol=0)
    torch.testing.assert_close(policy.chain_entropy(condition), old_entropy, atol=0, rtol=0)
    actual.mean().backward()
    expected.mean().backward()
    torch.testing.assert_close(condition.grad, old_condition.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(chain.grad, old_chain.grad, atol=1e-5, rtol=1e-5)
    for p, q in zip(policy.parameters(), old.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-5)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_sampling_entropy_preserves_rng_and_matches_reference():
    torch.manual_seed(23)
    policy = DiffusionPolicy(
        4, [-1, -2], [1, 2], [16], num_steps=5, beta_start=0.01, beta_end=0.1, min_std=0.1
    )
    condition = torch.randn(3, 4)
    generator = torch.Generator().manual_seed(99)
    _, chain, _ = policy.sample(condition, deterministic=False, generator=generator)
    state = generator.get_state().clone()
    torch.testing.assert_close(
        policy.chain_entropy(condition), reference(policy, condition, chain)[1], atol=0, rtol=0
    )
    assert torch.equal(state, generator.get_state())


@pytest.mark.parametrize("kind", ["score", "baseline"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_sample_policy_matches_second_pass(kind, deterministic):
    from sb_pomdp.baselines import _PolicyValueHeads
    from sb_pomdp.policies import ScoreBeliefActorCritic

    torch.manual_seed(31)
    if kind == "score":
        model = ScoreBeliefActorCritic(
            observation_dim=2,
            state_dim=3,
            action_kind="continuous",
            action_feature_dim=2,
            discrete_actions=None,
            action_low=[-2, -1],
            action_high=[1, 3],
            model_config={
                "num_particles": 3,
                "langevin_steps": 1,
                "langevin_step_size": 0.02,
                "particle_clip": 0.0,
                "score_clip": 0.0,
                "energy_hidden": [5],
                "d_model": 8,
                "num_heads": 2,
                "num_transformer_layers": 1,
                "transformer_ff_dim": 12,
                "dropout": 0.0,
                "policy_hidden": [9],
                "diffusion_steps": 3,
                "diffusion_beta_start": 0.01,
                "diffusion_beta_end": 0.1,
                "diffusion_min_std": 0.05,
                "encoder_kind": "transformer",
                "continuous_policy_kind": "diffusion",
                "belief_gradient_mode": "full",
            },
        )
        features = torch.randn(6, 8)
    else:
        model = _PolicyValueHeads(
            8,
            action_kind="continuous",
            discrete_actions=None,
            action_low=[-2, -1],
            action_high=[1, 3],
            hidden_dims=[9],
            continuous_policy_kind="diffusion",
            diffusion_steps=3,
            diffusion_beta_start=0.01,
            diffusion_beta_end=0.1,
            diffusion_min_std=0.05,
        )
        features = torch.randn(2, 3, 8)
    policy = model.diffusion_policy
    flat = features.reshape(-1, 8)
    old_generator = torch.Generator().manual_seed(77)
    action, chain, log_prob = policy.sample(
        flat, deterministic=deterministic, generator=old_generator
    )
    entropy = reference(policy, flat, chain)[1]
    count = []
    handle = policy.denoiser.register_forward_hook(lambda *args: count.append(1))
    generator = torch.Generator().manual_seed(77)
    sample = model.sample_policy(features, deterministic=deterministic, generator=generator)
    handle.remove()
    assert len(count) == policy.num_steps
    torch.testing.assert_close(sample.action.reshape_as(action), action, rtol=0, atol=0)
    torch.testing.assert_close(sample.diffusion_chain.reshape_as(chain), chain, rtol=0, atol=0)
    torch.testing.assert_close(sample.log_prob.reshape_as(log_prob), log_prob, rtol=0, atol=0)
    torch.testing.assert_close(sample.entropy.reshape_as(entropy), entropy, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), old_generator.get_state())
