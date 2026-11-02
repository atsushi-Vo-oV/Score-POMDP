"""Compare batched denoiser evaluation with the current sequential implementation.

Run with PYTHONPATH=src python agent/check_diffusion_batching.py. Numerical
results can differ by device and PyTorch version; this script does not assert.
"""

import copy

import torch
from torch.distributions import Independent, Normal

from sb_pomdp.policies import DiffusionPolicy


def candidate(policy, condition, chain):
    batch = condition.shape[0]
    steps = torch.arange(policy.num_steps - 1, -1, -1, device=condition.device)
    embedding = policy.time_embedding(steps.repeat_interleave(batch), policy.num_steps).to(
        condition.dtype
    )
    inputs = torch.cat(
        (
            condition.unsqueeze(0).expand(policy.num_steps, -1, -1),
            chain[:, :-1].transpose(0, 1),
            embedding.reshape(policy.num_steps, batch, -1),
        ),
        dim=-1,
    )
    predicted = policy.denoiser(inputs.reshape(policy.num_steps * batch, -1)).reshape(
        policy.num_steps, batch, policy.action_dim
    )
    log_probs = []
    for i, step in enumerate(reversed(range(policy.num_steps))):
        alpha = policy.alphas[step].to(condition.dtype)
        alpha_bar = policy.alpha_bars[step].to(condition.dtype)
        beta = policy.betas[step].to(condition.dtype)
        mean = chain[:, i] - beta * predicted[i] / torch.sqrt(1.0 - alpha_bar)
        mean = mean / torch.sqrt(alpha)
        variance = policy.posterior_variance[step].to(condition.dtype)
        std = torch.sqrt(variance.clamp_min(policy.min_std**2))
        distribution = Independent(Normal(mean, torch.ones_like(mean) * std), 1)
        log_probs.append(distribution.log_prob(chain[:, i + 1]))
    return torch.stack(log_probs, dim=1)


def report(name, pairs, atol):
    max_abs = max((a - b).abs().max().item() for a, b in pairs)
    max_rel = max(
        ((a - b).abs() / b.abs().clamp_min(torch.finfo(b.dtype).tiny)).max().item()
        for a, b in pairs
    )
    passed = all(torch.allclose(a, b, atol=atol, rtol=atol) for a, b in pairs)
    print(
        f"{name}: max_abs={max_abs:.9g}, max_rel={max_rel:.9g}, atol=rtol={atol:g}, pass={passed}"
    )


def main():
    torch.manual_seed(17)
    policy = DiffusionPolicy(
        4, [-1, -2], [1, 2], [32, 32], num_steps=5, beta_start=0.01, beta_end=0.1, min_std=0.05
    )
    old = copy.deepcopy(policy)
    condition = torch.randn(7, 4, requires_grad=True)
    with torch.no_grad():
        _, chain, _ = policy.sample(condition, deterministic=False)
    chain = chain.detach().requires_grad_()
    old_condition = condition.detach().clone().requires_grad_()
    old_chain = chain.detach().clone().requires_grad_()
    actual = candidate(policy, condition, chain)
    expected, _ = old.log_prob_chain(old_condition, old_chain)
    report("log_prob", [(actual, expected)], 1e-6)
    actual.mean().backward()
    expected.mean().backward()
    pairs = [(condition.grad, old_condition.grad), (chain.grad, old_chain.grad)]
    pairs.extend((p.grad, q.grad) for p, q in zip(policy.parameters(), old.parameters()))
    report("gradients", pairs, 1e-5)


if __name__ == "__main__":
    main()
