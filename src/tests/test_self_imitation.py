from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest
import torch

from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.self_imitation import (
    SIL_METRIC_FIELDS, SelfImitationBuffer, imitation_weights, positive_sample,
    replay_log_prob, sil_config, update_self_imitation,
)
from sb_pomdp.train import METRIC_FIELDS, build_optimizer, model_for_environment, train_single
from sb_pomdp.train_baselines import train_baseline_single
from sb_pomdp.envs import make_env


def tiny_config(tmp_path, *, enabled=True, total=2, horizon=3):
    return load_config("config/trysvgdlight_tagr_tf-svgd.json").with_overrides({
        "experiment.output_dir": str(tmp_path / "unused"), "experiment.device": "cpu",
        "experiment.deterministic": True, "experiment.eval_interval_updates": 999,
        "experiment.eval_episodes": 1, "experiment.checkpoint_interval_updates": 999,
        "environment.tasks.disk_tag.horizon": horizon,
        "environment.tasks.disk_tag.step_reward": 10.0,
        "model.num_particles": 2, "model.langevin_steps": 1,
        "model.d_model": 4, "model.num_heads": 1, "model.transformer_ff_dim": 8,
        "model.energy_hidden": [4], "model.policy_hidden": [8],
        "model.prediction_pretrain_steps": 0, "model.prediction_interval_updates": 0,
        "model.prediction_phase_updates": 0,
        "ppo.total_steps": total * 4, "ppo.num_envs": 1, "ppo.rollout_steps": 4,
        "ppo.minibatch_size": 4, "ppo.sequence_microbatch_size": 4, "ppo.epochs": 1,
        "ppo.self_imitation.enabled": enabled, "ppo.self_imitation.buffer_size": 20,
        "ppo.self_imitation.batch_size": 4, "ppo.self_imitation.updates_per_ppo": 2,
        "comparison.gru_encoder_hidden": [4], "comparison.gru_hidden_dim": 4,
    })


def artifacts(path):
    path.mkdir()
    return SeedArtifacts(path=path, task="disk_tag", seed=3)


def checkpoint(artifact):
    return torch.load(artifact.checkpoint_path(tag="latest", create_parent=False), map_location="cpu", weights_only=False)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("exponent", [0.0, 0.6, 4.0])
def test_priority_never_selects_nonpositive_or_threshold_advantage(exponent):
    cfg = sil_config({})
    cfg.update(batch_size=2000, priority_exponent=exponent, min_advantage=0.1)
    advantage = torch.tensor([-1.0, 0.0, 0.1, 0.2, 2.0])
    selected = positive_sample(advantage, cfg, generator=torch.Generator().manual_seed(5))
    assert set(selected.tolist()) <= {3, 4}
    if exponent > 0:
        assert (selected == 4).sum() > (selected == 3).sum()
    assert positive_sample(torch.zeros(8), cfg).numel() == 0


def test_exp_clips_before_exponentiation_and_does_not_backpropagate_weights():
    cfg = sil_config({})
    cfg.update(weighting="exp", max_weight=20.0)
    advantages = torch.tensor([-1000.0, 0.0, 1.0, 1e30], requires_grad=True)
    weights = imitation_weights(advantages, cfg, scale=torch.tensor(0.0))
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights, torch.tensor([0.0, 0.0, 20.0, 20.0]))
    assert not weights.requires_grad


def test_episode_returns_cross_update_boundary_ring_and_bootstrap():
    replay = SelfImitationBuffer(3, 1, 0.5)
    c = torch.tensor([[3.0]], requires_grad=True)
    for reward, ended in [(1.0, False), (2.0, False)]:
        replay.observe(c, c, torch.zeros(1), torch.tensor([reward]), torch.tensor([ended]))
    assert replay.size == 0
    saved = replay.state_dict()
    resumed = SelfImitationBuffer(3, 1, 0.5)
    resumed.load_state_dict(saved)
    resumed.observe(c, c, torch.zeros(1), torch.tensor([4.0]), torch.tensor([True]), bootstrap=torch.tensor([8.0]))
    torch.testing.assert_close(resumed.data["returns"], torch.tensor([4.0, 6.0, 8.0]))
    assert resumed.pending == [[]]
    assert not resumed.data["condition"].requires_grad
    resumed.observe(c, c, torch.zeros(1), torch.tensor([9.0]), torch.tensor([True]))
    assert resumed.size == 3 and resumed.position == 1
    assert resumed.data["returns"][0] == 9.0


@pytest.mark.parametrize("weighting", ["relu", "exp"])
def test_sil_only_updates_heads_and_reuses_sampled_diffusion_log_prob(tmp_path, weighting):
    config = tiny_config(tmp_path).to_dict()
    config["ppo"]["self_imitation"]["weighting"] = weighting
    model = model_for_environment(make_env("disk_tag", config["environment"]["tasks"]["disk_tag"], seed=3), config["model"], torch.device("cpu"))
    c = torch.randn(6, 4, requires_grad=True)
    with torch.no_grad():
        actions, chains, sample_lp = model.diffusion_policy.sample(c, deterministic=False)
    batch = {"condition": c.detach(), "actions": actions, "chains": chains}
    torch.testing.assert_close(replay_log_prob(model, batch), sample_lp.sum(-1), rtol=0, atol=0)
    replay = SelfImitationBuffer(20, 6, 0.99)
    replay.observe(c, c, actions, torch.full((6,), 20.0), torch.ones(6, dtype=torch.bool), chains=chains)
    before = copy.deepcopy(model.state_dict())
    metrics = update_self_imitation(model, build_optimizer(model, config["ppo"]), replay, config["ppo"])
    assert metrics["sil_updates"] == 2 and metrics["sil_selected_transitions"] == 8
    assert metrics["sil_mean_weight"] > 0
    assert c.grad is None
    for name, value in model.state_dict().items():
        if name.startswith(("belief.", "encoder.")):
            assert torch.equal(value, before[name]), name
    for prefix in ("diffusion_policy.", "value_head."):
        assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items() if name.startswith(prefix))


@pytest.mark.parametrize("method", ["score_transformer", "gru"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_one_update_smoke_and_opt_in_csv(tmp_path, method, enabled):
    config = tiny_config(tmp_path, enabled=enabled)
    output = artifacts(tmp_path / "run")
    train = train_single if method == "score_transformer" else train_baseline_single
    train(config, task="disk_tag", seed=3, artifacts=output, method=method, result_dir="run", segment_updates=1, verbose=False)
    with output.metrics_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == list(METRIC_FIELDS + (SIL_METRIC_FIELDS if enabled else ()))
    state = checkpoint(output)
    assert ("self_imitation" in state) == enabled
    if enabled:
        assert float(rows[0]["sil_updates"]) == 2
        assert float(rows[0]["sil_selected_transitions"]) == 8
        assert state["self_imitation"]["size"] == 3
        assert len(state["self_imitation"]["pending"][0]) == 1


@pytest.mark.parametrize("method", ["score_transformer", "gru"])
@pytest.mark.parametrize("weighting", ["relu", "exp"])
def test_sil_segment_resume_is_exact_with_pending_episodes(tmp_path, method, weighting):
    config = tiny_config(tmp_path).with_overrides({"ppo.self_imitation.weighting": weighting})
    uninterrupted, split = artifacts(tmp_path / "full"), artifacts(tmp_path / "split")
    train = train_single if method == "score_transformer" else train_baseline_single
    args = dict(task="disk_tag", seed=3, method=method, result_dir="run", evaluate_at_end=False, verbose=False)
    train(config, artifacts=uninterrupted, **args)
    train(config, artifacts=split, segment_updates=1, **args)
    train(config, artifacts=split, resume_checkpoint=split.checkpoint_path(tag="latest"), **args)
    expected, actual = checkpoint(uninterrupted), checkpoint(split)
    for key in ("model", "optimizer", "self_imitation", "torch_rng_state"):
        assert_tree_equal(expected[key], actual[key])
    for left, right in zip(expected["trainer_state"]["metric_rows"], actual["trainer_state"]["metric_rows"]):
        for field in SIL_METRIC_FIELDS:
            assert left[field] == right[field]


@pytest.mark.parametrize("method", ["score_transformer", "gru"])
def test_sil_skips_pretraining_actions_and_prediction_updates(tmp_path, method):
    config = tiny_config(tmp_path).with_overrides({
        "model.prediction_pretrain_steps": 4,
        "model.prediction_interval_updates": 1, "model.prediction_phase_updates": 1,
    })
    output = artifacts(tmp_path / "phases")
    train = train_single if method == "score_transformer" else train_baseline_single
    train(config, task="disk_tag", seed=3, method=method, result_dir="run", artifacts=output, verbose=False)
    state = checkpoint(output)
    rows = state["trainer_state"]["metric_rows"]
    assert len(rows) == 4  # pretrain, PPO, prediction, PPO
    assert rows[0]["sil_buffer_size"] == 0
    assert rows[0]["sil_updates"] == rows[2]["sil_updates"] == 0
    assert rows[2]["sil_buffer_size"] == 8  # all eight post-pretrain transitions ended
    assert rows[1]["sil_updates"] == rows[3]["sil_updates"] == 2


def test_disabled_sil_does_not_consume_rng_or_change_optimizer(tmp_path):
    cfg = tiny_config(tmp_path, enabled=False)
    model = model_for_environment(make_env("disk_tag", cfg.environment_for_task("disk_tag"), seed=3), cfg["model"], torch.device("cpu"))
    optimizer = build_optimizer(model, cfg["ppo"])
    before, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    replay = SelfImitationBuffer(20, 1, 0.99)
    replay.observe(torch.ones(1, 4), torch.ones(1, 4), torch.zeros(1, 2), torch.ones(1), torch.ones(1, dtype=torch.bool), chains=torch.ones(1, 6, 2))
    result = update_self_imitation(model, optimizer, replay, cfg["ppo"])
    assert result["sil_updates"] == 0
    assert not optimizer.state
    assert torch.equal(rng, torch.get_rng_state())
    assert_tree_equal(before, model.state_dict())


def test_corrupt_replay_checkpoint_fails_before_loading():
    replay = SelfImitationBuffer(4, 1, 0.99)
    replay.observe(torch.ones(1, 4), torch.ones(1, 4), torch.zeros(1), torch.ones(1), torch.ones(1, dtype=torch.bool))
    state = replay.state_dict()
    state["data"]["returns"][0] = float("nan")
    with pytest.raises(ValueError, match="tensor"):
        SelfImitationBuffer(4, 1, 0.99).load_state_dict(state)


@pytest.mark.parametrize("override", [
    {"ppo.self_imitation.weighting": "bad"}, {"ppo.self_imitation.temperature": 0},
    {"ppo.self_imitation.batch_size": 0}, {"ppo.self_imitation.enabled": 1},
    {"ppo.self_imitation.priority_exponent": -1}, {"ppo.self_imitation.max_weight": float("inf")},
    {"ppo.critic_warmup_updates": -1}, {"model.freeze_modules": [""]},
    {"model.reset_value_head": "yes"},
    {"ppo.self_imitation.enabled": True, "model.critic_kind": "action"},
    {"ppo.self_imitation.enabled": True, "ppo.algorithm": "p3o_wml"},
])
def test_invalid_opt_in_config_is_rejected(override):
    with pytest.raises(ConfigError):
        load_config("config/local.json").with_overrides(override)
