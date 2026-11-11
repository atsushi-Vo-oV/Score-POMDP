from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from sb_pomdp.config import load_config
from sb_pomdp.envs import make_env
from sb_pomdp.initialization import initialize_model, load_bc_actor, load_initial_weights
from sb_pomdp.policies import DiffusionPolicy
from sb_pomdp.prediction_schedule import PredictionSchedule
from sb_pomdp.train import build_optimizer, model_for_environment, seed_everything, train_single
from test_self_imitation import artifacts, assert_tree_equal, checkpoint, tiny_config


def make_model(config):
    return model_for_environment(make_env("disk_tag", config.environment_for_task("disk_tag"), seed=3), config["model"], torch.device("cpu"))


def policy(dim=32):
    return DiffusionPolicy(dim, [-1.0, -1.0], [1.0, 1.0], [96, 96], num_steps=5, beta_start=0.01, beta_end=0.2, min_std=0.05)


@pytest.mark.parametrize("seed", [10, 11])
def test_shipped_bc_actor_already_folded_exactly_reproduces_followup_inference(seed):
    path = Path(f"assets/bc_actors/followup_bc_disk_tag_score_transformer_{seed}_L5_std0.05.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected, actual = policy(), policy()
    expected.load_state_dict(payload["state_dict"])
    load_bc_actor(actual, path)
    c = torch.randn(40, 32)
    for deterministic in (True, False):
        ref = expected.sample(c, deterministic=deterministic, generator=torch.Generator().manual_seed(9))
        result = actual.sample(c, deterministic=deterministic, generator=torch.Generator().manual_seed(9))
        for a, b in zip(ref, result):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_unfolded_bc_statistics_are_folded_once(tmp_path):
    expected, actual = policy(4), policy(4)
    xm, xs = torch.randn(4), torch.rand(4) + 0.5
    path = tmp_path / "bc.pt"
    torch.save(dict(state_dict=expected.state_dict(), meta=dict(normalization_folded=False, steps=5, min_std=0.05), xm=xm, xs=xs), path)
    load_bc_actor(actual, path)
    c = torch.randn(20, 4)
    # The exporter uses w/xs and b - w@(xm/xs); compare both epsilon and actions.
    inputs = torch.randn(20, 22)
    normalized = inputs.clone()
    normalized[:, :4] = (inputs[:, :4] - xm) / xs
    torch.testing.assert_close(actual.denoiser(inputs), expected.denoiser(normalized), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.sample(c, deterministic=True)[0], expected.sample((c-xm)/xs, deterministic=True)[0], rtol=2e-5, atol=2e-6)
    load_bc_actor(actual, path)
    torch.testing.assert_close(actual.sample(c, deterministic=True)[0], expected.sample((c-xm)/xs, deterministic=True)[0], rtol=2e-5, atol=2e-6)


def test_rejected_bc_state_does_not_partially_change_actor(tmp_path):
    actor = policy(4)
    original = copy.deepcopy(actor.state_dict())
    malformed = copy.deepcopy(original)
    malformed["denoiser.0.weight"].fill_(10)
    malformed["denoiser.0.bias"] = torch.zeros(999)
    path = tmp_path / "bad.pt"
    torch.save(dict(state_dict=malformed, meta={}, xm=torch.zeros(4), xs=torch.ones(4)), path)
    with pytest.raises(ValueError, match="incompatible"):
        load_bc_actor(actor, path)
    assert_tree_equal(actor.state_dict(), original)


def test_weight_initialization_ignores_optimizer_progress_and_rng(tmp_path):
    config = tiny_config(tmp_path, enabled=False)
    source, target = make_model(config), make_model(config)
    path = tmp_path / "source.pt"
    payload = {"model": source.state_dict(), "config": config.to_dict(),
               "optimizer": "not an optimizer", "update": 90000,
               "trainer_state": "not a trainer", "torch_rng_state": torch.tensor([99])}
    payload["config"]["ppo"]["gamma"] = 0.8
    payload["config"]["model"]["prediction_interval_updates"] = 999
    torch.save(payload, path)
    rng = torch.get_rng_state().clone()
    load_initial_weights(target, path, config["model"])
    assert_tree_equal(source.state_dict(), target.state_dict())
    assert torch.equal(torch.get_rng_state(), rng)
    assert not build_optimizer(target, config["ppo"]).state
    payload["config"]["model"]["num_heads"] = 2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="structure"):
        load_initial_weights(target, path, config["model"])


def test_freezing_excludes_optimizer_and_rejects_typo(tmp_path):
    config = tiny_config(tmp_path, enabled=False)
    model = make_model(config)
    initialize_model(model, {**config["model"], "freeze_modules": ["belief", "encoder"]}, resume=False)
    optimizer = build_optimizer(model, config["ppo"])
    included = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, p in model.named_parameters():
        assert (id(p) in included) == p.requires_grad
        if name.startswith(("belief", "encoder")):
            assert not p.requires_grad
    with pytest.raises(ValueError, match="matches no parameters"):
        initialize_model(model, {"freeze_modules": ["beleif"]}, resume=False)


def test_warmup_adds_updates_and_counts_only_ppo():
    schedule = PredictionSchedule.from_config({"prediction_pretrain_steps": 8, "prediction_interval_updates": 2, "prediction_phase_updates": 1}, rollout_batch_size=4, ppo_updates=5, critic_warmup_updates=3)
    assert [schedule.phase(i) for i in range(1, 13)] == ["pretrain"]*2 + ["critic"]*3 + ["ppo", "ppo", "predict", "ppo", "ppo", "predict", "ppo"]
    assert schedule.total_updates == 12 and schedule.ppo_updates_done(5) == 0
    assert schedule.ppo_updates_done(12) == 5


def test_frozen_weights_and_actor_survive_warmup_then_actor_trains(tmp_path):
    config = tiny_config(tmp_path, enabled=False, total=2).with_overrides({
        "model.freeze_modules": ["belief", "encoder"], "ppo.critic_warmup_updates": 1,
    })
    seed_everything(3, True)
    initial = copy.deepcopy(make_model(config).state_dict())
    output = artifacts(tmp_path / "run")
    train_single(config, task="disk_tag", seed=3, artifacts=output, segment_updates=1, verbose=False)
    warm = checkpoint(output)
    assert warm["update"] == 1 and warm["global_step"] == 4
    for name, value in initial.items():
        if not name.startswith("value_head"):
            assert torch.equal(value, warm["model"][name]), name
    assert any(not torch.equal(v, warm["model"][n]) for n, v in initial.items() if n.startswith("value_head"))
    train_single(config, task="disk_tag", seed=3, artifacts=output, segment_updates=1,
                 resume_checkpoint=output.checkpoint_path(tag="latest"), verbose=False)
    trained = checkpoint(output)
    for name, value in initial.items():
        if name.startswith(("belief", "encoder")):
            assert torch.equal(value, trained["model"][name]), name
    assert any(not torch.equal(v, trained["model"][n]) for n, v in initial.items() if n.startswith("diffusion_policy.denoiser"))


def test_reset_value_preserves_fresh_constructor_and_resume_never_reopens_asset(tmp_path):
    config = tiny_config(tmp_path, enabled=False)
    source, target = make_model(config), make_model(config)
    original_value = copy.deepcopy(target.value_head.state_dict())
    path = tmp_path / "source.pt"
    torch.save({"model": source.state_dict(), "config": config.to_dict()}, path)
    cfg = {**config["model"], "init_from_checkpoint": str(path), "reset_value_head": True}
    initialize_model(target, cfg, resume=False)
    assert_tree_equal(target.value_head.state_dict(), original_value)
    assert_tree_equal(target.diffusion_policy.state_dict(), source.diffusion_policy.state_dict())
    path.unlink()
    before = copy.deepcopy(target.state_dict())
    initialize_model(target, {**cfg, "actor_init_path": "missing.pt"}, resume=True)
    assert_tree_equal(before, target.state_dict())


def test_fresh_campaign_starts_at_one_even_from_old_update(tmp_path):
    config = tiny_config(tmp_path, enabled=False)
    source = artifacts(tmp_path / "source")
    train_single(config, task="disk_tag", seed=3, artifacts=source, segment_updates=1, verbose=False)
    path = source.checkpoint_path(tag="latest")
    initialized = config.with_overrides({"model.init_from_checkpoint": str(path)})
    output = artifacts(tmp_path / "new")
    train_single(initialized, task="disk_tag", seed=9, artifacts=output, segment_updates=1, verbose=False)
    new = checkpoint(output)
    assert new["update"] == 1 and new["global_step"] == 4 and new["seed"] == 9
    assert len(new["trainer_state"]["metric_rows"]) == 1
