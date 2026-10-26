"""Interleaved next-observation prediction phases: schedule, exploration actions and the shared loss."""

from __future__ import annotations

import pytest
import torch

from sb_pomdp.config import ConfigError, load_config, validate_config
from sb_pomdp.envs import make_env
from sb_pomdp.networks import ConditionMixturePredictor
from sb_pomdp.ppo import condition_prediction_loss
from sb_pomdp.prediction_schedule import PredictionSchedule, random_actions


def test_schedule_interleaves_pretrain_ppo_and_prediction_blocks() -> None:
    schedule = PredictionSchedule(pretrain_updates=2, interval_updates=3, phase_updates=1, ppo_updates=7)
    phases = [schedule.phase(u) for u in range(1, schedule.total_updates + 1)]
    assert phases == ["pretrain"] * 2 + ["ppo"] * 3 + ["predict"] + ["ppo"] * 3 + ["predict"] + ["ppo"]
    assert schedule.total_updates == 2 + 7 + 2 and schedule.ppo_updates_done(schedule.total_updates) == 7
    with pytest.raises(ValueError):
        schedule.phase(schedule.total_updates + 1)


def test_schedule_from_config_rounds_pretrain_steps_and_is_off_by_default() -> None:
    model = load_config("config/core4_tps.json").to_dict()["model"]
    off = PredictionSchedule.from_config(model, rollout_batch_size=2048, ppo_updates=100)
    assert not off.enabled and off.total_updates == 100 and off.phase(1) == "ppo"
    on = PredictionSchedule.from_config(
        dict(model, prediction_pretrain_steps=200_000, prediction_interval_updates=25, prediction_phase_updates=5),
        rollout_batch_size=2048,
        ppo_updates=100,
    )
    assert on.pretrain_updates == 98 and on.interleaved_phases == 3 and on.total_updates == 98 + 100 + 15
    no_tail = [on.phase(u) for u in range(on.total_updates - 24, on.total_updates + 1)]
    assert no_tail == ["ppo"] * 25  # no prediction phase after the last PPO block


def test_random_actions_respect_the_action_space() -> None:
    torch.manual_seed(0)
    tag = make_env("disk_tag", {"horizon": 10, "observation_noise_std": 0.0})
    continuous = random_actions(tag.action_spec, 500, device=torch.device("cpu"))
    assert continuous.shape == (500, 2) and continuous.min() >= -1.0 and continuous.max() <= 1.0
    assert continuous.std() > 0.5  # uniform on [-1, 1] has std 0.577
    hell = make_env("heaven_hell", {"horizon": 10, "observation_noise_std": 0.0})
    discrete = random_actions(hell.action_spec, 400, device=torch.device("cpu"))
    assert discrete.dtype == torch.long and set(discrete.tolist()) == {0, 1, 2, 3}


def test_condition_prediction_loss_targets_the_next_observation_and_masks_episode_ends() -> None:
    torch.manual_seed(0)
    predictor = ConditionMixturePredictor(4, 2, 3, [8], components=2)
    steps, group = 5, 2
    condition = torch.randn(steps * group, 4)
    observations = torch.randn(steps, group, 3)
    previous_actions = torch.randn(steps, group, 2)
    dones = torch.zeros(steps, group, dtype=torch.bool)
    dones[2, 1] = True
    loss = condition_prediction_loss(predictor, condition, observations=observations,
                                     previous_actions=previous_actions, dones=dones)
    rows = [(t, g) for t in range(steps - 1) for g in range(group) if (t, g) != (2, 1)]
    expected = -torch.stack([
        predictor.log_likelihood(condition.reshape(steps, group, 4)[t, g][None], previous_actions[t + 1, g][None],
                                 observations[t + 1, g][None])[0]
        for t, g in rows
    ]).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_config_requires_interval_and_phase_length_together() -> None:
    resolved = load_config("config/core4_tps.json").to_dict()
    resolved["model"]["prediction_interval_updates"] = 25
    with pytest.raises(ConfigError):
        validate_config(resolved)
    resolved["model"]["prediction_phase_updates"] = 5
    validate_config(resolved)


def test_the_comparison_driver_counts_prediction_updates_toward_completion() -> None:
    from sb_pomdp.compare import _configured_updates

    plain = load_config("config/core4_tps.json")
    assert _configured_updates(plain) == 100
    scheduled = load_config("config/trysvgd_mirror2_tf-ula.json")
    assert _configured_updates(scheduled) == 98 + 200 + 35  # a resumed shard is complete only at 333
