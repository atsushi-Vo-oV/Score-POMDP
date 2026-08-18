from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.baselines import GRUActorCritic
from sb_pomdp.config import load_config
from sb_pomdp.train_baselines import (
    BaselineRollout,
    RecurrentReplayPrefix,
    replay_recurrent_sequence,
    train_baseline_single,
)

ROOT = Path(__file__).resolve().parents[2]
TASKS = (
    "masked_cartpole",
    "masked_pendulum",
    "masked_mountain_car_continuous",
    "light_dark",
)
METHODS = ("observation_mlp", "gru", "oracle_state")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("task", TASKS)
def test_one_update_baseline_writes_complete_artifacts(
    tmp_path: Path,
    method: str,
    task: str,
) -> None:
    config = load_config(ROOT / "config" / "local.json")
    # Keep this deliberately short so atomic temporary filenames stay below
    # the legacy Windows MAX_PATH limit used by CI.
    artifact_path = tmp_path / "s"
    artifact_path.mkdir()
    artifacts = SeedArtifacts(
        path=artifact_path,
        task=task,
        seed=0,
        method=method,
    )
    relative_result = f"{method}/{task}/seed_000"

    summary = train_baseline_single(
        config,
        method=method,  # type: ignore[arg-type]
        task=task,
        seed=0,
        artifacts=artifacts,
        max_updates=1,
        verbose=False,
        result_dir=relative_result,
    )

    assert summary["status"] == "completed"
    assert summary["global_steps"] == 16
    assert summary["result_dir"] == relative_result
    assert summary["parameter_count"] > 0
    assert artifacts.resolved_config_path.is_file()
    assert artifacts.metadata_path.is_file()
    assert artifacts.metrics_path.is_file()
    assert artifacts.arrays_path.is_file()
    assert (artifacts.path / "episodes.csv").is_file()
    assert artifacts.checkpoint_path(tag="best", create_parent=False).is_file()
    assert artifacts.checkpoint_path(tag="latest", create_parent=False).is_file()
    assert artifacts.checkpoint_path(step=16, create_parent=False).is_file()

    # A saved per-seed config remains strict and independently reloadable.
    load_config(artifacts.resolved_config_path)
    checkpoint = torch.load(
        artifacts.checkpoint_path(tag="latest", create_parent=False),
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["method"] == method
    assert checkpoint["task"] == task
    assert checkpoint["global_step"] == 16
    assert all(torch.isfinite(tensor).all() for tensor in checkpoint["model"].values())

    with artifacts.arrays_path.open("rb") as handle, np.load(handle, allow_pickle=False) as arrays:
        assert arrays["update"].tolist() == [1]
        assert arrays["global_step"].tolist() == [16]
        assert np.isfinite(arrays["policy_loss"]).all()
        assert np.isfinite(arrays["value_loss"]).all()
        for field in (
            "initial_energy_gradient_norm",
            "observation_energy_gradient_norm",
            "transition_energy_gradient_norm",
            "initial_score_l2_mean",
            "recursive_score_l2_mean",
            "initial_particle_l2_mean",
            "recursive_particle_l2_mean",
            "initial_belief_particle_count",
            "recursive_belief_particle_count",
            "score_nonfinite_count",
            "particle_nonfinite_count",
        ):
            assert np.isnan(arrays[field]).all()

    with (artifacts.path / "evaluation.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    deterministic_mode = (
        "greedy"
        if task == "masked_cartpole"
        else "mean_chain"
        if method == "gru"
        else "mean_action"
    )
    assert [row["mode"] for row in rows] == [deterministic_mode, "stochastic"]
    assert {int(row["evaluation_seed"]) for row in rows} == {1_000_000}
    for mode in (deterministic_mode, "stochastic"):
        trajectory_path = artifacts.path / f"evaluation_{mode}_trajectories_000001.npz"
        with (
            trajectory_path.open("rb") as handle,
            np.load(handle, allow_pickle=False) as trajectories,
        ):
            assert "episode_000_observations" in trajectories
            assert "episode_000_actions" in trajectories
            assert "episode_000_rewards" in trajectories
            assert "episode_000_states" in trajectories


def _recurrent_replay_rollout(prefix: RecurrentReplayPrefix) -> BaselineRollout:
    return BaselineRollout(
        inputs=torch.tensor([[[0.3]], [[-0.2]]]),
        actions=torch.zeros(2, 1, dtype=torch.long),
        old_log_prob=torch.zeros(2, 1),
        old_values=torch.zeros(2, 1),
        rewards=torch.zeros(2, 1),
        dones=torch.zeros(2, 1, dtype=torch.bool),
        advantages=torch.ones(2, 1),
        returns=torch.zeros(2, 1),
        episode_starts=torch.zeros(2, 1, dtype=torch.bool),
        pre_tanh_actions=None,
        diffusion_chains=None,
        hidden_states=None,
        recurrent_prefixes=(prefix,),
    )


def test_recurrent_full_replay_crosses_rollout_boundary_but_tbptt_one_does_not() -> None:
    torch.manual_seed(19)
    model = GRUActorCritic(
        1,
        action_kind="discrete",
        hidden_dims=[4],
        recurrent_hidden_dim=3,
        discrete_actions=2,
    )
    prefix_values = torch.tensor([[0.1], [0.2]])
    starts = torch.tensor([True, False])

    full_prefix_inputs = prefix_values.clone().requires_grad_()
    full_features = replay_recurrent_sequence(
        model,
        _recurrent_replay_rollout(
            RecurrentReplayPrefix(full_prefix_inputs, starts),
        ),
        torch.tensor([0]),
        temporal_gradient_mode="full",
    )
    (full_prefix_gradient,) = torch.autograd.grad(full_features[-1].sum(), full_prefix_inputs)
    assert torch.count_nonzero(full_prefix_gradient) > 0

    one_step_prefix_inputs = prefix_values.clone().requires_grad_()
    one_step_features = replay_recurrent_sequence(
        model,
        _recurrent_replay_rollout(
            RecurrentReplayPrefix(one_step_prefix_inputs, starts),
        ),
        torch.tensor([0]),
        temporal_gradient_mode="tbptt_1",
    )
    (one_step_prefix_gradient,) = torch.autograd.grad(
        one_step_features[-1].sum(),
        one_step_prefix_inputs,
        allow_unused=True,
    )
    assert one_step_prefix_gradient is None
    torch.testing.assert_close(one_step_features, full_features)


def test_baseline_trainer_rejects_method_artifact_mismatch(tmp_path: Path) -> None:
    config = load_config(ROOT / "config" / "local.json")
    artifact_path = tmp_path / "s"
    artifact_path.mkdir()
    artifacts = SeedArtifacts(
        path=artifact_path,
        task="masked_cartpole",
        seed=0,
        method="gru",
    )

    with pytest.raises(ValueError, match="does not match"):
        train_baseline_single(
            config,
            method="observation_mlp",
            task="masked_cartpole",
            seed=0,
            artifacts=artifacts,
            max_updates=1,
            verbose=False,
            result_dir="observation_mlp/masked_cartpole/seed_000",
        )
