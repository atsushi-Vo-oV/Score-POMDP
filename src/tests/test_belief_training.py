"""model.belief_training = prediction_only.

The belief and its readout learn from the next-observation loss alone (on every rollout), and PPO
reaches the actor / critic heads through a stop-gradient on their inputs.  ``joint`` (the default)
is the historical behaviour: PPO and the prediction phases both shape the belief.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from test_p3o_wml import OBS_DIM, PARTICLES, _rollout_for

import sb_pomdp.ppo as ppo_module
from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.policies import ScoreBeliefActorCritic
from sb_pomdp.ppo import update_ppo
from sb_pomdp.run_experiments import run_all
from sb_pomdp.train import build_optimizer
from sb_pomdp.train_baselines import train_baseline_single

ROOT = Path(__file__).resolve().parents[2]
PREDICTION = {"model.prediction_interval_updates": 25, "model.prediction_phase_updates": 5}


def _config(belief_training: str, **overrides):
    return load_config(ROOT / "config" / "local.json").with_overrides(
        {"model.belief_training": belief_training, **PREDICTION, **overrides}
    )


def _model(belief_training: str):
    config = _config(belief_training, **{"model.num_particles": PARTICLES, "model.langevin_steps": 2})
    model_config = dict(config["model"])
    ppo_config = dict(config["ppo"])
    ppo_config.update({"epochs": 1, "minibatch_size": 8, "sequence_microbatch_size": 4})
    torch.manual_seed(0)
    model = ScoreBeliefActorCritic(
        observation_dim=OBS_DIM,
        state_dim=2,
        action_kind="discrete",
        action_feature_dim=2,
        discrete_actions=2,
        action_low=None,
        action_high=None,
        model_config=model_config,
    )
    assert model.prediction_predictor is not None
    return model, model_config, ppo_config


def _is_representation(name: str) -> bool:
    return name.startswith("belief.") or name.startswith("encoder.")


def _is_head(name: str) -> bool:
    return name.startswith("value_head") or name.startswith("categorical_head")


def _snapshot(model, predicate):
    return {n: p.detach().clone() for n, p in model.named_parameters() if predicate(n)}


def _max_move(model, before) -> float:
    return max(float((p - before[n]).abs().max()) for n, p in model.named_parameters() if n in before)


def test_the_option_is_validated():
    _config("prediction_only")
    _config("joint")
    with pytest.raises(ConfigError):
        _config("something_else")
    with pytest.raises(ConfigError):  # prediction_only needs the prediction phases
        load_config(ROOT / "config" / "local.json").with_overrides({"model.belief_training": "prediction_only"})


def test_prediction_only_keeps_ppo_out_of_the_representation(monkeypatch):
    model, model_config, ppo_config = _model("prediction_only")
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    optimizer = build_optimizer(model, ppo_config)
    before_representation = _snapshot(model, _is_representation)
    before_heads = _snapshot(model, _is_head)
    # With the prediction loss switched off nothing may reach the belief or its readout.
    monkeypatch.setattr(ppo_module, "condition_prediction_loss", lambda predictor, condition, **kw: condition.sum() * 0.0)
    metrics = update_ppo(model, optimizer, rollout, model_config, ppo_config)
    assert np.isfinite(metrics.policy_loss)
    assert _max_move(model, before_representation) == 0.0
    assert _max_move(model, before_heads) > 0.0
    monkeypatch.undo()
    # With the prediction loss on, the representation moves (through that loss only).
    model, model_config, ppo_config = _model("prediction_only")
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    optimizer = build_optimizer(model, ppo_config)
    before_representation = _snapshot(model, _is_representation)
    update_ppo(model, optimizer, rollout, model_config, ppo_config)
    assert _max_move(model, before_representation) > 0.0


def test_joint_lets_ppo_reach_the_representation(monkeypatch):
    model, model_config, ppo_config = _model("joint")
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    optimizer = build_optimizer(model, ppo_config)
    before_representation = _snapshot(model, _is_representation)
    monkeypatch.setattr(ppo_module, "condition_prediction_loss", lambda predictor, condition, **kw: condition.sum() * 0.0)
    update_ppo(model, optimizer, rollout, model_config, ppo_config)
    assert _max_move(model, before_representation) > 0.0


def test_score_model_trains_one_update(tmp_path):
    config = _config("prediction_only", **{"experiment.output_dir": str(tmp_path)})
    task = "light_dark"
    run, summaries = run_all(config, tasks=[task], seeds=[0], max_updates=1, label="pytest-predonly", verbose=False)
    assert summaries[0]["status"] == "completed"
    with np.load(run.root / task / "seed_000" / "arrays.npz", allow_pickle=False) as arrays:
        assert np.isfinite(arrays["gradient_norm"]).all()


@pytest.mark.parametrize("method", ["gru", "particle_filter"])
def test_recurrent_baseline_trains_one_update(tmp_path, method):
    config = _config("prediction_only")
    task = "light_dark"
    artifact_path = tmp_path / "s"
    artifact_path.mkdir()
    artifacts = SeedArtifacts(path=artifact_path, task=task, seed=0, method=method)
    summary = train_baseline_single(
        config,
        method=method,  # type: ignore[arg-type]
        task=task,
        seed=0,
        artifacts=artifacts,
        max_updates=1,
        verbose=False,
        result_dir=f"{method}/{task}/seed_000",
    )
    assert summary["status"] == "completed"
    checkpoint = torch.load(
        artifacts.checkpoint_path(tag="latest", create_parent=False), map_location="cpu", weights_only=False
    )
    assert checkpoint["config"]["model"]["belief_training"] == "prediction_only"
    assert all(torch.isfinite(tensor).all() for tensor in checkpoint["model"].values())
