"""model.policy_observation_input: the actor and critic read ``[condition, o_t]``.

The option is shared by every compared model (score belief, GRU / RNN, particle filter): the
raw current observation is appended to the actor / critic input while the belief condition and
the next-observation prediction head stay as they were.  With the option off nothing changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.envs import make_env
from sb_pomdp.run_experiments import run_all
from sb_pomdp.train import (
    BeliefContext,
    _policy_from_particles,
    _sample_belief,
    belief_latent_dim,
    model_for_environment,
)
from sb_pomdp.train_baselines import (
    _append_input_noise,
    _make_model,
    train_baseline_single,
)

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cpu")


def _config(enabled: bool = True, **overrides):
    return load_config(ROOT / "config" / "local.json").with_overrides(
        {"model.policy_observation_input": enabled, **overrides}
    )


def _environment(config, task: str):
    return make_env(task, config["environment"]["tasks"][task], seed=0)


def _first_linear(module: torch.nn.Module) -> torch.nn.Linear:
    return next(m for m in module.modules() if isinstance(m, torch.nn.Linear))


def test_the_option_is_validated():
    _config()  # loads
    with pytest.raises(ConfigError):
        _config("yes")
    with pytest.raises(ConfigError):
        _config(**{"model.policy_head_kind": "alpha_lse", "model.encoder_kind": "alpha_pool"})


@pytest.mark.parametrize("task", ["light_dark", "masked_pendulum"])
def test_score_heads_read_the_observation_and_the_prediction_head_does_not(task):
    config = _config(**{"model.prediction_pretrain_steps": 16})
    environment = _environment(config, task)
    model_config = dict(config["model"])
    torch.manual_seed(0)
    model = model_for_environment(environment, model_config, DEVICE).eval()
    reference = model_for_environment(environment, dict(model_config, policy_observation_input=False), DEVICE)
    d_model = int(model_config["d_model"])
    obs_dim = int(environment.observation_dim)
    assert model.control_observation_dim == obs_dim and reference.control_observation_dim == 0
    assert _first_linear(model.value_head).in_features == d_model + obs_dim
    assert _first_linear(reference.value_head).in_features == d_model
    # The prediction head still conditions on the bare belief condition (+ action features).
    assert model.prediction_predictor is not None
    assert _first_linear(model.prediction_predictor).in_features == d_model + environment.action_spec.feature_dim

    observation, _ = environment.reset(seed=0)
    batch = 3
    context = BeliefContext(
        observation=torch.as_tensor(np.stack([observation] * batch), dtype=torch.float32),
        previous_particles=torch.zeros(batch, int(model_config["num_particles"]), belief_latent_dim(environment, model_config)),
        previous_action=torch.zeros(batch, environment.action_spec.feature_dim),
        initial=torch.ones(batch, dtype=torch.bool),
    )
    particles, scores = _sample_belief(model, context, model_config)
    with pytest.raises(ValueError):
        _policy_from_particles(model, particles, scores, deterministic=True)
    action, log_prob, _, _, _, value = _policy_from_particles(
        model, particles, scores, deterministic=True, observation=context.observation
    )
    assert action.shape[0] == batch and value.shape == (batch,) and torch.isfinite(value).all()
    # The reference model keeps working without an observation.
    reference_particles, reference_scores = _sample_belief(reference, context, dict(model_config, policy_observation_input=False))
    _policy_from_particles(reference, reference_particles, reference_scores, deterministic=True)


def test_score_model_trains_one_update(tmp_path):
    config = _config(**{"experiment.output_dir": str(tmp_path)})
    task = "light_dark"
    run, summaries = run_all(
        config, tasks=[task], seeds=[0], max_updates=1, label="pytest-obs", verbose=False
    )
    assert summaries[0]["status"] == "completed"
    with np.load(run.root / task / "seed_000" / "arrays.npz", allow_pickle=False) as arrays:
        assert np.isfinite(arrays["gradient_norm"]).all()
        assert np.isfinite(arrays["policy_loss"]).all()


@pytest.mark.parametrize("method", ["gru", "rnn", "particle_filter"])
def test_recurrent_baseline_heads_read_the_observation(method):
    task = "light_dark"
    config = _config()
    environment = _environment(config, task)
    torch.manual_seed(0)
    model = _make_model(environment, method, dict(config["model"]), dict(config["comparison"]), DEVICE).eval()
    reference = _make_model(
        environment, method, dict(config["model"], policy_observation_input=False), dict(config["comparison"]), DEVICE
    )
    obs_dim = int(environment.observation_dim)
    d_model = int(config["model"]["d_model"])
    assert model.control_observation_dim == obs_dim and reference.control_observation_dim == 0
    assert _first_linear(model.heads.value_head).in_features == d_model + obs_dim
    assert _first_linear(reference.heads.value_head).in_features == d_model

    observation, _ = environment.reset(seed=0)
    batch = 2
    inputs = torch.cat(
        (
            torch.as_tensor(np.stack([observation] * batch), dtype=torch.float32),
            torch.zeros(batch, environment.action_spec.feature_dim),
        ),
        dim=-1,
    )
    inputs = _append_input_noise(inputs, model)
    hidden = model.initial_hidden(batch, device=DEVICE)
    starts = torch.ones(batch, dtype=torch.bool)
    with torch.no_grad():
        sample, value, next_hidden = model.act_step(inputs, hidden, starts, deterministic=True)
    assert value.shape == (batch,) and torch.isfinite(value).all()
    assert next_hidden.shape == hidden.shape
    # control_features appends exactly the raw observation.
    features, _ = model.step(inputs, hidden, starts)
    control = model.control_features(features, inputs)
    assert control.shape == (batch, d_model + obs_dim)
    assert torch.allclose(control[:, d_model:], inputs[:, :obs_dim])
    with pytest.raises(ValueError):
        model.control_features(features, inputs[:1])


@pytest.mark.parametrize("method", ["gru", "particle_filter"])
def test_recurrent_baseline_trains_one_update(tmp_path, method):
    task = "light_dark"
    config = _config()
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
    assert all(torch.isfinite(tensor).all() for tensor in checkpoint["model"].values())
    assert checkpoint["config"]["model"]["policy_observation_input"] is True
