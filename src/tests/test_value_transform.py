"""Critic target transform: exact inverse, identity default, and trainer wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.train_baselines import train_baseline_single
from sb_pomdp.value_transform import ValueTransform, symexp, symlog

ROOT = Path(__file__).resolve().parents[2]


def test_symlog_and_symexp_are_exact_inverses_and_compress_tails() -> None:
    values = torch.tensor([-1e4, -100.0, -1.0, 0.0, 0.5, 30.0, 1e4], dtype=torch.float64)
    assert torch.allclose(symexp(symlog(values)), values, rtol=1e-9, atol=1e-9)
    assert symlog(torch.tensor(-1e4)).item() == pytest.approx(-9.2104, abs=1e-3)
    assert symlog(torch.tensor(-10.0)).item() == pytest.approx(-2.3979, abs=1e-3)
    assert torch.all(torch.sign(symlog(values)) == torch.sign(values))


def test_identity_transform_returns_the_same_tensor_object() -> None:
    transform = ValueTransform("none")
    tensor = torch.randn(5)
    assert transform.is_identity
    assert transform.to_raw(tensor) is tensor
    assert transform.to_target(tensor) is tensor
    assert ValueTransform.from_config({}).is_identity
    with pytest.raises(ValueError):
        ValueTransform("popart")


def test_symlog_transform_round_trips_through_raw_units() -> None:
    transform = ValueTransform.from_config({"value_target_transform": "symlog"})
    returns = torch.tensor([-3000.0, -2.0, 4.0])
    assert torch.allclose(transform.to_raw(transform.to_target(returns)), returns, atol=1e-3)


def test_config_schema_validates_the_transform_key() -> None:
    config = load_config("config/local.json")
    assert config.to_dict()["ppo"]["value_target_transform"] == "none"
    with pytest.raises(ConfigError):
        config.with_overrides({"ppo.value_target_transform": "popart"})
    accepted = config.with_overrides({"ppo.value_target_transform": "symlog"})
    assert accepted["ppo"]["value_target_transform"] == "symlog"


@pytest.mark.parametrize("method", ("gru", "observation_mlp"))
def test_baseline_trainer_runs_one_update_with_symlog_targets(tmp_path: Path, method: str) -> None:
    config = load_config(ROOT / "config" / "local.json").with_overrides(
        {"ppo.value_target_transform": "symlog"}
    )
    artifact_path = tmp_path / "s"
    artifact_path.mkdir()
    artifacts = SeedArtifacts(path=artifact_path, task="masked_cartpole", seed=0, method=method)
    summary = train_baseline_single(
        config,
        method=method,  # type: ignore[arg-type]
        task="masked_cartpole",
        seed=0,
        artifacts=artifacts,
        max_updates=1,
        verbose=False,
        result_dir=f"{method}/masked_cartpole/seed_000",
    )
    assert summary["status"] == "completed"
