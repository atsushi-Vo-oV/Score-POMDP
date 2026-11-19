"""Experiment 31: enlarged arms with the same representation size (256 numbers per step) and matched acting compute."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from sb_pomdp.config import load_config


def _generator():
    spec = importlib.util.spec_from_file_location("rep256_generate", Path("scripts/rep256_generate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_arms_differ_from_their_source_only_in_the_listed_keys():
    generator = _generator()
    assets = list(generator.assets())
    assert len(assets) == 63
    for arm, task, name, source, label, campaign, stem, keys, method, updates, _ in assets:
        config, base = load_config(f"config/{name}.json").to_dict(), load_config(f"config/{source}.json").to_dict()
        assert config["experiment"]["label"] == label and campaign == f"{label}-v1"
        assert config["model"]["num_particles"] == 16
        if method is None:
            assert config["model"]["belief_latent_dim"] == 16 and config["model"]["energy_hidden"] == [32]
            assert config["model"]["transition_persistence_units"] == "pair"
            assert config["model"]["transition_persistence_init"] == (4.0 if "tinit" in arm else None)
            assert config["model"]["transition_persistence_kernel"] == (4.0 if "tinitmix" in arm else None)
            assert config["model"]["initial_score_projection_precision"] == (4.0 if "tinitq" in arm else None)
            assert config["comparison"] == base["comparison"]
            for key in keys:
                config["model"][key] = base["model"][key]
        else:
            comparison = config["comparison"]
            assert comparison["methods"] == [method] and comparison["gru_hidden_dim"] == 256
            assert comparison["pf_particle_dim"] == 16
            assert comparison["gru_encoder_hidden"] == generator.WIDTHS[task]["gru_encoder_hidden"]
            assert comparison["pf_hidden"] == generator.WIDTHS[task]["pf_hidden"]
            assert config["model"] == base["model"]
            for key in ("methods", "gru_encoder_hidden", "gru_hidden_dim", "pf_hidden", "pf_observation_function", "pf_observation_hidden"):
                config["comparison"][key] = base["comparison"][key]
        config["experiment"]["label"] = base["experiment"]["label"]
        assert config == base
        for seed in range(10, 15):
            text = Path(f"jobs/rep256/{stem}_s{seed}.sh").read_text(encoding="utf-8")
            assert f'SB_POMDP_CONFIG="config/{name}.json"' in text
            assert f'SB_POMDP_CAMPAIGN_ID="{campaign}"' in text
            assert f"SB_POMDP_SHARD_INDEX={seed - 9}" in text and f"SB_POMDP_SEGMENT_UPDATES={updates}" in text
            assert text.rstrip().endswith("exec bash jobs/genkai_production.sh")


@pytest.mark.parametrize("task", ["mirror2", "tagr", "cpv"])
def test_every_arm_carries_256_numbers(task):
    from sb_pomdp.envs import make_env  # noqa: PLC0415
    from sb_pomdp.train import model_for_environment  # noqa: PLC0415
    from sb_pomdp.train_baselines import _make_model  # noqa: PLC0415

    device = torch.device("cpu")
    light = load_config(f"config/trysvgdlight2_{task}_tf-svgd-tinit-r256.json").to_dict()
    (name,) = light["experiment"]["tasks"]
    environment = make_env(name, light["environment"]["tasks"][name], seed=0)
    model = model_for_environment(environment, light["model"], device)
    assert model.belief.state_dim * light["model"]["num_particles"] == 256
    # the exact paired form fits every task at this size
    d = model.belief.state_dim
    g = torch.Generator().manual_seed(1)
    query, previous = torch.randn(32, d, generator=g), torch.randn(32, d, generator=g)
    action = torch.zeros(32, model.belief.action_feature_dim)
    out = model.belief.transition_energy(torch.cat((query, previous, action), dim=-1))
    assert torch.allclose(out[:, 1:], 4.0 * (previous - query), atol=1e-4)
    gru_config = load_config(f"config/trysvgd_{task}_gru-r256.json").to_dict()
    gru = _make_model(environment, "gru", dict(gru_config["model"]), dict(gru_config["comparison"]), device)
    hidden = gru.initial_hidden(1, device=device)
    assert sum(t.numel() for t in (hidden if isinstance(hidden, (tuple, list)) else (hidden,))) == 256
    pf_config = load_config(f"config/trysvgd_{task}_pf_v3-r256.json").to_dict()
    assert pf_config["model"]["num_particles"] * pf_config["comparison"]["pf_particle_dim"] == 256
