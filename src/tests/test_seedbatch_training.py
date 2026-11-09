from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from sb_pomdp.compare import run_comparisons
from sb_pomdp.config import load_config
from sb_pomdp.seedbatch import run_seedbatch
from sb_pomdp.config import ConfigError
from sb_pomdp.train import _load_resume_checkpoint, model_for_environment
from sb_pomdp.envs import make_env


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(root, **overrides):
    return load_config("config/trysvgdlight_mirror2_tf-svgd.json").with_overrides({
        "experiment.output_dir": str(root), "experiment.device": "cpu",
        "experiment.deterministic": True, "experiment.seeds": [10, 11],
        "experiment.eval_episodes": 1, "experiment.eval_interval_updates": 1,
        "experiment.checkpoint_interval_updates": 1,
        "environment.tasks.mountain_hike.horizon": 7,
        "model.num_particles": 4, "model.langevin_steps": 2,
        "model.energy_hidden": [8], "model.d_model": 8, "model.num_heads": 2,
        "model.transformer_ff_dim": 16, "model.policy_hidden": [8],
        "model.diffusion_steps": 2, "model.prediction_predictor_hidden": [8],
        "model.prediction_pretrain_steps": 0, "model.prediction_interval_updates": 0,
        "model.prediction_phase_updates": 0,
        "ppo.total_steps": 32, "ppo.num_envs": 2, "ppo.rollout_steps": 8,
        "ppo.epochs": 2, "ppo.minibatch_size": 16, "ppo.sequence_microbatch_size": 8,
        **overrides,
    })


def seed_dir(root, seed):
    return root / f"full/score_transformer/mountain_hike/seed_{seed:03d}"


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_csv_close(left, right, *, atol=3e-4, rtol=3e-4):
    a, b = rows(left), rows(right)
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.keys() == rb.keys()
        for key in ra:
            if key in {"steps_per_second", "gpu_peak_memory_mb"}:
                continue
            try:
                fa, fb = float(ra[key]), float(rb[key])
            except ValueError:
                assert ra[key] == rb[key]
            else:
                np.testing.assert_allclose(fa, fb, atol=atol, rtol=rtol, equal_nan=True,
                                           err_msg=f"{left.name} {key}: {ra[key]} vs {rb[key]}")


@pytest.mark.parametrize("sampler,proposal,bound,latent", [
    ("svgd", "noise", 2.0, None), ("ula", "noise", None, 3),
    ("svgd", "affine", "log", 3), ("ula", "uniform", "log", None),
])
def test_matches_existing_compare(tmp_path, sampler, proposal, bound, latent):
    config = tiny_config(tmp_path, **{
        "model.particle_sampler": sampler, "model.langevin_proposal": proposal,
        "model.direct_score_bound": bound, "model.belief_latent_dim": latent,
        "model.initial_score_prior": 0.2,
    })
    run_seedbatch(config, campaign_id="batched", verbose=False)
    for seed in [10, 11]:
        run, _, _ = run_comparisons(
            config, methods=["score_transformer"], tasks=["mountain_hike"], seeds=[seed],
            gradient_modes=["full"], max_updates=None, label=f"single-{seed}", verbose=False,
        )
        batched = seed_dir(tmp_path / f"campaigns/batched/shard{seed - 9:03d}", seed)
        single = seed_dir(run.root, seed)
        for name in ("metrics.csv", "evaluation.csv", "episodes.csv"):
            assert_csv_close(batched / name, single / name)
        left = torch.load(batched / "checkpoints/latest.pt", weights_only=False)
        right = torch.load(single / "checkpoints/latest.pt", weights_only=False)
        assert left.keys() == right.keys()
        assert torch.equal(left["torch_rng_state"], right["torch_rng_state"])
        for name in left["model"]:
            torch.testing.assert_close(left["model"][name], right["model"][name], rtol=3e-4, atol=3e-5)


def test_four_env_32_steps_prediction_schedule_matches_compare(tmp_path):
    config = tiny_config(tmp_path, **{
        "ppo.num_envs": 4, "ppo.rollout_steps": 32, "ppo.total_steps": 256,
        "ppo.minibatch_size": 64, "ppo.sequence_microbatch_size": 32,
        "model.prediction_pretrain_steps": 128, "model.prediction_interval_updates": 1,
        "model.prediction_phase_updates": 1, "environment.tasks.mountain_hike.horizon": 19,
        "experiment.eval_interval_updates": 2,
        "ppo.bootstrap_on_truncation": True,
    })
    summaries = run_seedbatch(config, campaign_id="phases", verbose=False)
    assert [s["updates"] for s in summaries] == [4, 4]
    for seed in [10, 11]:
        run, _, _ = run_comparisons(
            config, methods=["score_transformer"], tasks=["mountain_hike"], seeds=[seed],
            gradient_modes=["full"], max_updates=None, label=f"phases-{seed}", verbose=False,
        )
        batched = seed_dir(tmp_path / f"campaigns/phases/shard{seed - 9:03d}", seed)
        for name in ("metrics.csv", "evaluation.csv", "episodes.csv"):
            assert_csv_close(batched / name, seed_dir(run.root, seed) / name)
        assert {row["mode"] for row in rows(batched / "evaluation.csv")} == {"mean_chain", "stochastic"}


def test_segment_resume_exact_and_completed_retry_noop(tmp_path):
    config = tiny_config(tmp_path, **{
        "model.prediction_pretrain_steps": 16, "model.prediction_interval_updates": 1,
        "model.prediction_phase_updates": 1,
    })
    run_seedbatch(config, campaign_id="whole", verbose=False)
    for _ in range(4):
        run_seedbatch(config, campaign_id="segmented", resume=True, segment_updates=1, verbose=False)
    for seed in [10, 11]:
        whole = seed_dir(tmp_path / f"campaigns/whole/shard{seed - 9:03d}", seed)
        segmented = seed_dir(tmp_path / f"campaigns/segmented/shard{seed - 9:03d}", seed)
        for name in ("metrics.csv", "evaluation.csv", "episodes.csv"):
            assert_csv_close(whole / name, segmented / name, atol=0, rtol=0)
        left = torch.load(whole / "checkpoints/latest.pt", weights_only=False)
        right = torch.load(segmented / "checkpoints/latest.pt", weights_only=False)
        assert torch.equal(left["torch_rng_state"], right["torch_rng_state"])
        for name in left["model"]:
            assert torch.equal(left["model"][name], right["model"][name]), name
        # Existing checkpoint validation and ordinary model construction load
        # these unmodified v4 checkpoints, including Adam state and trainer state.
        _load_resume_checkpoint(segmented / "checkpoints/latest.pt", config=right["config"],
                                task="mountain_hike", seed=seed, method="score_transformer",
                                rollout_batch_size=16, configured_updates=4)
        environment = make_env("mountain_hike", config.environment_for_task("mountain_hike"), seed=seed)
        model = model_for_environment(environment, right["config"]["model"], torch.device("cpu"))
        model.load_state_dict(right["model"], strict=True)
    paths = [p for p in (tmp_path / "campaigns/segmented").rglob("*") if p.is_file()]
    contents = {p: p.read_bytes() for p in paths}
    run_seedbatch(config, campaign_id="segmented", resume=True, segment_updates=1, verbose=False)
    assert contents == {p: p.read_bytes() for p in paths}


def test_seed_order_and_singleton_independence(tmp_path):
    config = tiny_config(tmp_path)
    run_seedbatch(config, campaign_id="pair", seeds=[11, 10], verbose=False)
    run_seedbatch(config, campaign_id="alone", seeds=[10], verbose=False)
    paired = seed_dir(tmp_path / "campaigns/pair/shard001", 10)
    alone = seed_dir(tmp_path / "campaigns/alone/shard001", 10)
    for name in ("metrics.csv", "evaluation.csv", "episodes.csv"):
        assert_csv_close(paired / name, alone / name)
    assert seed_dir(tmp_path / "campaigns/pair/shard002", 11).is_dir()
    from sb_pomdp.aggregate import aggregate_runs
    combined = aggregate_runs(
        [tmp_path / f"campaigns/pair/shard{i:03d}" for i in [1, 2]],
        expected_config=config, output_dir=tmp_path / "aggregate", campaign_id="pair",
    )
    assert combined.root.is_dir()


@pytest.mark.parametrize("key,value", [
    ("model.score_parameterization", "energy"), ("model.particle_sampler", "smi"),
    ("model.belief_cuda_graphs", True), ("model.belief_activation_checkpoint", True),
    ("model.policy_observation_input", True), ("model.prediction_head", "belief_mixture"),
    ("model.critic_kind", "action"), ("ppo.algorithm", "td3"),
    ("model.langevin_proposal", "network"), ("model.reward_prediction_coef", 0.1),
    ("model.belief_training", "prediction_only"),
])
def test_unsupported_rejected_before_artifacts(tmp_path, key, value):
    config = tiny_config(tmp_path, **{"model.prediction_pretrain_steps": 16, key: value})
    with pytest.raises(ConfigError, match="seedbatch"):
        run_seedbatch(config, campaign_id="unsupported", verbose=False)
    assert not (tmp_path / "campaigns").exists()


def test_analytic_direct_and_adamw_supported(tmp_path):
    config = tiny_config(tmp_path, **{
        "model.score_implementation": "analytic", "ppo.weight_decay": 0.01,
        "ppo.langevin_schedule_lr_multiplier": 2.0,
        "environment.tasks.mountain_hike.horizon": 1,
    })
    run_seedbatch(config, campaign_id="initial", verbose=False)
    for seed in [10, 11]:
        run, _, _ = run_comparisons(
            config, methods=["score_transformer"], tasks=["mountain_hike"], seeds=[seed],
            gradient_modes=["full"], max_updates=None, label=f"initial-{seed}", verbose=False,
        )
        batched = seed_dir(tmp_path / f"campaigns/initial/shard{seed - 9:03d}", seed)
        assert_csv_close(batched / "metrics.csv", seed_dir(run.root, seed) / "metrics.csv")
        left = torch.load(batched / "checkpoints/latest.pt", weights_only=False)
        right = torch.load(seed_dir(run.root, seed) / "checkpoints/latest.pt", weights_only=False)
        assert left["optimizer"]["state"].keys() == right["optimizer"]["state"].keys()
        for name in left["model"]:
            torch.testing.assert_close(left["model"][name], right["model"][name], rtol=3e-4, atol=3e-5)
