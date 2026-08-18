from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from sb_pomdp.config import load_config
from sb_pomdp.envs import make_env
from sb_pomdp.fetch_results import _verify_staged_result
from sb_pomdp.run_experiments import run_all
from sb_pomdp.train import main as train_main
from sb_pomdp.train import model_for_environment, seed_everything


def test_all_four_tasks_train_and_write_reloadable_artifacts(tmp_path) -> None:
    config = load_config("config/local.json").with_overrides(
        {"experiment.output_dir": str(tmp_path)}
    )
    tasks = list(config["experiment"]["tasks"])
    run, summaries = run_all(
        config,
        tasks=tasks,
        seeds=[0],
        max_updates=1,
        label="pytest-smoke",
        verbose=False,
    )

    assert len(summaries) == 4
    assert all(summary["status"] == "completed" for summary in summaries)
    assert all(not Path(summary["result_dir"]).is_absolute() for summary in summaries)
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    load_config(run.resolved_config_path)

    for task in tasks:
        seed_directory = run.root / task / "seed_000"
        load_config(seed_directory / "resolved_config.json")
        with (seed_directory / "evaluation.csv").open("r", encoding="utf-8", newline="") as handle:
            evaluation_rows = list(csv.DictReader(handle))
        deterministic_mode = "greedy" if task == "masked_cartpole" else "mean_chain"
        assert {row["mode"] for row in evaluation_rows} == {
            deterministic_mode,
            "stochastic",
        }
        assert {row["evaluation_seed"] for row in evaluation_rows} == {"1000000"}
        assert len(evaluation_rows) == 2
        for mode in (deterministic_mode, "stochastic"):
            assert (seed_directory / f"evaluation_{mode}_trajectories_000001.npz").is_file()
        with np.load(seed_directory / "arrays.npz", allow_pickle=False) as arrays:
            assert arrays["global_step"].tolist() == [16]
            assert np.isfinite(arrays["gradient_norm"]).all()
            for field in (
                "initial_energy_gradient_norm",
                "observation_energy_gradient_norm",
                "transition_energy_gradient_norm",
                "initial_score_l2_mean",
                "recursive_score_l2_mean",
                "initial_particle_l2_mean",
                "recursive_particle_l2_mean",
            ):
                assert np.isfinite(arrays[field]).all()
            assert arrays["initial_belief_particle_count"].tolist() == [8]
            assert arrays["recursive_belief_particle_count"].tolist() == [56]
            assert arrays["score_nonfinite_count"].tolist() == [0]
            assert arrays["particle_nonfinite_count"].tolist() == [0]
        log_text = (seed_directory / "run.log").read_text(encoding="utf-8")
        for phase in (
            "rollout_collection",
            "ppo_update",
            "recondition",
            "evaluation",
        ):
            assert f"phase={phase} status=start update=1/1" in log_text
            assert f"phase={phase} status=end update=1/1 seconds=" in log_text
        checkpoint = torch.load(
            seed_directory / "checkpoints" / "latest.pt",
            map_location="cpu",
            weights_only=False,
        )

        seed_everything(0, deterministic=True)
        environment = make_env(task, config.environment_for_task(task), seed=0)
        fresh_model = model_for_environment(environment, config["model"], torch.device("cpu"))
        fresh = fresh_model.state_dict()
        trained = checkpoint["model"]
        initial_delta = sum(
            float((trained[name] - value).abs().sum())
            for name, value in fresh.items()
            if name.startswith("belief.initial_energy")
        )
        f_delta = sum(
            float((trained[name] - value).abs().sum())
            for name, value in fresh.items()
            if name.startswith("belief.observation_energy")
        )
        u_delta = sum(
            float((trained[name] - value).abs().sum())
            for name, value in fresh.items()
            if name.startswith("belief.transition_energy")
        )
        assert initial_delta > 0
        assert f_delta > 0
        assert u_delta > 0


def test_single_task_cli_writes_complete_root_artifacts(tmp_path, capsys) -> None:
    exit_code = train_main(
        [
            "--config",
            "config/local.json",
            "--task",
            "masked_cartpole",
            "--seed",
            "0",
            "--max-updates",
            "1",
            "--override",
            f"experiment.output_dir={tmp_path}",
        ]
    )

    assert exit_code == 0
    run_root = Path(capsys.readouterr().out.strip().splitlines()[-1])
    _verify_staged_result(run_root)
    load_config(run_root / "resolved_config.json")
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["runs"][0]["result_dir"] == "masked_cartpole/seed_000"


def test_full_and_one_step_modes_train_across_a_rollout_boundary(tmp_path) -> None:
    checkpoints: dict[str, dict[str, torch.Tensor]] = {}
    for gradient_mode in ("full", "tbptt_1"):
        config = load_config("config/local.json").with_overrides(
            {
                "experiment.output_dir": str(tmp_path / gradient_mode),
                "model.belief_gradient_mode": gradient_mode,
                "ppo.total_steps": 32,
                "environment.tasks.masked_mountain_car_continuous.horizon": 20,
            }
        )
        run, summaries = run_all(
            config,
            tasks=["masked_mountain_car_continuous"],
            seeds=[0],
            max_updates=2,
            label=f"pytest-{gradient_mode}",
            verbose=False,
        )

        assert summaries[0]["status"] == "completed"
        seed_root = run.root / "masked_mountain_car_continuous" / "seed_000"
        with np.load(seed_root / "arrays.npz", allow_pickle=False) as arrays:
            assert arrays["global_step"].tolist() == [16, 32]
            assert np.isfinite(arrays["gradient_norm"]).all()
            # Horizon 20 exceeds one eight-step rollout.  Thus only the two
            # time-zero beliefs (four particles each) are initial in update 1;
            # update 2 contains no reset.  This catches a one-step shift in the
            # diagnostic reset mask without depending on learned norm values.
            assert arrays["initial_belief_particle_count"].tolist() == [8, 0]
            assert arrays["recursive_belief_particle_count"].tolist() == [56, 64]
        checkpoints[gradient_mode] = torch.load(
            seed_root / "checkpoints" / "latest.pt",
            map_location="cpu",
            weights_only=False,
        )["model"]

    assert any(
        not torch.equal(checkpoints["full"][name], checkpoints["tbptt_1"][name])
        for name in checkpoints["full"]
        if name.startswith("belief.")
    )
