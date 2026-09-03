from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sb_pomdp.artifacts import ArtifactError, CsvAppender, RunArtifacts, truncate_csv_rows
from sb_pomdp.config import (
    ConfigError,
    ExperimentConfig,
    apply_overrides,
    load_config,
    normalize_legacy_resolved_config,
    parse_cli_overrides,
    validate_config,
)
from sb_pomdp.envs import make_env

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ``_valid_config`` is deliberately written in the pre-campaign schema.  These are
# the post-campaign keys the normalizer fills in, at the values that reproduce the
# campaign's behaviour exactly.
_LEGACY_PPO_DEFAULTS: dict[str, object] = {
    "bootstrap_on_truncation": False,
    "value_target_transform": "none",
    "langevin_schedule_lr_multiplier": 1.0,
    "algorithm": "ppo",
    "p3o_eta": 1.0,
    "p3o_resample_interval": 5,
    "p3o_demo_slots": 0,
}
_LEGACY_MODEL_DEFAULTS: dict[str, object] = {
    "langevin_temperature": 1.0,
    "langevin_temperature_learnable": False,
    "langevin_warm_start": False,
    "langevin_step_size_learnable": False,
    "langevin_schedule_bound": "clamp",
    "policy_head_kind": "mlp",
    "alpha_pieces": 16,
    "alpha_temperature": 1.0,
    "alpha_use_scores": False,
    "observation_prediction_coef": 0.0,
    "observation_predictor_hidden": [64],
    "energy_network_kind": "mlp",
    "kan_grid_size": 8,
    "kan_spline_order": 3,
    "kan_grid_range": 3.0,
    "kan_match_parameters": True,
    "head_network_kind": "mlp",
    "trunk_network_kind": "mlp",
    "basis_order": 4,
    "basis_projection_dim": 16,
}
_LEGACY_COMPARISON_DEFAULTS: dict[str, object] = {
    "pf_hidden": [64],
    "pf_particle_dim": 8,
    "pf_soft_alpha": 0.9,
    "alpha_feedforward_dim": None,
    "rnn_encoder_hidden": [64],
    "rnn_hidden_dim": 32,
}
_LEGACY_LIGHT_DARK_DEFAULTS: dict[str, object] = {
    "noise_gain": 0.5,
    "terminal_cost": 0.0,
    "process_noise_std": 0.0,
    "light_position": 5.0,
    "initial_mean": 2.0,
    "initial_std": 0.5,
    "state_cost": 0.5,
    "action_cost": 0.5,
    "goal_radius": 0.25,
    "goal_bonus": 0.0,
}


def _valid_config() -> dict[str, object]:
    return {
        "experiment": {
            "name": "score-belief-smoke",
            "label": "local",
            "tasks": ["tiger", "light_dark", "continuous_navigation"],
            "seeds": [0, 7],
            "device": "cpu",
            "deterministic": True,
            "output_dir": "results",
            "eval_episodes": 2,
            "eval_interval_updates": 1,
            "checkpoint_interval_updates": 2,
            "log_interval_updates": 1,
        },
        "environment": {
            "tasks": {
                "tiger": {"horizon": 20, "observation_noise_std": 0.1},
                "light_dark": {
                    "horizon": 30,
                    "observation_noise_std": 0.0,
                    "dimension": 5,
                },
                "continuous_navigation": {
                    "horizon": 40,
                    "observation_noise_std": 0.2,
                },
            }
        },
        "model": {
            "num_particles": 8,
            "langevin_steps": 2,
            "langevin_step_size": 0.05,
            "particle_clip": 5.0,
            "score_clip": 10.0,
            "energy_hidden": [32, 32],
            "d_model": 32,
            "num_heads": 4,
            "num_transformer_layers": 1,
            "transformer_ff_dim": 64,
            "dropout": 0.0,
            "policy_hidden": [32],
            "diffusion_steps": 4,
            "diffusion_beta_start": 0.0001,
            "diffusion_beta_end": 0.02,
            "diffusion_min_std": 0.05,
            "diffusion_advantage_discount": 0.95,
            "encoder_kind": "transformer",
            "continuous_policy_kind": "diffusion",
            "belief_gradient_mode": "full",
        },
        "ppo": {
            "total_steps": 128,
            "num_envs": 2,
            "rollout_steps": 16,
            "epochs": 2,
            "minibatch_size": 16,
            "sequence_microbatch_size": 16,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.0,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
        },
        "comparison": {
            "belief_gradient_modes": ["full", "tbptt_1"],
            "methods": [
                "score_transformer",
                "score_deepsets",
                "score_gaussian",
                "observation_mlp",
                "gru",
                "oracle_state",
            ],
            "gru_encoder_hidden": [24, 12],
            "gru_hidden_dim": 10,
        },
    }


def _write_config(path: Path, config: dict[str, object]) -> Path:
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _config_with_legacy_defaults() -> dict[str, object]:
    """Return ``_valid_config`` with every post-campaign key stated explicitly."""

    config = _valid_config()
    config["ppo"].update(_LEGACY_PPO_DEFAULTS)  # type: ignore[union-attr]
    config["model"].update(_LEGACY_MODEL_DEFAULTS)  # type: ignore[union-attr]
    config["comparison"].update(_LEGACY_COMPARISON_DEFAULTS)  # type: ignore[union-attr]
    config["environment"]["tasks"]["light_dark"].update(  # type: ignore[index]
        _LEGACY_LIGHT_DARK_DEFAULTS
    )
    return config


@pytest.mark.parametrize("name", ["local.json", "production.json"])
def test_shipped_comparison_configs_disable_entropy_bonus(name: str) -> None:
    config = load_config(_PROJECT_ROOT / "config" / name)

    assert config["ppo"]["entropy_coef"] == 0.0


def test_load_config_and_resolve_task(tmp_path: Path) -> None:
    source = _write_config(tmp_path / "config.json", _valid_config())

    config = load_config(source)

    assert isinstance(config, ExperimentConfig)
    assert config.source == source.resolve()
    assert config["model"]["num_particles"] == 8
    # The stored task block carries the Light-Dark reward keys at their legacy
    # defaults even though the file on disk omits them.
    assert config.environment_for_task("light_dark") == {
        "horizon": 30,
        "observation_noise_std": 0.0,
        "dimension": 5,
        **_LEGACY_LIGHT_DARK_DEFAULTS,
    }
    resolved = config.for_task("light_dark")
    assert resolved["environment"]["selected"] == {
        "name": "light_dark",
        "horizon": 30,
        "observation_noise_std": 0.0,
        "dimension": 5,
        **_LEGACY_LIGHT_DARK_DEFAULTS,
    }
    assert set(resolved["environment"]["tasks"]) == {
        "tiger",
        "light_dark",
        "continuous_navigation",
    }
    with pytest.raises(ConfigError, match="unknown task"):
        config.environment_for_task("missing")


def test_cli_and_nested_overrides_are_validated_without_mutating_source() -> None:
    original = _valid_config()
    config = ExperimentConfig(original)

    cli = parse_cli_overrides(
        [
            "ppo.learning_rate=0.001",
            "experiment.device=cuda",
            "experiment.deterministic=false",
        ]
    )
    changed = config.with_overrides(cli)
    nested = config.with_overrides({"model": {"num_particles": 12}})
    mixed = config.with_overrides({"ppo.learning_rate": 0.002, "experiment": {"device": "cuda:0"}})

    assert changed["ppo"]["learning_rate"] == 0.001
    assert changed["experiment"]["device"] == "cuda"
    assert changed["experiment"]["deterministic"] is False
    assert nested["model"]["num_particles"] == 12
    assert mixed["ppo"]["learning_rate"] == 0.002
    assert mixed["experiment"]["device"] == "cuda:0"
    assert config["ppo"]["learning_rate"] == 0.0003
    assert original["model"]["num_particles"] == 8

    exposed = config["ppo"]
    exposed["learning_rate"] = 0
    assert config["ppo"]["learning_rate"] == 0.0003

    with pytest.raises(ConfigError, match="does not exist"):
        config.with_overrides({"ppo.typo": 1})
    with pytest.raises(ConfigError, match="must be >"):
        config.with_overrides({"ppo.learning_rate": 0})
    with pytest.raises(ConfigError, match="PATH=VALUE"):
        parse_cli_overrides(["ppo.learning_rate"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("unexpected", {}), "unknown keys"),
        (lambda value: value.pop("ppo"), "missing keys"),
        (lambda value: value["model"].__setitem__("d_model", 30), "divisible"),
        (lambda value: value["model"].__setitem__("dropout", 0.1), "must be 0.0"),
        (
            lambda value: value["model"].__setitem__("encoder_kind", "attention"),
            "encoder_kind",
        ),
        (
            lambda value: value["model"].__setitem__("continuous_policy_kind", "beta"),
            "continuous_policy_kind",
        ),
        (
            lambda value: value["model"].__setitem__("belief_gradient_mode", "semi"),
            "belief_gradient_mode",
        ),
        (
            lambda value: value["model"].__setitem__("diffusion_advantage_discount", 0),
            "must be >",
        ),
        (lambda value: value["ppo"].__setitem__("minibatch_size", 7), "divisible"),
        (lambda value: value["ppo"].__setitem__("minibatch_size", 8), "at least"),
        (lambda value: value["ppo"].__setitem__("total_steps", 129), "must be divisible"),
        (lambda value: value["ppo"].__setitem__("target_kl", 0.03), "unknown keys"),
        (
            lambda value: value["ppo"].__setitem__("bootstrap_on_truncation", "true"),
            "bootstrap_on_truncation must be a boolean",
        ),
        (
            lambda value: value["comparison"].__setitem__("methods", ["unknown"]),
            "unknown methods",
        ),
        (
            lambda value: value["comparison"].__setitem__(
                "belief_gradient_modes", ["full", "semi"]
            ),
            "unknown modes",
        ),
        (
            lambda value: value["comparison"].__setitem__(
                "belief_gradient_modes", ["tbptt_1"]
            ),
            "must be enabled",
        ),
        (lambda value: value["comparison"].__setitem__("gru_encoder_hidden", []), "non-empty"),
        (lambda value: value["experiment"].__setitem__("seeds", [True]), "must be an integer"),
        (lambda value: value["experiment"].__setitem__("seeds", [2**32]), "must be <="),
        (lambda value: value["experiment"].__setitem__("device", "cuad"), "device must be"),
        (
            lambda value: value["experiment"].__setitem__("tasks", ["tiger."]),
            "safe filename",
        ),
        (
            lambda value: value["environment"]["tasks"]["tiger"].__setitem__(
                "observation_noise_std", -0.1
            ),
            "observation_noise_std",
        ),
        (
            lambda value: value["environment"]["tasks"]["light_dark"].__setitem__(
                "dimension", 0
            ),
            "dimension",
        ),
        (
            lambda value: value["environment"]["tasks"]["tiger"].__setitem__(
                "dimension", 3
            ),
            "unknown keys",
        ),
        (
            lambda value: value["environment"]["tasks"]["tiger"].__setitem__(
                "goal_bonus", 50.0
            ),
            "unknown keys",
        ),
        (
            lambda value: value["environment"]["tasks"]["light_dark"].__setitem__(
                "goal_bonus", -1.0
            ),
            r"light_dark\.goal_bonus must be >= 0",
        ),
        (
            lambda value: value["environment"]["tasks"]["light_dark"].__setitem__(
                "goal_bonus", "50"
            ),
            r"light_dark\.goal_bonus must be a number",
        ),
        (
            lambda value: value["environment"]["tasks"]["light_dark"].__setitem__(
                "goal_radius", 0.0
            ),
            r"light_dark\.goal_radius must be > 0",
        ),
        (
            lambda value: value["environment"]["tasks"]["light_dark"].__setitem__(
                "state_cost", -0.5
            ),
            r"light_dark\.state_cost must be >= 0",
        ),
        (
            lambda value: value["environment"]["tasks"]["light_dark"].__setitem__(
                "initial_std", -0.5
            ),
            r"light_dark\.initial_std must be >= 0",
        ),
        (
            lambda value: value["environment"]["tasks"].pop("light_dark"),
            "missing from environment.tasks",
        ),
    ],
)
def test_invalid_configs_fail_fast(mutation: object, message: str) -> None:
    value = _valid_config()
    mutation(value)  # type: ignore[operator]
    with pytest.raises(ConfigError, match=message):
        ExperimentConfig(value)


def test_legacy_config_normalizes_to_the_campaign_defaults() -> None:
    legacy = _valid_config()
    assert "bootstrap_on_truncation" not in legacy["ppo"]  # type: ignore[operator]
    assert "goal_bonus" not in legacy["environment"]["tasks"]["light_dark"]  # type: ignore[index]
    explicit = _config_with_legacy_defaults()

    normalized = normalize_legacy_resolved_config(legacy)

    # A resolved config written before the key existed loads, validates, and
    # compares equal to a current one that leaves the key at its default.
    validate_config(legacy)
    assert normalized == explicit
    assert ExperimentConfig(legacy).to_dict() == ExperimentConfig(explicit).to_dict()
    assert normalize_legacy_resolved_config(explicit) == explicit
    assert legacy == _valid_config()

    enabled = _valid_config()
    enabled["ppo"]["bootstrap_on_truncation"] = True  # type: ignore[index]
    assert normalize_legacy_resolved_config(enabled) != normalized
    rewarded = _valid_config()
    rewarded["environment"]["tasks"]["light_dark"]["goal_bonus"] = 50.0  # type: ignore[index]
    assert normalize_legacy_resolved_config(rewarded) != normalized
    with pytest.raises(ConfigError, match="config must be an object"):
        normalize_legacy_resolved_config(["ppo"])  # type: ignore[arg-type]


def test_light_dark_reward_keys_are_only_added_to_light_dark_tasks() -> None:
    aliased = _valid_config()
    tasks = aliased["environment"]["tasks"]  # type: ignore[index]
    tasks["Light-Dark-ND"] = tasks.pop("light_dark")
    aliased["experiment"]["tasks"] = [  # type: ignore[index]
        "tiger",
        "Light-Dark-ND",
        "continuous_navigation",
    ]

    normalized = normalize_legacy_resolved_config(aliased)
    normalized_tasks = normalized["environment"]["tasks"]

    # The alias normalization matches ``make_env`` rather than the literal name.
    assert normalized_tasks["Light-Dark-ND"]["goal_bonus"] == 0.0
    assert normalized_tasks["tiger"] == {"horizon": 20, "observation_noise_std": 0.1}
    assert normalized_tasks["continuous_navigation"] == {
        "horizon": 40,
        "observation_noise_std": 0.2,
    }


def test_light_dark_reward_keys_reach_the_environment_factory_arguments() -> None:
    config = _config_with_legacy_defaults()
    task = config["environment"]["tasks"]["light_dark"]  # type: ignore[index]
    task.update({"goal_bonus": 50.0, "goal_radius": 0.5, "light_position": 4.0})

    resolved = ExperimentConfig(config).environment_for_task("light_dark")

    # ``make_env`` forwards exactly this mapping into ``LightDarkNDEnv``.
    assert resolved["goal_bonus"] == 50.0
    assert resolved["goal_radius"] == 0.5
    assert resolved["light_position"] == 4.0
    assert make_env("light_dark", resolved, seed=0).goal_bonus == 50.0


def test_post_campaign_keys_stay_overridable_on_a_legacy_config() -> None:
    config = load_config(_PROJECT_ROOT / "config" / "local.json")
    assert config["ppo"]["bootstrap_on_truncation"] is False
    assert config["environment"]["tasks"]["light_dark"]["goal_bonus"] == 0.0
    assert config.with_overrides({"ppo.bootstrap_on_truncation": True})["ppo"][
        "bootstrap_on_truncation"
    ]
    assert (
        config.with_overrides({"environment.tasks.light_dark.goal_bonus": 50.0})[
            "environment"
        ]["tasks"]["light_dark"]["goal_bonus"]
        == 50.0
    )
    assert (
        apply_overrides(_valid_config(), {"ppo.bootstrap_on_truncation": True})["ppo"][
            "bootstrap_on_truncation"
        ]
        is True
    )


def test_json_parser_rejects_duplicate_and_nonstandard_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"experiment": {}, "experiment": {}}', encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate JSON key"):
        load_config(duplicate)

    nonstandard = tmp_path / "nan.json"
    nonstandard.write_text('{"experiment": NaN}', encoding="utf-8")
    with pytest.raises(ConfigError, match="non-standard JSON number"):
        load_config(nonstandard)


def test_artifact_tree_json_manifest_and_checkpoint_paths(tmp_path: Path) -> None:
    run = RunArtifacts.create(
        tmp_path / "results",
        "local smoke",
        timestamp="20260808T120000",
    )
    seed = run.for_seed("light_dark", 7)

    assert run.root.name == "20260808T120000_local-smoke"
    assert seed.path == run.root / "light_dark" / "seed_007"
    config_payload = {"seed": 7, "path": Path("relative/path")}
    metadata_payload = {"duration": np.float32(1.25)}
    seed.write_resolved_config(config_payload)
    seed.write_metadata(metadata_payload)
    run.write_manifest({"tasks": ["light_dark"], "complete": False})
    run.write_summary({"mean_return": 2.5})
    run.write_resolved_config(ExperimentConfig(_valid_config()))
    run.write_metadata({"host": "local"})

    assert json.loads(seed.resolved_config_path.read_text(encoding="utf-8")) == {
        "path": str(Path("relative/path")),
        "seed": 7,
    }
    assert json.loads(seed.metadata_path.read_text(encoding="utf-8"))["duration"] == 1.25
    assert json.loads(run.manifest_path.read_text(encoding="utf-8"))["tasks"] == ["light_dark"]
    assert json.loads(run.summary_json_path.read_text(encoding="utf-8"))["mean_return"] == 2.5
    assert seed.checkpoint_path(12) == seed.path / "checkpoints" / "step_000000012.pt"
    assert seed.torch_checkpoint_path(tag="best") == seed.path / "checkpoints" / "best.pt"
    assert seed.checkpoint_path() == seed.path / "checkpoints" / "latest.pt"


def test_explicit_run_root_can_be_created_reopened_and_atomically_rewritten(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign" / "shard001"
    created = RunArtifacts.create_at(root, "resumable shard")
    created.write_resolved_config({"version": 1})
    created.write_summary_csv(
        [{"status": "partial", "updates": 10}],
        ("status", "updates"),
    )

    reopened = RunArtifacts.open(root, "resumable shard")
    reopened.write_summary_csv(
        [{"status": "completed", "updates": 20}],
        ("status", "updates"),
    )

    assert reopened.root == created.root
    with reopened.summary_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"status": "completed", "updates": "20"}]
    with pytest.raises(FileExistsError):
        RunArtifacts.create_at(root, "resumable shard")
    with pytest.raises(ArtifactError, match="does not exist"):
        RunArtifacts.open(tmp_path / "missing", "resumable shard")


def test_csv_append_uses_and_enforces_fixed_header(tmp_path: Path) -> None:
    run = RunArtifacts.create(tmp_path, "csv", timestamp="20260808T120001")
    seed = run.for_seed("tiger", 0)
    appender = seed.csv_appender(("step", "return"))

    appender.append({"return": 1.5, "step": 1})
    appender.append_many(
        [
            {"step": 2, "return": 2.5},
            {"step": 3, "return": 3.5},
        ]
    )
    run.append_summary(
        {"task": "tiger", "mean_return": 2.0},
        ("task", "mean_return"),
    )

    with seed.metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"step": "1", "return": "1.5"},
        {"step": "2", "return": "2.5"},
        {"step": "3", "return": "3.5"},
    ]
    with pytest.raises(ArtifactError, match="missing=.*return"):
        appender.append({"step": 4})
    with pytest.raises(ArtifactError, match="header mismatch"):
        CsvAppender(seed.metrics_path, ("return", "step")).append({"step": 4, "return": 4.0})


def test_csv_append_many_validates_all_rows_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    appender = CsvAppender(path, ("step", "return"))
    appender.append({"step": 1, "return": 1.5})
    before = path.read_bytes()

    with pytest.raises(ArtifactError, match=r"row 1.*missing=.*return"):
        appender.append_many(
            [
                {"step": 2, "return": 2.5},
                {"step": 3},
            ]
        )

    assert path.read_bytes() == before
    assert appender.append_many([]) == path
    assert path.read_bytes() == before


def test_truncate_csv_rows_keeps_header_and_only_rewrites_when_rows_are_dropped(
    tmp_path: Path,
) -> None:
    run = RunArtifacts.create(tmp_path, "truncate", timestamp="20260818T120000")
    seed = run.for_seed("tiger", 1)
    appender = seed.csv_appender(("update", "value"))
    appender.append_many([{"update": index, "value": index / 2} for index in range(1, 4)])
    complete = seed.metrics_path.read_bytes()

    assert truncate_csv_rows(seed.metrics_path, lambda row: int(row["update"]) <= 3) == 0
    assert seed.metrics_path.read_bytes() == complete
    assert seed.truncate_csv(lambda row: int(row["update"]) <= 1) == 2
    appender.append({"update": 2, "value": 1.0})

    with seed.metrics_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"update": "1", "value": "0.5"},
            {"update": "2", "value": "1.0"},
        ]
    assert not list(seed.path.glob("._*.tmp"))
    assert truncate_csv_rows(seed.path / "missing.csv", lambda row: False) == 0


def test_truncate_csv_rows_drops_an_interrupted_final_row_and_rejects_torn_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episodes.csv"
    CsvAppender(path, ("global_step", "episode_return")).append_many(
        [
            {"global_step": 4, "episode_return": 1.0},
            {"global_step": 8, "episode_return": 2.0},
        ]
    )
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("12,")

    assert truncate_csv_rows(path, lambda row: int(row["global_step"]) <= 8) == 1
    with path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"global_step": "4", "episode_return": "1.0"},
            {"global_step": "8", "episode_return": "2.0"},
        ]

    torn = tmp_path / "torn.csv"
    torn.write_text("update,value\r\n1\r\n2,0.5\r\n", encoding="utf-8", newline="")
    with pytest.raises(ArtifactError, match="has 1 values; expected 2"):
        truncate_csv_rows(torn, lambda row: True)


def test_npz_never_requires_pickle(tmp_path: Path) -> None:
    run = RunArtifacts.create(tmp_path, "npz", timestamp="20260808T120002")
    seed = run.for_seed("continuous_navigation", 11)

    path = seed.save_npz(
        trajectories=np.arange(12, dtype=np.float32).reshape(3, 4),
        returns=[1.0, 2.0, 3.0],
    )
    with np.load(path, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["trajectories"], np.arange(12).reshape(3, 4))
        np.testing.assert_allclose(loaded["returns"], [1.0, 2.0, 3.0])

    with pytest.raises(ArtifactError, match="object dtype"):
        seed.save_npz({"unsafe": np.asarray([{"value": 1}], dtype=object)})
    with pytest.raises(ArtifactError, match="reserved"):
        seed.save_npz({"file": np.arange(2)})


def test_artifact_paths_cannot_escape_and_existing_runs_are_not_overwritten(
    tmp_path: Path,
) -> None:
    run = RunArtifacts.create(tmp_path, "safe", timestamp="20260808T120003")
    with pytest.raises(ArtifactError, match="unsafe task name"):
        run.for_seed("../escape", 0)
    with pytest.raises(ArtifactError, match="non-negative"):
        run.for_seed("tiger", -1)
    with pytest.raises(FileExistsError):
        RunArtifacts.create(tmp_path, "safe", timestamp="20260808T120003")

    created = run.for_seed("tiger", 0)
    with pytest.raises(FileExistsError):
        run.for_seed("tiger", 0)
    assert created.path.is_relative_to(run.root)


def test_comparison_artifacts_use_method_task_seed_hierarchy(tmp_path: Path) -> None:
    run = RunArtifacts.create(tmp_path, "comparison", timestamp="20260808T120004")
    seed = run.for_method_seed("score_transformer", "tiger", 7)

    assert seed.method == "score_transformer"
    assert seed.path.relative_to(run.root).as_posix() == "score_transformer/tiger/seed_007"
    with pytest.raises(ArtifactError, match="unsafe method"):
        run.for_method_seed("../escape", "tiger", 0)


def test_gradient_comparison_artifacts_use_four_level_hierarchy(tmp_path: Path) -> None:
    run = RunArtifacts.create(tmp_path, "comparison", timestamp="20260808T120005")
    seed = run.for_gradient_mode_method_seed("tbptt_1", "score_transformer", "tiger", 7)

    assert seed.gradient_mode == "tbptt_1"
    assert seed.method == "score_transformer"
    assert seed.task == "tiger"
    assert seed.seed == 7
    assert (
        seed.path.relative_to(run.root).as_posix()
        == "tbptt_1/score_transformer/tiger/seed_007"
    )
    with pytest.raises(ArtifactError, match="unsafe gradient mode"):
        run.for_gradient_mode_method_seed("../escape", "score_transformer", "tiger", 0)
    with pytest.raises(FileExistsError):
        run.for_gradient_mode_method_seed("tbptt_1", "score_transformer", "tiger", 7)


def test_experiment_config_copies_input() -> None:
    source = _valid_config()
    # ``_valid_config`` is written in the pre-campaign schema, so the stored copy
    # additionally carries every post-campaign key at its legacy default.
    expected = copy.deepcopy(_config_with_legacy_defaults())
    config = ExperimentConfig(source)
    source["model"]["num_particles"] = 999  # type: ignore[index]
    assert config.to_dict() == expected
