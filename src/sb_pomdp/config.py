"""Strict, dependency-free configuration loading for the experiments.

The configuration format deliberately has a small, closed schema.  This catches
misspelled hyperparameters before a (potentially expensive) cluster run starts.
Use :meth:`ExperimentConfig.with_overrides` for runtime changes rather than
mutating a JSON file in place.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class ConfigError(ValueError):
    """Raised when a configuration cannot be parsed or fails validation."""


_TOP_LEVEL_KEYS = frozenset({"experiment", "environment", "model", "ppo", "comparison"})
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?$")
_DEVICE = re.compile(r"^(?:auto|cpu|cuda(?::[0-9]+)?)$")

_EXPERIMENT_KEYS = frozenset(
    {
        "name",
        "label",
        "tasks",
        "seeds",
        "device",
        "deterministic",
        "output_dir",
        "eval_episodes",
        "eval_interval_updates",
        "checkpoint_interval_updates",
        "log_interval_updates",
    }
)

_ENVIRONMENT_KEYS = frozenset({"tasks"})
_ENVIRONMENT_TASK_KEYS = frozenset({"horizon", "observation_noise_std"})
_LIGHT_DARK_ALIASES = frozenset({"lightdark", "lightdarknd"})

# Light-Dark reward and geometry knobs promoted from hard-coded constants after
# the production campaign was submitted.  Every default is the constant that
# :class:`sb_pomdp.envs.LightDarkNDEnv` used at submission time, so a task block
# that omits them reproduces the campaign's reward exactly.  ``goal_bonus`` is
# the one genuinely new term: it is added once, at the terminating step.
_LIGHT_DARK_LEGACY_TASK_DEFAULTS: dict[str, JSONValue] = {
    "light_position": 5.0,
    "initial_mean": 2.0,
    "initial_std": 0.5,
    "state_cost": 0.5,
    "action_cost": 0.5,
    "goal_radius": 0.25,
    "goal_bonus": 0.0,
}
_LIGHT_DARK_TASK_KEYS = (
    _ENVIRONMENT_TASK_KEYS
    | frozenset({"dimension"})
    | frozenset(_LIGHT_DARK_LEGACY_TASK_DEFAULTS)
)

_COMPARISON_KEYS = frozenset(
    {"methods", "belief_gradient_modes", "gru_encoder_hidden", "gru_hidden_dim"}
)
BELIEF_GRADIENT_MODES = frozenset({"full", "tbptt_1"})
COMPARISON_METHODS = frozenset(
    {
        "score_transformer",
        "score_deepsets",
        "score_gaussian",
        "observation_mlp",
        "gru",
        "oracle_state",
    }
)

_MODEL_KEYS = frozenset(
    {
        "num_particles",
        "langevin_steps",
        "langevin_step_size",
        "particle_clip",
        "score_clip",
        "energy_hidden",
        "d_model",
        "num_heads",
        "num_transformer_layers",
        "transformer_ff_dim",
        "dropout",
        "policy_hidden",
        "diffusion_steps",
        "diffusion_beta_start",
        "diffusion_beta_end",
        "diffusion_min_std",
        "diffusion_advantage_discount",
        "encoder_kind",
        "continuous_policy_kind",
        "belief_gradient_mode",
    }
)

_PPO_KEYS = frozenset(
    {
        "total_steps",
        "num_envs",
        "rollout_steps",
        "epochs",
        "minibatch_size",
        "sequence_microbatch_size",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_coef",
        "value_coef",
        "entropy_coef",
        "max_grad_norm",
        "normalize_advantage",
        "bootstrap_on_truncation",
    }
)

# Top-level ``SECTION.KEY`` entries added after the production campaign was
# submitted.  Every entry maps to the value that reproduces the campaign's
# behaviour, so a configuration or a ``resolved_config.json`` written before the
# key existed is *equal* to a current one that leaves the key alone.  Extend
# this table - or ``_LIGHT_DARK_LEGACY_TASK_DEFAULTS`` for per-task keys, and
# nothing else - when a new key has to stay comparable with the campaign.
_LEGACY_DEFAULTS: dict[str, dict[str, JSONValue]] = {
    "ppo": {"bootstrap_on_truncation": False},
}


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise ConfigError(f"non-standard JSON number is not allowed: {value}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path} keys must be strings")
    return value


def _closed_keys(value: Any, path: str, expected: frozenset[str]) -> Mapping[str, Any]:
    section = _mapping(value, path)
    actual = set(section)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    problems: list[str] = []
    if missing:
        problems.append(f"missing keys {missing}")
    if unknown:
        problems.append(f"unknown keys {unknown}")
    if problems:
        raise ConfigError(f"{path}: " + "; ".join(problems))
    return section


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if value < minimum:
        comparator = "positive" if minimum == 1 else f">= {minimum}"
        raise ConfigError(f"{path} must be {comparator}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{path} must be <= {maximum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path} must be finite")
    if minimum is not None:
        bad = result < minimum if minimum_inclusive else result <= minimum
        if bad:
            operator = ">=" if minimum_inclusive else ">"
            raise ConfigError(f"{path} must be {operator} {minimum}")
    if maximum is not None:
        bad = result > maximum if maximum_inclusive else result >= maximum
        if bad:
            operator = "<=" if maximum_inclusive else "<"
            raise ConfigError(f"{path} must be {operator} {maximum}")
    return result


def _positive_number(value: Any, path: str) -> float:
    return _number(value, path, minimum=0.0, minimum_inclusive=False)


def _positive_integer_list(value: Any, path: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty array")
    return [_integer(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _validate_experiment(value: Any) -> Mapping[str, Any]:
    section = _closed_keys(value, "experiment", _EXPERIMENT_KEYS)
    _nonempty_string(section["name"], "experiment.name")
    _nonempty_string(section["label"], "experiment.label")
    device = _nonempty_string(section["device"], "experiment.device").lower()
    if not _DEVICE.fullmatch(device):
        raise ConfigError("experiment.device must be auto, cpu, cuda, or cuda:<index>")
    _nonempty_string(section["output_dir"], "experiment.output_dir")
    _boolean(section["deterministic"], "experiment.deterministic")

    tasks = section["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ConfigError("experiment.tasks must be a non-empty array")
    task_names = [
        _nonempty_string(task, f"experiment.tasks[{index}]") for index, task in enumerate(tasks)
    ]
    if any(not _SAFE_COMPONENT.fullmatch(task) for task in task_names):
        raise ConfigError("experiment.tasks entries must be safe filename components")
    if len(task_names) != len(set(task_names)):
        raise ConfigError("experiment.tasks must not contain duplicates")

    seeds = section["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("experiment.seeds must be a non-empty array")
    checked_seeds = [
        _integer(seed, f"experiment.seeds[{index}]", minimum=0, maximum=2**32 - 1)
        for index, seed in enumerate(seeds)
    ]
    if len(checked_seeds) != len(set(checked_seeds)):
        raise ConfigError("experiment.seeds must not contain duplicates")

    for key in (
        "eval_episodes",
        "eval_interval_updates",
        "checkpoint_interval_updates",
        "log_interval_updates",
    ):
        _integer(section[key], f"experiment.{key}")
    return section


def _normalized_task_name(task_name: str) -> str:
    """Return the alias-insensitive task key used by :func:`sb_pomdp.envs.make_env`."""

    return "".join(character for character in task_name.lower() if character.isalnum())


def _validate_light_dark_task(task: Mapping[str, Any], path: str) -> None:
    """Validate the Light-Dark specific reward and geometry keys.

    The bounds mirror :class:`sb_pomdp.envs.LightDarkNDEnv` so an unusable value
    fails here rather than after the run directory has been created.
    """

    _integer(task["dimension"], f"{path}.dimension", maximum=64)
    _number(task["light_position"], f"{path}.light_position")
    _number(task["initial_mean"], f"{path}.initial_mean")
    _number(task["initial_std"], f"{path}.initial_std", minimum=0.0)
    for key in ("state_cost", "action_cost", "goal_bonus"):
        _number(task[key], f"{path}.{key}", minimum=0.0)
    _positive_number(task["goal_radius"], f"{path}.goal_radius")


def _validate_environment(value: Any, requested_tasks: Sequence[str]) -> Mapping[str, Any]:
    section = _closed_keys(value, "environment", _ENVIRONMENT_KEYS)
    tasks = _mapping(section["tasks"], "environment.tasks")
    if not tasks:
        raise ConfigError("environment.tasks must not be empty")

    for task_name, task_value in tasks.items():
        _nonempty_string(task_name, "environment.tasks key")
        if not _SAFE_COMPONENT.fullmatch(task_name):
            raise ConfigError("environment task names must be safe filename components")
        is_light_dark = _normalized_task_name(task_name) in _LIGHT_DARK_ALIASES
        expected_keys = _LIGHT_DARK_TASK_KEYS if is_light_dark else _ENVIRONMENT_TASK_KEYS
        path = f"environment.tasks.{task_name}"
        task = _closed_keys(task_value, path, expected_keys)
        _integer(task["horizon"], f"{path}.horizon")
        _number(
            task["observation_noise_std"],
            f"{path}.observation_noise_std",
            minimum=0.0,
        )
        if is_light_dark:
            _validate_light_dark_task(task, path)

    missing = [task for task in requested_tasks if task not in tasks]
    if missing:
        raise ConfigError(f"experiment.tasks are missing from environment.tasks: {missing}")
    return section


def _validate_model(value: Any) -> Mapping[str, Any]:
    section = _closed_keys(value, "model", _MODEL_KEYS)
    for key in (
        "num_particles",
        "langevin_steps",
        "d_model",
        "num_heads",
        "num_transformer_layers",
        "transformer_ff_dim",
        "diffusion_steps",
    ):
        _integer(section[key], f"model.{key}")
    for key in ("langevin_step_size", "diffusion_min_std"):
        _positive_number(section[key], f"model.{key}")
    for key in ("particle_clip", "score_clip"):
        _number(section[key], f"model.{key}", minimum=0.0)
    _positive_integer_list(section["energy_hidden"], "model.energy_hidden")
    _positive_integer_list(section["policy_hidden"], "model.policy_hidden")
    dropout = _number(
        section["dropout"],
        "model.dropout",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    if dropout != 0.0:
        raise ConfigError(
            "model.dropout must be 0.0 because PPO re-evaluates saved policy likelihoods"
        )
    beta_start = _number(
        section["diffusion_beta_start"],
        "model.diffusion_beta_start",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    beta_end = _number(
        section["diffusion_beta_end"],
        "model.diffusion_beta_end",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    if beta_start >= beta_end:
        raise ConfigError("model.diffusion_beta_start must be less than model.diffusion_beta_end")
    _number(
        section["diffusion_advantage_discount"],
        "model.diffusion_advantage_discount",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    if section["d_model"] % section["num_heads"] != 0:
        raise ConfigError("model.d_model must be divisible by model.num_heads")
    encoder_kind = _nonempty_string(section["encoder_kind"], "model.encoder_kind")
    if encoder_kind not in {"transformer", "deep_sets"}:
        raise ConfigError("model.encoder_kind must be transformer or deep_sets")
    continuous_policy_kind = _nonempty_string(
        section["continuous_policy_kind"], "model.continuous_policy_kind"
    )
    if continuous_policy_kind not in {"diffusion", "gaussian"}:
        raise ConfigError("model.continuous_policy_kind must be diffusion or gaussian")
    belief_gradient_mode = _nonempty_string(
        section["belief_gradient_mode"], "model.belief_gradient_mode"
    )
    if belief_gradient_mode not in BELIEF_GRADIENT_MODES:
        raise ConfigError("model.belief_gradient_mode must be full or tbptt_1")
    return section


def _validate_ppo(value: Any) -> Mapping[str, Any]:
    section = _closed_keys(value, "ppo", _PPO_KEYS)
    for key in (
        "total_steps",
        "num_envs",
        "rollout_steps",
        "epochs",
        "minibatch_size",
        "sequence_microbatch_size",
    ):
        _integer(section[key], f"ppo.{key}")
    for key in ("learning_rate", "clip_coef", "value_coef", "max_grad_norm"):
        _positive_number(section[key], f"ppo.{key}")
    for key in ("gamma", "gae_lambda"):
        _number(
            section[key],
            f"ppo.{key}",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
    _number(section["entropy_coef"], "ppo.entropy_coef", minimum=0.0)
    _boolean(section["normalize_advantage"], "ppo.normalize_advantage")
    _boolean(section["bootstrap_on_truncation"], "ppo.bootstrap_on_truncation")

    rollout_batch = section["num_envs"] * section["rollout_steps"]
    if section["minibatch_size"] > rollout_batch:
        raise ConfigError("ppo.minibatch_size must not exceed num_envs * rollout_steps")
    if rollout_batch % section["minibatch_size"] != 0:
        raise ConfigError("num_envs * rollout_steps must be divisible by ppo.minibatch_size")
    if section["minibatch_size"] < section["rollout_steps"]:
        raise ConfigError(
            "ppo.minibatch_size must be at least ppo.rollout_steps for belief sequence replay"
        )
    if section["minibatch_size"] % section["rollout_steps"] != 0:
        raise ConfigError(
            "ppo.minibatch_size must be divisible by ppo.rollout_steps for belief sequence replay"
        )
    if section["sequence_microbatch_size"] < section["rollout_steps"]:
        raise ConfigError("ppo.sequence_microbatch_size must be at least ppo.rollout_steps")
    if section["sequence_microbatch_size"] > section["minibatch_size"]:
        raise ConfigError("ppo.sequence_microbatch_size must not exceed ppo.minibatch_size")
    if section["sequence_microbatch_size"] % section["rollout_steps"] != 0:
        raise ConfigError("ppo.sequence_microbatch_size must be divisible by ppo.rollout_steps")
    if section["minibatch_size"] % section["sequence_microbatch_size"] != 0:
        raise ConfigError("ppo.minibatch_size must be divisible by ppo.sequence_microbatch_size")
    if section["total_steps"] < rollout_batch:
        raise ConfigError("ppo.total_steps must be at least num_envs * rollout_steps")
    if section["total_steps"] % rollout_batch != 0:
        raise ConfigError("ppo.total_steps must be divisible by num_envs * rollout_steps")
    return section


def _validate_comparison(
    value: Any,
    *,
    model_gradient_mode: str,
) -> Mapping[str, Any]:
    section = _closed_keys(value, "comparison", _COMPARISON_KEYS)
    methods = section["methods"]
    if not isinstance(methods, list) or not methods:
        raise ConfigError("comparison.methods must be a non-empty array")
    checked_methods = [
        _nonempty_string(method, f"comparison.methods[{index}]")
        for index, method in enumerate(methods)
    ]
    unknown = sorted(set(checked_methods) - COMPARISON_METHODS)
    if unknown:
        raise ConfigError(f"comparison.methods contains unknown methods: {unknown}")
    if len(checked_methods) != len(set(checked_methods)):
        raise ConfigError("comparison.methods must not contain duplicates")
    gradient_modes = section["belief_gradient_modes"]
    if not isinstance(gradient_modes, list) or not gradient_modes:
        raise ConfigError("comparison.belief_gradient_modes must be a non-empty array")
    checked_gradient_modes = [
        _nonempty_string(mode, f"comparison.belief_gradient_modes[{index}]")
        for index, mode in enumerate(gradient_modes)
    ]
    unknown_gradient_modes = sorted(set(checked_gradient_modes) - BELIEF_GRADIENT_MODES)
    if unknown_gradient_modes:
        raise ConfigError(
            f"comparison.belief_gradient_modes contains unknown modes: {unknown_gradient_modes}"
        )
    if len(checked_gradient_modes) != len(set(checked_gradient_modes)):
        raise ConfigError("comparison.belief_gradient_modes must not contain duplicates")
    if model_gradient_mode not in checked_gradient_modes:
        raise ConfigError(
            "model.belief_gradient_mode must be enabled by comparison.belief_gradient_modes"
        )
    _positive_integer_list(
        section["gru_encoder_hidden"],
        "comparison.gru_encoder_hidden",
    )
    _integer(section["gru_hidden_dim"], "comparison.gru_hidden_dim")
    return section


def _fill_light_dark_task_defaults(config: dict[str, Any]) -> None:
    """Insert the Light-Dark post-campaign task keys in *config*, in place."""

    environment = config.get("environment")
    if not isinstance(environment, dict):
        return
    tasks = environment.get("tasks")
    if not isinstance(tasks, dict):
        return
    for task_name, task in tasks.items():
        if not isinstance(task_name, str) or not isinstance(task, dict):
            continue
        if _normalized_task_name(task_name) not in _LIGHT_DARK_ALIASES:
            continue
        for key, default in _LIGHT_DARK_LEGACY_TASK_DEFAULTS.items():
            task.setdefault(key, deepcopy(default))


def normalize_legacy_resolved_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of *config* with post-campaign keys at their defaults.

    The schema is closed, so every key it knows about must be present.  A
    configuration file or a ``resolved_config.json`` artifact written before a
    key was introduced simply lacks it.  Filling in ``_LEGACY_DEFAULTS`` and
    ``_LIGHT_DARK_LEGACY_TASK_DEFAULTS`` here - and only here - keeps such an
    artifact loadable and makes it compare *equal* to a current configuration
    that never touched the key, which is what the resume identity check in
    :mod:`sb_pomdp.compare` and the exact-match validation in
    :mod:`sb_pomdp.aggregate` require of the running campaign.  Sections that
    are absent or malformed are left untouched for the schema validator to
    report.
    """

    if not isinstance(config, Mapping):
        raise ConfigError("config must be an object")
    result = deepcopy(dict(config))
    for section, defaults in _LEGACY_DEFAULTS.items():
        values = result.get(section)
        if not isinstance(values, dict):
            continue
        for key, default in defaults.items():
            values.setdefault(key, deepcopy(default))
    _fill_light_dark_task_defaults(result)
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate *config* against the complete experiment schema.

    Keys listed in ``_LEGACY_DEFAULTS`` or ``_LIGHT_DARK_LEGACY_TASK_DEFAULTS``
    may be omitted; they are validated at their legacy default.  ``ConfigError``
    messages include the failing dotted path so errors remain actionable when a
    configuration is supplied by a batch job.
    """

    root = _closed_keys(normalize_legacy_resolved_config(config), "config", _TOP_LEVEL_KEYS)
    experiment = _validate_experiment(root["experiment"])
    _validate_environment(root["environment"], experiment["tasks"])
    model = _validate_model(root["model"])
    _validate_ppo(root["ppo"])
    _validate_comparison(
        root["comparison"],
        model_gradient_mode=str(model["belief_gradient_mode"]),
    )


def parse_cli_overrides(items: Sequence[str]) -> dict[str, Any]:
    """Parse ``SECTION.KEY=VALUE`` command-line values.

    Right-hand sides use JSON syntax when possible (numbers, booleans, arrays,
    and objects); otherwise they are treated as plain strings.  The resulting
    mapping is suitable for :func:`apply_overrides` or :func:`load_config`.
    """

    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"override must have PATH=VALUE form: {item!r}")
        path, raw_value = item.split("=", 1)
        path = path.strip()
        if not path or any(not part for part in path.split(".")):
            raise ConfigError(f"invalid override path: {path!r}")
        if path in parsed:
            raise ConfigError(f"duplicate override path: {path!r}")
        try:
            value = json.loads(
                raw_value,
                object_pairs_hook=_duplicate_checked_object,
                parse_constant=_reject_nonstandard_number,
            )
        except json.JSONDecodeError:
            value = raw_value
        parsed[path] = value
    return parsed


def _set_existing_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if any(not part for part in parts):
        raise ConfigError(f"invalid override path: {path!r}")
    current: Any = config
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"override path does not exist: {path!r}")
        current = current[part]
    leaf = parts[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise ConfigError(f"override path does not exist: {path!r}")
    current[leaf] = deepcopy(value)


def _flatten_nested_overrides(
    value: Mapping[str, Any], prefix: str = ""
) -> Iterator[tuple[str, Any]]:
    for key, nested_value in value.items():
        if not isinstance(key, str) or not key:
            raise ConfigError("override keys must be non-empty strings")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(nested_value, Mapping):
            yield from _flatten_nested_overrides(nested_value, path)
        else:
            yield path, nested_value


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated deep copy of *config* with runtime overrides.

    Overrides may be dotted (``{"ppo.learning_rate": 1e-4}``) or nested
    (``{"ppo": {"learning_rate": 1e-4}}``).  Only existing paths can be
    replaced, which keeps misspellings fail-fast.  Legacy defaults are filled in
    first so a post-campaign key stays overridable on an older configuration.
    """

    result = normalize_legacy_resolved_config(config)
    pairs: list[tuple[str, Any]] = []
    for key, value in overrides.items():
        if not isinstance(key, str):
            raise ConfigError("override keys must be strings")
        if "." in key or not isinstance(value, Mapping):
            pairs.append((key, value))
        else:
            pairs.extend(_flatten_nested_overrides(value, key))
    for path, value in pairs:
        _set_existing_dotted(result, path, value)
    validate_config(result)
    return result


class ExperimentConfig(Mapping[str, Any]):
    """Validated configuration with mapping-compatible access."""

    def __init__(self, data: Mapping[str, Any], *, source: Path | None = None) -> None:
        copied = normalize_legacy_resolved_config(dict(data))
        validate_config(copied)
        self._data = copied
        self.source = source

    def __getitem__(self, key: str) -> Any:
        # Nested dictionaries are mutable; returning them directly would let
        # callers bypass the validation performed at construction time.
        return deepcopy(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        source = f", source={str(self.source)!r}" if self.source is not None else ""
        return f"ExperimentConfig({self._data!r}{source})"

    def to_dict(self) -> dict[str, Any]:
        """Return a deep mutable copy suitable for JSON serialization."""

        return deepcopy(self._data)

    def environment_for_task(self, task_name: str) -> dict[str, Any]:
        """Return a copy of one task's environment settings."""

        try:
            task = self._data["environment"]["tasks"][task_name]
        except KeyError as exc:
            available = sorted(self._data["environment"]["tasks"])
            raise ConfigError(f"unknown task {task_name!r}; available tasks: {available}") from exc
        return deepcopy(task)

    def for_task(self, task_name: str) -> dict[str, Any]:
        """Return the full config with an explicit selected-environment view.

        The complete task catalogue remains under ``environment.tasks``.  The
        selected settings are additionally exposed as ``environment.selected``
        with a ``name`` field.  This is a transient convenience view rather than
        a schema-valid resolved configuration artifact.
        """

        result = self.to_dict()
        selected = {"name": task_name, **self.environment_for_task(task_name)}
        result["environment"]["selected"] = selected
        return result

    def with_overrides(self, overrides: Mapping[str, Any] | Sequence[str]) -> ExperimentConfig:
        """Return a new config with validated runtime overrides applied."""

        normalized = (
            parse_cli_overrides(overrides) if not isinstance(overrides, Mapping) else overrides
        )
        return ExperimentConfig(apply_overrides(self._data, normalized), source=self.source)


def load_config(
    path: str | Path,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
) -> ExperimentConfig:
    """Load strict JSON from *path* and return a validated configuration."""

    source = Path(path).expanduser()
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read config {source}: {exc}") from exc
    try:
        data = json.loads(
            raw,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_nonstandard_number,
        )
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"invalid JSON in {source} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError("config must be a JSON object")

    config = ExperimentConfig(data, source=source.resolve())
    if overrides is not None:
        config = config.with_overrides(overrides)
    return config


__all__ = [
    "BELIEF_GRADIENT_MODES",
    "COMPARISON_METHODS",
    "ConfigError",
    "ExperimentConfig",
    "apply_overrides",
    "load_config",
    "normalize_legacy_resolved_config",
    "parse_cli_overrides",
    "validate_config",
]
