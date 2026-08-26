"""Small, dependency-free partially observed classic-control environments.

The environments in this module intentionally expose the true simulator state only
through ``info["state"]``.  The agent observation is a noisy, masked projection of
that state.  Their API follows the useful subset of the Gymnasium API, without
requiring Gymnasium as a dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True, eq=False)
class ActionSpec:
    """Metadata and NumPy-only validation/encoding for an action space."""

    kind: Literal["discrete", "continuous"]
    shape: tuple[int, ...]
    n: int | None = None
    low: np.ndarray | None = None
    high: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.kind == "discrete":
            if (
                self.n is None
                or isinstance(self.n, (bool, np.bool_))
                or not isinstance(self.n, (int, np.integer))
                or self.n < 2
            ):
                raise ValueError("A discrete ActionSpec requires n >= 2")
            object.__setattr__(self, "n", int(self.n))
            if self.shape != ():
                raise ValueError("A discrete ActionSpec must have scalar shape ()")
            if self.low is not None or self.high is not None:
                raise ValueError("A discrete ActionSpec does not use low/high bounds")
            return

        if self.kind != "continuous":
            raise ValueError(f"Unknown action kind: {self.kind!r}")
        if not self.shape or any(d <= 0 for d in self.shape):
            raise ValueError("A continuous ActionSpec needs a non-empty positive shape")
        if self.low is None or self.high is None:
            raise ValueError("A continuous ActionSpec requires low/high bounds")

        low = np.asarray(self.low, dtype=np.float32)
        high = np.asarray(self.high, dtype=np.float32)
        if low.shape == ():
            low = np.full(self.shape, low.item(), dtype=np.float32)
        if high.shape == ():
            high = np.full(self.shape, high.item(), dtype=np.float32)
        if low.shape != self.shape or high.shape != self.shape:
            raise ValueError("Action bounds must be scalar or match shape")
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("Action bounds must be finite")
        if np.any(low >= high):
            raise ValueError("Every low action bound must be smaller than high")

        low = low.copy()
        high = high.copy()
        low.setflags(write=False)
        high.setflags(write=False)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @classmethod
    def discrete(cls, n: int) -> ActionSpec:
        return cls(kind="discrete", shape=(), n=n)

    @classmethod
    def continuous(
        cls,
        low: float | np.ndarray,
        high: float | np.ndarray,
        shape: tuple[int, ...] = (1,),
    ) -> ActionSpec:
        return cls(kind="continuous", shape=shape, low=np.asarray(low), high=np.asarray(high))

    @property
    def is_discrete(self) -> bool:
        return self.kind == "discrete"

    @property
    def is_continuous(self) -> bool:
        return self.kind == "continuous"

    @property
    def feature_dim(self) -> int:
        """Width produced by :meth:`encode` for one action."""

        if self.is_discrete:
            assert self.n is not None
            return self.n
        return int(np.prod(self.shape))

    def validate(self, action: Any) -> int | np.ndarray:
        """Validate one action and return it in canonical simulator form.

        Discrete actions become a Python ``int``.  Continuous actions become a
        float32 array with exactly ``shape``.  Invalid values raise ``ValueError``.
        """

        if self.is_discrete:
            array = np.asarray(action)
            if array.shape != () or np.issubdtype(array.dtype, np.bool_):
                raise ValueError("A discrete action must be one integer scalar")
            try:
                value_float = float(array)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("A discrete action must be numeric") from exc
            if not np.isfinite(value_float) or value_float != np.floor(value_float):
                raise ValueError("A discrete action must be an integer")
            value = int(value_float)
            assert self.n is not None
            if not 0 <= value < self.n:
                raise ValueError(f"Discrete action {value} is outside [0, {self.n})")
            return value

        try:
            array = np.asarray(action, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("A continuous action must be numeric") from exc
        if array.shape == () and self.shape == (1,):
            array = array.reshape(1)
        if array.shape != self.shape:
            raise ValueError(f"Continuous action must have shape {self.shape}, got {array.shape}")
        assert self.low is not None and self.high is not None
        if not np.all(np.isfinite(array)):
            raise ValueError("Continuous action must be finite")
        if np.any(array < self.low) or np.any(array > self.high):
            raise ValueError("Continuous action is outside its bounds")
        return array.copy()

    def contains(self, action: Any) -> bool:
        """Return whether ``action`` passes :meth:`validate`."""

        try:
            self.validate(action)
        except (TypeError, ValueError):
            return False
        return True

    def encode(self, action_array: Any) -> np.ndarray:
        """Encode scalar/batched actions for use as model features.

        Discrete values are one-hot encoded.  Continuous values are affinely
        normalized from their declared bounds to ``[-1, 1]``.
        """

        if self.is_discrete:
            raw = np.asarray(action_array)
            if np.issubdtype(raw.dtype, np.bool_):
                raise ValueError("Boolean values are not discrete actions")
            try:
                numeric = raw.astype(np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Discrete actions must be numeric") from exc
            if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
                raise ValueError("Discrete actions must contain integers")
            indices = numeric.astype(np.int64)
            assert self.n is not None
            if np.any(indices < 0) or np.any(indices >= self.n):
                raise ValueError(f"Discrete actions must be in [0, {self.n})")
            return np.eye(self.n, dtype=np.float32)[indices]

        try:
            values = np.asarray(action_array, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Continuous actions must be numeric") from exc
        if values.shape == () and self.shape == (1,):
            values = values.reshape(1)
        action_ndim = len(self.shape)
        if values.ndim < action_ndim or values.shape[-action_ndim:] != self.shape:
            raise ValueError(
                f"Continuous actions must end in shape {self.shape}, got {values.shape}"
            )
        assert self.low is not None and self.high is not None
        if not np.all(np.isfinite(values)):
            raise ValueError("Continuous actions must be finite")
        if np.any(values < self.low) or np.any(values > self.high):
            raise ValueError("Continuous actions are outside their bounds")
        encoded = 2.0 * (values - self.low) / (self.high - self.low) - 1.0
        return np.asarray(encoded, dtype=np.float32)

    def sample(self, rng: np.random.Generator) -> int | np.ndarray:
        """Sample one valid action with the supplied generator."""

        if self.is_discrete:
            assert self.n is not None
            return int(rng.integers(self.n))
        assert self.low is not None and self.high is not None
        return rng.uniform(self.low, self.high).astype(np.float32)


class _BaseMaskedEnv:
    """Shared episode bookkeeping for the simulators."""

    state_dim: int
    observation_dim: int
    oracle_dim: int
    action_spec: ActionSpec

    def __init__(
        self,
        *,
        horizon: int,
        observation_noise_std: float = 0.05,
        seed: int | None = None,
    ) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
            raise TypeError("horizon must be a positive integer")
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if not np.isfinite(observation_noise_std) or observation_noise_std < 0:
            raise ValueError("observation_noise_std must be finite and non-negative")
        self.horizon = int(horizon)
        self.observation_noise_std = float(observation_noise_std)
        self._rng = np.random.default_rng(seed)
        self._state: np.ndarray | None = None
        self._elapsed_steps = 0
        self._needs_reset = True

    @property
    def state(self) -> np.ndarray:
        """A defensive copy of the current latent state."""

        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        return self._state.copy()

    def state_dict(self) -> dict[str, Any]:
        """Return all mutable simulator state needed for an exact continuation."""

        return {
            "format_version": 1,
            "environment_type": type(self).__qualname__,
            "state": None if self._state is None else self._state.copy(),
            "elapsed_steps": self._elapsed_steps,
            "needs_reset": self._needs_reset,
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a :meth:`state_dict` after validating it without partial mutation."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("environment state_dict must be a mapping")
        required = {
            "format_version",
            "environment_type",
            "state",
            "elapsed_steps",
            "needs_reset",
            "rng_state",
        }
        missing = required - set(state_dict)
        extra = set(state_dict) - required
        if missing or extra:
            raise ValueError(
                "invalid environment state_dict keys; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if state_dict["format_version"] != 1:
            raise ValueError(
                "unsupported environment state_dict format_version: "
                f"{state_dict['format_version']!r}"
            )
        expected_type = type(self).__qualname__
        if state_dict["environment_type"] != expected_type:
            raise ValueError(
                "environment state_dict type mismatch: "
                f"expected {expected_type!r}, got {state_dict['environment_type']!r}"
            )

        elapsed_steps = state_dict["elapsed_steps"]
        if (
            isinstance(elapsed_steps, (bool, np.bool_))
            or not isinstance(elapsed_steps, (int, np.integer))
            or not 0 <= int(elapsed_steps) <= self.horizon
        ):
            raise ValueError("environment elapsed_steps must be an integer within the horizon")
        elapsed_steps = int(elapsed_steps)
        needs_reset = state_dict["needs_reset"]
        if not isinstance(needs_reset, (bool, np.bool_)):
            raise TypeError("environment needs_reset must be boolean")
        needs_reset = bool(needs_reset)

        raw_state = state_dict["state"]
        restored_state: np.ndarray | None
        if raw_state is None:
            if elapsed_steps != 0 or not needs_reset:
                raise ValueError(
                    "an uninitialized environment must have elapsed_steps=0 and needs_reset=True"
                )
            restored_state = None
        else:
            if not isinstance(raw_state, np.ndarray):
                raise ValueError("environment latent state must be a NumPy array or None")
            if raw_state.shape != (self.state_dim,):
                raise ValueError(
                    "environment latent state has the wrong shape: "
                    f"expected {(self.state_dim,)}, got {raw_state.shape}"
                )
            if raw_state.dtype != np.float64 or not np.all(np.isfinite(raw_state)):
                raise ValueError("environment latent state must be a finite float64 array")
            if elapsed_steps == self.horizon and not needs_reset:
                raise ValueError("an environment at its horizon must require reset")
            restored_state = raw_state.copy()

        rng_state = state_dict["rng_state"]
        if not isinstance(rng_state, Mapping):
            raise TypeError("environment rng_state must be a mapping")
        expected_bit_generator = type(self._rng.bit_generator).__name__
        if rng_state.get("bit_generator") != expected_bit_generator:
            raise ValueError(
                "environment RNG bit generator mismatch: "
                f"expected {expected_bit_generator!r}, got {rng_state.get('bit_generator')!r}"
            )
        try:
            bit_generator = type(self._rng.bit_generator)()
            bit_generator.state = deepcopy(dict(rng_state))
            restored_rng = np.random.Generator(bit_generator)
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise ValueError("invalid environment rng_state") from exc

        self._state = restored_state
        self._elapsed_steps = elapsed_steps
        self._needs_reset = needs_reset
        self._rng = restored_rng

    def _start_reset(self, seed: int | None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._elapsed_steps = 0
        self._needs_reset = False

    def _before_step(self) -> None:
        if self._needs_reset or self._state is None:
            raise RuntimeError("Call reset() before step(), and after an episode ends")

    def _finish_step(self, terminated: bool) -> tuple[bool, bool]:
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.horizon and not terminated
        self._needs_reset = bool(terminated or truncated)
        return bool(terminated), bool(truncated)

    def _info(self) -> dict[str, Any]:
        assert self._state is not None
        return {"state": self._state.copy(), "elapsed_steps": self._elapsed_steps}

    def _add_observation_noise(self, values: np.ndarray) -> np.ndarray:
        if self.observation_noise_std:
            values = values + self._rng.normal(
                loc=0.0,
                scale=self.observation_noise_std,
                size=values.shape,
            )
        return np.asarray(values, dtype=np.float32)


class MaskedCartPoleEnv(_BaseMaskedEnv):
    """Cart-pole with only normalized cart position and pole angle observed."""

    state_dim = 4
    observation_dim = 2
    oracle_dim = 4

    def __init__(
        self,
        *,
        horizon: int = 500,
        observation_noise_std: float = 0.05,
        seed: int | None = None,
        gravity: float = 9.8,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        half_length: float = 0.5,
        force_mag: float = 10.0,
        tau: float = 0.02,
        x_threshold: float = 2.4,
        theta_threshold_radians: float = 12.0 * np.pi / 180.0,
    ) -> None:
        super().__init__(
            horizon=horizon,
            observation_noise_std=observation_noise_std,
            seed=seed,
        )
        positive = {
            "gravity": gravity,
            "mass_cart": mass_cart,
            "mass_pole": mass_pole,
            "half_length": half_length,
            "force_mag": force_mag,
            "tau": tau,
            "x_threshold": x_threshold,
            "theta_threshold_radians": theta_threshold_radians,
        }
        if any(not np.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError("CartPole physical parameters and thresholds must be positive")
        self.gravity = float(gravity)
        self.mass_cart = float(mass_cart)
        self.mass_pole = float(mass_pole)
        self.total_mass = self.mass_cart + self.mass_pole
        self.half_length = float(half_length)
        self.polemass_length = self.mass_pole * self.half_length
        self.force_mag = float(force_mag)
        self.tau = float(tau)
        self.x_threshold = float(x_threshold)
        self.theta_threshold_radians = float(theta_threshold_radians)
        self.action_spec = ActionSpec.discrete(2)

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        x, _, theta, _ = self._state
        signal = np.array(
            [x / self.x_threshold, theta / self.theta_threshold_radians],
            dtype=np.float64,
        )
        return self._add_observation_noise(signal)

    def oracle_features(self) -> np.ndarray:
        """Return bounded, noise-free features of the complete simulator state."""

        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        x, x_dot, theta, theta_dot = self._state
        return np.asarray(
            [
                np.clip(x / self.x_threshold, -1.0, 1.0),
                np.tanh(x_dot),
                np.clip(theta / self.theta_threshold_radians, -1.0, 1.0),
                np.tanh(theta_dot),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._start_reset(seed)
        self._state = self._rng.uniform(low=-0.05, high=0.05, size=4).astype(np.float64)
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._before_step()
        selected = self.action_spec.validate(action)
        assert isinstance(selected, int) and self._state is not None
        x, x_dot, theta, theta_dot = self._state
        force = self.force_mag if selected == 1 else -self.force_mag
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.half_length * (4.0 / 3.0 - self.mass_pole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        self._state = np.array(
            [
                x + self.tau * x_dot,
                x_dot + self.tau * x_acc,
                theta + self.tau * theta_dot,
                theta_dot + self.tau * theta_acc,
            ],
            dtype=np.float64,
        )
        new_x, _, new_theta, _ = self._state
        terminated = bool(
            abs(new_x) > self.x_threshold or abs(new_theta) > self.theta_threshold_radians
        )
        terminated, truncated = self._finish_step(terminated)
        return self._observation(), 1.0, terminated, truncated, self._info()


class MaskedPendulumEnv(_BaseMaskedEnv):
    """Pendulum with angular velocity hidden from the observation."""

    state_dim = 2
    observation_dim = 2
    oracle_dim = 3

    def __init__(
        self,
        *,
        horizon: int = 200,
        observation_noise_std: float = 0.05,
        seed: int | None = None,
        max_speed: float = 8.0,
        max_torque: float = 2.0,
        dt: float = 0.05,
        gravity: float = 10.0,
        mass: float = 1.0,
        length: float = 1.0,
    ) -> None:
        super().__init__(
            horizon=horizon,
            observation_noise_std=observation_noise_std,
            seed=seed,
        )
        positive = {
            "max_speed": max_speed,
            "max_torque": max_torque,
            "dt": dt,
            "gravity": gravity,
            "mass": mass,
            "length": length,
        }
        if any(not np.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError("Pendulum physical parameters and limits must be positive")
        self.max_speed = float(max_speed)
        self.max_torque = float(max_torque)
        self.dt = float(dt)
        self.gravity = float(gravity)
        self.mass = float(mass)
        self.length = float(length)
        self.action_spec = ActionSpec.continuous(-self.max_torque, self.max_torque)

    @staticmethod
    def _normalize_angle(theta: float) -> float:
        return float((theta + np.pi) % (2.0 * np.pi) - np.pi)

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        theta = self._state[0]
        signal = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        return self._add_observation_noise(signal)

    def oracle_features(self) -> np.ndarray:
        """Return angle-periodic, normalized full-state features without noise."""

        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        theta, theta_dot = self._state
        return np.asarray(
            [np.cos(theta), np.sin(theta), theta_dot / self.max_speed],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._start_reset(seed)
        self._state = np.array(
            [self._rng.uniform(-np.pi, np.pi), self._rng.uniform(-1.0, 1.0)],
            dtype=np.float64,
        )
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._before_step()
        selected = self.action_spec.validate(action)
        assert isinstance(selected, np.ndarray) and self._state is not None
        torque = float(selected[0])
        theta, theta_dot = self._state
        cost = self._normalize_angle(theta) ** 2 + 0.1 * theta_dot**2 + 0.001 * torque**2
        acceleration = 3.0 * self.gravity / (2.0 * self.length) * np.sin(theta) + 3.0 * torque / (
            self.mass * self.length**2
        )
        new_theta_dot = float(
            np.clip(theta_dot + acceleration * self.dt, -self.max_speed, self.max_speed)
        )
        new_theta = theta + new_theta_dot * self.dt
        self._state = np.array([new_theta, new_theta_dot], dtype=np.float64)
        terminated, truncated = self._finish_step(False)
        return self._observation(), -float(cost), terminated, truncated, self._info()


class MaskedMountainCarContinuousEnv(_BaseMaskedEnv):
    """Continuous mountain car with velocity hidden from the observation."""

    state_dim = 2
    observation_dim = 1
    oracle_dim = 2

    def __init__(
        self,
        *,
        horizon: int = 999,
        observation_noise_std: float = 0.05,
        seed: int | None = None,
        min_position: float = -1.2,
        max_position: float = 0.6,
        max_speed: float = 0.07,
        goal_position: float = 0.45,
        goal_velocity: float = 0.0,
        power: float = 0.0015,
    ) -> None:
        super().__init__(
            horizon=horizon,
            observation_noise_std=observation_noise_std,
            seed=seed,
        )
        values = (min_position, max_position, max_speed, goal_position, goal_velocity, power)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("MountainCar parameters must be finite")
        if min_position >= max_position:
            raise ValueError("min_position must be below max_position")
        if max_speed <= 0 or power <= 0:
            raise ValueError("max_speed and power must be positive")
        if not min_position < goal_position <= max_position:
            raise ValueError("goal_position must lie inside the position interval")
        if goal_velocity < 0 or goal_velocity > max_speed:
            raise ValueError("goal_velocity must lie in [0, max_speed]")
        self.min_position = float(min_position)
        self.max_position = float(max_position)
        self.max_speed = float(max_speed)
        self.goal_position = float(goal_position)
        self.goal_velocity = float(goal_velocity)
        self.power = float(power)
        self.action_spec = ActionSpec.continuous(-1.0, 1.0)

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        position = self._state[0]
        normalized_position = (
            2.0 * (position - self.min_position) / (self.max_position - self.min_position) - 1.0
        )
        return self._add_observation_noise(np.array([normalized_position], dtype=np.float64))

    def oracle_features(self) -> np.ndarray:
        """Return normalized position and the otherwise hidden velocity."""

        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        position, velocity = self._state
        normalized_position = (
            2.0 * (position - self.min_position) / (self.max_position - self.min_position) - 1.0
        )
        return np.asarray(
            [normalized_position, velocity / self.max_speed],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._start_reset(seed)
        self._state = np.array([self._rng.uniform(-0.6, -0.4), 0.0], dtype=np.float64)
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._before_step()
        selected = self.action_spec.validate(action)
        assert isinstance(selected, np.ndarray) and self._state is not None
        force = float(selected[0])
        position, velocity = self._state
        velocity += force * self.power - 0.0025 * np.cos(3.0 * position)
        velocity = float(np.clip(velocity, -self.max_speed, self.max_speed))
        position += velocity
        position = float(np.clip(position, self.min_position, self.max_position))
        if position <= self.min_position and velocity < 0.0:
            velocity = 0.0
        self._state = np.array([position, velocity], dtype=np.float64)

        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = (100.0 if terminated else 0.0) - 0.1 * force**2
        terminated, truncated = self._finish_step(terminated)
        return self._observation(), float(reward), terminated, truncated, self._info()


class LightDarkNDEnv(_BaseMaskedEnv):
    """N-dimensional Light-Dark continuous-control task.

    This is the direct N-dimensional extension of the planar Light-Dark
    problem: the state follows ``x' = x + u`` and observations become most
    accurate on the hyperplane ``x[0] = light_position``.  The action bounds
    are the only material addition needed by the bounded continuous policies
    used in this project.

    The reward is ``-(state_cost * ||x||^2 + action_cost * ||u||^2)`` charged on
    the pre-transition state, plus ``goal_bonus`` once on the step that reaches
    ``||x'|| <= goal_radius`` and terminates the episode.  ``goal_bonus``
    defaults to ``0.0``, which is the pure cost-truncation reward the task
    shipped with; a positive value is what makes localizing at the light worth
    its detour, because only a localized agent can enter the goal ball
    deliberately.
    """

    def __init__(
        self,
        *,
        horizon: int = 30,
        observation_noise_std: float = 0.05,
        seed: int | None = None,
        dimension: int = 2,
        light_position: float = 5.0,
        initial_mean: float = 2.0,
        initial_std: float = 0.5,
        max_action: float = 1.0,
        state_cost: float = 0.5,
        action_cost: float = 0.5,
        goal_radius: float = 0.25,
        goal_bonus: float = 0.0,
        noise_gain: float = 0.5,
        terminal_cost: float = 0.0,
        process_noise_std: float = 0.0,
    ) -> None:
        super().__init__(
            horizon=horizon,
            observation_noise_std=observation_noise_std,
            seed=seed,
        )
        if (
            isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, (int, np.integer))
            or dimension <= 0
        ):
            raise ValueError("dimension must be a positive integer")
        finite_parameters = {
            "light_position": light_position,
            "initial_mean": initial_mean,
            "initial_std": initial_std,
            "max_action": max_action,
            "state_cost": state_cost,
            "action_cost": action_cost,
            "goal_radius": goal_radius,
            "goal_bonus": goal_bonus,
            "noise_gain": noise_gain,
            "terminal_cost": terminal_cost,
            "process_noise_std": process_noise_std,
        }
        if any(not np.isfinite(value) for value in finite_parameters.values()):
            raise ValueError("Light-Dark parameters must be finite")
        if noise_gain < 0 or terminal_cost < 0 or process_noise_std < 0:
            raise ValueError(
                "noise_gain, terminal_cost, and process_noise_std must be non-negative"
            )
        if initial_std < 0:
            raise ValueError("initial_std must be non-negative")
        if max_action <= 0:
            raise ValueError("max_action must be positive")
        if state_cost < 0 or action_cost < 0:
            raise ValueError("state_cost and action_cost must be non-negative")
        if goal_radius <= 0:
            raise ValueError("goal_radius must be positive")
        if goal_bonus < 0:
            raise ValueError("goal_bonus must be non-negative")

        self.dimension = int(dimension)
        self.state_dim = self.dimension
        self.observation_dim = self.dimension
        self.oracle_dim = self.dimension
        self.light_position = float(light_position)
        self.initial_mean = float(initial_mean)
        self.initial_std = float(initial_std)
        self.max_action = float(max_action)
        self.state_cost = float(state_cost)
        self.action_cost = float(action_cost)
        self.goal_radius = float(goal_radius)
        self.goal_bonus = float(goal_bonus)
        self.noise_gain = float(noise_gain)
        self.terminal_cost = float(terminal_cost)
        self.process_noise_std = float(process_noise_std)
        self.action_spec = ActionSpec.continuous(
            -self.max_action,
            self.max_action,
            shape=(self.dimension,),
        )

    def observation_std(self) -> float:
        """Return the state-dependent isotropic observation standard deviation."""

        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        variance = (
            self.noise_gain * (self.light_position - float(self._state[0])) ** 2
            + self.observation_noise_std**2
        )
        return float(np.sqrt(variance))

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        standard_deviation = self.observation_std()
        if standard_deviation == 0.0:
            return np.asarray(self._state, dtype=np.float32)
        noise = self._rng.normal(
            loc=0.0,
            scale=standard_deviation,
            size=self.dimension,
        )
        return np.asarray(self._state + noise, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info["effective_observation_std"] = self.observation_std()
        info["distance_to_goal"] = float(np.linalg.norm(self._state))
        return info

    def oracle_features(self) -> np.ndarray:
        """Return bounded, noise-free features of the complete position."""

        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        scale = max(abs(self.light_position), self.max_action, 1.0)
        return np.asarray(np.tanh(self._state / scale), dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._start_reset(seed)
        self._state = self._rng.normal(
            loc=self.initial_mean,
            scale=self.initial_std,
            size=self.dimension,
        ).astype(np.float64)
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._before_step()
        selected = self.action_spec.validate(action)
        assert isinstance(selected, np.ndarray) and self._state is not None
        previous_state = self._state
        cost = self.state_cost * float(np.dot(previous_state, previous_state))
        cost += self.action_cost * float(np.dot(selected, selected))
        self._state = previous_state + selected.astype(np.float64)
        if self.process_noise_std > 0.0:
            # Guarded so the default keeps the historical RNG stream untouched.
            self._state = self._state + self._rng.normal(
                loc=0.0, scale=self.process_noise_std, size=self.dimension
            )

        terminated = bool(np.linalg.norm(self._state) <= self.goal_radius)
        reward = -float(cost)
        if terminated and self.goal_bonus:
            # Guarding on the bonus keeps the default reward bit-identical to
            # the pure cost-truncation definition, down to the sign of a zero.
            reward += self.goal_bonus
        terminated, truncated = self._finish_step(terminated)
        if truncated and self.terminal_cost:
            # A P3O-style dense final grading of the post-transition state; its
            # expectation under the belief prices the residual uncertainty
            # directly, which is what makes localisation smoothly rewarding.
            reward -= self.terminal_cost * float(np.dot(self._state, self._state))
        return self._observation(), reward, terminated, truncated, self._info()


_CARTPOLE_KEYS = {
    "horizon",
    "observation_noise_std",
    "gravity",
    "mass_cart",
    "mass_pole",
    "half_length",
    "force_mag",
    "tau",
    "x_threshold",
    "theta_threshold_radians",
}
_PENDULUM_KEYS = {
    "horizon",
    "observation_noise_std",
    "max_speed",
    "max_torque",
    "dt",
    "gravity",
    "mass",
    "length",
}
_MOUNTAIN_CAR_KEYS = {
    "horizon",
    "observation_noise_std",
    "min_position",
    "max_position",
    "max_speed",
    "goal_position",
    "goal_velocity",
    "power",
}
_LIGHT_DARK_KEYS = {
    "horizon",
    "observation_noise_std",
    "dimension",
    "light_position",
    "initial_mean",
    "initial_std",
    "max_action",
    "state_cost",
    "action_cost",
    "goal_radius",
    "goal_bonus",
    "noise_gain",
    "terminal_cost",
    "process_noise_std",
}


def _factory_kwargs(config: Mapping[str, Any] | None, allowed: set[str]) -> dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping or None")
    ignored_metadata = {"name", "task", "seed"}
    unknown = set(config) - allowed - ignored_metadata
    if unknown:
        joined = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"Unsupported environment config key(s): {joined}")
    return {key: config[key] for key in allowed if key in config}


def make_env(
    name: str,
    config: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> _BaseMaskedEnv:
    """Construct a masked environment by a forgiving task-name alias."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Environment name must be a non-empty string")
    key = "".join(character for character in name.lower() if character.isalnum())

    if key in {"maskedcartpole", "cartpole", "cartpolev0", "cartpolev1"}:
        kwargs = _factory_kwargs(config, _CARTPOLE_KEYS)
        return MaskedCartPoleEnv(seed=seed, **kwargs)
    if key in {"maskedpendulum", "pendulum", "pendulumv0", "pendulumv1"}:
        kwargs = _factory_kwargs(config, _PENDULUM_KEYS)
        return MaskedPendulumEnv(seed=seed, **kwargs)
    if key in {
        "maskedmountaincarcontinuous",
        "mountaincarcontinuous",
        "mountaincarcontinuousv0",
    }:
        kwargs = _factory_kwargs(config, _MOUNTAIN_CAR_KEYS)
        return MaskedMountainCarContinuousEnv(seed=seed, **kwargs)
    if key in {"lightdark", "lightdarknd"}:
        kwargs = _factory_kwargs(config, _LIGHT_DARK_KEYS)
        return LightDarkNDEnv(seed=seed, **kwargs)

    raise ValueError(
        f"Unknown environment {name!r}; expected masked_cartpole, masked_pendulum, "
        "masked_mountain_car_continuous, or light_dark"
    )
