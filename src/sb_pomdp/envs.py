"""Small, dependency-free partially observed classic-control environments.

The environments in this module intentionally expose the true simulator state only
through ``info["state"]``.  The agent observation is a noisy, masked projection of
that state.  Their API follows the useful subset of the Gymnasium API, without
requiring Gymnasium as a dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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
        initial_angle_range: float = np.pi,
        initial_speed_range: float = 1.0,
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
        # Start distribution: theta ~ U(-angle_range, angle_range) around upright (theta = 0)
        # and theta_dot ~ U(-speed_range, speed_range).  The defaults (pi, 1) are the gym
        # swing-up start; a small angle range gives the balancing variant.
        for name, value in (("initial_angle_range", initial_angle_range), ("initial_speed_range", initial_speed_range)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if initial_angle_range > np.pi:
            raise ValueError("initial_angle_range cannot exceed pi")
        self.max_speed = float(max_speed)
        self.max_torque = float(max_torque)
        self.dt = float(dt)
        self.gravity = float(gravity)
        self.mass = float(mass)
        self.length = float(length)
        self.initial_angle_range = float(initial_angle_range)
        self.initial_speed_range = float(initial_speed_range)
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
            [
                self._rng.uniform(-self.initial_angle_range, self.initial_angle_range),
                self._rng.uniform(-self.initial_speed_range, self.initial_speed_range),
            ],
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
    "initial_angle_range",
    "initial_speed_range",
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

def _bivariate_normal(
    x: np.ndarray, y: np.ndarray, sigma_x: float, sigma_y: float, mu_x: float, mu_y: float
) -> np.ndarray:
    """matplotlib.mlab.bivariate_normal with zero correlation (normalised density)."""

    z = ((x - mu_x) / sigma_x) ** 2 + ((y - mu_y) / sigma_y) ** 2
    return np.exp(-0.5 * z) / (2.0 * np.pi * sigma_x * sigma_y)


_MOUNTAIN_HIKE_MEANS = ((0.5, 0.5), (0.0, 0.2), (-0.375, -0.5))
_MOUNTAIN_HIKE_SIGMAS = ((0.75, 0.1), (0.1, 0.75), (0.75, 0.1))
_MOUNTAIN_HIKE_FACTORS = (1.0, 0.8, 0.55)


def mountain_hike_terrain(
    x: np.ndarray | float,
    y: np.ndarray | float,
    *,
    hill_height: float = 4.0,
    shaping_power: float = 4.0,
    ridges: Sequence[int] | None = None,
) -> np.ndarray:
    """The Mountain Hike reward map ``r(x, y)`` in internal ``[-1, 1]^2`` coordinates.

    Reproduces ``DeathValleyEnv.get_reward`` of the DVRL reference code
    (Igl et al. 2018): the element-wise maximum over three axis-aligned
    Gaussian ridges (``factor * pdf / 3``), shaped as ``1 - (1 - Z)^p``, scaled
    by the hill height, shifted by ``-(0.5 + h)``, plus the linear tilt
    ``(x + y) / 4``.  Note that the reference code passes the ``sigma``
    entries to ``bivariate_normal`` as standard deviations.  ``ridges`` keeps a
    subset of the three ridges (indices 0, 1, 2; default all) with the per-ridge
    scale ``factor / 3`` unchanged, so the map equals the reference wherever a
    kept ridge dominates.
    """

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    z = np.zeros(np.broadcast(x_arr, y_arr).shape, dtype=np.float64)
    count = len(_MOUNTAIN_HIKE_MEANS)
    selected = range(count) if ridges is None else [int(index) for index in ridges]
    if any(index < 0 or index >= count for index in selected):
        raise ValueError("ridges must index the three reference ridges (0, 1, 2)")
    for index in selected:
        (mu_x, mu_y), (sigma_x, sigma_y), factor = (
            _MOUNTAIN_HIKE_MEANS[index],
            _MOUNTAIN_HIKE_SIGMAS[index],
            _MOUNTAIN_HIKE_FACTORS[index],
        )
        z = np.maximum(
            z, factor * _bivariate_normal(x_arr, y_arr, sigma_x, sigma_y, mu_x, mu_y) / count
        )
    z = 1.0 - np.power(1.0 - z, shaping_power)
    z = z * hill_height - (0.5 + hill_height) + (x_arr + y_arr) / 4.0
    return z


class MountainHikeEnv(_BaseMaskedEnv):
    """Mountain Hike (Igl et al. 2018, DVRL): noisy 2D terrain navigation.

    Faithful to the reference ``DeathValleyEnv`` with the ``mountainHike.yaml``
    constants, expressed in the paper's coordinates (internal box scaled by
    ``box_scale = 10``): the agent starts near ``(-8.5, -8.5)``, moves by a
    norm-clipped step of at most ``max_action`` (0.5), suffers Gaussian
    transition noise (std 0.25), observes its position with Gaussian noise of
    ``observation_noise_std`` (the paper sweeps 0 / 1.5 / 3), and is rewarded by
    the terrain map every step (minus ``action_cost * ||a||``), with a flat
    penalty outside the box.  Episodes are time-limited (75 steps); the goal
    region neither pays nor terminates by default, exactly like the reference
    configuration.  The latent state is the true position, never observed.
    """

    state_dim = 2
    observation_dim = 2
    oracle_dim = 2

    def __init__(
        self,
        *,
        horizon: int = 75,
        observation_noise_std: float = 3.0,
        seed: int | None = None,
        box_scale: float = 10.0,
        transition_std: float = 0.25,
        max_action: float = 0.5,
        start_mean: Sequence[float] = (-8.5, -8.5),
        start_std: float = 1.0,
        action_cost: float = 0.01,
        outside_box_cost: float = -1.5,
        hill_height: float = 4.0,
        shaping_power: float = 4.0,
        goal_position: Sequence[float] = (7.0, 5.0),
        goal_radius: float = 1.0,
        goal_reward: float = 0.0,
        goal_end: bool = False,
        teleport_probability: float = 0.0,
        observation_symmetry: str = "none",
        mirror_symmetry: bool = False,
        signpost_width: float = 2.0,
        signpost_noise_std: float = 0.2,
        start_distribution: str = "normal",
        terrain_ridges: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            horizon=horizon,
            observation_noise_std=observation_noise_std,
            seed=seed,
        )
        start = np.asarray(start_mean, dtype=np.float64).reshape(-1)
        goal = np.asarray(goal_position, dtype=np.float64).reshape(-1)
        if start.shape != (2,) or goal.shape != (2,):
            raise ValueError("start_mean and goal_position must be 2-vectors")
        for name, value in (
            ("box_scale", box_scale),
            ("max_action", max_action),
            ("hill_height", hill_height),
            ("shaping_power", shaping_power),
            ("goal_radius", goal_radius),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("transition_std", transition_std),
            ("start_std", start_std),
            ("action_cost", action_cost),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not np.isfinite(outside_box_cost) or not np.isfinite(goal_reward):
            raise ValueError("outside_box_cost and goal_reward must be finite")
        self.box_scale = float(box_scale)
        self.transition_std = float(transition_std)
        self.max_action = float(max_action)
        self.start_mean = start
        self.start_std = float(start_std)
        self.action_cost = float(action_cost)
        self.outside_box_cost = float(outside_box_cost)
        self.hill_height = float(hill_height)
        self.shaping_power = float(shaping_power)
        self.goal_position = goal
        self.goal_radius = float(goal_radius)
        self.goal_reward = float(goal_reward)
        self.goal_end = bool(goal_end)
        if not 0.0 <= float(teleport_probability) < 1.0:
            raise ValueError("teleport_probability must be in [0, 1)")
        # Kidnapped-robot variant: with this probability the transition is replaced
        # by a uniform relocation inside the box, so the belief must be rebuilt.
        self.teleport_probability = float(teleport_probability)
        self._teleported = False
        if observation_symmetry not in ("none", "abs"):
            raise ValueError("observation_symmetry must be 'none' or 'abs'")
        # 'abs': the agent observes |s| + noise, so the observation is invariant to
        # the sign of each coordinate and the position belief is four-fold ambiguous
        # until the (asymmetric) terrain reward or the start prior disambiguates it.
        self.observation_symmetry = str(observation_symmetry)
        # Mirror worlds: the body frame is the world frame reflected by an unknown
        # m in {+-1}^2 drawn at reset.  The agent observes m*s (+ noise) and its action
        # moves the world state by m*a, so every mirror hypothesis explains every
        # observation sequence; only the signpost near the origin (two extra observation
        # components m * exp(-|s|^2 / 2 w^2) + noise) and the asymmetric terrain reward
        # tell the four worlds apart.  The start must then be symmetric, or the first
        # observation would reveal the frame.
        if not isinstance(mirror_symmetry, (bool, np.bool_)):
            raise ValueError("mirror_symmetry must be boolean")
        if start_distribution not in ("normal", "mirror_corners", "uniform"):
            raise ValueError("start_distribution must be normal, mirror_corners or uniform")
        if not np.isfinite(signpost_width) or signpost_width <= 0:
            raise ValueError("signpost_width must be positive")
        if not np.isfinite(signpost_noise_std) or signpost_noise_std < 0:
            raise ValueError("signpost_noise_std must be non-negative")
        self.mirror_symmetry = bool(mirror_symmetry)
        self.signpost_width = float(signpost_width)
        self.signpost_noise_std = float(signpost_noise_std)
        self.start_distribution = str(start_distribution)
        if terrain_ridges is None:
            self.terrain_ridges: tuple[int, ...] | None = None
        else:
            ridges = tuple(int(index) for index in terrain_ridges)
            if not ridges or len(set(ridges)) != len(ridges) or any(i not in (0, 1, 2) for i in ridges):
                raise ValueError("terrain_ridges must be a non-empty subset of (0, 1, 2)")
            self.terrain_ridges = ridges
        if self.mirror_symmetry:
            if self.observation_symmetry != "none":
                raise ValueError("mirror_symmetry and observation_symmetry cannot be combined")
            if self.start_distribution == "normal":
                raise ValueError(
                    "mirror_symmetry needs a symmetric start (mirror_corners or uniform); "
                    "a fixed start corner would reveal the mirror frame"
                )
            self.observation_dim = 4
        self._mirror = np.ones(2, dtype=np.float64)
        self.action_spec = ActionSpec.continuous(-self.max_action, self.max_action, shape=(2,))

    def terrain_reward(self, position: np.ndarray) -> float:
        """Reward map evaluated at a position given in paper (scaled) coordinates."""

        internal = np.asarray(position, dtype=np.float64) / self.box_scale
        return float(
            mountain_hike_terrain(
                internal[0],
                internal[1],
                hill_height=self.hill_height,
                shaping_power=self.shaping_power,
                ridges=self.terrain_ridges,
            )
        )

    def signpost_signal(self, position: np.ndarray | None = None) -> float:
        """Noise-free signpost strength exp(-|s|^2 / 2 w^2) at a world position."""

        if position is None:
            assert self._state is not None
            position = self._state
        squared = float(np.dot(position, position))
        return float(np.exp(-0.5 * squared / self.signpost_width**2))

    def _outside_box(self) -> bool:
        assert self._state is not None
        return bool(np.any(np.abs(self._state) > self.box_scale))

    def _reached_goal(self) -> bool:
        assert self._state is not None
        return bool(np.linalg.norm(self._state - self.goal_position) < self.goal_radius)

    def _observed_state(self) -> np.ndarray:
        """State as seen by the sensor: the raw position, or |position| under 'abs' symmetry."""

        assert self._state is not None
        if self.mirror_symmetry:
            return self._mirror * self._state
        if self.observation_symmetry == "abs":
            return np.abs(self._state)
        return self._state

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        observed = self._add_observation_noise(np.asarray(self._observed_state(), dtype=np.float64))
        if not self.mirror_symmetry:
            return observed
        signpost = self._mirror * self.signpost_signal()
        if self.signpost_noise_std > 0.0:
            signpost = signpost + self._rng.normal(loc=0.0, scale=self.signpost_noise_std, size=2)
        return np.concatenate([observed, np.asarray(signpost, dtype=np.float32)]).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info["terrain_reward"] = self.terrain_reward(self._state)
        info["outside_box"] = self._outside_box()
        info["distance_to_goal"] = float(np.linalg.norm(self._state - self.goal_position))
        info["teleported"] = bool(self._teleported)
        if self.mirror_symmetry:
            info["mirror"] = self._mirror.copy()
            info["signpost"] = self.signpost_signal()
        return info

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        if self.mirror_symmetry:
            state["mirror"] = self._mirror.copy()
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("environment state_dict must be a mapping")
        payload = dict(state_dict)
        mirror = payload.pop("mirror", None)
        if self.mirror_symmetry:
            if mirror is None:
                raise ValueError("a mirror-symmetric environment state_dict needs 'mirror'")
            mirror = np.asarray(mirror, dtype=np.float64)
            if mirror.shape != (2,) or not np.all(np.isin(mirror, (-1.0, 1.0))):
                raise ValueError("environment mirror must be a 2-vector of +-1")
        elif mirror is not None:
            raise ValueError("'mirror' is only valid for mirror-symmetric environments")
        super().load_state_dict(payload)
        if self.mirror_symmetry:
            self._mirror = mirror.copy()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        if self.mirror_symmetry:
            state["mirror"] = self._mirror.copy()
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("environment state_dict must be a mapping")
        payload = dict(state_dict)
        mirror = payload.pop("mirror", None)
        if self.mirror_symmetry:
            if mirror is None:
                raise ValueError("a mirror-symmetric environment state_dict needs 'mirror'")
            mirror = np.asarray(mirror, dtype=np.float64)
            if mirror.shape != (2,) or not np.all(np.isin(mirror, (-1.0, 1.0))):
                raise ValueError("environment mirror must be a 2-vector of +-1")
        elif mirror is not None:
            raise ValueError("'mirror' is only valid for mirror-symmetric environments")
        super().load_state_dict(payload)
        if self.mirror_symmetry:
            self._mirror = mirror.copy()

    def oracle_features(self) -> np.ndarray:
        """Noise-free position squashed to (-1, 1)."""

        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        return np.asarray(np.tanh(self._state / self.box_scale), dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._start_reset(seed)
        self._teleported = False
        if self.start_distribution == "uniform":
            self._state = self._rng.uniform(-self.box_scale, self.box_scale, size=2).astype(np.float64)
        else:
            centre = self.start_mean
            if self.start_distribution == "mirror_corners":
                # One of the four mirror images of the start corner, independent of the frame.
                centre = centre * self._rng.choice(np.array([-1.0, 1.0]), size=2)
            self._state = self._rng.normal(
                loc=centre,
                scale=self.start_std,
                size=2,
            ).astype(np.float64)
        if self.mirror_symmetry:
            self._mirror = self._rng.choice(np.array([-1.0, 1.0]), size=2).astype(np.float64)
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._before_step()
        selected = self.action_spec.validate(action)
        assert isinstance(selected, np.ndarray) and self._state is not None
        proposed = np.asarray(selected, dtype=np.float64)
        norm = float(np.linalg.norm(proposed))
        action_penalty = self.action_cost * norm
        if norm > self.max_action:
            proposed = proposed / norm * self.max_action
        if self.mirror_symmetry:
            proposed = self._mirror * proposed  # commanded in the body frame
        noise = (
            self._rng.normal(loc=0.0, scale=self.transition_std, size=2)
            if self.transition_std > 0.0
            else np.zeros(2)
        )
        if self.teleport_probability > 0.0 and self._rng.random() < self.teleport_probability:
            self._state = self._rng.uniform(-self.box_scale, self.box_scale, size=2)
            self._teleported = True
        else:
            self._state = self._state + proposed + noise
            self._teleported = False
        reached = self._reached_goal()
        if reached and self.goal_reward != 0.0:
            reward = self.goal_reward
        elif self._outside_box():
            reward = self.outside_box_cost * self.hill_height
        else:
            reward = self.terrain_reward(self._state)
        terminated, truncated = self._finish_step(bool(self.goal_end and reached))
        return self._observation(), float(reward - action_penalty), terminated, truncated, self._info()


_MOUNTAIN_HIKE_KEYS = {
    "horizon",
    "observation_noise_std",
    "box_scale",
    "transition_std",
    "max_action",
    "start_mean",
    "start_std",
    "action_cost",
    "outside_box_cost",
    "hill_height",
    "shaping_power",
    "goal_position",
    "goal_radius",
    "goal_reward",
    "goal_end",
    "teleport_probability",
    "observation_symmetry",
    "mirror_symmetry",
    "signpost_width",
    "signpost_noise_std",
    "start_distribution",
    "terrain_ridges",
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


class CarFlagEnv(_BaseMaskedEnv):
    """Car-Flag (Nguyen et al., CoRL 2022; Baisero & Amato, AAMAS 2022).

    A car on a line accelerates with a continuous force in [-1, 1].  The green
    flag sits at +1 or -1 with equal probability and the red flag at the other
    end; reaching either ends the episode.  The side of the green flag is the
    only hidden variable: the observation reveals it (+1 right, -1 left) only
    while the car is within ``priest_range`` of the blue flag at
    ``priest_position``, and reports 0 elsewhere.  Constants follow the public
    implementation (hai-h-nguyen/pomdp-domains, ``pdomains/car_flag.py``); its
    asymmetric reward is replaced by the paper definition: ``-step_cost`` per
    step, ``+flag_reward`` at the green flag and ``-flag_reward`` at the red one.

    State ``(position, velocity, side)``; observation ``(position,
    velocity / max_speed, signal)`` plus optional Gaussian noise.
    """

    state_dim = 3
    observation_dim = 3
    oracle_dim = 3

    def __init__(
        self,
        *,
        horizon: int = 160,
        observation_noise_std: float = 0.0,
        seed: int | None = None,
        max_speed: float = 0.07,
        power: float = 0.0015,
        position_limit: float = 1.1,
        flag_position: float = 1.0,
        priest_position: float = 0.5,
        priest_range: float = 0.2,
        start_range: float = 0.2,
        step_cost: float = 0.01,
        flag_reward: float = 1.0,
    ) -> None:
        super().__init__(
            horizon=horizon,
            observation_noise_std=observation_noise_std,
            seed=seed,
        )
        values = (
            max_speed,
            power,
            position_limit,
            flag_position,
            priest_position,
            priest_range,
            start_range,
            step_cost,
            flag_reward,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Car-Flag parameters must be finite")
        if max_speed <= 0 or power <= 0 or priest_range <= 0 or flag_reward <= 0:
            raise ValueError("max_speed, power, priest_range and flag_reward must be positive")
        if not 0 < flag_position <= position_limit:
            raise ValueError("flag_position must lie in (0, position_limit]")
        if not abs(priest_position) + priest_range < flag_position:
            raise ValueError("the information region must lie strictly between the flags")
        if not 0 <= start_range < flag_position or step_cost < 0:
            raise ValueError("start_range must lie in [0, flag_position) and step_cost must be >= 0")
        self.max_speed = float(max_speed)
        self.power = float(power)
        self.position_limit = float(position_limit)
        self.flag_position = float(flag_position)
        self.priest_position = float(priest_position)
        self.priest_range = float(priest_range)
        self.start_range = float(start_range)
        self.step_cost = float(step_cost)
        self.flag_reward = float(flag_reward)
        self.action_spec = ActionSpec.continuous(-1.0, 1.0)

    def _signal(self) -> float:
        assert self._state is not None
        position, _, side = self._state
        if abs(position - self.priest_position) <= self.priest_range:
            return float(side)
        return 0.0

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        position, velocity, _ = self._state
        values = np.array([position, velocity / self.max_speed, self._signal()], dtype=np.float64)
        return self._add_observation_noise(values)

    def oracle_features(self) -> np.ndarray:
        """Return position, normalized velocity and the hidden green-flag side."""

        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        position, velocity, side = self._state
        return np.asarray([position, velocity / self.max_speed, side], dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._start_reset(seed)
        side = 1.0 if self._rng.random() < 0.5 else -1.0
        position = self._rng.uniform(-self.start_range, self.start_range)
        self._state = np.array([position, 0.0, side], dtype=np.float64)
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._before_step()
        selected = self.action_spec.validate(action)
        assert isinstance(selected, np.ndarray) and self._state is not None
        position, velocity, side = self._state
        velocity = float(np.clip(velocity + float(selected[0]) * self.power, -self.max_speed, self.max_speed))
        position = float(np.clip(position + velocity, -self.position_limit, self.position_limit))
        if abs(position) >= self.position_limit and position * velocity > 0.0:
            velocity = 0.0
        self._state = np.array([position, velocity, side], dtype=np.float64)

        reward = -self.step_cost
        terminated = abs(position) >= self.flag_position
        if terminated:
            reached_green = (position > 0.0) == (side > 0.0)
            reward += self.flag_reward if reached_green else -self.flag_reward
        terminated, truncated = self._finish_step(bool(terminated))
        return self._observation(), float(reward), terminated, truncated, self._info()


class HeavenHellEnv(_BaseMaskedEnv):
    """Heaven-Hell (Geffner & Bonet 1998; Thrun 2000) in the corridor layout of
    gym-pomdps ``heavenhell_<n>`` (Baisero & Amato, AAMAS 2022, use n = 3, 4).

    Cells (``n = corridor_length``)::

        left end 2n ... n+1   n   2n+1 ... 3n right end
                              :
                              0  (start)
                            3n+1  3n+2 ... 4n+1 (priest)

    Heaven is the left or the right end with equal probability and hell the
    other.  Actions N, S, E, W move along the corridors (moves into a wall do
    nothing).  Entering an end gives +1 (heaven) or -1 (hell) and terminates.
    The observation is the gym-pomdps symbol, one-hot encoded: the cell index
    for every cell except the priest, where it is the heaven side (``left`` /
    ``right``) instead, so the side is revealed only there (4n+3 symbols: 15
    for n = 3, 19 for n = 4, as in the paper).  gym-pomdps pays the reward for
    the action taken *in* the end cell; here it is paid on entering it, which
    shifts the return by one discount step and nothing else.

    State ``(x, y, side)`` in cell coordinates (fork at (0, n), priest at
    (n, -1)); side = -1 for heaven on the left, +1 on the right.
    """

    state_dim = 3
    oracle_dim = 3
    _NORTH, _SOUTH, _EAST, _WEST = 0, 1, 2, 3

    def __init__(
        self,
        *,
        horizon: int = 100,
        observation_noise_std: float = 0.0,
        seed: int | None = None,
        corridor_length: int = 3,
    ) -> None:
        super().__init__(
            horizon=horizon,
            observation_noise_std=observation_noise_std,
            seed=seed,
        )
        if (
            isinstance(corridor_length, bool)
            or not isinstance(corridor_length, (int, np.integer))
            or corridor_length < 1
        ):
            raise ValueError("corridor_length must be a positive integer")
        n = int(corridor_length)
        self.corridor_length = n
        self.num_cells = 2 + 4 * n
        self.priest_cell = 4 * n + 1
        self.left_end = 2 * n
        self.right_end = 3 * n
        self.observation_dim = self.num_cells + 1  # (num_cells - 1) cell symbols + left + right
        self.action_spec = ActionSpec.discrete(4)
        self._coordinates = np.zeros((self.num_cells, 2), dtype=np.float64)
        for cell in range(n + 1):
            self._coordinates[cell] = (0.0, float(cell))
        for offset in range(1, n + 1):
            self._coordinates[n + offset] = (-float(offset), float(n))
            self._coordinates[2 * n + offset] = (float(offset), float(n))
        for offset in range(n + 1):
            self._coordinates[3 * n + 1 + offset] = (float(offset), -1.0)
        self._moves = self._transition_table()

    def _transition_table(self) -> np.ndarray:
        """``moves[cell, action]`` -> next cell, transcribed from the gym-pomdps generator."""

        n = self.corridor_length
        moves = np.tile(np.arange(self.num_cells)[:, None], (1, 4))
        for cell in range(self.num_cells):
            if 0 <= cell < n:
                moves[cell, self._NORTH] = cell + 1
            elif cell == 3 * n + 1:
                moves[cell, self._NORTH] = 0
            if 0 < cell <= n:
                moves[cell, self._SOUTH] = cell - 1
            elif cell == 0:
                moves[cell, self._SOUTH] = 3 * n + 1
            if n + 1 <= cell < 2 * n:
                moves[cell, self._EAST] = cell - 1
            if cell == n:
                moves[cell, self._EAST] = 2 * n + 1
            if 2 * n < cell < 3 * n or 3 * n < cell < self.num_cells - 1:
                moves[cell, self._EAST] = cell + 1
            if n <= cell < 2 * n:
                moves[cell, self._WEST] = cell + 1
            elif cell == 2 * n + 1:
                moves[cell, self._WEST] = n
            elif 2 * n < cell <= 3 * n or 3 * n + 1 < cell < self.num_cells:
                moves[cell, self._WEST] = cell - 1
        return moves

    def _cell(self) -> int:
        assert self._state is not None
        x, y, _ = self._state
        matches = np.flatnonzero(np.all(self._coordinates == (x, y), axis=1))
        assert matches.size == 1
        return int(matches[0])

    def _symbol(self) -> int:
        assert self._state is not None
        cell = self._cell()
        if cell == self.priest_cell:
            return self.num_cells - 1 + (0 if self._state[2] < 0 else 1)
        return cell

    def _observation(self) -> np.ndarray:
        one_hot = np.zeros(self.observation_dim, dtype=np.float64)
        one_hot[self._symbol()] = 1.0
        return self._add_observation_noise(one_hot)

    def oracle_features(self) -> np.ndarray:
        """Return the cell coordinates scaled by the corridor length and the heaven side."""

        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        x, y, side = self._state
        n = float(self.corridor_length)
        return np.asarray([x / n, y / n, side], dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._start_reset(seed)
        side = -1.0 if self._rng.random() < 0.5 else 1.0
        x, y = self._coordinates[0]
        self._state = np.array([x, y, side], dtype=np.float64)
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._before_step()
        selected = self.action_spec.validate(action)
        assert isinstance(selected, int) and self._state is not None
        side = float(self._state[2])
        cell = int(self._moves[self._cell(), selected])
        x, y = self._coordinates[cell]
        self._state = np.array([x, y, side], dtype=np.float64)
        reward = 0.0
        terminated = cell in (self.left_end, self.right_end)
        if terminated:
            heaven = self.left_end if side < 0 else self.right_end
            reward = 1.0 if cell == heaven else -1.0
        terminated, truncated = self._finish_step(bool(terminated))
        return self._observation(), reward, terminated, truncated, self._info()


def disk_tag_ghost_step(
    ghosts: np.ndarray,
    speeds: np.ndarray,
    player: np.ndarray,
    turn_noise: np.ndarray,
    *,
    arena_radius: float,
    view_half_angle: float,
    view_range: float,
    wander_ratio: float,
    wander_turn_std: float,
) -> np.ndarray:
    """One step of the disk-tag ghosts, batched over any leading dimensions.

    ``ghosts[..., i, :] = (x, y, heading, mode, target_x, target_y)`` with mode 0 = wander and
    1 = pursue; ``speeds[..., i]`` is the pursuit speed, ``player[..., :]`` the player position
    after its move and ``turn_noise[..., i]`` standard normal draws.  A ghost that sees the player
    (within ``view_range`` and ``view_half_angle`` of its heading) runs at it and remembers the
    spot; one that has lost sight runs to the remembered spot and wanders again once there.  A
    wandering ghost moves at ``wander_ratio * speed`` with a heading random walk and turns back to
    the centre when it would leave the disk.  The DiskTagEnv simulator and filters share this.
    """

    position = ghosts[..., 0:2]
    heading = ghosts[..., 2]
    mode = ghosts[..., 3]
    target = ghosts[..., 4:6]
    offset = player[..., None, :] - position
    distance = np.linalg.norm(offset, axis=-1)
    bearing = (np.arctan2(offset[..., 1], offset[..., 0]) - heading + math.pi) % (2.0 * math.pi) - math.pi
    sees = (distance <= view_range) & (np.abs(bearing) <= view_half_angle)
    target = np.where(sees[..., None], player[..., None, :], target)
    mode = np.where(sees, 1.0, mode)

    to_target = target - position
    target_distance = np.linalg.norm(to_target, axis=-1)
    pursuing = mode > 0.5
    arrived = pursuing & ~sees & (target_distance <= speeds)
    pursue_heading = np.where(target_distance > 1e-12, np.arctan2(to_target[..., 1], to_target[..., 0]), heading)
    new_heading = np.where(pursuing, pursue_heading, heading + wander_turn_std * turn_noise)
    step = np.where(pursuing, np.minimum(speeds, target_distance), wander_ratio * speeds)
    new_position = position + step[..., None] * np.stack([np.cos(new_heading), np.sin(new_heading)], axis=-1)
    radius = np.linalg.norm(new_position, axis=-1)
    outside = radius > arena_radius
    scale = np.where(outside, arena_radius / np.maximum(radius, 1e-12), 1.0)
    new_position = new_position * scale[..., None]
    inward = np.arctan2(-new_position[..., 1], -new_position[..., 0])
    new_heading = np.where(outside & ~pursuing, inward, new_heading)
    new_heading = (new_heading + math.pi) % (2.0 * math.pi) - math.pi
    mode = np.where(arrived, 0.0, mode)
    return np.concatenate([new_position, new_heading[..., None], mode[..., None], target], axis=-1)


class DiskTagEnv(_BaseMaskedEnv):
    """Tag in a disk: evade ``num_ghosts`` ghosts that the player sees only as a count in a view cone.

    The player moves inside a disk of radius ``arena_radius`` with heading ``phi`` and speed
    ``v`` in ``[0, max_speed]``; the continuous action ``(accel, turn)`` in ``[-1, 1]^2``
    changes them by ``accel * acceleration`` and ``turn * turn_rate`` per step.  A player that
    would leave the disk is projected back onto its rim.

    Every ghost starts uniform in the disk with a uniform heading, wandering, and draws its own
    pursuit speed ``U(ghost_speed_min, ghost_speed_max) * max_speed`` once per episode.  Ghosts have
    their own wide view cone (``ghost_view_angle_deg`` around their heading, range
    ``ghost_view_range``); see :func:`disk_tag_ghost_step` for wander / pursue / search.  The
    player starts uniform in the disk at least ``start_distance`` from every ghost.  A ghost within
    ``capture_radius`` ends the episode and adds ``capture_reward`` (a penalty when negative);
    every step survived pays ``step_reward``.  The default 0.1 per step is learnable by PPO;
    0.01 left even the oracle-state policy at its initial return after 800k steps.

    The observation is the player's own state ``(x, y, cos phi, sin phi, v / max_speed)`` and the
    fraction of ghosts inside the player's view cone (``view_angle_deg`` around ``phi``, range
    ``view_range``).  Everything about the ghosts is hidden; they are exchangeable, so the belief
    over them is a permutation-symmetric, multimodal distribution.

    ``observation_mode="ranges"`` replaces the view-cone count by the noisy distance to every ghost
    (fixed ghost order, all directions, no range limit): ``d_i + eps_i`` with
    ``eps_i ~ N(0, (range_noise_base + range_noise_scale * d_i)^2)``.  A range carries no bearing,
    so each ghost's belief is a ring (and a mirror pair about the player's path), resolved over
    time by moving.

    State ``(x, y, phi, v, [g_x, g_y, g_heading, g_mode, g_target_x, g_target_y] * N, s_1..s_N)``.
    """

    def __init__(
        self,
        *,
        horizon: int = 200,
        observation_noise_std: float = 0.0,
        seed: int | None = None,
        num_ghosts: int = 3,
        arena_radius: float = 1.0,
        max_speed: float = 0.05,
        acceleration: float = 0.01,
        turn_rate: float = 0.35,
        ghost_speed_min: float = 0.8,
        ghost_speed_max: float = 1.0,
        ghost_view_angle_deg: float = 180.0,
        ghost_view_range: float = 1.0,
        ghost_wander_ratio: float = 0.5,
        ghost_wander_turn_std: float = 0.3,
        capture_radius: float = 0.08,
        view_angle_deg: float = 60.0,
        view_range: float = 0.6,
        start_distance: float = 0.6,
        step_reward: float = 0.1,
        capture_reward: float = 0.0,
        observation_mode: str = "count",
        range_noise_base: float = 0.05,
        range_noise_scale: float = 0.2,
    ) -> None:
        super().__init__(horizon=horizon, observation_noise_std=observation_noise_std, seed=seed)
        if observation_mode not in ("count", "ranges"):
            raise ValueError("observation_mode must be count or ranges")
        if not (np.isfinite(range_noise_base) and np.isfinite(range_noise_scale)) or min(range_noise_base, range_noise_scale) < 0:
            raise ValueError("range noise parameters must be finite and non-negative")
        self.observation_mode = str(observation_mode)
        self.range_noise_base = float(range_noise_base)
        self.range_noise_scale = float(range_noise_scale)
        if isinstance(num_ghosts, bool) or not isinstance(num_ghosts, (int, np.integer)) or num_ghosts < 1:
            raise ValueError("num_ghosts must be a positive integer")
        values = (arena_radius, max_speed, acceleration, turn_rate, ghost_speed_min, ghost_speed_max,
                  ghost_view_angle_deg, ghost_view_range, ghost_wander_ratio, ghost_wander_turn_std,
                  capture_radius, view_angle_deg, view_range, start_distance, step_reward, capture_reward)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Disk tag parameters must be finite")
        if min(arena_radius, max_speed, acceleration, turn_rate, capture_radius, view_range, ghost_view_range) <= 0:
            raise ValueError("radii, speeds, acceleration, turn rate and view ranges must be positive")
        if not 0 < ghost_speed_min <= ghost_speed_max:
            raise ValueError("ghost speeds must satisfy 0 < ghost_speed_min <= ghost_speed_max")
        if not 0 < view_angle_deg <= 360 or not 0 < ghost_view_angle_deg <= 360:
            raise ValueError("view angles must lie in (0, 360]")
        if not 0 <= ghost_wander_ratio <= 1 or ghost_wander_turn_std < 0:
            raise ValueError("ghost_wander_ratio must lie in [0, 1] and ghost_wander_turn_std must be >= 0")
        if not capture_radius < start_distance < arena_radius:
            raise ValueError("start_distance must lie between capture_radius and arena_radius")
        self.num_ghosts = int(num_ghosts)
        self.arena_radius = float(arena_radius)
        self.max_speed = float(max_speed)
        self.acceleration = float(acceleration)
        self.turn_rate = float(turn_rate)
        self.ghost_speed_min = float(ghost_speed_min)
        self.ghost_speed_max = float(ghost_speed_max)
        self.ghost_view_half_angle = math.radians(float(ghost_view_angle_deg)) / 2.0
        self.ghost_view_range = float(ghost_view_range)
        self.ghost_wander_ratio = float(ghost_wander_ratio)
        self.ghost_wander_turn_std = float(ghost_wander_turn_std)
        self.capture_radius = float(capture_radius)
        self.view_half_angle = math.radians(float(view_angle_deg)) / 2.0
        self.view_range = float(view_range)
        self.start_distance = float(start_distance)
        self.step_reward = float(step_reward)
        self.capture_reward = float(capture_reward)
        self.state_dim = 4 + 7 * self.num_ghosts
        self.observation_dim = 5 + (self.num_ghosts if self.observation_mode == "ranges" else 1)
        self.oracle_dim = 5 + 4 * self.num_ghosts
        self.action_spec = ActionSpec.continuous(-1.0, 1.0, shape=(2,))

    def ghost_params(self) -> dict[str, float]:
        """Keyword arguments of :func:`disk_tag_ghost_step` for this environment."""

        return {
            "arena_radius": self.arena_radius,
            "view_half_angle": self.ghost_view_half_angle,
            "view_range": self.ghost_view_range,
            "wander_ratio": self.ghost_wander_ratio,
            "wander_turn_std": self.ghost_wander_turn_std,
        }

    def _ghost_block(self) -> np.ndarray:
        assert self._state is not None
        return self._state[4 : 4 + 6 * self.num_ghosts].reshape(self.num_ghosts, 6)

    def _ghosts(self) -> np.ndarray:
        return self._ghost_block()[:, 0:2]

    def _speeds(self) -> np.ndarray:
        assert self._state is not None
        return self._state[4 + 6 * self.num_ghosts :]

    def visible_count(self) -> int:
        """Number of ghosts inside the player's view cone."""

        assert self._state is not None
        x, y, phi = self._state[:3]
        offsets = self._ghosts() - np.array([x, y])
        distances = np.linalg.norm(offsets, axis=1)
        bearing = np.arctan2(offsets[:, 1], offsets[:, 0]) - phi
        bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
        return int(np.sum((distances <= self.view_range) & (np.abs(bearing) <= self.view_half_angle)))

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        x, y, phi, speed = self._state[:4]
        own = [x, y, math.cos(phi), math.sin(phi), speed / self.max_speed]
        if self.observation_mode == "ranges":
            distances = np.linalg.norm(self._ghosts() - np.array([x, y]), axis=1)
            noise = self._rng.normal(size=self.num_ghosts) * (self.range_noise_base + self.range_noise_scale * distances)
            values = np.concatenate([own, distances + noise]).astype(np.float64)
        else:
            values = np.array(own + [self.visible_count() / self.num_ghosts], dtype=np.float64)
        return self._add_observation_noise(values)

    def oracle_features(self) -> np.ndarray:
        """Return the player state, ghost positions, pursuit flags and speeds (relative to max_speed)."""

        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        x, y, phi, speed = self._state[:4]
        block = self._ghost_block()
        return np.concatenate([[x, y, math.cos(phi), math.sin(phi), speed / self.max_speed],
                               block[:, 0:2].ravel(), block[:, 3], self._speeds() / self.max_speed]).astype(np.float32)

    def _uniform_disk(self, count: int) -> np.ndarray:
        radius = self.arena_radius * np.sqrt(self._rng.random(count))
        angle = self._rng.uniform(-math.pi, math.pi, count)
        return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._start_reset(seed)
        while True:
            ghosts = self._uniform_disk(self.num_ghosts)
            players = self._uniform_disk(64)
            far = np.all(np.linalg.norm(players[:, None, :] - ghosts[None], axis=2) >= self.start_distance, axis=1)
            if far.any():
                player = players[int(np.argmax(far))]
                break
        phi = self._rng.uniform(-math.pi, math.pi)
        headings = self._rng.uniform(-math.pi, math.pi, self.num_ghosts)
        speeds = self._rng.uniform(self.ghost_speed_min, self.ghost_speed_max, self.num_ghosts) * self.max_speed
        block = np.concatenate([ghosts, headings[:, None], np.zeros((self.num_ghosts, 1)), ghosts], axis=1)
        self._state = np.concatenate([player, [phi, 0.0], block.ravel(), speeds]).astype(np.float64)
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._before_step()
        selected = self.action_spec.validate(action)
        assert isinstance(selected, np.ndarray) and self._state is not None
        x, y, phi, speed = self._state[:4]
        speed = float(np.clip(speed + float(selected[0]) * self.acceleration, 0.0, self.max_speed))
        phi = float((phi + float(selected[1]) * self.turn_rate + math.pi) % (2.0 * math.pi) - math.pi)
        player = np.array([x + speed * math.cos(phi), y + speed * math.sin(phi)])
        norm = float(np.linalg.norm(player))
        if norm > self.arena_radius:
            player *= self.arena_radius / norm
        speeds = self._speeds().copy()
        block = disk_tag_ghost_step(self._ghost_block(), speeds, player, self._rng.standard_normal(self.num_ghosts),
                                    **self.ghost_params())
        self._state = np.concatenate([player, [phi, speed], block.ravel(), speeds]).astype(np.float64)
        terminated = bool(np.any(np.linalg.norm(block[:, 0:2] - player, axis=1) <= self.capture_radius))
        reward = self.step_reward + (self.capture_reward if terminated else 0.0)
        terminated, truncated = self._finish_step(terminated)
        return self._observation(), reward, terminated, truncated, self._info()


_DISK_TAG_KEYS = {
    "horizon",
    "observation_noise_std",
    "num_ghosts",
    "arena_radius",
    "max_speed",
    "acceleration",
    "turn_rate",
    "ghost_speed_min",
    "ghost_speed_max",
    "ghost_view_angle_deg",
    "ghost_view_range",
    "ghost_wander_ratio",
    "ghost_wander_turn_std",
    "capture_radius",
    "view_angle_deg",
    "view_range",
    "start_distance",
    "step_reward",
    "capture_reward",
    "observation_mode",
    "range_noise_base",
    "range_noise_scale",
}

_CAR_FLAG_KEYS = {
    "horizon",
    "observation_noise_std",
    "max_speed",
    "power",
    "position_limit",
    "flag_position",
    "priest_position",
    "priest_range",
    "start_range",
    "step_cost",
    "flag_reward",
}
_HEAVEN_HELL_KEYS = {"horizon", "observation_noise_std", "corridor_length"}


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
    if key in {"mountainhike", "mountainhikev0", "deathvalley", "deathvalleyv0"}:
        kwargs = _factory_kwargs(config, _MOUNTAIN_HIKE_KEYS)
        return MountainHikeEnv(seed=seed, **kwargs)
    if key in {"carflag", "carflagv0"}:
        kwargs = _factory_kwargs(config, _CAR_FLAG_KEYS)
        return CarFlagEnv(seed=seed, **kwargs)
    if key in {"heavenhell", "heavenhellv0"}:
        kwargs = _factory_kwargs(config, _HEAVEN_HELL_KEYS)
        return HeavenHellEnv(seed=seed, **kwargs)
    if key in {"disktag", "tag"}:
        kwargs = _factory_kwargs(config, _DISK_TAG_KEYS)
        return DiskTagEnv(seed=seed, **kwargs)

    raise ValueError(
        f"Unknown environment {name!r}; expected masked_cartpole, masked_pendulum, "
        "masked_mountain_car_continuous, light_dark, mountain_hike, car_flag, heaven_hell, or disk_tag"
    )
