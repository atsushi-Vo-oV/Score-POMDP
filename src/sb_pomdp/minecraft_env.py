"""Minecraft "walk in one direction" task (MineRL 1.0) as a masked environment.

The Minecraft instance runs in its own interpreter (the MineRL virtual environment) and is driven over a pipe
by :class:`MinecraftWalkEnv` (``backend="minerl"``, see ``agent/minecraft/sbmc/server.py``); the agent sees a
128x128 point of view encoded to a latent vector by a trained convolutional VAE (``vae_checkpoint``, see
``agent/minecraft/scripts/train_vae.py``), never its position.  Reward per step = displacement along the
heading fixed at reset (blocks), as in the standalone ``WalkForward`` environment.  ``backend="fake"`` is an
in-process stand-in with the same interface (a random-walk-free kinematic agent in a bumpy plane rendered to
a synthetic image statistic) for tests and CPU smoke runs; it needs neither Minecraft nor a VAE.

State ``(x, z, yaw, h_x, h_z, progress)``: world position, heading angle, the heading unit vector fixed at reset
and the cumulative progress.  Oracle features ``(progress / scale, lateral / scale, cos(yaw - yaw0), sin(yaw - yaw0))``.
A Minecraft world cannot be restored from a checkpoint: ``load_state_dict`` restores the bookkeeping but the
next ``reset`` starts a fresh world (resumed runs continue with new episodes).
"""
from __future__ import annotations

import contextlib
import math
import os
import pickle
import struct
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .envs import ActionSpec, _BaseMaskedEnv

NUM_ACTIONS = 9  # noop, forward, forward+sprint, forward+jump, back, turn left, turn right, forward-left, forward-right
IMAGE_SHAPE = (128, 128, 3)


class _VAEEncoder:
    """The encoder half of ``train_vae.VAE`` (same architecture), loaded from its checkpoint; CPU inference."""

    def __init__(self, checkpoint: str):
        import torch
        from torch import nn

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        latent = int(payload["latent"])
        channels = [3, 32, 64, 128, 256]
        conv = nn.Sequential(*[m for i in range(4) for m in (nn.Conv2d(channels[i], channels[i + 1], 4, 2, 1), nn.SiLU())])
        fc = nn.Linear(256 * 8 * 8, 2 * latent)
        state = {k[len("encoder."):]: v for k, v in payload["model"].items() if k.startswith("encoder.")}
        conv.load_state_dict({k[len("conv."):]: v for k, v in state.items() if k.startswith("conv.")})
        fc.load_state_dict({k[len("fc."):]: v for k, v in state.items() if k.startswith("fc.")})
        self.conv, self.fc, self.latent = conv.eval(), fc.eval(), latent
        self.mu_mean = payload["mu_mean"].float()
        self.mu_std = payload["mu_std"].float().clamp_min(1e-3)
        self.torch = torch

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(np.ascontiguousarray(frame)).permute(2, 0, 1)[None].float() / 255.0
            mu = self.fc(self.conv(x).flatten(1))[:, : self.latent]
            return ((mu[0] - self.mu_mean) / self.mu_std).numpy().astype(np.float32)


# X display bookkeeping of the private Xvfb servers (module level so tests can redirect it).
_X_LOCK_DIR = Path("/tmp")
_X_SOCKET_DIR = Path("/tmp/.X11-unix")
_X_DISPLAY_MIN, _X_DISPLAY_MAX = 200, 900
_XVFB_ATTEMPTS = 200


def _pid_alive(pid: int) -> bool:
    """Whether a process exists (via ``/proc``; elsewhere every pid counts as alive, so no lock is ever removed)."""
    if pid <= 0:
        return False
    proc = Path("/proc")
    return (proc / str(pid)).exists() if proc.is_dir() else True


class _PipeBackend:
    """Drive ``sbmc/server.py`` (Minecraft in the MineRL venv) over stdin/stdout."""

    def __init__(self, python: str, root: str, *, max_steps: int, action_repeat: int, seed: int, new_world: bool):
        command = [python, "-m", "sbmc.server", "--max-steps", str(max_steps), "--action-repeat", str(action_repeat),
                   "--seed", str(seed), "--new-world-every-reset", str(int(new_world))]
        env = dict(os.environ, PYTHONPATH=root, CUDA_VISIBLE_DEVICES="")
        self.xvfb: subprocess.Popen | None = None
        self.log = open(Path(root) / "logs" / f"server_{os.getpid()}_{seed}.log", "ab")  # noqa: SIM115
        if not os.environ.get("DISPLAY"):
            # A nested xvfb-run hangs on its display lock; start a private Xvfb instead.
            env["DISPLAY"] = self._start_xvfb(seed)
        self.process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log)
        hello = self._recv()
        if not hello or not hello.get("ready"):
            raise RuntimeError(f"Minecraft server did not start: {hello!r}")
        self.num_actions = int(hello["num_actions"])

    @staticmethod
    def _stale_display(number: int) -> bool:
        """True when the lock of display ``number`` names a process that no longer exists.

        An Xvfb that was killed (SIGKILL) leaves ``/tmp/.X<N>-lock`` and ``/tmp/.X11-unix/X<N>`` behind, and a
        new Xvfb refuses that display.  A lock whose pid is dead can be removed and the number reused.
        """
        try:
            pid = int((_X_LOCK_DIR / f".X{number}-lock").read_text().strip() or "0")
        except (OSError, ValueError):
            return False
        return pid > 0 and not _pid_alive(pid)

    def _start_xvfb(self, seed: int) -> str:
        """Start an X virtual framebuffer on a free display number and return ``:N``.

        The candidates are drawn from the whole range so that the leftovers of earlier servers of this process
        (every evaluation starts and stops its own instances) cannot exhaust a fixed band of numbers; a stale
        lock (dead pid) is removed first instead of being skipped.
        """
        import random
        import time

        rng = random.Random(os.getpid() * 7 + seed * 13 + time.time_ns())
        for _ in range(_XVFB_ATTEMPTS):
            number = rng.randrange(_X_DISPLAY_MIN, _X_DISPLAY_MAX)
            lock = _X_LOCK_DIR / f".X{number}-lock"
            if lock.exists():
                if not self._stale_display(number):
                    continue
                with contextlib.suppress(OSError):
                    lock.unlink()
                with contextlib.suppress(OSError):
                    (_X_SOCKET_DIR / f"X{number}").unlink()
            proc = subprocess.Popen(["Xvfb", f":{number}", "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
                                    stdout=self.log, stderr=self.log)
            time.sleep(1.5)
            if proc.poll() is None:
                self.xvfb = proc
                return f":{number}"
        raise RuntimeError("could not start Xvfb for the Minecraft server")

    def _stop_xvfb(self) -> None:
        """Stop the private Xvfb gracefully (SIGTERM lets it remove its lock file and socket)."""
        proc, self.xvfb = self.xvfb, None
        if proc is None:
            return
        with contextlib.suppress(Exception):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    def _send(self, payload: Mapping[str, Any]) -> None:
        data = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
        assert self.process.stdin is not None
        self.process.stdin.write(struct.pack("<I", len(data)) + data)
        self.process.stdin.flush()

    def _recv(self):
        assert self.process.stdout is not None
        header = self.process.stdout.read(4)
        if len(header) < 4:
            return None
        (length,) = struct.unpack("<I", header)
        return pickle.loads(self.process.stdout.read(length))

    def reset(self):
        self._send({"cmd": "reset"})
        return self._recv()

    def step(self, action: int):
        self._send({"cmd": "step", "action": int(action)})
        return self._recv()

    def close(self) -> None:
        try:
            self._send({"cmd": "close"})
            self.process.wait(timeout=30)
        except Exception:  # noqa: BLE001 - a hung server is simply killed
            self.process.kill()
        self._stop_xvfb()
        with contextlib.suppress(Exception):
            self.log.close()


class _FakeBackend:
    """Kinematic stand-in: a walker on a plane with random obstacles; the frame is a synthetic 128x128 image whose
    statistics depend on the pose, so a trivial 'encoder' (mean colour of 4 x 4 blocks) yields a 48-d observation."""

    def __init__(self, *, max_steps: int, action_repeat: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.max_steps, self.action_repeat = max_steps, action_repeat
        self.num_actions = NUM_ACTIONS
        self.pos = np.zeros(2)
        self.yaw = 0.0
        self.heading = np.array([0.0, 1.0])
        self.start = np.zeros(2)
        self.steps = 0
        self.obstacles = np.zeros((0, 2))

    @staticmethod
    def _heading(yaw_deg: float) -> np.ndarray:
        yaw = math.radians(yaw_deg)
        return np.array([-math.sin(yaw), math.cos(yaw)])

    def _frame(self) -> np.ndarray:
        # a horizon line whose height depends on nothing, ground texture scrolling with position and a wall
        # of obstacles whose screen position depends on the pose: enough structure for a latent to be useful
        h, w = IMAGE_SHAPE[:2]
        sky = np.broadcast_to((np.arange(h)[:, None] < h // 2).astype(np.float32), (h, w))
        phase = (self.pos[0] * 7 + self.pos[1] * 5 + self.yaw / 4.0) % 16
        texture = np.broadcast_to(0.5 + 0.5 * np.sin((np.arange(w)[None, :] + phase) / 2.0), (h, w))
        frame = np.stack([sky * 0.6 + (1 - sky) * texture * 0.4, sky * 0.8 + (1 - sky) * texture * 0.6, sky * 1.0 + (1 - sky) * 0.2], -1).copy()
        # a sun whose screen column follows the yaw, so turning alone changes the image
        sun = int((self.yaw + 180.0) / 360.0 * w) % w
        frame[8:24, max(0, sun - 8): sun + 8] = (1.0, 0.95, 0.6)
        for o in self.obstacles:
            rel = o - self.pos
            forward = float(np.dot(rel, self.heading))
            if forward <= 0.5:
                continue
            right = float(rel[0] * self.heading[1] - rel[1] * self.heading[0])
            column = int(w / 2 + right / forward * w / 2)
            size = int(min(h // 2, 40 / forward))
            if 0 <= column < w:
                frame[h // 2 - size: h // 2 + size, max(0, column - size // 2): column + size // 2] = (0.4, 0.25, 0.1)
        return (frame * 255).astype(np.uint8)

    def reset(self):
        self.pos = self.rng.uniform(-50, 50, 2)
        self.yaw = float(self.rng.uniform(-180, 180))
        self.heading = self._heading(self.yaw)
        self.start = self.pos.copy()
        self.steps = 0
        self.obstacles = self.pos + self.rng.uniform(-30, 30, (12, 2))
        return {"obs": self._frame(), "reward": 0.0, "done": False, "info": {"xz": self.pos.tolist(), "yaw": self.yaw, "progress": 0.0}}

    def step(self, action: int):
        before = self.pos.copy()
        turn = {5: -5.0, 6: 5.0, 7: -5.0, 8: 5.0}.get(action, 0.0) * self.action_repeat
        self.yaw = (self.yaw + turn + 180) % 360 - 180
        speed = {1: 0.22, 2: 0.28, 3: 0.2, 4: -0.2, 7: 0.22, 8: 0.22}.get(action, 0.0) * self.action_repeat
        move = self._heading(self.yaw) * speed
        proposed = self.pos + move
        blocked = any(np.linalg.norm(proposed - o) < 1.0 for o in self.obstacles)
        if not blocked:
            self.pos = proposed
        self.steps += 1
        reward = float(np.dot(self.pos - before, self.heading))
        done = self.steps >= self.max_steps
        info = {"xz": self.pos.tolist(), "yaw": self.yaw, "progress": float(np.dot(self.pos - self.start, self.heading)), "action": int(action)}
        return {"obs": self._frame(), "reward": reward, "done": done, "info": info}

    def close(self) -> None:
        pass


def _fake_encoder(frame: np.ndarray) -> np.ndarray:
    blocks = frame.reshape(4, 32, 4, 32, 3).mean(axis=(1, 3)).reshape(-1) / 255.0  # 4 x 4 x 3 = 48
    return (blocks * 2.0 - 1.0).astype(np.float32)


class MinecraftWalkEnv(_BaseMaskedEnv):
    """See the module docstring."""

    state_dim = 6
    oracle_dim = 4

    def __init__(
        self,
        *,
        horizon: int = 200,
        observation_noise_std: float = 0.0,
        seed: int | None = None,
        action_repeat: int = 4,
        latent_dim: int = 32,
        vae_checkpoint: str = "",
        minecraft_python: str = "",
        minecraft_root: str = "",
        new_world_every_reset: bool = True,
        progress_scale: float = 10.0,
        backend: str = "minerl",
    ) -> None:
        super().__init__(horizon=horizon, observation_noise_std=observation_noise_std, seed=seed)
        if backend not in {"minerl", "fake"}:
            raise ValueError("backend must be minerl or fake")
        if int(action_repeat) <= 0 or int(latent_dim) <= 0:
            raise ValueError("action_repeat and latent_dim must be positive")
        self.backend_kind = backend
        self.action_repeat = int(action_repeat)
        self.progress_scale = float(progress_scale)
        self.action_spec = ActionSpec.discrete(NUM_ACTIONS)
        self._episode_seed = int(self._rng.integers(0, 2**31 - 1)) if seed is None else int(seed)
        if backend == "fake":
            self.observation_dim = 48
            self._encode = _fake_encoder
            self._backend: Any = _FakeBackend(max_steps=horizon, action_repeat=self.action_repeat, seed=self._episode_seed)
        else:
            if not vae_checkpoint or not minecraft_python or not minecraft_root:
                raise ValueError("the minerl backend needs vae_checkpoint, minecraft_python and minecraft_root")
            encoder = _VAEEncoder(vae_checkpoint)
            if encoder.latent != int(latent_dim):
                raise ValueError(f"vae_checkpoint has latent {encoder.latent}, config says {latent_dim}")
            self.observation_dim = int(latent_dim)
            self._encode = encoder
            self._backend = _PipeBackend(minecraft_python, minecraft_root, max_steps=horizon, action_repeat=self.action_repeat,
                                         seed=self._episode_seed, new_world=bool(new_world_every_reset))
        self._yaw0 = 0.0
        self._last_frame: np.ndarray | None = None

    # -- state bookkeeping -------------------------------------------------------------------------
    def _set_state(self, info: Mapping[str, Any], heading: np.ndarray, progress: float) -> None:
        x, z = info["xz"]
        yaw = float(info.get("yaw", self._yaw0))
        self._state = np.array([x, z, yaw, heading[0], heading[1], progress], dtype=np.float64)

    def _observation(self, frame: np.ndarray) -> np.ndarray:
        self._last_frame = frame
        return self._add_observation_noise(np.asarray(self._encode(frame), dtype=np.float64))

    def last_frame(self) -> np.ndarray | None:
        """The most recent raw 128x128 frame (for viewers; not part of the observation)."""
        return None if self._last_frame is None else self._last_frame.copy()

    def oracle_features(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("The environment has not been reset")
        x, z, yaw, hx, hz, progress = self._state
        lateral = (x - self._start[0]) * hz - (z - self._start[1]) * hx
        rel = math.radians(yaw - self._yaw0)
        return np.array([progress / self.progress_scale, lateral / self.progress_scale, math.cos(rel), math.sin(rel)], dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info["progress"] = float(self._state[5])
        return info

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._start_reset(seed)
        result = self._backend.reset()
        info = result["info"]
        heading = np.asarray(info.get("yaw_heading", _FakeBackend._heading(float(info.get("yaw", 0.0)))), dtype=np.float64)
        self._yaw0 = float(info.get("yaw", 0.0))
        self._start = np.asarray(info["xz"], dtype=np.float64)
        self._set_state(info, heading, 0.0)
        return self._observation(result["obs"]), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._before_step()
        selected = int(self.action_spec.validate(action))
        result = self._backend.step(selected)
        info = result["info"]
        assert self._state is not None
        heading = self._state[3:5]
        self._set_state(info, heading, float(info.get("progress", self._state[5] + result["reward"])))
        terminated, truncated = self._finish_step(bool(result["done"]) and self._elapsed_steps + 1 < self.horizon)
        return self._observation(result["obs"]), float(result["reward"]), terminated, truncated, self._info()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        # The world itself is not restorable: the restored bookkeeping describes an
        # episode that no longer exists, so ``needs_reset`` turns true and the
        # trainers' resume path (``train.reset_unresumable_environments``) starts a
        # fresh episode here, discarding the interrupted one.
        self._needs_reset = True

    def close(self) -> None:
        self._backend.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        with contextlib.suppress(Exception):
            self.close()


_MINECRAFT_KEYS = {
    "horizon", "observation_noise_std", "action_repeat", "latent_dim", "vae_checkpoint", "minecraft_python",
    "minecraft_root", "new_world_every_reset", "progress_scale", "backend",
}
