"""SBWalk: a MineRL 1.0 task.  Image in, discrete move out, reward = progress along the initial heading.

Episode: a freshly generated world (random spawn), the agent is turned to a random yaw at reset, and the
heading unit vector h is fixed from that yaw.  Reward per agent step = <p_t - p_{t-1}, h> in blocks
(horizontal displacement projected on the initial heading; walking back is negative).  Observation =
RGB 128x128 point of view.  Position / yaw come from MineRL's location stats and are oracle-only: they
are returned in `info`, never in the observation.
"""
from __future__ import annotations

import math
import sys
import time
from typing import List

import gym
import numpy as np
from minerl.herobraine.env_specs.human_controls import SimpleHumanEmbodimentEnvSpec
from minerl.herobraine.hero import handlers as H, mc
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler

# Minecraft draws the HUD (hearts, hunger, hotbar) over the bottom ~30 rows; render taller and keep the top square.
RENDER_RESOLUTION = (128, 240)  # (width, height) handed to the Minecraft video producer; toasts sit in the top rows, HUD in the bottom rows (guiScale 1)
RESOLUTION = (128, 128)  # (width, height) of the observation after cropping the HUD away

# World generation on a loaded machine can exceed MineRL's socket timeout (SOCKTIME, 60 s by default);
# give the Minecraft process five minutes before a call is declared hung.
try:
    from minerl.env import _multiagent as _minerl_multiagent

    _minerl_multiagent.SOCKTIME = max(float(_minerl_multiagent.SOCKTIME), 300.0)
except Exception:  # pragma: no cover - older / newer MineRL layouts
    pass
CROP_TOP = 64
# Largest horizontal displacement (blocks) a player can make in one Minecraft tick (sprint-jumping is
# about 0.4; falling adds nothing horizontally).  A larger per-step displacement is a location glitch.
MAX_STEP_DISPLACEMENT = 0.6
# (name, keyboard keys held, camera yaw delta in degrees per Minecraft tick)
DISCRETE_ACTIONS = (
    ("noop", (), 0.0),
    ("forward", ("forward",), 0.0),
    ("forward_sprint", ("forward", "sprint"), 0.0),
    ("forward_jump", ("forward", "jump"), 0.0),
    ("back", ("back",), 0.0),
    ("turn_left", (), -5.0),
    ("turn_right", (), 5.0),
    ("forward_left", ("forward",), -5.0),
    ("forward_right", ("forward",), 5.0),
)


class SBWalkEnvSpec(SimpleHumanEmbodimentEnvSpec):
    def __init__(self, name="SBWalk-v0", max_episode_steps=4000, biome="plains", new_world_every_reset=True):
        self.biome = biome
        self.new_world_every_reset = new_world_every_reset
        super().__init__(name, max_episode_steps=max_episode_steps, resolution=RENDER_RESOLUTION,
                         fov_range=[70, 70], gamma_range=[2, 2], guiscale_range=[1, 1], cursor_size_range=[0, 0])

    def create_observables(self) -> List[TranslationHandler]:
        return [H.POVObservation(self.resolution), H.ObservationFromCurrentLocation()]

    def create_rewardables(self) -> List[TranslationHandler]:
        return []

    def create_agent_start(self) -> List[Handler]:
        return super().create_agent_start() + [H.PreferredSpawnBiome(self.biome), H.DoneOnDeath()]

    def create_agent_handlers(self) -> List[Handler]:
        return []

    def create_monitors(self) -> List[TranslationHandler]:
        return []

    def create_server_world_generators(self) -> List[Handler]:
        return [H.DefaultWorldGenerator(force_reset=self.new_world_every_reset)]

    def create_server_quit_producers(self) -> List[Handler]:
        return [H.ServerQuitFromTimeUp(self.max_episode_steps * mc.MS_PER_STEP), H.ServerQuitWhenAnyAgentFinishes()]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        return [H.TimeInitialCondition(allow_passage_of_time=False, start_time=6000),
                H.SpawningInitialCondition(allow_spawning=False)]

    def is_from_folder(self, folder: str) -> bool:
        return False

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return False

    def get_docstring(self):
        return __doc__


_REGISTERED = set()


def register(spec: SBWalkEnvSpec):
    if spec.name not in _REGISTERED:
        spec.register()
        _REGISTERED.add(spec.name)


class WalkForward(gym.Env):
    """Discrete-action wrapper around the MineRL env with the heading-progress reward."""

    metadata = {"render.modes": []}

    def __init__(self, max_steps=200, action_repeat=4, seed=0, new_world_every_reset=True, turn_at_reset=True):
        self.spec_obj = SBWalkEnvSpec(max_episode_steps=(max_steps + 50) * action_repeat,
                                      new_world_every_reset=new_world_every_reset)
        register(self.spec_obj)
        self.env = gym.make(self.spec_obj.name)
        self.max_steps = max_steps
        self.action_repeat = action_repeat
        self.turn_at_reset = turn_at_reset
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(0, 255, (RESOLUTION[1], RESOLUTION[0], 3), np.uint8)
        self.action_space = gym.spaces.Discrete(len(DISCRETE_ACTIONS))
        self.heading = np.zeros(2)
        self.position = np.zeros(2)
        self.start_position = np.zeros(2)
        self.steps = 0

    # Minecraft yaw: 0 = +z (south), 90 = -x (west); the horizontal unit vector the player faces.
    @staticmethod
    def _heading(yaw_deg: float) -> np.ndarray:
        yaw = math.radians(yaw_deg)
        return np.array([-math.sin(yaw), math.cos(yaw)])

    @staticmethod
    def _xz(obs) -> np.ndarray:
        loc = obs["location_stats"]
        return np.array([float(loc["xpos"]), float(loc["zpos"])])

    def _minerl_action(self, index: int, camera_yaw=None):
        _name, keys, yaw = DISCRETE_ACTIONS[index]
        action = self.env.action_space.noop()
        for key in keys:
            action[key] = 1
        action["camera"] = np.array([0.0, yaw if camera_yaw is None else camera_yaw], dtype=np.float32)
        return action

    @staticmethod
    def _frame(obs) -> np.ndarray:
        """The HUD-free top square of the rendered point of view."""
        return np.ascontiguousarray(obs["pov"][CROP_TOP:CROP_TOP + RESOLUTION[1], :RESOLUTION[0]])

    def _restart(self, reason: Exception) -> None:
        """Replace a hung / crashed Minecraft instance (MineRL socket timeouts happen on world generation)."""
        print(f"[sbmc] restarting Minecraft after {type(reason).__name__}: {reason}", file=sys.stderr, flush=True)
        try:
            self.env.close()
        except Exception:
            pass
        time.sleep(5.0)
        self.env = gym.make(self.spec_obj.name)

    def reset(self):
        obs = None
        for attempt in range(4):
            try:
                obs = self.env.reset()
                break
            except Exception as exc:  # noqa: BLE001 - any MineRL failure gets a fresh instance
                if attempt == 3:
                    raise
                self._restart(exc)
        if self.turn_at_reset:
            # turn to a uniformly random yaw over a few ticks (camera deltas are clipped per tick)
            total = float(self.rng.uniform(-180.0, 180.0))
            for _ in range(4):
                obs, _, done, _ = self.env.step(self._minerl_action(0, camera_yaw=total / 4))
                if done:
                    break
        yaw = float(obs["location_stats"]["yaw"])
        self.heading = self._heading(yaw)
        self.position = self._xz(obs)
        self.start_position = self.position.copy()
        self.steps = 0
        return self._frame(obs)

    def step(self, index: int):
        done, obs = False, None
        try:
            for _ in range(self.action_repeat):
                obs, _, done, _ = self.env.step(self._minerl_action(int(index)))
                if done:
                    break
        except Exception as exc:  # noqa: BLE001 - end the episode, the next reset gets a fresh instance
            self._restart(exc)
            self.steps = self.max_steps
            info = {"xz": self.position.copy(), "progress": float(np.dot(self.position - self.start_position, self.heading)),
                    "yaw": float("nan"), "ypos": float("nan"), "action": DISCRETE_ACTIONS[int(index)][0], "restarted": True}
            return np.zeros((RESOLUTION[1], RESOLUTION[0], 3), np.uint8), 0.0, True, info
        position = self._xz(obs)
        displacement = float(np.dot(position - self.position, self.heading))
        ended_by_minecraft = bool(done)
        if ended_by_minecraft:
            # MineRL ended the episode itself (DoneOnDeath, time up): the location stats of that final
            # observation are not the player's position any more (respawn / teardown values hundreds of
            # thousands of blocks away were observed), so the step earns nothing and keeps the last position.
            reward = 0.0
            position = self.position.copy()
        elif abs(displacement) > MAX_STEP_DISPLACEMENT * self.action_repeat:
            # A displacement no walking or falling player can make in action_repeat ticks: a location glitch.
            print(f"[sbmc] implausible displacement {displacement:.1f} blocks in one step ignored", file=sys.stderr, flush=True)
            reward = 0.0
            position = self.position.copy()
        else:
            reward = displacement
        self.position = position
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        info = {"xz": position.copy(), "progress": float(np.dot(position - self.start_position, self.heading)),
                "yaw": float(obs["location_stats"]["yaw"]), "ypos": float(obs["location_stats"]["ypos"]),
                "action": DISCRETE_ACTIONS[int(index)][0], "ended_by_minecraft": ended_by_minecraft}
        return self._frame(obs), reward, bool(done), info

    def close(self):
        self.env.close()
