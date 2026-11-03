"""Collect SBWalk frames with a random (action-persistent) policy for VAE training.
Usage: collect_frames.py <out.npz> <frames> <seed> [--max-steps 200] [--repeat 4]
Each process runs one Minecraft instance; run several processes in parallel with different seeds.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from sbmc.walk_env import WalkForward  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("frames", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--repeat", type=int, default=4)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    env = WalkForward(max_steps=a.max_steps, action_repeat=a.repeat, seed=a.seed)
    frames, actions, rewards, episode_ids, xz, yaw = [], [], [], [], [], []
    episode = 0
    t0 = time.time()
    while len(frames) < a.frames:
        obs = env.reset()
        action = 1
        for t in range(a.max_steps):
            # persist each action for a geometric number of steps so the camera sweeps and the agent walks
            if rng.random() < 0.3:
                action = int(rng.integers(env.action_space.n))
            frames.append(obs)
            actions.append(action)
            obs, r, done, info = env.step(action)
            rewards.append(r)
            episode_ids.append(episode)
            xz.append(info["xz"])
            yaw.append(info["yaw"])
            if done or len(frames) >= a.frames:
                break
        episode += 1
        print(f"episode {episode} frames {len(frames)} elapsed {time.time() - t0:.0f}s", flush=True)
        if episode % 5 == 0:
            save(a.out + ".partial.npz", frames, actions, rewards, episode_ids, xz, yaw)
    env.close()
    save(a.out, frames, actions, rewards, episode_ids, xz, yaw)
    print("saved", a.out, len(frames), "frames")


def save(path, frames, actions, rewards, episode_ids, xz, yaw) -> None:
    np.savez_compressed(path, frames=np.asarray(frames, dtype=np.uint8), actions=np.asarray(actions, np.int64),
                        rewards=np.asarray(rewards, np.float32), episode=np.asarray(episode_ids, np.int64),
                        xz=np.asarray(xz, np.float32), yaw=np.asarray(yaw, np.float32))


if __name__ == "__main__":
    main()
