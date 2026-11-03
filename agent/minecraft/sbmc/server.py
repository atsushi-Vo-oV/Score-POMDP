"""Minecraft environment server: runs one SBWalk instance in the MineRL venv and serves reset / step over
stdin / stdout (length-prefixed pickles), so a training process in another venv can drive it.

Protocol (client -> server): {"cmd": "reset"} | {"cmd": "step", "action": int} | {"cmd": "close"}
Server -> client: {"obs": uint8 [H, W, 3], "reward": float, "done": bool, "info": {...}} (reset: reward 0, done False)
Launch: python -m sbmc.server --max-steps 200 --action-repeat 4 --seed 0  (stdout is the data channel; logs go to stderr)
"""
from __future__ import annotations

import argparse
import os
import pickle
import struct
import sys


def _send(stream, payload) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)
    stream.flush()


def _recv(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    (length,) = struct.unpack("<I", header)
    return pickle.loads(stream.read(length))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--new-world-every-reset", type=int, default=1)
    args = parser.parse_args()
    # Minecraft and MineRL print to stdout; move the data channel to a private descriptor first.
    data_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    data_in = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
    sys.stdout = sys.stderr
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sbmc.walk_env import WalkForward  # noqa: E402

    env = WalkForward(max_steps=args.max_steps, action_repeat=args.action_repeat, seed=args.seed,
                      new_world_every_reset=bool(args.new_world_every_reset))
    _send(data_out, {"ready": True, "num_actions": int(env.action_space.n), "obs_shape": list(env.observation_space.shape)})
    while True:
        message = _recv(data_in)
        if message is None or message.get("cmd") == "close":
            break
        if message["cmd"] == "reset":
            obs = env.reset()
            _send(data_out, {"obs": obs, "reward": 0.0, "done": False,
                             "info": {"xz": env.position.tolist(), "yaw_heading": env.heading.tolist(), "progress": 0.0}})
        elif message["cmd"] == "step":
            obs, reward, done, info = env.step(int(message["action"]))
            info = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info.items()}
            _send(data_out, {"obs": obs, "reward": float(reward), "done": bool(done), "info": info})
        else:
            _send(data_out, {"error": f"unknown command {message!r}"})
    env.close()


if __name__ == "__main__":
    main()
