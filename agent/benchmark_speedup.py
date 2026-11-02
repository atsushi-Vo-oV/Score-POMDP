"""Run the same small CPU campaign with HEAD or the working-tree implementation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=("before", "after", "both"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    if args.version == "both":
        summary = {"platform": platform.platform(), "torch": torch.__version__, "threads": 1}
        for version in ("before", "after"):
            log = root / "agent" / f"speedup-{version}.log"
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), version],
                    cwd=root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            phases = {}
            for phase, update, seconds in re.findall(
                r"phase=(\w+) status=end update=(\d+)/\d+ seconds=([\d.]+)",
                log.read_text(encoding="utf-8"),
            ):
                phases.setdefault(phase, []).append(
                    {"update": int(update), "seconds": float(seconds)}
                )
            summary[version] = phases
        (root / "agent" / "speedup-timings.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact_root = str(root / "agent" / "speedup-benchmark")
        if sys.platform == "win32":
            artifact_root = "\\\\?\\" + artifact_root
        checkpoints = [
            torch.load(
                next(max((Path(artifact_root) / version).iterdir()).rglob("latest.pt")),
                map_location="cpu",
            )
            for version in ("before", "after")
        ]
        before, after = checkpoints
        differences = {
            name: (value - after["model"][name]).abs().max().item()
            for name, value in before["model"].items()
            if value.numel()
        }
        comparison = {
            "model_max_abs": max(differences.values()),
            "largest_parameter": max(differences, key=differences.get),
            "model_allclose_1e-5": all(
                torch.allclose(value, after["model"][name], atol=1e-5, rtol=1e-5)
                for name, value in before["model"].items()
            ),
            "torch_rng_equal": torch.equal(before["torch_rng_state"], after["torch_rng_state"]),
            "python_rng_equal": before["python_rng_state"] == after["python_rng_state"],
            "optimizer_steps_equal": all(
                torch.equal(state["step"], after["optimizer"]["state"][key]["step"])
                for key, state in before["optimizer"]["state"].items()
            ),
        }
        (root / "agent" / "speedup-checkpoint-comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.version == "before":
        # Load the committed modules in memory, without reverting shared files.
        for name in (
            "networks",
            "belief",
            "policies",
            "baselines",
            "ppo",
            "train",
            "train_baselines",
        ):
            path = root / "src" / "sb_pomdp" / f"{name}.py"
            source = subprocess.check_output(
                ["git", "show", f"HEAD:src/sb_pomdp/{name}.py"], cwd=root
            ).decode("utf-8")
            qualified = f"sb_pomdp.{name}"
            spec = importlib.util.spec_from_file_location(qualified, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 - committed reference
    from sb_pomdp.compare import main as compare

    output = str(root / "agent" / "speedup-benchmark" / args.version)
    if sys.platform == "win32":
        output = "\\\\?\\" + output
    command = [
        "--config",
        "config/trysvgd_tagr_tf-ula.json",
        "--methods",
        "score_transformer",
        "--seeds",
        "10",
        "--label",
        f"speed-{args.version}",
    ]
    for override in (
        "experiment.device=cpu",
        "ppo.num_envs=4",
        "ppo.rollout_steps=16",
        "ppo.minibatch_size=64",
        "ppo.sequence_microbatch_size=64",
        "ppo.total_steps=128",
        "model.prediction_pretrain_steps=64",
        "experiment.eval_episodes=1",
        f"experiment.output_dir={json.dumps(output)}",
    ):
        command.extend(("--override", override))
    return compare(command)


if __name__ == "__main__":
    raise SystemExit(main())
