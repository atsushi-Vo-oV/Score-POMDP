"""Measure one end-to-end PPO update after one warmup update (no evaluation)."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from uuid import uuid4

import torch

from ..compare import run_comparisons
from ..config import load_config
from ..train import resolve_device
from .runner import run_seedbatch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/trysvgdlight_mirror2_tf-svgd.json")
    parser.add_argument("--output-dir", default="results/seedbatch-bench")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)
    if args.repeats < 1 or any(n < 1 for n in args.counts):
        parser.error("counts and repeats must be positive")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    config = load_config(args.config, args.override)
    size = config["ppo"]["num_envs"] * config["ppo"]["rollout_steps"]
    changes = {
        "experiment.output_dir": args.output_dir,
        "experiment.eval_interval_updates": 1_000_000,
        "experiment.checkpoint_interval_updates": 1_000_000,
        "model.prediction_pretrain_steps": 0,
        "model.prediction_interval_updates": 0,
        "model.prediction_phase_updates": 0,
        "ppo.total_steps": size * 3,
    }
    if args.device:
        changes["experiment.device"] = args.device
    config = config.with_overrides(changes)
    device = resolve_device(config["experiment"]["device"])
    seeds = config["experiment"]["seeds"]
    if max(args.counts) > len(seeds) or len(config["experiment"]["tasks"]) != 1:
        parser.error("config must provide enough seeds and exactly one task")
    task = config["experiment"]["tasks"][0]
    token = uuid4().hex[:10]
    measurements = []

    def timed(operation):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - start

    for count in args.counts:
        for repeat in range(args.repeats):
            campaign = f"bench-{token}-n{count}-r{repeat}"

            def batched():
                return run_seedbatch(config, campaign_id=campaign, seeds=seeds[:count],
                                     resume=True, segment_updates=1, verbose=False)

            def individual():
                for index, seed in enumerate(seeds[:count], 1):
                    run_comparisons(
                        config, methods=["score_transformer"], tasks=[task], seeds=[seed],
                        gradient_modes=["full"], max_updates=None, label=f"single-{seed}",
                        verbose=False, resume=True, segment_updates=1,
                        run_dir=Path(args.output_dir) / f"{campaign}-single" / f"shard{index:03d}",
                    )

            operations = [("seedbatch", batched)]
            if not args.skip_single:
                operations.append(("individual", individual))
            # Alternate measurement order to reduce systematic first-run bias.
            if repeat % 2:
                operations.reverse()
            for mode, operation in operations:
                operation()  # warmup update 1
                elapsed = timed(operation)  # measured update 2, configured target 3
                row = {"mode": mode, "seeds": count, "repeat": repeat,
                       "seconds": elapsed, "seconds_per_seed": elapsed / count}
                measurements.append(row)
                print(json.dumps(row), flush=True)
    medians = []
    for count in args.counts:
        modes = {row["mode"] for row in measurements if row["seeds"] == count}
        row = {"seeds": count}
        for mode in sorted(modes):
            row[mode + "_seconds"] = statistics.median(
                item["seconds"] for item in measurements if item["seeds"] == count and item["mode"] == mode)
        if "individual" in modes:
            row["throughput_speedup"] = row["individual_seconds"] / row["seedbatch_seconds"]
        medians.append(row)
    result = {
        "torch": torch.__version__, "device": str(device), "threads": torch.get_num_threads(),
        "definition": "One resumed PPO update after one warmup; includes setup, checkpoint load/save and CSV I/O; excludes evaluation. Individual seeds run sequentially in the same process.",
        "config": config.to_dict(), "measurements": measurements, "medians": medians,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"medians": medians}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
