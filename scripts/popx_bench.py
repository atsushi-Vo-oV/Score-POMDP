"""End-to-end GPU timing for PPO/CEM/full on tagr and mirror2.

Default stress cadence executes selection, fitness and periodic evaluation every
update. --cadence production retains their normal intervals (measure >=25 PPO
updates); startup pretraining is omitted in either profile, explicitly recorded.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

import torch
from sb_pomdp.popexplore import run_population
from sb_pomdp.popexplore.config import load_config


def benchmark(args):
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("warmup/repeats must be positive")
    if args.cadence == "production" and args.repeats < 25:
        raise ValueError("production cadence requires >=25 measured PPO updates to include periodic evaluation")
    if not args.cpu_small and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for math timing; --cpu-small is only a plumbing smoke test")
    if args.cpu_small:
        torch.set_num_threads(1)
    root = Path(args.output_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for task in args.tasks:
        for arm in args.arms:
            cfg = load_config(f"config/popx_scratch_{task}_{arm}.json")
            env_task = cfg.base["experiment"]["tasks"][0]
            changes = {"experiment.output_dir":str(root), "experiment.seeds":[10],
                       "experiment.device":"cpu" if args.cpu_small else "cuda",
                       "model.prediction_pretrain_steps":0,
                       "ppo.total_steps":(args.warmup+args.repeats)*64*128}
            if args.fitness_from_rollout:
                changes["popexplore.fitness_from_rollout"] = True
            if args.cadence == "stress":
                changes.update({"popexplore.selection_interval":1, "experiment.eval_interval_updates":1,
                                "experiment.checkpoint_interval_updates":1})
            if args.cpu_small:
                changes.update({"ppo.num_envs":4, "ppo.rollout_steps":3, "ppo.epochs":1,
                                "ppo.total_steps":(args.warmup+args.repeats)*12,
                                "ppo.minibatch_size":12, "ppo.sequence_microbatch_size":6,
                                "popexplore.population_size":2 if arm not in {"ppo", "gru_ppo"} else 1,
                                "popexplore.fitness_episodes":1, "popexplore.qd_capacity":4,
                                "experiment.eval_episodes":1, f"environment.tasks.{env_task}.horizon":5,
                                "model.num_particles":3, "model.langevin_steps":1, "model.energy_hidden":[4],
                                "model.d_model":8, "model.num_heads":2, "model.transformer_ff_dim":8,
                                "model.policy_hidden":[8], "model.diffusion_steps":2, "model.prediction_predictor_hidden":[8]})
                if arm.startswith("gru_"):
                    changes.update({"comparison.gru_encoder_hidden":[8], "comparison.gru_hidden_dim":6})
            cfg = cfg.with_overrides(changes)
            start = time.perf_counter()
            run_population(cfg, campaign_id=f"bench-{task}-{arm}", verbose=False)
            wall = time.perf_counter() - start
            run = root / f"campaigns/bench-{task}-{arm}/shard001/full/{cfg.method}/{env_task}/seed_010"
            with (run / "timing.csv").open(encoding="utf-8", newline="") as f:
                timings = list(csv.DictReader(f))
            measured = [r for r in timings if r["phase"] == "ppo"][args.warmup:]
            assert len(measured) == args.repeats
            assert any(int(r["evaluated"]) for r in measured)
            if cfg.pop["population_size"] > 1 and cfg.pop["update_rule"] != "none" and not args.fitness_from_rollout:
                assert any(int(r["selection_replaced"]) for r in measured)
            seconds = [float(r["seconds"]) for r in measured]
            result = dict(task=task, arm=arm, seconds=seconds, mean_seconds=statistics.mean(seconds),
                          median_seconds=statistics.median(seconds), min_seconds=min(seconds), max_seconds=max(seconds),
                          samples=measured, total_wall_seconds=wall,
                          untimed_setup_summary_seconds=wall-sum(float(r["seconds"]) for r in timings),
                          config=cfg.to_dict())
            results.append(result)
    payload = dict(device="cpu-small" if args.cpu_small else torch.cuda.get_device_name(), torch=torch.__version__,
                   cadence=args.cadence, warmup=args.warmup, repeats=args.repeats,
                   scope="continuous synchronized updates: rollout + learning + fitness + selection + evaluation + checkpoint; excludes construction; pretrain omitted",
                   results=results)
    output = root / "benchmark.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), results=[{k:r[k] for k in ("task","arm","mean_seconds","median_seconds")} for r in results])))
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cadence", choices=["stress","production"], default="stress")
    p.add_argument("--tasks", nargs="+", choices=["tagr","mirror2","carflag"], default=["tagr","mirror2"])
    p.add_argument("--arms", nargs="+", choices=["full","no_f","no_a2","no_c","no_d","no_qd","cem","es","pbt","a2","d","ppo","gru_ppo","gru_full"], default=["ppo","cem","full"])
    p.add_argument("--cpu-small", action="store_true")
    p.add_argument("--fitness-from-rollout", action="store_true",
                   help="experimental behavior-return fitness; no extra fitness episodes, missing data skips selection")
    benchmark(p.parse_args())


if __name__ == "__main__":
    main()
