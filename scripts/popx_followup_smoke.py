"""Small CPU scratch runs of full/PPO/no_f with learned belief and paired seeds."""
import argparse
import json
from pathlib import Path
import time
import torch
from sb_pomdp.popexplore import run_population
from sb_pomdp.popexplore.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10,11])
    parser.add_argument("--tasks", nargs="+", default=["carflag","mirror2","tagr"])
    parser.add_argument("--ppo-updates", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.output_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    timings = []
    for task in args.tasks:
        for arm in ("full", "ppo", "no_f"):
            cfg = load_config(f"config/popx_scratch_{task}_{arm}.json")
            env_task = cfg.base["experiment"]["tasks"][0]
            cfg = cfg.with_overrides({
                "experiment.output_dir":str(root), "experiment.device":"cpu", "experiment.deterministic":True,
                "experiment.seeds":args.seeds, "experiment.eval_interval_updates":2,
                "experiment.checkpoint_interval_updates":2, "experiment.eval_episodes":2,
                f"environment.tasks.{env_task}.horizon":11,
                "ppo.num_envs":8, "ppo.rollout_steps":8, "ppo.epochs":1,
                "ppo.minibatch_size":64, "ppo.sequence_microbatch_size":32, "ppo.total_steps":args.ppo_updates*64,
                "model.num_particles":3, "model.langevin_steps":1, "model.energy_hidden":[8],
                "model.d_model":8, "model.num_heads":2, "model.transformer_ff_dim":16,
                "model.policy_hidden":[8], "model.diffusion_steps":2, "model.prediction_predictor_hidden":[8],
                "model.prediction_pretrain_steps":64, "model.prediction_interval_updates":2,
                "model.prediction_phase_updates":1,
                "popexplore.population_size":4 if arm != "ppo" else 1,
                "popexplore.fitness_episodes":2, "popexplore.qd_capacity":8,
                "popexplore.rnd_hidden":8, "popexplore.rnd_features":4})
            start = time.perf_counter()
            summaries = run_population(cfg, campaign_id=f"followup-{task}-{arm}", verbose=False)
            result = dict(task=task, arm=arm, seconds=time.perf_counter()-start, summaries=summaries, config=cfg.to_dict())
            timings.append(result)
            print(json.dumps({k:result[k] for k in ("task","arm","seconds")}), flush=True)
    (root / "timings.json").write_text(json.dumps(timings, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
