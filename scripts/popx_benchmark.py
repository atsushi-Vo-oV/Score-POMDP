"""Fixed-data CPU scaling: warmup + three segmented updates, excluding evaluation.

The wall clock includes construction/resume/checkpoint I/O. Only A1+C are enabled
to isolate population overhead; this is not a full-QD production time estimate.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from sb_pomdp.config import load_config
from sb_pomdp.popexplore import PopConfig, run_population


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True)
    p.add_argument("--counts", nargs="+", type=int, default=[1,2,4,16])
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    base = load_config("config/trysvgdlight_tagr_tf-svgd.json").with_overrides({
        "experiment.output_dir": str(root), "experiment.device":"cpu", "experiment.deterministic":True,
        "experiment.seeds":[20], "experiment.eval_interval_updates":1000,
        "experiment.checkpoint_interval_updates":1000, "experiment.eval_episodes":1,
        "environment.tasks.disk_tag.horizon":11,
        "model.num_particles":4, "model.langevin_steps":2, "model.energy_hidden":[8],
        "model.d_model":8, "model.num_heads":2, "model.transformer_ff_dim":16,
        "model.policy_hidden":[8], "model.diffusion_steps":2,
        "model.prediction_pretrain_steps":0, "model.prediction_interval_updates":0,
        "model.prediction_phase_updates":0,
        "ppo.num_envs":16, "ppo.rollout_steps":8, "ppo.epochs":1,
        "ppo.minibatch_size":128, "ppo.sequence_microbatch_size":128,
        "ppo.total_steps":(args.repeats+2)*128,
    })
    results = []
    for count in args.counts:
        cfg = PopConfig({**base.to_dict(), "popexplore":{"population_size":count, "init_std":.01 if count > 1 else 0.0}})
        campaign = f"bench-n{count}"
        run_population(cfg, campaign_id=campaign, segment_updates=1, resume=True, verbose=False)
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            run_population(cfg, campaign_id=campaign, segment_updates=1, resume=True, verbose=False)
            times.append(time.perf_counter()-start)
        results.append(dict(population_size=count, num_envs=16, seconds=times,
                            median_seconds=statistics.median(times)))
    payload = dict(torch=torch.__version__, threads=1, scope="A1+C fixed 16 environments; no selection/evaluation", results=results, config=base.to_dict())
    (root/"benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
