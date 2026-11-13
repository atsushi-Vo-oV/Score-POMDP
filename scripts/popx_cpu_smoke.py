"""CPU mechanism smoke using the supplied real tagr checkpoint (not efficacy evidence)."""
import argparse
import json
from pathlib import Path
import time

import torch

from sb_pomdp.popexplore import load_config, run_population


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--updates", type=int, default=30)
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for arm in ("full", "ppo"):
        config = load_config(f"config/popx_frozen_tagr_{arm}_s10.json", {
            "model.init_from_checkpoint": args.checkpoint,
            "experiment.output_dir": str(output), "experiment.device":"cpu",
            "experiment.deterministic":True, "experiment.seeds":[20],
            "experiment.eval_episodes":2, "experiment.eval_interval_updates":10,
            "experiment.checkpoint_interval_updates":10,
            "ppo.num_envs":4, "ppo.rollout_steps":16, "ppo.minibatch_size":64,
            "ppo.sequence_microbatch_size":64, "ppo.total_steps":args.updates*64, "ppo.epochs":1,
            "popexplore.population_size":4 if arm == "full" else 1,
            "popexplore.fitness_episodes":2, "popexplore.qd_capacity":8,
        })
        started = time.perf_counter()
        summary = run_population(config, campaign_id=f"popx-tagr-cpu-{arm}-v1", resume=True, verbose=True)
        results.append(dict(arm=arm, seconds=time.perf_counter()-started, summary=summary,
                            updates=args.updates, config=config.to_dict()))
        (output / "timings.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k:v for k,v in r.items() if k != "config"} for r in results], ensure_ascii=False))


if __name__ == "__main__":
    main()
