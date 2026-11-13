"""Population-exploration arms on the v2 default of the light TF-SVGD (try-svgd-light2, experiment 23).

Reads the 36 light configs written by popx_generate.py (which stay untouched), applies the five v2 model keys and writes
config/popx2_scratch_<task>_<arm>.json, jobs/popx/popx2_scratch_<task>_<arm>_r<seed>.sh (campaign popx2-<task>-<arm>-v1)
and jobs/popx/manifest_v2.json.  The two GRU arms have no score belief, so their popx jobs are reused unchanged.
"""
import json
from pathlib import Path

from sb_pomdp.popexplore.config import PopConfig
from sb_pomdp.popexplore.experiments import PRODUCTION_ARMS
from sb_pomdp.popexplore.runner import configured_updates

V2 = {"langevin_proposal": "uniform", "langevin_initial_noise": "uniform", "langevin_observation_anchor": False,
      "direct_score_bound": None, "initial_score_prior": 0.0}


def main():
    root = Path(__file__).absolute().parents[1]
    manifest = []
    for task in ("carflag", "mirror2", "tagr"):
        default = json.loads((root / f"config/trysvgdlight2_{task}_tf-svgd.json").read_text(encoding="utf-8"))
        assert all(default["model"][k] == v for k, v in V2.items())
        for arm in PRODUCTION_ARMS:
            cfg = json.loads((root / f"config/popx_scratch_{task}_{arm}.json").read_text(encoding="utf-8"))
            cfg["model"].update(V2)
            updates = configured_updates(PopConfig(cfg))
            assert updates == (1293 if task == "tagr" else 333)
            name = f"popx2_scratch_{task}_{arm}"
            config, campaign = f"config/{name}.json", f"popx2-{task}-{arm}-v1"
            (root / config).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
            for bulk, seed in enumerate(range(10, 15), 1):
                job = f"jobs/popx/{name}_r{seed}.sh"
                script = f"""#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S
# Population exploration on the v2 default of the light TF-SVGD (uniform chain start at every step, no observation
# anchor, unbounded score without the t = 0 -x term); otherwise identical to popx_scratch_{task}_{arm}.
set -euo pipefail
export SB_POMDP_ENTRY=popexplore
export SB_POMDP_CONFIG={config}
export SB_POMDP_CAMPAIGN_ID={campaign}
export SB_POMDP_SHARD_INDEX={bulk}
export SB_POMDP_BULK_INDEX={bulk}
export SB_POMDP_SEGMENT_UPDATES="${{SB_POMDP_SEGMENT_UPDATES:-{updates}}}"
if [[ "${{SB_POMDP_SOURCE_ONLY:-0}}" != 1 ]]; then
  cd "${{PJM_O_WORKDIR:-$PWD}}"
  exec bash jobs/popx/launch.sh
fi
"""
                (root / job).write_text(script, encoding="utf-8", newline="\n")
                manifest.append(dict(group="scratch", task=task, arm=arm, seed=seed, config=config, campaign=campaign,
                                     bulk_index=bulk, shard_index=bulk, updates=updates, job=job))
    (root / "jobs/popx/manifest_v2.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Generated", len({r["config"] for r in manifest}), "configs and", len(manifest), "jobs on the v2 default")


if __name__ == "__main__":
    main()
