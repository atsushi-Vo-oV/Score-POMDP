"""try-svgd-light2: the v2 default of the light TF-SVGD (uniform start at every step, no anchor, unbounded score)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sb_pomdp.config import load_config

TASKS = {"mirror2": 333, "tps": 213, "tagr": 1293, "carflag": 333}
V2 = {"langevin_proposal": "uniform", "langevin_initial_noise": "uniform", "langevin_observation_anchor": False,
      "direct_score_bound": None, "initial_score_prior": 0.0}


@pytest.mark.parametrize("task", sorted(TASKS))
def test_v2_default_differs_from_v1_only_in_the_chain_start_anchor_and_bound(task):
    v1 = load_config(f"config/trysvgdlight_{task}_tf-svgd.json").to_dict()
    v2 = load_config(f"config/trysvgdlight2_{task}_tf-svgd.json").to_dict()
    assert v1["model"]["langevin_observation_anchor"] is True and v1["model"]["direct_score_bound"] == 10.0
    assert {k: v2["model"][k] for k in V2} == V2
    assert v2["model"]["langevin_warm_start"] is False and v2["model"]["score_parameterization"] == "direct"
    for key in V2:
        v2["model"][key] = v1["model"][key]
    v2["experiment"]["label"] = v1["experiment"]["label"]
    assert v2 == v1


@pytest.mark.parametrize("task", sorted(TASKS))
def test_v2_jobs(task):
    for seed in range(10, 15):
        text = Path(f"jobs/trysvgd_light2/trysvgdlight2_{task}_svgd_s{seed}.sh").read_text(encoding="utf-8")
        assert f'SB_POMDP_CONFIG="config/trysvgdlight2_{task}_tf-svgd.json"' in text
        assert f'SB_POMDP_CAMPAIGN_ID="trysvgdlight2-{task}-tf-svgd-v1"' in text
        assert f"SB_POMDP_SHARD_INDEX={seed - 9}" in text and f"SB_POMDP_SEGMENT_UPDATES={TASKS[task]}" in text
        assert text.rstrip().endswith("exec bash jobs/genkai_production.sh")
