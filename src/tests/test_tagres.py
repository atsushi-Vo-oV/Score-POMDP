"""tagres: the elliptic tag game with slower ghosts (0.5-0.7 of the player's top speed), experiment 25;
tagresn: tagres with the range noise doubled (std = 0.10 + 0.40 d), experiment 26."""
from __future__ import annotations

from pathlib import Path

import pytest

from sb_pomdp.config import load_config

V2 = {"langevin_proposal": "uniform", "langevin_initial_noise": "uniform", "langevin_observation_anchor": False,
      "direct_score_bound": None, "initial_score_prior": 0.0}
# new config: (source config, campaign, job stem, built on the v2 default)
ARMS = {"trysvgdlight2_tagres_tf-svgd": ("trysvgdlight_tagre_tf-svgd", "trysvgdlight2-tagres-tf-svgd-env64-k64-v1", "trysvgdlight2_tagres_svgd", True),
        "trysvgdlight_tagres_tf-svgd": ("trysvgdlight_tagre_tf-svgd", "trysvgdlight-tagres-tf-svgd-env64-k64-v1", "trysvgdlight_tagres_svgd", False),
        "trysvgd_tagres_gru": ("trysvgd_tagre_gru", "trysvgd-tagres-gru-env64-k64-v1", "trysvgd_tagres_gru", False),
        "trysvgd_tagres_pf_v3": ("trysvgd_tagre_pf_v3", "trysvgd-tagres-pf-v3-env64-k64-v1", "trysvgd_tagres_pf_v3", False)}


@pytest.mark.parametrize("task_name", ["tagres", "tagresn"])
@pytest.mark.parametrize("name", sorted(ARMS))
def test_slow_elliptic_tag_differs_from_tagre_only_in_the_ghost_speed(name, task_name):
    source, campaign, job, v2 = ARMS[name]
    name, campaign, job = (text.replace("tagres", task_name) for text in (name, campaign, job))
    slow, base = load_config(f"config/{name}.json").to_dict(), load_config(f"config/{source}.json").to_dict()
    task = slow["environment"]["tasks"]["disk_tag"]
    assert task["observation_mode"] == "elliptic_ranges"
    assert (task["ghost_speed_min"], task["ghost_speed_max"]) == (0.5, 0.7)
    assert (base["environment"]["tasks"]["disk_tag"]["ghost_speed_min"], base["environment"]["tasks"]["disk_tag"]["ghost_speed_max"]) == (0.8, 1.0)
    task.update(ghost_speed_min=0.8, ghost_speed_max=1.0)
    reference = base["environment"]["tasks"]["disk_tag"]
    assert (reference["range_noise_base"], reference["range_noise_scale"]) == (0.05, 0.2)
    if task_name == "tagresn":  # the noisy version doubles both terms of the range noise and changes nothing else
        assert (task["range_noise_base"], task["range_noise_scale"]) == (0.10, 0.40)
        task.update(range_noise_base=0.05, range_noise_scale=0.2)
    slow["experiment"]["label"] = base["experiment"]["label"]
    if v2:  # the v2 default changes exactly these five keys of the light model
        assert {key: slow["model"][key] for key in V2} == V2
        for key in V2:
            slow["model"][key] = base["model"][key]
    assert slow == base
    for seed in range(10, 15):
        text = Path(f"jobs/{task_name}/{job}_s{seed}.sh").read_text(encoding="utf-8")
        assert f'SB_POMDP_CONFIG="config/{name}.json"' in text and f'SB_POMDP_CAMPAIGN_ID="{campaign}"' in text
        assert f"SB_POMDP_SHARD_INDEX={seed - 9}" in text and "SB_POMDP_SEGMENT_UPDATES=2420" in text
        assert text.rstrip().endswith("exec bash jobs/genkai_production.sh")
