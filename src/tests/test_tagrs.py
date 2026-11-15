"""tagrs: the tag game with slower ghosts (0.5-0.7 of the player's top speed), experiment 24."""
from __future__ import annotations

from pathlib import Path

import pytest

from sb_pomdp.config import load_config

ARMS = {"trysvgdlight2_tagrs_tf-svgd": ("trysvgdlight2_tagr_tf-svgd", "trysvgdlight2-tagrs-tf-svgd-v1", "trysvgdlight2_tagrs_svgd"),
        "trysvgdlight_tagrs_tf-svgd": ("trysvgdlight_tagr_tf-svgd", "trysvgdlight-tagrs-tf-svgd-v1", "trysvgdlight_tagrs_svgd"),
        "trysvgd_tagrs_gru": ("trysvgd_tagr_gru", "trysvgd-tagrs-gru-v1", "trysvgd_tagrs_gru"),
        "trysvgd_tagrs_pf_v3": ("trysvgd_tagr_pf_v3", "trysvgd-tagrs-pf-v3-v1", "trysvgd_tagrs_pf_v3")}


@pytest.mark.parametrize("name", sorted(ARMS))
def test_slow_tag_differs_from_tagr_only_in_the_ghost_speed(name):
    source, campaign, job = ARMS[name]
    slow, base = load_config(f"config/{name}.json").to_dict(), load_config(f"config/{source}.json").to_dict()
    task = slow["environment"]["tasks"]["disk_tag"]
    assert (task["ghost_speed_min"], task["ghost_speed_max"]) == (0.5, 0.7)
    assert (base["environment"]["tasks"]["disk_tag"]["ghost_speed_min"], base["environment"]["tasks"]["disk_tag"]["ghost_speed_max"]) == (0.8, 1.0)
    task.update(ghost_speed_min=0.8, ghost_speed_max=1.0)
    slow["experiment"]["label"] = base["experiment"]["label"]
    assert slow == base
    for seed in range(10, 15):
        text = Path(f"jobs/tagrs/{job}_s{seed}.sh").read_text(encoding="utf-8")
        assert f'SB_POMDP_CONFIG="config/{name}.json"' in text and f'SB_POMDP_CAMPAIGN_ID="{campaign}"' in text
        assert f"SB_POMDP_SHARD_INDEX={seed - 9}" in text and "SB_POMDP_SEGMENT_UPDATES=1293" in text
