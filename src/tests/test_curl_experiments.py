from pathlib import Path

import pytest

from sb_pomdp.config import load_config
from sb_pomdp.prediction_schedule import PredictionSchedule


@pytest.mark.parametrize("task,updates,ppo_updates", [("mirror2", 333, 200), ("tagr", 1293, 1000)])
@pytest.mark.parametrize("arm,coefficient,schedule", [
    ("mon", 0., "constant"), ("c001", .01, "constant"), ("c01", .1, "constant"),
    ("r01", .1, "ramp"), ("r1", 1., "ramp"),
])
def test_curl_configs_jobs_and_math_overrides(task, updates, ppo_updates, arm, coefficient, schedule):
    stem = f"trysvgdlight_{task}_tf-svgd-curl-{arm}"
    cfg = load_config(f"config/{stem}.json")
    original = load_config(f"config/trysvgdlight_{task}_tf-svgd.json").to_dict()
    actual = cfg.to_dict()
    assert actual["experiment"]["seeds"] == [10, 11, 12, 13, 14]
    assert actual["model"]["score_curl_penalty"] == coefficient
    assert actual["model"]["score_curl_monitor"] == (arm == "mon")
    assert actual["model"]["score_curl_schedule"] == schedule
    assert actual["model"]["score_curl_samples"] == 8
    assert actual["model"]["score_curl_estimator"] == "auto"
    assert actual["model"]["score_curl_probes"] == 4
    # Every other training/evaluation setting is inherited exactly from the control.
    actual["experiment"]["label"] = original["experiment"]["label"]
    for key in actual["model"]:
        if key.startswith("score_curl_"):
            actual["model"][key] = original["model"][key]
    assert actual == original
    accelerated = cfg.with_overrides({"model.score_implementation": "analytic",
        "model.belief_cuda_graphs": True, "ppo.sequence_microbatch_size": 512})
    plan = PredictionSchedule.from_config(accelerated["model"], rollout_batch_size=2048,
                                          ppo_updates=ppo_updates)
    assert plan.total_updates == updates
    assert len(list(Path("jobs/curl").glob(f"{stem}_s*.sh"))) == 5
    for seed in range(10, 15):
        path = Path(f"jobs/curl/{stem}_s{seed}.sh")
        script = path.read_text(encoding="utf-8")
        assert b"\r" not in path.read_bytes()
        for line in [f"export SB_POMDP_CONFIG=config/{stem}.json",
                     f"export SB_POMDP_CAMPAIGN_ID=trysvgdlight-{task}-tf-svgd-curl-{arm}-v1",
                     f"export SB_POMDP_SHARD_INDEX={seed-9}",
                     f"export SB_POMDP_BULK_INDEX={seed-9}",
                     f"export SB_POMDP_SEGMENT_UPDATES={updates}",
                     "exec bash jobs/genkai_production.sh"]:
            assert line in script
