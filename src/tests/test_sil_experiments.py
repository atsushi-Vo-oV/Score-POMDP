from pathlib import Path

import pytest

from sb_pomdp.compare import _configured_updates
from sb_pomdp.config import load_config
from sb_pomdp.prediction_schedule import PredictionSchedule


@pytest.mark.parametrize("seed", [10, 11])
@pytest.mark.parametrize("variant", ["orig", "bc", "bcreset"])
def test_bcresume_experiment_mapping(seed, variant):
    path = Path(f"config/bcresume_tagr_s{seed}_{variant}.json")
    cfg = load_config(path)
    expected_updates = 100 if variant == "orig" else 110
    assert cfg["experiment"]["seeds"] == [seed]
    assert cfg["experiment"]["eval_interval_updates"] == cfg["experiment"]["eval_episodes"] == 10
    assert cfg["model"]["freeze_modules"] == ["belief", "encoder"]
    assert cfg["model"]["reset_value_head"] == (variant == "bcreset")
    assert f"shard{seed-9:03d}/full/score_transformer/disk_tag/seed_{seed:03d}" in cfg["model"]["init_from_checkpoint"]
    if variant != "orig":
        assert Path(cfg["model"]["actor_init_path"]).is_file()
    schedule = PredictionSchedule.from_config(cfg["model"], rollout_batch_size=2048, ppo_updates=100, critic_warmup_updates=cfg["ppo"]["critic_warmup_updates"])
    assert schedule.total_updates == expected_updates
    assert _configured_updates(cfg) == expected_updates
    assert schedule.ppo_updates_done(expected_updates) == 100
    assert all(schedule.phase(u) in {"ppo", "critic"} for u in range(1, expected_updates+1))
    script = Path(f"jobs/bcresume/s{seed}_{variant}.sh").read_text()
    for line in (
        f"export SB_POMDP_CONFIG={path.as_posix()}",
        f"export SB_POMDP_CAMPAIGN_ID=bcresume-tagr-{variant}-v1",
        f"export SB_POMDP_SHARD_INDEX={seed-9}",
        f"export SB_POMDP_SEGMENT_UPDATES={expected_updates}",
        "export SB_POMDP_BULK_INDEX=1", "exec bash jobs/genkai_production.sh",
    ):
        assert line in script


@pytest.mark.parametrize("stem,updates", [
    ("trysvgdlight_tagr_tf-svgd-sil", 1293),
    ("trysvgdlight_tagr_tf-svgd-silexp", 1293),
    ("trysvgd_tagr_gru-sil", 1293),
    ("trysvgdlight_carflag_tf-svgd-long-sil", 2493),
    ("trysvgdlight_tagr_tf-svgd-lam99", 1293),
])
def test_sil_experiment_and_job_counts(stem, updates):
    cfg = load_config(f"config/{stem}.json")
    assert cfg["experiment"]["seeds"] == [10, 11, 12, 13, 14]
    ppo = cfg["ppo"]
    schedule = PredictionSchedule.from_config(cfg["model"], rollout_batch_size=ppo["num_envs"]*ppo["rollout_steps"], ppo_updates=ppo["total_steps"]//(ppo["num_envs"]*ppo["rollout_steps"]))
    assert schedule.total_updates == updates
    assert _configured_updates(cfg) == updates
    assert ppo["self_imitation"]["enabled"] == ("lam99" not in stem)
    if "lam99" in stem:
        assert ppo["gae_lambda"] == 0.99
    if "silexp" in stem:
        assert ppo["self_imitation"]["weighting"] == "exp"
    for shard, seed in enumerate(cfg["experiment"]["seeds"], 1):
        script = Path(f"jobs/sil/{stem}_s{seed}.sh").read_text()
        assert f"export SB_POMDP_CONFIG=config/{stem}.json" in script
        assert f"export SB_POMDP_SHARD_INDEX={shard}" in script
        assert f"export SB_POMDP_SEGMENT_UPDATES={updates}" in script
