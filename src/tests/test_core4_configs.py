"""core4 experiment group: gru, rnn, DPFRL PF and the k16f score transformer at a matched budget.

core4 (mh / ld2d / pendbal) and core4mm (the multimodal tasks mirror / mirror2 / tps).
"""

from __future__ import annotations

import pytest

from sb_pomdp.config import load_config


@pytest.mark.parametrize("task", ["mh", "ld2d", "pendbal", "mirror", "mirror2", "tps"])
def test_core4_configs_hold_the_group_definition(task: str) -> None:
    resolved = load_config(f"config/core4_{task}.json").to_dict()
    assert resolved["comparison"]["methods"] == ["score_transformer", "gru", "rnn", "particle_filter"]
    assert resolved["comparison"]["pf_variant"] == "dpfrl"
    assert resolved["comparison"]["pf_hidden"] == [108, 192]  # the parameter-matched DPFRL baseline
    model = resolved["model"]
    # k16f: 16 particles, 12 Langevin steps, energy [8] (one SiLU hidden layer), d_model 32, ff 64 (1.11 MFLOP/step)
    assert (model["num_particles"], model["langevin_steps"], model["energy_hidden"]) == (16, 12, [8])
    assert (model["d_model"], model["transformer_ff_dim"], model["num_heads"]) == (32, 64, 4)
    # v2 belief design for the transformer
    assert model["langevin_warm_start"] is True and model["langevin_observation_anchor"] is True
    assert model["observation_prediction_coef"] > 0 and model["reward_prediction_coef"] > 0
    assert resolved["experiment"]["seeds"] == [10, 11, 12, 13, 14]
    assert resolved["experiment"]["label"] == f"core4-{task}"
    if task == "pendbal":
        assert resolved["ppo"]["epochs"] == 10 and resolved["ppo"]["gamma"] == 0.9
        assert resolved["environment"]["tasks"]["masked_pendulum"]["initial_angle_range"] == 0.5


@pytest.mark.parametrize("task", ["mirror", "mirror2"])
def test_core4_mirror_tasks_keep_the_persistent_four_modal_belief(task: str) -> None:
    hike = load_config(f"config/core4_{task}.json").to_dict()["environment"]["tasks"]["mountain_hike"]
    assert hike["mirror_symmetry"] is True
    assert hike["start_distribution"] == "mirror_corners"
    assert hike["terrain_ridges"] == [0]  # ridge 1 only: knowing m is worth ~2.3 reward/step
    assert hike["horizon"] == 100 and hike["observation_noise_std"] == 3.0
    assert hike["signpost_noise_std"] == 0.2
    if task == "mirror":  # exactly the lf-grid task
        assert hike["signpost_width"] == 2.0
        assert hike.get("outside_box_cost", -1.5) == -1.5
    else:  # revised: stronger box penalty (-5 x hill 4 = -20/step) and a wider signpost
        assert hike["signpost_width"] == 4.0
        assert hike["outside_box_cost"] == -5.0


def test_core4_tps_task_is_the_teleport_symmetric_hike() -> None:
    hike = load_config("config/core4_tps.json").to_dict()["environment"]["tasks"]["mountain_hike"]
    assert hike["teleport_probability"] == 0.03
    assert hike["observation_symmetry"] == "abs"
    assert hike["horizon"] == 75 and hike["observation_noise_std"] == 3.0
    assert not hike.get("mirror_symmetry", False)
