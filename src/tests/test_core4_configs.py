"""core4 experiment group: gru, rnn, DPFRL PF and the k16f score transformer at a matched budget."""

from __future__ import annotations

import pytest

from sb_pomdp.config import load_config


@pytest.mark.parametrize("task", ["mh", "ld2d", "pendbal"])
def test_core4_configs_hold_the_group_definition(task: str) -> None:
    resolved = load_config(f"config/core4_{task}.json").to_dict()
    assert resolved["comparison"]["methods"] == ["score_transformer", "gru", "rnn", "particle_filter"]
    assert resolved["comparison"]["pf_variant"] == "dpfrl"
    model = resolved["model"]
    # k16f: 16 particles, 12 Langevin steps, energy [8] (one SiLU hidden layer), d_model 32, ff 64 (1.11 MFLOP/step)
    assert (model["num_particles"], model["langevin_steps"], model["energy_hidden"]) == (16, 12, [8])
    assert (model["d_model"], model["transformer_ff_dim"], model["num_heads"]) == (32, 64, 4)
    # v2 belief design for the transformer
    assert model["langevin_warm_start"] is True and model["langevin_observation_anchor"] is True
    assert model["observation_prediction_coef"] > 0 and model["reward_prediction_coef"] > 0
    assert resolved["experiment"]["seeds"] == [10, 11, 12, 13, 14]
    if task == "pendbal":
        assert resolved["ppo"]["epochs"] == 10 and resolved["ppo"]["gamma"] == 0.9
        assert resolved["environment"]["tasks"]["masked_pendulum"]["initial_angle_range"] == 0.5
