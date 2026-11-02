import pytest
import torch

from sb_pomdp.baselines import GRUActorCritic
from sb_pomdp.train_baselines import (
    RecurrentReplayHistory,
    _recondition_recurrent_hidden,
    _replay_recurrent_prefix,
)


@pytest.mark.parametrize("cell", ["gru", "rnn"])
def test_recondition_matches_sequential(cell):
    torch.manual_seed(5)
    model = GRUActorCritic(
        3,
        action_kind="discrete",
        discrete_actions=2,
        hidden_dims=[7],
        recurrent_hidden_dim=5,
        recurrent_layers=2,
        recurrent_cell=cell,
    )
    histories = []
    for length in (3, 1, 0, 2):
        history = RecurrentReplayHistory()
        for step in range(length):
            history.inputs.append(torch.randn(3))
            history.episode_starts.append(torch.tensor(step == 0))
        histories.append(history)
    inputs = torch.zeros(4, 3)
    starts = torch.ones(4, dtype=torch.bool)
    rng = torch.get_rng_state().clone()
    expected = torch.cat(
        [
            _replay_recurrent_prefix(
                model,
                history.snapshot(input_template=inputs, start_template=starts),
                temporal_gradient_mode="tbptt_1",
            )
            for history in histories
        ],
        dim=1,
    )
    actual = _recondition_recurrent_hidden(
        model, histories, input_template=inputs, start_template=starts
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not actual.requires_grad
    histories[0].episode_starts[1] = torch.tensor(True)
    with pytest.raises(ValueError, match="exactly one episode start"):
        _recondition_recurrent_hidden(
            model, histories, input_template=inputs, start_template=starts
        )


def test_dropout_preserves_sequential_random_stream():
    torch.manual_seed(9)
    model = GRUActorCritic(
        3,
        action_kind="discrete",
        discrete_actions=2,
        hidden_dims=[7],
        recurrent_hidden_dim=5,
        recurrent_layers=2,
    )
    model.gru.dropout = 0.4
    histories = [
        RecurrentReplayHistory(
            [torch.randn(3) for _ in range(length)],
            [torch.tensor(step == 0) for step in range(length)],
        )
        for length in (3, 1)
    ]
    inputs, starts = torch.zeros(2, 3), torch.ones(2, dtype=torch.bool)
    rng = torch.get_rng_state().clone()
    expected = torch.cat(
        [
            _replay_recurrent_prefix(
                model,
                history.snapshot(input_template=inputs, start_template=starts),
                temporal_gradient_mode="tbptt_1",
            )
            for history in histories
        ],
        dim=1,
    )
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual = _recondition_recurrent_hidden(
        model, histories, input_template=inputs, start_template=starts
    )
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
