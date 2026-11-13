"""Access the native heads without changing their checkpoint parameter names."""
from ..baselines import GRUActorCritic


def is_gru(model):
    return isinstance(model, GRUActorCritic) and model.recurrent_cell == "gru"


def heads(model):
    return model.heads if is_gru(model) else model


def actor(model):
    return heads(model).diffusion_policy


def set_actor(model, policy):
    heads(model).diffusion_policy = policy


def make_model(environment, config, device):
    if config["comparison"]["methods"] == ["gru"]:
        from ..train_baselines import _make_model
        return _make_model(environment, "gru", config["model"], config["comparison"], device)
    from ..train import model_for_environment
    return model_for_environment(environment, config["model"], device)
