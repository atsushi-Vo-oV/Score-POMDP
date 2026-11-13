"""Shared-belief actor population experiments, isolated from compare/train."""
from .config import PopConfig, load_config


def run_population(*args, **kwargs):
    from .runner import run_population as run
    return run(*args, **kwargs)


__all__ = ["PopConfig", "load_config", "run_population"]
