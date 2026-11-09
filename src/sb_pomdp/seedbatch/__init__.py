"""Independent-seed vectorized training, isolated from the original trainers."""

from .runner import run_seedbatch

__all__ = ["run_seedbatch"]
