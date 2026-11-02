"""Run analytic comparisons and record observed absolute errors by tolerance."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch.set_num_threads(1)
original = torch.testing.assert_close
maxima = {}


def record(actual, expected, **kwargs):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor) and actual.numel():
        key = f"{actual.dtype}:atol={kwargs.get('atol')}:rtol={kwargs.get('rtol')}"
        maxima[key] = max(maxima.get(key, 0.0), (actual - expected).detach().abs().max().item())
    original(actual, expected, **kwargs)


if __name__ == "__main__":
    torch.testing.assert_close = record
    try:
        status = pytest.main(["-q", str(ROOT / "src/tests/test_belief_analytic.py"), "-p", "no:cacheprovider"])
    finally:
        torch.testing.assert_close = original
    report = {"pytest_exit_code": int(status), "max_absolute_errors": maxima}
    (ROOT / "agent/option-c-exactness.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(status)
