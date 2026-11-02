"""A failed or partial campaign must never be reported as a warm timing."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[2] / "agent" / "benchmark_option_c.py"
    spec = importlib.util.spec_from_file_location("benchmark_option_c", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_separates_prediction_and_ppo_warmup(benchmark):
    log = "prediction phase=pretrain update=1 next_observation_nll=0.1\n"
    log += "\n".join(
        f"phase=ppo_update status=end update={update}/5 seconds={seconds}"
        for update, seconds in enumerate([82.0, 40.0, 3.0, 4.0, 5.0], start=1)
    )
    _, ppo = benchmark.summarize_phases(log, 1, 3)
    assert ppo["pretrain"] == [{"update": 1, "seconds": 82.0}]
    assert ppo["warmup"] == [{"update": 2, "seconds": 40.0}]
    assert [item["update"] for item in ppo["measured"]] == [3, 4, 5]
    assert ppo["measured_mean_seconds"] == 4.0


@pytest.mark.parametrize("returncode", [0, 1, -9])
def test_benchmark_reports_failed_or_incomplete_child(benchmark, monkeypatch, tmp_path, capsys,
                                                     returncode):
    class Child:
        stdout = ("phase=ppo_update status=end update=1/5 seconds=82.0\n",)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return returncode

    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(sys, "argv", ["benchmark_option_c.py", "--output", str(tmp_path)])
    assert benchmark.main() == 1
    summary = json.loads((tmp_path / "timings.json").read_text(encoding="utf-8"))
    for setting in ("autograd", "analytic", "inductor"):
        result = summary["settings"][setting]
        assert "phases" not in result
        expected_error = "incomplete benchmark" if returncode == 0 else f"code {returncode}"
        assert expected_error in result["error"]
        assert (tmp_path / f"{setting}.log").read_text(encoding="utf-8") == Child.stdout[0]
    output = capsys.readouterr()
    assert "[inductor] phase=ppo_update" in output.out
    assert "ERROR: setting=inductor:" in output.err
