from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _PROJECT_ROOT / "jobs" / "submit_one_shard.py"
_COMBINED_DEBUG_PATH = _PROJECT_ROOT / "jobs" / "genkai_debug_all.sh"
_SHORT_DEBUG_PATH = _PROJECT_ROOT / "jobs" / "genkai_debug_short.sh"
_HEAVY_DEBUG_PATH = _PROJECT_ROOT / "jobs" / "genkai_debug_heavy.sh"
_HEAVY_C_DEBUG_PATH = _PROJECT_ROOT / "jobs" / "genkai_debug_heavy_c.sh"
_SINGLE_DEBUG_PATH = _PROJECT_ROOT / "jobs" / "genkai_debug.sh"
_PRODUCTION_PATH = _PROJECT_ROOT / "jobs" / "genkai_production.sh"
_PRODUCTION_CONFIG_PATH = _PROJECT_ROOT / "config" / "production.json"
_PRODUCTION_CAMPAIGN_PATH = _PROJECT_ROOT / "jobs" / "submit_production_campaign.sh"
_PRODUCTION_CHAIN_PATH = _PROJECT_ROOT / "jobs" / "submit_production_chain.sh"
_SPEC = importlib.util.spec_from_file_location("genkai_submit_one_shard", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SUBMIT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SUBMIT
_SPEC.loader.exec_module(_SUBMIT)


def test_combined_debug_job_uses_one_full_gpu_for_matched_conditions() -> None:
    script = _COMBINED_DEBUG_PATH.read_text(encoding="utf-8")

    assert "#PJM -L rscgrp=b-batch" in script
    assert "#PJM -L gpu=1" in script
    assert "#PJM -L elapse=02:00:00" in script
    assert "SHARDS=(3 11 7 15)" in script
    assert 'bash jobs/genkai_debug.sh' in script
    assert 'SB_POMDP_SHARD_INDEX="$shard"' in script
    assert 'bash jobs/genkai_debug.sh >"$log_file" 2>&1' in script
    assert 'LOG_DIR="$PROJECT_ROOT/logs/genkai-debug/$CAMPAIGN_ID"' in script
    assert 'FAILED_SHARDS+=("$shard")' in script
    assert 'if [[ -n "${PJM_BULKNUM:-}" ]]' in script
    assert 'CAMPAIGN_ID="${SB_POMDP_CAMPAIGN_ID:-}"' in script


def test_short_debug_uses_four_full_gpus_for_all_conditions_under_30_minutes() -> None:
    script = _SHORT_DEBUG_PATH.read_text(encoding="utf-8")

    assert "#PJM -L rscgrp=b-batch" in script
    assert "#PJM -L node=1" in script
    assert "#PJM -L elapse=00:30:00" in script
    assert "#PJM -L gpu=" not in script
    assert "#PJM -L vnode-core=" not in script
    assert "WATCHDOG_SECONDS=1680" in script
    assert "GPUS=(\"${AVAILABLE_GPUS[@]:0:4}\")" in script
    assert "SHARDS=(3 11 7 15 4 2 1 12 10 9 8 6 5 16 14 13)" in script
    assert sorted({int(value) for value in script.split("SHARDS=(", 1)[1].split(")", 1)[0].split()}) == list(
        range(1, 17)
    )
    assert 'CUDA_VISIBLE_DEVICES="$gpu"' in script
    assert "SB_POMDP_SHORT_DEBUG=1" in script
    assert 'timeout --signal=TERM --kill-after=15s "${remaining}s"' in script
    assert 'run_worker "$worker_index" "${GPUS[$worker_index]}" &' in script
    assert 'if ! wait "$worker_pid"' in script
    assert "python" not in script.lower()


def test_heavy_debug_job_fixes_the_single_slowest_condition() -> None:
    script = _HEAVY_DEBUG_PATH.read_text(encoding="utf-8")

    assert "#PJM -L rscgrp=b-batch" in script
    assert "#PJM -L gpu=1" in script
    assert "#PJM -L elapse=02:00:00" in script
    assert "export SB_POMDP_SHARD_INDEX=3" in script
    assert 'exec bash jobs/genkai_debug.sh' in script
    assert 'CAMPAIGN_ID="${SB_POMDP_CAMPAIGN_ID:-}"' in script
    assert 'if [[ -n "${PJM_BULKNUM:-}" ]]' in script
    assert "python" not in script.lower()


def test_c_batch_heavy_debug_preserves_the_training_configuration() -> None:
    script = _HEAVY_C_DEBUG_PATH.read_text(encoding="utf-8")
    debug = _SINGLE_DEBUG_PATH.read_text(encoding="utf-8")

    assert "#PJM -L rscgrp=c-batch" in script
    assert "#PJM -L gpu=1" in script
    assert "#PJM -L elapse=02:00:00" in script
    assert "export SB_POMDP_SHARD_INDEX=3" in script
    assert "export SB_POMDP_OMP_NUM_THREADS=14" in script
    assert 'exec bash jobs/genkai_debug.sh' in script
    assert "--override model." not in script
    assert "--override ppo." not in script
    assert 'export OMP_NUM_THREADS="${SB_POMDP_OMP_NUM_THREADS:-16}"' in debug
    assert "python" not in script.lower()


def test_debug_matches_production_compute_environment() -> None:
    debug = _SINGLE_DEBUG_PATH.read_text(encoding="utf-8")
    production = _PRODUCTION_PATH.read_text(encoding="utf-8")
    production_config = json.loads(_PRODUCTION_CONFIG_PATH.read_text(encoding="utf-8"))

    shared_lines = (
        "#PJM -L rscgrp=b-batch",
        "#PJM -L gpu=1",
        "module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2",
        "--config config/production.json",
    )
    for line in shared_lines:
        assert line in debug
        assert line in production

    assert "--override model." not in debug
    assert debug.count("--override ppo.num_envs=4") == 1
    assert debug.count("--override ppo.epochs=1") == 1
    assert 'if [[ "$SHORT_DEBUG" == "1" ]]' in debug
    assert 'SHORT_DEBUG="${SB_POMDP_SHORT_DEBUG:-0}"' in debug
    assert 'export OMP_NUM_THREADS="${SB_POMDP_OMP_NUM_THREADS:-16}"' in debug
    assert "export OMP_NUM_THREADS=16" in production
    assert production_config["ppo"]["num_envs"] == 16
    assert production_config["ppo"]["epochs"] == 4
    assert production_config["ppo"]["rollout_steps"] == 128
    assert production_config["ppo"]["minibatch_size"] == 512
    assert production_config["ppo"]["sequence_microbatch_size"] == 128
    assert production_config["ppo"]["entropy_coef"] == 0.0


def test_genkai_debug_shell_scripts_have_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable on this platform")
    for path in (
        _SINGLE_DEBUG_PATH,
        _SHORT_DEBUG_PATH,
        _COMBINED_DEBUG_PATH,
        _HEAVY_DEBUG_PATH,
        _HEAVY_C_DEBUG_PATH,
    ):
        subprocess.run(
            [bash, "-n", str(path)],
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def test_production_job_segments_only_full_score_shards_in_one_fixed_root() -> None:
    script = _PRODUCTION_PATH.read_text(encoding="utf-8")

    assert 'SEGMENT_UPDATES="${SB_POMDP_SEGMENT_UPDATES:-}"' in script
    assert 'if (( 10#$SHARD_INDEX > 20 )); then' in script
    assert '--segment-updates "$SEGMENT_UPDATES"' in script
    assert '--run-dir "results/campaigns/$CAMPAIGN_ID/shard${SHARD_TAG}"' in script
    assert "--resume" in script
    assert 'flock -n 9' in script
    assert "--override model." not in script
    assert "--override ppo." not in script


def test_production_lock_stamps_its_owner_and_warns_about_a_foreign_stamp() -> None:
    """flock stays authoritative; the owner stamp is best-effort double protection."""

    script = _PRODUCTION_PATH.read_text(encoding="utf-8")

    assert 'LOCK_FILE="$LOCK_DIR/shard${SHARD_TAG}.lock"' in script
    assert 'read -r PREVIOUS_HOST PREVIOUS_PID PREVIOUS_EPOCH _ <"$LOCK_FILE"' in script
    assert '"$THIS_HOST" "$$" "$(date -u +%s)"' in script
    assert "WARNING: $LOCK_FILE still names host $PREVIOUS_HOST" in script
    assert "exit 75" in script
    # The stamp is read before the descriptor opens, and read-write opening keeps
    # a refused acquisition from truncating the current holder's stamp.
    assert 'exec 9<>"$LOCK_FILE"' in script
    assert 'exec 9>"$LOCK_FILE"' not in script
    assert script.index("read -r PREVIOUS_HOST") < script.index('exec 9<>"$LOCK_FILE"')
    # Only an acquired lock may be cleared on exit.
    assert script.index("flock -n 9") < script.index("""trap ': >"$LOCK_FILE"' EXIT""")


def test_production_job_script_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable on this platform")
    subprocess.run(
        [bash, "-n", str(_PRODUCTION_PATH)],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_shell_campaign_launcher_dry_run_submits_one_seed_in_parallel() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable on this platform")
    completed = subprocess.run(
        [
            bash,
            "jobs/submit_production_campaign.sh",
            "--dry-run",
            "--seed-offset",
            "0",
            "pytest-campaign",
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout

    pjsub_lines = [line for line in output.splitlines() if line.startswith("pjsub ")]
    assert len(pjsub_lines) == 4 * 25 + 12
    assert sum(" --step " in line for line in pjsub_lines) == 4 * 25
    assert sum(line.startswith("pjsub -z jid --step") for line in pjsub_lines) == 4
    assert output.count("jobs/genkai_production.sh") == 4 * 25 + 12
    step_lines = [line for line in output.splitlines() if line.startswith("pjsub --step")]
    assert len(step_lines) == 4 * 24
    assert all("jid=STEP_JOB_ID" in line for line in step_lines)
    assert "SB_POMDP_SEGMENT_UPDATES=10" in output
    assert "SB_POMDP_SHARD_INDEX=1" in output
    assert "SB_POMDP_SHARD_INDEX=76" in output
    assert "python" not in _PRODUCTION_CAMPAIGN_PATH.read_text(encoding="utf-8").lower()


def test_step_launcher_attaches_one_script_per_pjsub_call_by_job_id() -> None:
    campaign_script = _PRODUCTION_CAMPAIGN_PATH.read_text(encoding="utf-8")
    chain_script = _PRODUCTION_CHAIN_PATH.read_text(encoding="utf-8")

    assert 'bash jobs/submit_production_chain.sh "${chain_arguments[@]}"' in campaign_script
    assert 'pjsub -z jid --step' in chain_script
    assert '"$first_subjob_id" =~ ^([0-9]+)_0$' in chain_script
    assert 'step_job_id="${BASH_REMATCH[1]}"' in chain_script
    assert 'previous_step=$((step - 1))' in chain_script
    assert (
        '--sparam "jid=$step_job_id,sn=$step,sd=ec!=0:all:$previous_step"'
        in chain_script
    )
    assert '"$JOB_SCRIPT"' in chain_script
    assert "$JOB_SCRIPT,$JOB_SCRIPT" not in chain_script


def _prepare_project(tmp_path: Path, phase_name: str, campaign: bytes = b"campaign-1\n") -> None:
    phase = _SUBMIT.PHASES[phase_name]
    job_path = tmp_path / phase.job_script
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text("#!/bin/bash\n", encoding="ascii")
    campaign_path = tmp_path / phase.campaign_file
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_path.write_bytes(campaign)


@pytest.mark.parametrize(
    ("phase_name", "index"),
    [("debug", "16"), ("stability", "48"), ("production", "80")],
)
def test_submit_one_shard_uses_one_normal_job_and_two_explicit_variables(
    tmp_path: Path,
    phase_name: str,
    index: str,
) -> None:
    _prepare_project(tmp_path, phase_name)
    captured: list[tuple[list[str], Path, bool]] = []

    def fake_run(command: list[str], *, cwd: Path, check: bool):
        captured.append((command, cwd, check))
        return subprocess.CompletedProcess(command, 0)

    result = _SUBMIT.submit_one_shard(
        phase_name,
        index,
        project_root=tmp_path,
        runner=fake_run,
        which=lambda _: "/usr/local/bin/pjsub",
    )

    assert result == 0
    assert captured == [
        (
            [
                "/usr/local/bin/pjsub",
                "-x",
                f"SB_POMDP_SHARD_INDEX={index},SB_POMDP_CAMPAIGN_ID=campaign-1",
                _SUBMIT.PHASES[phase_name].job_script,
            ],
            tmp_path.resolve(),
            False,
        )
    ]


@pytest.mark.parametrize(
    "index",
    ["", "0", "01", "+1", "1-2", "1,2", "1.0", " 1", "9999"],
)
def test_submit_one_shard_rejects_noncanonical_or_out_of_range_index(
    tmp_path: Path,
    index: str,
) -> None:
    with pytest.raises(ValueError, match="index"):
        _SUBMIT.submit_one_shard("production", index, project_root=tmp_path, dry_run=True)


def test_submit_one_shard_rejects_unknown_phase(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="phase"):
        _SUBMIT.submit_one_shard("all", "1", project_root=tmp_path, dry_run=True)


@pytest.mark.parametrize(
    "campaign",
    [b"", b"one\ntwo\n", b"bad value\n", b"bad,comma\n", b"bad=value\n", b"\xff\n"],
)
def test_submit_one_shard_rejects_invalid_campaign_file(
    tmp_path: Path,
    campaign: bytes,
) -> None:
    _prepare_project(tmp_path, "production", campaign)
    with pytest.raises(ValueError, match="campaign"):
        _SUBMIT.submit_one_shard("production", "11", project_root=tmp_path, dry_run=True)


def test_submit_one_shard_propagates_pjsub_exit_status(tmp_path: Path) -> None:
    _prepare_project(tmp_path, "production")

    def fake_run(command: list[str], *, cwd: Path, check: bool):
        return subprocess.CompletedProcess(command, 17)

    assert (
        _SUBMIT.submit_one_shard(
            "production",
            "11",
            project_root=tmp_path,
            runner=fake_run,
            which=lambda _: "/usr/local/bin/pjsub",
        )
        == 17
    )


def test_submit_one_shard_dry_run_does_not_require_or_call_pjsub(tmp_path: Path) -> None:
    _prepare_project(tmp_path, "production")

    def fail_run(*args, **kwargs):
        raise AssertionError("runner must not be called")

    assert (
        _SUBMIT.submit_one_shard(
            "production",
            "11",
            project_root=tmp_path,
            dry_run=True,
            runner=fail_run,
            which=lambda _: None,
        )
        == 0
    )
