#!/usr/bin/env python3.11
"""Submit exactly one Genkai comparison shard as a normal PJM job."""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

_CAMPAIGN_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?")
_INDEX_PATTERN = re.compile(r"[1-9][0-9]{0,2}")


@dataclass(frozen=True)
class Phase:
    job_script: str
    campaign_file: str
    maximum_index: int
    requested_point_ceiling: str


PHASES = {
    "debug": Phase(
        job_script="jobs/genkai_debug.sh",
        campaign_file="config/campaign-id-debug.txt",
        maximum_index=16,
        requested_point_ceiling="60 pt (2 h, one full GPU)",
    ),
    "stability": Phase(
        job_script="jobs/genkai_stability.sh",
        campaign_file="config/campaign-id-stability.txt",
        maximum_index=48,
        requested_point_ceiling="60 pt (2 h, one full GPU)",
    ),
    "production": Phase(
        job_script="jobs/genkai_production.sh",
        campaign_file="config/campaign-id.txt",
        maximum_index=80,
        requested_point_ceiling="5,040 pt (168 h, one full GPU)",
    ),
}


Runner = Callable[..., subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]]


def _parse_index(value: str, maximum: int) -> int:
    if not _INDEX_PATTERN.fullmatch(value):
        raise ValueError("index must be one canonical positive integer")
    index = int(value)
    if index > maximum:
        raise ValueError(f"index must be in 1..{maximum}")
    return index


def _read_campaign_id(path: Path) -> str:
    try:
        text = path.read_bytes().decode("ascii")
    except FileNotFoundError as exc:
        raise ValueError(f"campaign file does not exist: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("campaign ID must be ASCII") from exc
    lines = text.splitlines()
    if len(lines) != 1:
        raise ValueError("campaign file must contain exactly one line")
    campaign_id = lines[0]
    if len(campaign_id) > 128 or not _CAMPAIGN_PATTERN.fullmatch(campaign_id):
        raise ValueError("campaign ID contains unsupported characters or is too long")
    return campaign_id


def submit_one_shard(
    phase_name: str,
    index_text: str,
    *,
    project_root: Path,
    dry_run: bool = False,
    runner: Runner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> int:
    try:
        phase = PHASES[phase_name]
    except KeyError as exc:
        choices = ", ".join(PHASES)
        raise ValueError(f"phase must be one of: {choices}") from exc

    root = project_root.resolve()
    index = _parse_index(index_text, phase.maximum_index)
    job_script = root / phase.job_script
    if not job_script.is_file():
        raise ValueError(f"job script does not exist: {job_script}")
    campaign_id = _read_campaign_id(root / phase.campaign_file)

    exported = f"SB_POMDP_SHARD_INDEX={index},SB_POMDP_CAMPAIGN_ID={campaign_id}"
    pjsub = which("pjsub")
    if pjsub is None and not dry_run:
        raise ValueError("pjsub was not found; run this command on Genkai")
    command = [pjsub or "pjsub", "-x", exported, phase.job_script]

    print(f"phase: {phase_name}")
    print(f"shard index: {index}/{phase.maximum_index}")
    print(f"campaign ID: {campaign_id}")
    print(f"requested point ceiling: {phase.requested_point_ceiling}")
    print(f"command: {shlex.join(command)}")
    if dry_run:
        print("dry-run: no job submitted")
        return 0

    completed = runner(command, cwd=root, check=False)
    return completed.returncode


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=tuple(PHASES))
    parser.add_argument("index", help="one shard index, not a range or list")
    parser.add_argument("--dry-run", action="store_true", help="print without calling pjsub")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        return submit_one_shard(
            args.phase,
            args.index,
            project_root=project_root,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as exc:
        print(f"Submission failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
