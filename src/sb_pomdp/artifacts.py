"""Portable and overwrite-conscious experiment artifact management.

This module intentionally does not import PyTorch.  It only prepares checkpoint
paths; the training code remains responsible for calling ``torch.save``.
"""

from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


class ArtifactError(ValueError):
    """Raised when an artifact path or payload is unsafe or inconsistent."""


_SAFE_COMPONENT = re.compile(r"^[\w.-]+$", flags=re.UNICODE)
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _strict_component(value: str, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{description} must be a non-empty string")
    normalized = unicodedata.normalize("NFC", value)
    if (
        normalized in {".", ".."}
        or normalized.endswith((" ", "."))
        or not _SAFE_COMPONENT.fullmatch(normalized)
    ):
        raise ArtifactError(
            f"unsafe {description} {value!r}; use only letters, numbers, '.', '_', and '-'"
        )
    if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ArtifactError(f"{description} is a reserved Windows filename: {value!r}")
    return normalized


def _label_component(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError("label must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", value.strip())
    slug = re.sub(r"[^\w.-]+", "-", normalized, flags=re.UNICODE).strip(".-_")
    if not slug:
        raise ArtifactError(f"label has no usable filename characters: {value!r}")
    return _strict_component(slug, "label")


def _artifact_filename(name: str, suffix: str) -> str:
    base = name.removesuffix(suffix)
    return f"{_strict_component(base, 'artifact name')}{suffix}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object-dtype arrays cannot be written to JSON safely")
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"._{uuid4().hex[:12]}.json.tmp")
    serializable = dict(payload) if isinstance(payload, Mapping) else payload
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                serializable,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True, slots=True)
class CsvAppender:
    """Append mappings to a CSV file while enforcing one immutable header."""

    path: Path
    fieldnames: tuple[str, ...]

    def __init__(self, path: str | Path, fieldnames: Sequence[str]) -> None:
        fields = tuple(fieldnames)
        if not fields or any(not isinstance(field, str) or not field for field in fields):
            raise ArtifactError("CSV fieldnames must be non-empty strings")
        if len(fields) != len(set(fields)):
            raise ArtifactError("CSV fieldnames must not contain duplicates")
        object.__setattr__(self, "path", Path(path))
        object.__setattr__(self, "fieldnames", fields)

    def _read_header(self) -> tuple[str, ...] | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), None)
        return tuple(header) if header is not None else None

    def append(self, row: Mapping[str, Any]) -> Path:
        """Append exactly one row, rejecting missing/extra columns."""

        return self.append_many((row,))

    def append_many(self, rows: Sequence[Mapping[str, Any]]) -> Path:
        """Append rows with one open/flush/fsync after validating the full batch."""

        materialized = list(rows)
        if not materialized:
            return self.path

        expected_fields = set(self.fieldnames)
        for index, row in enumerate(materialized):
            if not isinstance(row, Mapping):
                raise ArtifactError(f"CSV row {index} must be a mapping")
            actual_fields = set(row)
            if actual_fields != expected_fields:
                missing = sorted(expected_fields - actual_fields)
                extra = sorted(actual_fields - expected_fields)
                raise ArtifactError(
                    f"CSV row {index} does not match header; missing={missing}, extra={extra}"
                )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing_header = self._read_header()
        if existing_header is not None and existing_header != self.fieldnames:
            raise ArtifactError(
                f"CSV header mismatch for {self.path}: "
                f"existing={existing_header}, requested={self.fieldnames}"
            )

        mode = "a" if existing_header is not None else "w"
        with self.path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="raise")
            if existing_header is None:
                writer.writeheader()
            writer.writerows(dict(row) for row in materialized)
            handle.flush()
            os.fsync(handle.fileno())
        return self.path


def append_csv(
    path: str | Path,
    row: Mapping[str, Any],
    fieldnames: Sequence[str],
) -> Path:
    """Convenience wrapper around :class:`CsvAppender`."""

    return CsvAppender(path, fieldnames).append(row)


def truncate_csv_rows(
    path: str | Path,
    keep: Callable[[Mapping[str, str]], bool],
) -> int:
    """Atomically drop the already-written rows that ``keep`` rejects.

    The header and every retained row keep their exact on-disk text, and the
    rewrite goes through a temporary file that is fsynced before ``os.replace``,
    so an interrupted truncation leaves either the original or the truncated
    file.  Nothing is written when every row is kept, and a missing, empty, or
    header-only file is left untouched.  A final row with fewer values than the
    header is an append that never completed and is removed as well; any other
    row width is a structural error.  Returns the number of removed rows.
    """

    csv_path = Path(path)
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = list(reader)
    if header is None:
        return 0
    fields = tuple(header)
    if not fields or len(fields) != len(set(fields)):
        raise ArtifactError(f"CSV header of {csv_path} is empty or contains duplicates")

    retained: list[list[str]] = []
    for index, row in enumerate(rows):
        if len(row) != len(fields):
            if len(row) < len(fields) and index == len(rows) - 1:
                continue  # interrupted append; the row was never committed
            raise ArtifactError(
                f"CSV row {index} of {csv_path} has {len(row)} values; expected {len(fields)}"
            )
        if keep(dict(zip(fields, row))):
            retained.append(row)
    removed = len(rows) - len(retained)
    if removed == 0:
        return 0

    temporary = csv_path.with_name(f"._{uuid4().hex[:12]}.csv.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(retained)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    return removed


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> Path:
    """Atomically replace one CSV with a fully validated set of rows."""

    fields = tuple(fieldnames)
    if not fields or len(fields) != len(set(fields)):
        raise ArtifactError("CSV fieldnames must be non-empty and unique")
    expected = set(fields)
    materialized = [dict(row) for row in rows]
    for index, row in enumerate(materialized):
        actual = set(row)
        if actual != expected:
            raise ArtifactError(
                f"CSV row {index} does not match header; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"._{uuid4().hex[:12]}.csv.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _save_npz(path: Path, arrays: Mapping[str, Any], *, compressed: bool = True) -> Path:
    if not arrays:
        raise ArtifactError("at least one array is required")
    converted: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        _strict_component(key, "array name")
        if key == "file":
            raise ArtifactError("array name 'file' is reserved by numpy.savez")
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"could not convert array {key!r}: {exc}") from exc
        if array.dtype.hasobject:
            raise ArtifactError(
                f"array {key!r} has object dtype; it would require allow_pickle=True to load"
            )
        converted[key] = array

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"._{uuid4().hex[:12]}.tmp.npz")
    try:
        saver = np.savez_compressed if compressed else np.savez
        saver(temporary, **converted)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True, slots=True)
class SeedArtifacts:
    """Artifact paths for one ``task``/``seed`` experiment."""

    path: Path
    task: str
    seed: int
    method: str | None = None
    gradient_mode: str | None = None

    @property
    def resolved_config_path(self) -> Path:
        return self.path / "resolved_config.json"

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.csv"

    @property
    def arrays_path(self) -> Path:
        return self.path / "arrays.npz"

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    def write_resolved_config(self, config: Mapping[str, Any]) -> Path:
        return _atomic_write_json(self.resolved_config_path, config)

    def write_metadata(self, metadata: Mapping[str, Any]) -> Path:
        return _atomic_write_json(self.metadata_path, metadata)

    def csv_appender(
        self,
        fieldnames: Sequence[str],
        *,
        name: str = "metrics",
    ) -> CsvAppender:
        filename = _artifact_filename(name, ".csv")
        return CsvAppender(self.path / filename, fieldnames)

    def append_metrics(
        self,
        row: Mapping[str, Any],
        fieldnames: Sequence[str] | None = None,
    ) -> Path:
        """Append a metric row, fixing the header on the first append.

        Supplying ``fieldnames`` on the first call is recommended.  If omitted,
        insertion order of the first row defines the fixed on-disk header; later
        calls read and enforce that header.
        """

        fields: Sequence[str]
        if fieldnames is not None:
            fields = fieldnames
        elif self.metrics_path.exists() and self.metrics_path.stat().st_size:
            with self.metrics_path.open("r", encoding="utf-8", newline="") as handle:
                fields = tuple(next(csv.reader(handle)))
        else:
            fields = tuple(row)
        return CsvAppender(self.metrics_path, fields).append(row)

    def truncate_csv(
        self,
        keep: Callable[[Mapping[str, str]], bool],
        *,
        name: str = "metrics",
    ) -> int:
        """Atomically drop rows of ``<name>.csv`` that ``keep`` rejects."""

        filename = _artifact_filename(name, ".csv")
        return truncate_csv_rows(self.path / filename, keep)

    def save_npz(
        self,
        arrays: Mapping[str, Any] | None = None,
        *,
        name: str = "arrays",
        compressed: bool = True,
        **named_arrays: Any,
    ) -> Path:
        """Atomically save non-object arrays readable with ``allow_pickle=False``."""

        payload = dict(arrays or {})
        overlap = set(payload) & set(named_arrays)
        if overlap:
            raise ArtifactError(f"duplicate array names: {sorted(overlap)}")
        payload.update(named_arrays)
        filename = _artifact_filename(name, ".npz")
        return _save_npz(self.path / filename, payload, compressed=compressed)

    def checkpoint_path(
        self,
        step: int | None = None,
        *,
        tag: str | None = None,
        create_parent: bool = True,
    ) -> Path:
        """Return a ``.pt`` path for ``torch.save`` without importing torch."""

        if step is not None and tag is not None:
            raise ArtifactError("provide either checkpoint step or tag, not both")
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ArtifactError("checkpoint step must be a non-negative integer")
            stem = f"step_{step:09d}"
        elif tag is not None:
            stem = _strict_component(tag, "checkpoint tag")
        else:
            stem = "latest"
        if create_parent:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoints_dir / f"{stem}.pt"

    def torch_checkpoint_path(
        self,
        step: int | None = None,
        *,
        tag: str | None = None,
        create_parent: bool = True,
    ) -> Path:
        """Explicit alias for :meth:`checkpoint_path`."""

        return self.checkpoint_path(step, tag=tag, create_parent=create_parent)


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """A complete timestamped result tree shared by tasks and seeds."""

    root: Path
    timestamp: str
    label: str

    @classmethod
    def create(
        cls,
        results_dir: str | Path,
        label: str,
        *,
        timestamp: str | datetime | None = None,
        exist_ok: bool = False,
    ) -> RunArtifacts:
        base = Path(results_dir).expanduser()
        safe_label = _label_component(label)
        if timestamp is None:
            timestamp_text = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        elif isinstance(timestamp, datetime):
            timestamp_text = timestamp.astimezone().strftime("%Y%m%dT%H%M%S")
        else:
            timestamp_text = timestamp
        safe_timestamp = _strict_component(timestamp_text, "timestamp")

        base.mkdir(parents=True, exist_ok=True)
        root = base / f"{safe_timestamp}_{safe_label}"
        root.mkdir(parents=False, exist_ok=exist_ok)
        return cls(root=root.resolve(), timestamp=safe_timestamp, label=safe_label)

    @classmethod
    def create_at(cls, root: str | Path, label: str) -> RunArtifacts:
        """Create an explicitly named run root for a resumable execution."""

        path = Path(root).expanduser()
        safe_label = _label_component(label)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(parents=False, exist_ok=False)
        return cls(root=path.resolve(), timestamp="fixed", label=safe_label)

    @classmethod
    def open(cls, root: str | Path, label: str) -> RunArtifacts:
        """Open an existing explicitly named run root without creating files."""

        path = Path(root).expanduser()
        if not path.is_dir():
            raise ArtifactError(f"run root does not exist or is not a directory: {path}")
        return cls(root=path.resolve(), timestamp="fixed", label=_label_component(label))

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.csv"

    @property
    def summary_json_path(self) -> Path:
        return self.root / "summary.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.root / "resolved_config.json"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    @property
    def comparison_path(self) -> Path:
        return self.root / "comparison.csv"

    @property
    def comparison_json_path(self) -> Path:
        return self.root / "comparison.json"

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        return _atomic_write_json(self.manifest_path, manifest)

    def write_summary(self, summary: Mapping[str, Any] | Sequence[Any]) -> Path:
        return _atomic_write_json(self.summary_json_path, summary)

    def write_resolved_config(self, config: Mapping[str, Any]) -> Path:
        return _atomic_write_json(self.resolved_config_path, config)

    def write_metadata(self, metadata: Mapping[str, Any]) -> Path:
        return _atomic_write_json(self.metadata_path, metadata)

    def write_comparison(self, comparison: Mapping[str, Any] | Sequence[Any]) -> Path:
        return _atomic_write_json(self.comparison_json_path, comparison)

    def write_summary_csv(
        self,
        rows: Sequence[Mapping[str, Any]],
        fieldnames: Sequence[str],
    ) -> Path:
        return _atomic_write_csv(self.summary_path, rows, fieldnames)

    def write_comparison_csv(
        self,
        rows: Sequence[Mapping[str, Any]],
        fieldnames: Sequence[str],
    ) -> Path:
        return _atomic_write_csv(self.comparison_path, rows, fieldnames)

    def comparison_appender(self, fieldnames: Sequence[str]) -> CsvAppender:
        return CsvAppender(self.comparison_path, fieldnames)

    def summary_appender(self, fieldnames: Sequence[str]) -> CsvAppender:
        return CsvAppender(self.summary_path, fieldnames)

    def append_summary(self, row: Mapping[str, Any], fieldnames: Sequence[str]) -> Path:
        return self.summary_appender(fieldnames).append(row)

    def task_dir(self, task: str) -> Path:
        safe_task = _strict_component(task, "task name")
        path = self.root / safe_task
        path.mkdir(parents=False, exist_ok=True)
        return path

    def for_seed(self, task: str, seed: int, *, exist_ok: bool = False) -> SeedArtifacts:
        """Create and return ``<task>/seed_NNN`` beneath this run."""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ArtifactError("seed must be a non-negative integer")
        safe_task = _strict_component(task, "task name")
        task_path = self.task_dir(safe_task)
        seed_path = task_path / f"seed_{seed:03d}"
        seed_path.mkdir(parents=False, exist_ok=exist_ok)
        return SeedArtifacts(path=seed_path, task=safe_task, seed=seed)

    def for_method_seed(
        self,
        method: str,
        task: str,
        seed: int,
        *,
        exist_ok: bool = False,
    ) -> SeedArtifacts:
        """Create ``<method>/<task>/seed_NNN`` for a comparison run."""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ArtifactError("seed must be a non-negative integer")
        safe_method = _strict_component(method, "method name")
        safe_task = _strict_component(task, "task name")
        method_path = self.root / safe_method
        method_path.mkdir(parents=False, exist_ok=True)
        task_path = method_path / safe_task
        task_path.mkdir(parents=False, exist_ok=True)
        seed_path = task_path / f"seed_{seed:03d}"
        seed_path.mkdir(parents=False, exist_ok=exist_ok)
        return SeedArtifacts(
            path=seed_path,
            task=safe_task,
            seed=seed,
            method=safe_method,
        )

    def for_gradient_mode_method_seed(
        self,
        mode: str,
        method: str,
        task: str,
        seed: int,
        *,
        exist_ok: bool = False,
    ) -> SeedArtifacts:
        """Create ``<mode>/<method>/<task>/seed_NNN`` for a gradient comparison."""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ArtifactError("seed must be a non-negative integer")
        safe_mode = _strict_component(mode, "gradient mode")
        safe_method = _strict_component(method, "method name")
        safe_task = _strict_component(task, "task name")
        mode_path = self.root / safe_mode
        mode_path.mkdir(parents=False, exist_ok=True)
        method_path = mode_path / safe_method
        method_path.mkdir(parents=False, exist_ok=True)
        task_path = method_path / safe_task
        task_path.mkdir(parents=False, exist_ok=True)
        seed_path = task_path / f"seed_{seed:03d}"
        seed_path.mkdir(parents=False, exist_ok=exist_ok)
        return SeedArtifacts(
            path=seed_path,
            task=safe_task,
            seed=seed,
            method=safe_method,
            gradient_mode=safe_mode,
        )

    create_seed = for_seed


def create_run_artifacts(
    results_dir: str | Path,
    label: str,
    *,
    timestamp: str | datetime | None = None,
    exist_ok: bool = False,
) -> RunArtifacts:
    """Functional wrapper for :meth:`RunArtifacts.create`."""

    return RunArtifacts.create(
        results_dir,
        label,
        timestamp=timestamp,
        exist_ok=exist_ok,
    )


__all__ = [
    "ArtifactError",
    "CsvAppender",
    "RunArtifacts",
    "SeedArtifacts",
    "append_csv",
    "create_run_artifacts",
    "truncate_csv_rows",
]
