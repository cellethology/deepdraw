"""Unit tests for report_failed_jobs helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from job_sub import report_failed_jobs as rfj


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_is_time_dir() -> None:
    assert rfj._is_time_dir("00-00-00")
    assert rfj._is_time_dir("23-59-59")
    assert not rfj._is_time_dir("0-00-00")
    assert not rfj._is_time_dir("00-00")
    assert not rfj._is_time_dir("abc-def-gh")


def test_resolve_sweep_dirs_prefers_timestamp_children(tmp_path: Path) -> None:
    (tmp_path / "20-49-41").mkdir()
    (tmp_path / "07-00-02").mkdir()
    (tmp_path / "misc").mkdir()

    sweeps = rfj._resolve_sweep_dirs(tmp_path)
    assert [path.name for path in sweeps] == ["07-00-02", "20-49-41"]


def test_iter_run_dirs_prefers_seed_dirs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset_a"
    (dataset_dir / "job_1" / "seed_1").mkdir(parents=True)
    (dataset_dir / "job_1" / "seed_0").mkdir(parents=True)
    _touch(dataset_dir / "job_2" / "run_config.log")

    run_dirs = rfj._iter_run_dirs(dataset_dir)
    assert [path.name for path in run_dirs] == ["seed_0", "seed_1"]


def test_find_failed_runs_filters_dataset_and_detects_reasons(tmp_path: Path) -> None:
    sweep = tmp_path / "2026-02-06" / "20-49-41"

    # Dataset A: one success, one error, one missing summary.
    _touch(sweep / "dataset_a" / "job_0" / "seed_0" / "summary.json")
    _touch(sweep / "dataset_a" / "job_1" / "seed_0" / "error.txt")
    (sweep / "dataset_a" / "job_2" / "seed_0").mkdir(parents=True)

    # Dataset B: failing run should be filtered out by dataset selector.
    _touch(sweep / "dataset_b" / "job_0" / "seed_0" / "error.txt")

    failures = rfj.find_failed_runs(
        sweep_dir=tmp_path / "2026-02-06",
        datasets=["dataset_a"],
    )

    assert len(failures) == 2
    reasons = sorted(item.reason for item in failures)
    assert reasons == ["error.txt + missing summary", "missing summary"]
    assert all(item.dataset == "dataset_a" for item in failures)


def test_main_exits_nonzero_when_failures_found(tmp_path: Path, monkeypatch) -> None:
    sweep = tmp_path / "2026-02-06" / "20-49-41"
    (sweep / "dataset_a" / "job_0" / "seed_0").mkdir(parents=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_failed_jobs.py",
            "--sweep-dir",
            str(tmp_path / "2026-02-06"),
            "--exit-nonzero",
        ],
    )
    with pytest.raises(SystemExit, match="1"):
        rfj.main()
