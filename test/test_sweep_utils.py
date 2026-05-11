"""Unit tests for Hydra sweep utilities."""

from __future__ import annotations

from pathlib import Path

from job_sub.utils import sweep_utils


def test_collect_user_overrides_drops_only_multirun_flags() -> None:
    argv = [
        "-m",
        "dataset=toy",
        "--multirun",
        "query_strategy=botorch_mes",
        "hydra.verbose=true",
    ]

    assert sweep_utils.collect_user_overrides(argv) == [
        "dataset=toy",
        "query_strategy=botorch_mes",
        "hydra.verbose=true",
    ]


def test_list_sweep_dirs_returns_timestamp_children(tmp_path: Path) -> None:
    (tmp_path / "2026-01-01" / "10-00-00").mkdir(parents=True)
    (tmp_path / "2026-01-01" / "not_a_dir.txt").write_text("x")
    (tmp_path / "2026-01-02" / "11-30-00").mkdir(parents=True)
    (tmp_path / "README.txt").write_text("not a date directory")
    (tmp_path / "loose_sweep").mkdir()

    sweeps = sweep_utils.list_sweep_dirs(tmp_path)

    assert sweeps == {
        tmp_path / "2026-01-01" / "10-00-00",
        tmp_path / "2026-01-02" / "11-30-00",
    }


def test_list_sweep_dirs_missing_base_is_empty(tmp_path: Path) -> None:
    assert sweep_utils.list_sweep_dirs(tmp_path / "missing") == set()
