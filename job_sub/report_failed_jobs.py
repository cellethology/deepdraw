"""
Report unsuccessful Hydra/SubmitIt runs under a sweep directory.

A run is considered unsuccessful if:
1. `error.txt` exists, or
2. expected summary file is missing (default: `summary.json`).

Examples:
    ./.venv/bin/python job_sub/report_failed_jobs.py \
        --sweep-dir job_sub/multirun/2026-02-06/20-49-41

    ./.venv/bin/python job_sub/report_failed_jobs.py \
        --sweep-dir job_sub/multirun/2026-02-06 \
        --dataset AD_part1_indices \
        --exit-nonzero
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FailedRun:
    """Container describing one unsuccessful run directory."""

    sweep: Path
    dataset: str
    run_dir: Path
    has_error: bool
    missing_summary: bool

    @property
    def reason(self) -> str:
        if self.has_error and self.missing_summary:
            return "error.txt + missing summary"
        if self.has_error:
            return "error.txt"
        return "missing summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help=(
            "Path to a sweep date directory (e.g., job_sub/multirun/2026-02-06) "
            "or a specific sweep timestamp directory "
            "(e.g., job_sub/multirun/2026-02-06/20-49-41)."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="NAME",
        help="Limit reporting to specific dataset names. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Expected summary filename for successful runs (default: summary.json).",
    )
    parser.add_argument(
        "--exit-nonzero",
        action="store_true",
        help="Exit with status 1 if any unsuccessful runs are found.",
    )
    return parser.parse_args()


def find_failed_runs(
    sweep_dir: Path,
    datasets: list[str] | None = None,
    summary_name: str = "summary.json",
) -> list[FailedRun]:
    """Return all unsuccessful runs under one or more sweep timestamps."""
    root = sweep_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {root}")

    dataset_filters = {item.strip() for item in (datasets or []) if item.strip()}
    selected_sweeps = _resolve_sweep_dirs(root)

    failures: list[FailedRun] = []
    for sweep in selected_sweeps:
        for dataset_dir in _iter_dataset_dirs(sweep):
            dataset_name = dataset_dir.name
            if dataset_filters and dataset_name not in dataset_filters:
                continue
            run_dirs = _iter_run_dirs(dataset_dir)
            for run_dir in run_dirs:
                has_error = (run_dir / "error.txt").exists()
                missing_summary = not (run_dir / summary_name).exists()
                if has_error or missing_summary:
                    failures.append(
                        FailedRun(
                            sweep=sweep,
                            dataset=dataset_name,
                            run_dir=run_dir,
                            has_error=has_error,
                            missing_summary=missing_summary,
                        )
                    )
    return failures


def _resolve_sweep_dirs(path: Path) -> list[Path]:
    """Resolve input as either a timestamp dir or a date dir containing timestamps."""
    child_dirs = [
        item
        for item in path.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    ]
    time_dirs = [item for item in child_dirs if _is_time_dir(item.name)]
    if time_dirs:
        return sorted(time_dirs)
    return [path]


def _iter_dataset_dirs(sweep_dir: Path) -> list[Path]:
    """Return dataset directories under a timestamp directory."""
    if not sweep_dir.exists():
        return []
    return sorted(
        [
            item
            for item in sweep_dir.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        ]
    )


def _iter_run_dirs(dataset_dir: Path) -> list[Path]:
    """Return run directories (seed dirs when present, else task dirs)."""
    seed_dirs = sorted(path for path in dataset_dir.rglob("seed_*") if path.is_dir())
    if seed_dirs:
        return seed_dirs

    candidate_dirs = []
    for path in dataset_dir.rglob("*"):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if (
            (path / "run_config.log").exists()
            or (path / "summary.json").exists()
            or (path / "error.txt").exists()
        ):
            candidate_dirs.append(path)

    if (
        (dataset_dir / "run_config.log").exists()
        or (dataset_dir / "summary.json").exists()
        or (dataset_dir / "error.txt").exists()
    ):
        candidate_dirs.append(dataset_dir)

    # Preserve deterministic output while removing duplicates.
    seen: set[Path] = set()
    unique_dirs: list[Path] = []
    for path in sorted(candidate_dirs):
        if path in seen:
            continue
        seen.add(path)
        unique_dirs.append(path)
    return unique_dirs


def _is_time_dir(name: str) -> bool:
    parts = name.split("-")
    if len(parts) != 3:
        return False
    return all(part.isdigit() and len(part) == 2 for part in parts)


def _print_report(failures: list[FailedRun], root: Path) -> None:
    if not failures:
        print("No unsuccessful runs found.")
        return

    print(f"Found {len(failures)} unsuccessful run(s):")
    for item in failures:
        run_path = item.run_dir
        try:
            run_path = item.run_dir.relative_to(root)
        except ValueError:
            pass
        print(f"- [{item.reason}] {run_path}")


def main() -> None:
    args = parse_args()
    root = args.sweep_dir.expanduser().resolve()
    failures = find_failed_runs(
        sweep_dir=args.sweep_dir,
        datasets=args.datasets,
        summary_name=args.summary_name,
    )
    _print_report(failures, root=root)
    if args.exit_nonzero and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
