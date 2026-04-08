"""
Aggregate summary.json files from Hydra sweeps into by-round CSVs.

Run this script after SubmitIt jobs finish to generate
combined_summaries.by_round.csv for every dataset directory under a
specific sweep timestamp directory.

Example:
    python job_sub/aggregate_summaries.py --sweep-dir job_sub/multirun/2026-01-11 \\
    python job_sub/aggregate_summaries.py --sweep-dir job_sub/multirun/2025-12-30 \\
        --summary-names summary.json,summary_2.json,summary_5.json
    python job_sub/aggregate_summaries.py --sweep-dir job_sub/multirun/2025-12-30 --overwrite
"""

import argparse
import ast
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_MARKER_NAME = ".summaries_combined"
_SUMMARY_BY_ROUND_KEY = "summary_by_round"


def combine_summaries(
    sweep_dir: Path,
    dataset_name: str | None = None,
    summary_names: Sequence[str] | None = None,
) -> dict[str, int]:
    """Combine summary files inside the sweep (per dataset) for multiple names."""
    search_dir = sweep_dir / dataset_name if dataset_name else sweep_dir
    if not search_dir.exists():
        search_dir = sweep_dir
    summary_names = summary_names or ["summary.json"]
    name_set = set(summary_names)
    summary_files = sorted(search_dir.rglob("*.json"))
    if not summary_files:
        return dict.fromkeys(summary_names, 0)

    round_rows_by_name = {name: [] for name in summary_names}
    for path in tqdm(summary_files, desc=f"Reading summaries for {dataset_name}"):
        if path.name not in name_set:
            continue
        try:
            data = json.loads(path.read_text())
            _apply_overrides_to_summary(data)
            data["summary_path"] = str(path)
            summary_by_round = data.get(_SUMMARY_BY_ROUND_KEY)
            if isinstance(summary_by_round, list):
                base = {k: v for k, v in data.items() if k != _SUMMARY_BY_ROUND_KEY}
                for idx, record in enumerate(summary_by_round):
                    if not isinstance(record, dict):
                        continue
                    round_record = dict(record)
                    round_record.setdefault("round", idx)
                    round_rows_by_name[path.name].append(
                        {
                            **base,
                            **round_record,
                        }
                    )
        except json.JSONDecodeError:
            continue

    counts: dict[str, int] = {}
    for summary_name, round_rows in round_rows_by_name.items():
        if not round_rows:
            counts[summary_name] = 0
            continue
        safe_stem = Path(summary_name).name.replace("/", "_")
        safe_stem = Path(safe_stem).stem
        round_df = pd.DataFrame(round_rows)
        round_output_name = (
            "combined_summaries.by_round.csv"
            if summary_name == "summary.json"
            else f"combined_summaries.{safe_stem}.by_round.csv"
        )
        round_output_csv = search_dir / round_output_name
        round_df.to_csv(round_output_csv, index=False)
        counts[summary_name] = len(round_rows)
        print(f"Combined {len(round_rows)} round summaries into {round_output_csv}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help=(
            "Path to a sweep date directory (e.g., job_sub/multirun/2025-12-30) "
            "or a specific sweep timestamp directory "
            "(e.g., job_sub/multirun/2025-12-30/15-30-03)."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="NAME",
        help="Limit aggregation to specific dataset names. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--summary-names",
        dest="summary_names",
        default=None,
        help=(
            "Comma-separated list of summary filenames "
            "(e.g. summary.json,summary_2.json)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        "--force",
        action="store_true",
        help="Recombine even if a dataset sweep already has a marker file.",
    )
    return parser.parse_args()


def aggregate_summaries(
    sweep_dir: Path | None = None,
    datasets: Sequence[str] | None = None,
    summary_names: Sequence[str] | None = None,
    overwrite: bool = False,
) -> int:
    """Combine summary.json files for completed sweeps.

    Args:
        sweep_dir: Path to a sweep date directory or a specific sweep timestamp directory.
        datasets: Optional iterable of dataset names to aggregate. If None, defaults to dataset configs.
        summary_names: Summary filenames to aggregate.
        overwrite: Recombine even if marker files exist.

    Returns:
        Number of dataset directories aggregated.
    """
    dataset_filters: set[str] | None = None
    if datasets:
        dataset_filters = {name.strip() for name in datasets if name}

    summary_names = summary_names or ["summary.json"]
    marker_names = {}
    for summary_name in summary_names:
        if summary_name == "summary.json":
            marker_names[summary_name] = _MARKER_NAME
        else:
            safe_name = summary_name.replace("/", "_")
            marker_names[summary_name] = f"{_MARKER_NAME}.{safe_name}"

    if sweep_dir is None:
        raise ValueError("sweep_dir is required.")
    sweep_dir = sweep_dir.expanduser().resolve()
    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}")
        return 0
    sweep_dirs = [
        path
        for path in sweep_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    time_dirs = [path for path in sweep_dirs if _is_time_dir(path.name)]
    if time_dirs:
        sweeps = set(time_dirs)
    else:
        sweeps = {sweep_dir}

    aggregated = 0
    for sweep_dir in sorted(sweeps):
        for dataset_dir in _iter_dataset_dirs(sweep_dir):
            dataset_name = dataset_dir.name
            if dataset_filters and dataset_name not in dataset_filters:
                continue
            pending = [
                name
                for name in summary_names
                if overwrite or not (dataset_dir / marker_names[name]).exists()
            ]
            if not pending:
                continue

            counts = combine_summaries(
                sweep_dir, dataset_name=dataset_name, summary_names=pending
            )
            for summary_name in pending:
                marker_path = dataset_dir / marker_names[summary_name]
                if counts.get(summary_name, 0) == 0 and not overwrite:
                    continue
                marker_path.write_text(
                    f"combined at {datetime.now().isoformat(timespec='seconds')}\n"
                )
                aggregated += 1

    if aggregated == 0:
        print("No new datasets required aggregation.")
    else:
        print(f"Aggregated summaries for {aggregated} dataset(s).")
    return aggregated


def _iter_dataset_dirs(sweep_dir: Path) -> list[Path]:
    """Return dataset directories under the given sweep timestamp directory."""
    if not sweep_dir.exists():
        return []
    dirs = [
        path
        for path in sweep_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    return dirs


def _is_time_dir(name: str) -> bool:
    parts = name.split("-")
    if len(parts) != 3:
        return False
    return all(part.isdigit() and len(part) == 2 for part in parts)


def _apply_overrides_to_summary(summary: dict[str, object]) -> None:
    override_map = _build_override_map(summary.get("hydra_overrides"))
    if not override_map:
        return
    column_overrides = [
        ("query_strategy", "query_strategy"),
        ("predictor", "predictor"),
        ("initial_selection_strategy", "initial_selection"),
        ("embedding_model", "embedding_model"),
    ]
    for override_key, column in column_overrides:
        override_value = override_map.get(override_key)
        if override_value:
            summary[column] = override_value


def _build_override_map(raw_overrides: object) -> dict[str, str]:
    overrides: list[str] = []
    if raw_overrides is None:
        return {}
    if isinstance(raw_overrides, str):
        parsed = _parse_override_string(raw_overrides)
        overrides.extend(parsed)
    elif isinstance(raw_overrides, Sequence):
        overrides.extend([str(item) for item in raw_overrides])
    else:
        overrides.append(str(raw_overrides))

    items: dict[str, str] = {}
    for entry in overrides:
        text = str(entry).strip()
        if text.startswith("+"):
            text = text[1:].strip()
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            continue
        items[key] = _strip_quotes(value.strip())
    return items


def _parse_override_string(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [raw]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    if not text:
        return []
    return [raw]


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def main() -> None:
    args = parse_args()
    summary_names = None
    if args.summary_names:
        summary_names = [
            name.strip() for name in args.summary_names.split(",") if name.strip()
        ]
    aggregate_summaries(
        sweep_dir=args.sweep_dir,
        datasets=args.datasets,
        summary_names=summary_names,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
