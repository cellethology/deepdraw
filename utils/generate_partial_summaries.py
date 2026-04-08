#!/usr/bin/env python3
"""
Generate summary_n.json files for partial runs based on results.csv.

Example:
  python utils/generate_partial_summaries.py job_sub/multirun/2026-01-01
  python utils/generate_partial_summaries.py job_sub/multirun/2026-01-01 --n 3,5 --overwrite
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

SUMMARY_METRIC_RULES = {
    "auc_true": ("max_accumulate", "normalized_true"),
    "avg_top": ("top_mean", "n_top"),
    "rounds_to_top": ("rounds_to_top", "n_top"),
    "overall_true": ("max_overall", "normalized_true"),
    "max_train_spearman": ("max_overall", "train_spearman"),
    "max_extreme_value_auc": ("max_overall", "extreme_value_auc"),
    "pool_spearman": ("last", "pool_spearman"),
}


def _parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_selected_ids(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [item for item in text.split(",") if item.strip()]
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    for row in rows:
        try:
            row["_round"] = int(row.get("round", 0))
        except (TypeError, ValueError):
            row["_round"] = 0
    return sorted(rows, key=lambda r: r["_round"])


def _compute_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {name: float("nan") for name in SUMMARY_METRIC_RULES}

    columns = set(rows[0].keys())
    n_top_col = None
    if "n_top" in columns:
        n_top_col = "n_top"
    elif "n_selected_in_top" in columns:
        n_top_col = "n_selected_in_top"

    selected_counts = [
        len(_parse_selected_ids(row.get("selected_sample_ids"))) for row in rows
    ]
    cumulative_selected = int(np.sum(np.cumsum(selected_counts)))

    summary: dict[str, float] = {}
    for metric_name, (rule, metric_column) in SUMMARY_METRIC_RULES.items():
        column = metric_column
        if metric_column == "n_top" and n_top_col is not None:
            column = n_top_col
        if column not in columns:
            summary[metric_name] = float("nan")
            continue

        values = np.array([_parse_float(row.get(column)) for row in rows], dtype=float)
        if rule == "top_mean":
            if cumulative_selected <= 0:
                summary[metric_name] = 0.0
            else:
                cumulative_sum = np.cumsum(values)
                summary[metric_name] = (
                    float(np.sum(cumulative_sum)) / cumulative_selected
                )
        elif rule == "mean":
            summary[metric_name] = float(np.nanmean(values))
        elif rule == "max_accumulate":
            cumulative_max = np.maximum.accumulate(values)
            summary[metric_name] = float(np.sum(cumulative_max)) / len(values)
        elif rule == "max_overall":
            finite = values[np.isfinite(values)]
            summary[metric_name] = (
                float(np.max(finite)) if finite.size else float("nan")
            )
        elif rule == "rounds_to_top":
            hits = np.where(values >= 1)[0]
            summary[metric_name] = float(hits[0] + 1) if hits.size else float("nan")
        elif rule == "last":
            summary[metric_name] = float(values[-1])
        else:
            raise ValueError(f"Unknown summary metric rule: {rule}")

    return summary


def _load_base_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _resolve_metrics_to_update(_: dict[str, Any]) -> list[str]:
    return list(SUMMARY_METRIC_RULES.keys())


def _iter_results(root: Path) -> Iterable[Path]:
    max_depth = 4  # root/<time>/<dataset>/<sweep>/seed_*
    root = root.resolve()
    for dirpath, dirnames, filenames in tqdm(
        os.walk(root), desc="Scanning directories", unit="dir"
    ):
        rel_parts = Path(dirpath).relative_to(root).parts
        depth = len(rel_parts)
        if Path(dirpath).name.startswith("seed_"):
            if "results.csv" in filenames:
                yield Path(dirpath) / "results.csv"
            dirnames[:] = []
            continue
        if depth >= max_depth:
            dirnames[:] = []


def _write_summary(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create summary_n.json files for partial runs based on results.csv."
        )
    )
    parser.add_argument(
        "multirun_dir",
        type=Path,
        help="Path to the multirun folder (e.g. job_sub/multirun/2026-01-01).",
    )
    parser.add_argument(
        "--n",
        type=str,
        default=None,
        help="Comma-separated list of n values to generate (default: all rounds).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing summary_n.json files.",
    )
    args = parser.parse_args()

    multirun_dir = args.multirun_dir
    if not multirun_dir.exists():
        raise SystemExit(f"Multirun dir not found: {multirun_dir}")

    explicit_ns: list[int] | None = None
    if args.n:
        explicit_ns = [int(item) for item in args.n.split(",") if item.strip()]

    results_paths = list(_iter_results(multirun_dir))
    for results_path in tqdm(results_paths, desc="Processing runs", unit="run"):
        run_dir = results_path.parent
        if explicit_ns and not args.overwrite:
            expected = [run_dir / f"summary_{n}.json" for n in explicit_ns if n >= 1]
            if expected and all(path.exists() for path in expected):
                continue
        rows = _load_rows(results_path)
        if not rows:
            continue

        base_summary = _load_base_summary(run_dir / "summary.json")
        metrics_to_update = _resolve_metrics_to_update(base_summary)

        max_rounds = len(rows)
        ns = explicit_ns or list(range(1, max_rounds + 1))
        for n in ns:
            if n < 1 or n > max_rounds:
                continue
            output_path = run_dir / f"summary_{n}.json"
            if output_path.exists() and not args.overwrite:
                continue
            partial_rows = rows[:n]
            computed = _compute_summary(partial_rows)
            summary = dict(base_summary)
            for key in metrics_to_update:
                summary[key] = computed.get(key, float("nan"))
            _write_summary(output_path, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
