"""Unit tests for partial summary generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from utils import generate_partial_summaries as gps


def test_parse_selected_ids_accepts_lists_literals_and_csv_text() -> None:
    assert gps._parse_selected_ids([1, 2]) == [1, 2]
    assert gps._parse_selected_ids("[3, 'x']") == [3, "x"]
    assert gps._parse_selected_ids("a,b,, c") == ["a", "b", " c"]
    assert gps._parse_selected_ids(5) == [5]
    assert gps._parse_selected_ids(None) == []


def test_compute_summary_uses_cumulative_metrics() -> None:
    rows = [
        {
            "round": "0",
            "normalized_true": "0.2",
            "n_top": "0",
            "selected_sample_ids": "[1, 2]",
            "train_spearman": "0.1",
            "extreme_value_auc": "",
        },
        {
            "round": "1",
            "normalized_true": "0.5",
            "n_top": "1",
            "selected_sample_ids": "[3, 4]",
            "train_spearman": "0.3",
            "extreme_value_auc": "0.4",
        },
    ]

    summary = gps._compute_summary(rows)

    assert summary["auc_true"] == pytest.approx(0.35)
    assert summary["avg_top"] == pytest.approx(1.0 / 6.0)
    assert summary["rounds_to_top"] == pytest.approx(2.0)
    assert summary["overall_true"] == pytest.approx(0.5)
    assert summary["max_train_spearman"] == pytest.approx(0.3)
    assert summary["max_extreme_value_auc"] == pytest.approx(0.4)


def test_compute_summary_returns_nan_for_empty_rows() -> None:
    summary = gps._compute_summary([])

    assert set(summary) == set(gps.SUMMARY_METRIC_RULES)
    assert all(value != value for value in summary.values())


def test_load_rows_sorts_by_round_and_defaults_bad_rounds(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "round,normalized_true,selected_sample_ids\n2,0.4,[3]\nbad,0.1,[1]\n1,0.3,[2]\n"
    )

    rows = gps._load_rows(csv_path)

    assert [row["_round"] for row in rows] == [0, 1, 2]


def test_main_writes_explicit_partial_summaries(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "20-00-00" / "dataset_a" / "job_0" / "seed_0"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"dataset_name": "dataset_a"}))
    (run_dir / "results.csv").write_text(
        "round,normalized_true,n_selected_in_top,selected_sample_ids\n"
        "0,0.2,0,variant_1\n"
        "1,0.7,1,variant_2\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_partial_summaries.py",
            str(tmp_path),
            "--n",
            "1,2,99",
        ],
    )

    assert gps.main() == 0

    summary_1 = json.loads((run_dir / "summary_1.json").read_text())
    summary_2 = json.loads((run_dir / "summary_2.json").read_text())
    assert summary_1["dataset_name"] == "dataset_a"
    assert summary_1["overall_true"] == pytest.approx(0.2)
    assert summary_2["overall_true"] == pytest.approx(0.7)
    assert summary_2["rounds_to_top"] == pytest.approx(2.0)
    assert not (run_dir / "summary_99.json").exists()


def test_main_rejects_missing_multirun_dir(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setattr(sys, "argv", ["generate_partial_summaries.py", str(missing)])

    with pytest.raises(SystemExit, match="Multirun dir not found"):
        gps.main()
