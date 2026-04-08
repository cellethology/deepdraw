"""Unit tests for by-round summary aggregation."""

import json

import pandas as pd

from job_sub.aggregate_summaries import combine_summaries


def test_combine_summaries_writes_only_by_round_csv(tmp_path):
    dataset_dir = tmp_path / "dataset_a" / "job_0" / "seed_0"
    dataset_dir.mkdir(parents=True)
    summary_path = dataset_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset_name": "dataset_a",
                "query_strategy": "BOTORCH_QLOG_NEI",
                "predictor": "BoTorchGPRegressor",
                "summary_by_round": [
                    {"round": 0, "normalized_true": 0.1},
                    {"round": 1, "normalized_true": 0.2},
                ],
            }
        )
    )

    counts = combine_summaries(
        sweep_dir=tmp_path,
        dataset_name="dataset_a",
        summary_names=["summary.json"],
    )

    assert counts["summary.json"] == 2

    by_round_path = tmp_path / "dataset_a" / "combined_summaries.by_round.csv"
    flat_path = tmp_path / "dataset_a" / "combined_summaries.csv"

    assert by_round_path.exists()
    assert not flat_path.exists()

    frame = pd.read_csv(by_round_path)
    assert len(frame) == 2
    assert "summary_path" in frame.columns
    assert frame["round"].tolist() == [0, 1]
