"""
Unit tests for RoundTracker class.
"""

import numpy as np
import pandas as pd
import pytest

from core.round_tracker import RoundTracker

DUMMY_METRICS = {
    "n_top": 1,
    "best_true": 1.0,
    "normalized_true": 0.5,
    "train_spearman": 0.9,
    "pool_spearman": 0.2,
    "extreme_value_auc": 0.8,
}


class TestRoundTracker:
    def test_initialization(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))
        assert tracker.rounds == []
        assert tracker.round_num == 0

    def test_track_round_basic(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))

        tracker.track_round(selected_indices=[0, 2], metrics=DUMMY_METRICS)

        assert len(tracker.rounds) == 1
        recorded = tracker.rounds[0]
        assert recorded["round"] == 0
        assert recorded["selected_sample_ids"] == [0, 2]
        assert recorded["unlabeled_pool_size"] == 1

    def test_track_round_missing_ids(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))

        with pytest.raises(IndexError):
            tracker.track_round(selected_indices=[3], metrics=DUMMY_METRICS)

    def test_compute_summary_metrics_with_no_rounds(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))

        with pytest.raises(
            ValueError,
            match="Cannot compute summary metrics: no rounds have been tracked yet",
        ):
            tracker.compute_summary_metrics()

    def test_compute_summary_metrics_returns_expected_values(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))
        tracker.track_round(
            selected_indices=[0],
            metrics={
                "normalized_true": 0.2,
                "n_top": 1,
                "train_spearman": 0.1,
                "pool_spearman": 0.2,
                "extreme_value_auc": 0.2,
            },
        )
        tracker.track_round(
            selected_indices=[1],
            metrics={
                "normalized_true": 0.4,
                "n_top": 0,
                "train_spearman": 0.3,
                "pool_spearman": 0.4,
                "extreme_value_auc": 0.4,
            },
        )
        tracker.track_round(
            selected_indices=[2],
            metrics={
                "normalized_true": 0.3,
                "n_top": 1,
                "train_spearman": 0.2,
                "pool_spearman": 0.6,
                "extreme_value_auc": 0.5,
            },
        )

        metrics = tracker.compute_summary_metrics()
        assert pytest.approx(1.0 / 3, rel=1e-6) == metrics["auc_true"]
        assert pytest.approx(2.0 / 3.0, rel=1e-6) == metrics["avg_top"]
        assert pytest.approx(1.0, rel=1e-6) == metrics["rounds_to_top"]
        assert pytest.approx(0.4, rel=1e-6) == metrics["overall_true"]
        assert pytest.approx(0.3, rel=1e-6) == metrics["max_train_spearman"]
        assert pytest.approx(0.6, rel=1e-6) == metrics["pool_spearman"]
        assert pytest.approx(0.5, rel=1e-6) == metrics["max_extreme_value_auc"]

    def test_compute_summary_metrics_missing_required_column(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1]))
        tracker.track_round(
            selected_indices=[0],
            metrics={"normalized_true": 0.2, "n_top": 1},
        )

        with pytest.raises(ValueError, match="Metric column"):
            tracker.compute_summary_metrics()

    def test_compute_summary_metrics_history_returns_expected_values(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))
        tracker.track_round(
            selected_indices=[0],
            metrics={
                "normalized_true": 0.2,
                "n_top": 1,
                "train_spearman": 0.1,
                "pool_spearman": 0.2,
                "extreme_value_auc": 0.2,
            },
        )
        tracker.track_round(
            selected_indices=[1],
            metrics={
                "normalized_true": 0.4,
                "n_top": 0,
                "train_spearman": 0.3,
                "pool_spearman": 0.4,
                "extreme_value_auc": 0.4,
            },
        )
        tracker.track_round(
            selected_indices=[2],
            metrics={
                "normalized_true": 0.3,
                "n_top": 1,
                "train_spearman": 0.2,
                "pool_spearman": 0.6,
                "extreme_value_auc": 0.5,
            },
        )

        history = tracker.compute_summary_metrics_history()
        expected = [
            {
                "auc_true": 0.2,
                "avg_top": 1.0,
                "rounds_to_top": 1.0,
                "overall_true": 0.2,
                "max_train_spearman": 0.1,
                "pool_spearman": 0.2,
                "max_extreme_value_auc": 0.2,
            },
            {
                "auc_true": 0.3,
                "avg_top": 2.0 / 3.0,
                "rounds_to_top": 1.0,
                "overall_true": 0.4,
                "max_train_spearman": 0.3,
                "pool_spearman": 0.4,
                "max_extreme_value_auc": 0.4,
            },
            {
                "auc_true": 1.0 / 3.0,
                "avg_top": 2.0 / 3.0,
                "rounds_to_top": 1.0,
                "overall_true": 0.4,
                "max_train_spearman": 0.3,
                "pool_spearman": 0.6,
                "max_extreme_value_auc": 0.5,
            },
        ]

        assert len(history) == len(expected)
        for idx, record in enumerate(history):
            assert record["round"] == idx
            for metric, value in expected[idx].items():
                assert pytest.approx(value, rel=1e-6) == record[metric]

    def test_save_to_csv(self, tmp_path):
        tracker = RoundTracker(sample_ids=np.array([0, 1]))
        tracker.track_round([0], DUMMY_METRICS)

        output_path = tmp_path / "rounds.csv"
        tracker.save_to_csv(output_path)

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert "selected_sample_ids" in df.columns
