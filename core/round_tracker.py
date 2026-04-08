"""
Variant tracking utilities for active learning experiments.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


SUMMARY_METRIC_RULES = {
    "auc_true": ("max_accumulate", "normalized_true"),
    "avg_top": ("top_mean", "n_top"),
    "rounds_to_top": ("rounds_to_top", "n_top"),
    "overall_true": ("max_overall", "normalized_true"),
    "max_train_spearman": ("max_overall", "train_spearman"),
    "max_extreme_value_auc": ("max_overall", "extreme_value_auc"),
    "pool_spearman": ("last", "pool_spearman"),
}


class RoundTracker:
    """
    Tracks selected samples across active learning rounds.
    """

    def __init__(self, sample_ids: np.ndarray) -> None:
        """
        Initialize the round tracker.

        Args:
            sample_ids: Identifiers for each sample in the dataset
        """
        self.sample_ids = sample_ids
        self.rounds: list[dict[str, Any]] = []
        self.round_num = 0

    def track_round(
        self,
        selected_indices: list[int],
        metrics: dict[str, float],
    ) -> None:
        """
        Track samples selected in a round.

        Args:
            selected_indices: Indices of selected samples
            metrics: Metrics for the round
        """
        train_size = (
            self.rounds[-1]["train_size"] + len(self.rounds[-1]["selected_sample_ids"])
            if self.rounds
            else 0
        )
        unlabeled_pool_size = len(self.sample_ids) - train_size - len(selected_indices)
        selected_ids = [int(self.sample_ids[idx]) for idx in selected_indices]
        self.rounds.append(
            {
                "round": self.round_num,
                "train_size": train_size,
                **metrics,
                "selected_sample_ids": selected_ids,
                "unlabeled_pool_size": unlabeled_pool_size,
            }
        )
        self.round_num += 1

    def compute_summary_metrics(self) -> dict[str, float]:
        """
        Compute summary metrics defined by `SUMMARY_METRIC_RULES`.
        """
        return self._compute_summary_metrics_for_rounds(self.rounds)

    def compute_summary_metrics_history(self) -> list[dict[str, float]]:
        """
        Compute summary metrics for each round prefix.
        """
        if not self.rounds:
            raise ValueError(
                "Cannot compute summary metrics: no rounds have been tracked yet"
            )

        history: list[dict[str, float]] = []
        for idx in range(len(self.rounds)):
            summary = self._compute_summary_metrics_for_rounds(self.rounds[: idx + 1])
            summary["round"] = self.rounds[idx]["round"]
            history.append(summary)
        return history

    def _compute_summary_metrics_for_rounds(
        self, rounds: list[dict[str, Any]]
    ) -> dict[str, float]:
        if not rounds:
            raise ValueError(
                "Cannot compute summary metrics: no rounds have been tracked yet"
            )

        cumulative_selected = np.sum(
            np.cumsum(np.array([len(round["selected_sample_ids"]) for round in rounds]))
        )

        summary_values: dict[str, float] = {}

        for metric_name, (rule, metric_column) in SUMMARY_METRIC_RULES.items():
            if metric_column not in rounds[0]:
                raise ValueError(f"Metric column {metric_name} not found in rounds")

            values = np.array([round[metric_column] for round in rounds])
            if rule == "top_mean":
                cumulative_sum = np.cumsum(values)
                summary_values[metric_name] = (
                    float(np.sum(cumulative_sum)) / cumulative_selected
                    if cumulative_selected > 0
                    else 0.0
                )
            elif rule == "mean":
                summary_values[metric_name] = float(np.nanmean(values))
            elif rule == "max_accumulate":
                cumulative_max_per_round = np.maximum.accumulate(values)
                summary_values[metric_name] = float(
                    np.sum(cumulative_max_per_round)
                ) / len(rounds)
            elif rule == "max_overall":
                finite = values[np.isfinite(values)]
                summary_values[metric_name] = (
                    float(np.max(finite)) if finite.size else float("nan")
                )
            elif rule == "rounds_to_top":
                hits = np.where(values >= 1)[0]
                summary_values[metric_name] = (
                    float(hits[0] + 1) if hits.size else float("nan")
                )
            elif rule == "last":
                summary_values[metric_name] = float(values[-1])
            else:
                raise ValueError(
                    f"Unknown summary metric rule '{rule}' for {metric_name}"
                )
        return summary_values

    def save_to_csv(self, output_path: Path) -> None:
        """
        Save rounds to CSV file.

        Args:
            output_path: Path to save rounds
        """
        df = pd.DataFrame(self.rounds)
        df.to_csv(output_path, index=False)
        logger.info(f"Rounds saved to {output_path}")
