"""
Custom metrics calculation for active learning experiments.
"""

import logging

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates and tracks custom metrics for active learning experiments.
    """

    def __init__(self, labels: np.ndarray) -> None:
        """
        Initialize the metrics calculator.

        Args:
            labels: Array of all target values in the dataset
        """
        self.labels = np.asarray(labels)
        self.cumulative_metrics: list[dict[str, float]] = []
        self._max_label = (
            float(np.max(self.labels)) if self.labels.size else float("nan")
        )
        self._sorted_label_indices = np.argsort(self.labels)
        self._top_cache: dict[float, np.ndarray] = {}
        self._bottom_cache: dict[float, np.ndarray] = {}

    def compute_metrics_for_round(
        self,
        selected_indices: np.ndarray,
        train_indices: np.ndarray,
        train_predictions: np.ndarray | None,
        pool_indices: np.ndarray,
        pool_predictions: np.ndarray | None,
        top_p: float = 0.01,
    ) -> dict[str, float]:
        """
        Compute metrics for a single round.

        Args:
            selected_indices: Indices of selected sequences
            train_indices: Indices used to fit the model
            train_predictions: Model predictions for the training set
            pool_indices: Indices remaining in the unlabeled pool
            pool_predictions: Model predictions for the pool set
            top_p: Percentage of top labels to consider
        """
        # Number of selected samples within the top performers
        n_top = self.n_selected_in_top(
            selected_indices=selected_indices,
            top_p=top_p,
        )

        # Best value ground truth
        best_value_true = np.max(self.labels[selected_indices])
        normalized_true_values = best_value_true / self._max_label

        train_spearman = self._compute_spearman(
            indices=train_indices,
            predictions=train_predictions,
        )

        dataset_indices = np.concatenate([train_indices, pool_indices])
        dataset_predictions = None
        if train_predictions is not None and pool_predictions is not None:
            dataset_predictions = np.concatenate(
                [np.asarray(train_predictions), np.asarray(pool_predictions)]
            )
        dataset_auc = self._compute_auc(
            indices=dataset_indices,
            predictions=dataset_predictions,
            top_p=top_p,
        )

        logger.info(
            "Normalized Best Label: %.3f, Extreme-value AUC: %.3f",
            normalized_true_values,
            dataset_auc,
        )

        return {
            "n_top": n_top,
            "best_true": self._round_metric(best_value_true),
            "normalized_true": self._round_metric(normalized_true_values),
            "train_spearman": self._round_metric(train_spearman),
            "extreme_value_auc": self._round_metric(dataset_auc),
        }

    def n_selected_in_top(
        self, selected_indices: np.ndarray, top_p: float = 0.01
    ) -> int:
        """
        Calculate how many selected samples fall into the top performers.

        Args:
            selected_indices: Indices of selected sequences
            top_p: Percentage of top labels to consider
        """
        top_labels = self._get_top_label_indices(top_p)
        return int(len(np.intersect1d(selected_indices, top_labels)))

    def _get_top_label_indices(self, top_p: float) -> np.ndarray:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")

        top_labels = self._top_cache.get(top_p)
        if top_labels is None:
            num_top = max(1, int(len(self.labels) * top_p))
            top_labels = self._sorted_label_indices[-num_top:]
            self._top_cache[top_p] = top_labels
        return top_labels

    def _compute_auc(
        self, indices: np.ndarray, predictions: np.ndarray | None, top_p: float
    ) -> float:
        if predictions is None or len(indices) < 2:
            return float("nan")
        top_labels = self._get_top_label_indices(top_p)
        bottom_labels = self._get_bottom_label_indices(top_p)
        mask = np.isin(indices, np.concatenate([top_labels, bottom_labels]))
        if np.count_nonzero(mask) < 2:
            return float("nan")
        preds = np.asarray(predictions).ravel()
        if preds.size != len(indices):
            return float("nan")
        filtered_indices = indices[mask]
        binary_labels = np.isin(filtered_indices, top_labels).astype(int)
        if np.min(binary_labels) == np.max(binary_labels):
            return float("nan")
        preds = preds[mask]
        try:
            auc = roc_auc_score(binary_labels, preds)
        except ValueError:
            return float("nan")
        return float(auc)

    def _get_bottom_label_indices(self, top_p: float) -> np.ndarray:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")

        bottom_labels = self._bottom_cache.get(top_p)
        if bottom_labels is None:
            num_bottom = max(1, int(len(self.labels) * top_p))
            bottom_labels = self._sorted_label_indices[:num_bottom]
            self._bottom_cache[top_p] = bottom_labels
        return bottom_labels

    def _compute_spearman(
        self, indices: np.ndarray, predictions: np.ndarray | None
    ) -> float:
        if predictions is None or len(indices) < 2:
            return float("nan")
        labels = np.asarray(self.labels)[indices]
        preds = np.asarray(predictions)
        corr = spearmanr(labels, preds).correlation
        if corr is None:
            return float("nan")
        return float(corr)

    def _round_metric(self, value: float, digits: int = 6) -> float:
        if np.isnan(value):
            return value
        return round(float(value), digits)
