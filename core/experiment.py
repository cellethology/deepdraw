"""
ActiveLearningExperiment class.

This class orchestrates the active learning loop using composed components.
"""

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from core.data_loader import DataLoader, Dataset
from core.initial_selection_strategies import (
    InitialSelectionStrategy,
)
from core.metrics_calculator import MetricsCalculator
from core.predictor_trainer import PredictorTrainer
from core.query_strategies import QueryStrategyBase
from core.round_tracker import RoundTracker

logger = logging.getLogger(__name__)


class ActiveLearningExperiment:
    """
    Active learning experiment for sequence-expression prediction.

    This class orchestrates the active learning loop using composed components
    for data loading, model training, metrics calculation, and result management.
    """

    def __init__(
        self,
        embeddings_path: str,
        metadata_path: str,
        initial_selection_strategy: InitialSelectionStrategy,
        query_strategy: QueryStrategyBase,
        predictor: RegressorMixin,
        batch_size: int = 8,
        random_seed: int = 42,
        feature_transforms: list[tuple[str, Any]] | None = None,
        target_transforms: list[tuple[str, Any]] | None = None,
        starting_batch_size: int | None = None,
        label_key: str | None = None,
        subset_ids_path: str | None = None,
    ) -> None:
        """
        Initialize the active learning experiment.

        Args:
            embeddings_path: Path to NPZ file containing embeddings (and optional IDs)
            metadata_csv_path: CSV file with labels aligned to embeddings
            query_strategy: Strategy for selecting next samples
            predictor: Regression model to fit during the loop
            batch_size: Number of samples to select in each round\
            random_seed: Random seed for reproducibility
            feature_transforms: List of (name, transformer) steps to apply to the *features*
            target_transforms: List of (name, transformer) steps to apply to the *targets*
            label_key: Column name in the metadata CSV containing target values
            starting_batch_size: Number of samples to sample initially. If None, will be set to batch_size.
            subset_ids_path: Optional path to a newline-delimited list of sample IDs to keep.
        """
        # Store configuration
        if label_key is None:
            raise ValueError("label_key must be provided for metadata loading.")

        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.query_strategy = query_strategy
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.label_key = label_key
        self.initial_selection_strategy = initial_selection_strategy
        self.subset_ids_path = subset_ids_path
        self.feature_transforms = feature_transforms
        self.target_transforms = target_transforms
        self.failure_info: dict[str, Any] | None = None
        if starting_batch_size is None:
            self.starting_batch_size = self.batch_size
        else:
            self.starting_batch_size = starting_batch_size

        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Initialize components
        self.data_loader = DataLoader(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            label_key=label_key,
            subset_ids_path=subset_ids_path,
        )

        # Load data
        self.dataset: Dataset = self.data_loader.load()

        # Initialize training pool
        self.train_indices = self._select_starting_batch()

        # Apply feature transforms globally (fit on all embeddings) to keep scaling consistent.
        feature_transforms_for_trainer = feature_transforms
        if feature_transforms:
            feature_pipeline = Pipeline(feature_transforms)
            self.dataset.embeddings = feature_pipeline.fit_transform(
                self.dataset.embeddings
            )
            self.feature_transformer_ = feature_pipeline
            feature_transforms_for_trainer = None
            logger.info(
                "Applied global feature transforms to all embeddings; "
                "skipping per-round feature fitting."
            )

        # Initialize trainer
        self.trainer = PredictorTrainer(
            predictor,
            feature_transform=feature_transforms_for_trainer,
            target_transform=target_transforms,
        )

        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(self.dataset.labels)

        # Initialize round tracker
        self.round_tracker = RoundTracker(
            sample_ids=self.dataset.sample_ids,
        )

        logger.info(f"Start experiment with {query_strategy.name}, seed={random_seed}")
        logger.info(
            f"Query strategy random state={getattr(query_strategy, 'random_state', None)}, Predictor random state={getattr(predictor, 'random_state', None)}"
        )

    def _select_starting_batch(self) -> tuple[list[int], list[int]]:
        """
        Sample the initial training pool using the configured strategy.

        Returns:
            List of indices for the initial training pool
        """
        total_samples = len(self.dataset.sample_ids)
        if self.starting_batch_size >= total_samples:
            logger.warning(
                "starting_batch_size >= total samples. Using all samples for training."
            )
            all_indices = list(range(total_samples))
            return all_indices

        selected_indices = self.initial_selection_strategy.select(dataset=self.dataset)

        logger.info(
            f"Initialized training pool with {len(selected_indices)} samples via "
            f"{self.initial_selection_strategy.name}, leaving "
            f"{total_samples - len(selected_indices)} unlabeled samples."
        )

        return selected_indices

    def _select_next_batch(self) -> list[int]:
        """
        Select next batch of samples based on the configured strategy.

        Returns:
            List of indices for next batch of samples
        """
        return self.query_strategy.select(self)

    def _evaluate_and_track(
        self,
        indices: list[int],
        top_p: float = 0.01,
        train_indices: np.ndarray | None = None,
        train_predictions: np.ndarray | None = None,
        pool_indices: np.ndarray | None = None,
        pool_predictions: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Evaluate and track the performance of the model on the given indices.

        Args:
            indices: Indices of the samples to evaluate
            top_p: Percentage of top labels to consider
            train_indices: Indices used to train the model
            train_predictions: Model predictions for the training set
            pool_indices: Indices remaining in the unlabeled pool
            pool_predictions: Model predictions for the pool set
        Returns:
            Metrics for the round
        """
        round_metrics = self.metrics_calculator.compute_metrics_for_round(
            selected_indices=np.asarray(indices),
            train_indices=np.asarray(train_indices)
            if train_indices is not None
            else np.array([]),
            train_predictions=train_predictions,
            pool_indices=np.asarray(pool_indices)
            if pool_indices is not None
            else np.array([]),
            pool_predictions=pool_predictions,
            top_p=top_p,
        )
        self.round_tracker.track_round(selected_indices=indices, metrics=round_metrics)
        return round_metrics

    def _get_round_predictions(
        self, requires_model: bool
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
        train_indices = np.asarray(self.train_indices)
        pool_indices = np.asarray(self.unlabeled_indices)
        if requires_model:
            train_predictions = self.trainer.predict(
                self.dataset.embeddings[train_indices, :]
            )
            pool_predictions = (
                self.trainer.predict(self.dataset.embeddings[pool_indices, :])
                if len(pool_indices) > 0
                else np.array([])
            )
        else:
            train_predictions = None
            pool_predictions = None
        return train_indices, train_predictions, pool_indices, pool_predictions

    def _set_failure_info(self, stage: str, round_num: int, exc: Exception) -> None:
        """Record failure metadata for downstream reporting/summary output."""
        self.failure_info = {
            "stage": stage,
            "round": round_num,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    def run_experiment(
        self, max_rounds: int = 30, top_p: float = 0.01
    ) -> list[dict[str, Any]]:
        """
        Run the active learning experiment.

        Args:
            max_rounds: Maximum number of active learning rounds after the initial selection
            top_p: Percentage of top labels to consider
        Returns:
            List of results for each round
        """
        logger.info(f"Starting active learning run with {max_rounds} max rounds")

        requires_model = getattr(self.query_strategy, "requires_model", True)

        logger.info("--- Initial selection ---")
        self._evaluate_and_track(
            indices=self.train_indices,
            top_p=top_p,
        )

        if not self.unlabeled_indices:
            logger.info(
                "All samples have been selected. Stopping. train=%d total=%d",
                len(self.train_indices),
                len(self.dataset.sample_ids),
            )
            return self.round_tracker.rounds

        if max_rounds <= 0:
            logger.info("max_rounds <= 0; skipping active learning rounds.")
            return self.round_tracker.rounds

        if self.batch_size <= 0:
            logger.info("batch_size <= 0; skipping active learning rounds.")
            return self.round_tracker.rounds

        for round_num in range(max_rounds):
            logger.info(f"--- Round {round_num + 1} ---")
            logger.info(
                "Pool size before selection: %d (train=%d total=%d)",
                len(self.unlabeled_indices),
                len(self.train_indices),
                len(self.dataset.sample_ids),
            )

            if requires_model:
                X_train = self.dataset.embeddings[self.train_indices, :]
                y_train = self.dataset.labels[self.train_indices]
                logger.info("Starting model training for round %d", round_num + 1)
                try:
                    self.trainer.train(X_train=X_train, y_train=y_train)
                except Exception as exc:
                    logger.exception("Training failed at round %d", round_num + 1)
                    self._set_failure_info("train", round_num + 1, exc)
                    logger.warning(
                        "Stopping early after training failure at round %d. "
                        "Returning partial results with %d completed rounds.",
                        round_num + 1,
                        len(self.round_tracker.rounds),
                    )
                    break
            elif round_num == 0:
                logger.info(
                    "Skipping model training because query strategy does not require a predictor."
                )

            try:
                (
                    train_indices,
                    train_predictions,
                    pool_indices,
                    pool_predictions,
                ) = self._get_round_predictions(requires_model)

                logger.info(
                    "Starting acquisition selection for round %d", round_num + 1
                )
                next_batch = self._select_next_batch()
            except Exception as exc:
                logger.exception("Selection failed at round %d", round_num + 1)
                self._set_failure_info("select", round_num + 1, exc)
                logger.warning(
                    "Stopping early after selection failure at round %d. "
                    "Returning partial results with %d completed rounds.",
                    round_num + 1,
                    len(self.round_tracker.rounds),
                )
                break
            if not next_batch:
                logger.info(
                    "No new samples selected. Stopping. pool=%d train=%d total=%d",
                    len(self.unlabeled_indices),
                    len(self.train_indices),
                    len(self.dataset.sample_ids),
                )
                break

            self._evaluate_and_track(
                indices=next_batch,
                top_p=top_p,
                train_indices=train_indices,
                train_predictions=train_predictions,
                pool_indices=pool_indices,
                pool_predictions=pool_predictions,
            )

            self.train_indices.extend(next_batch)

            if len(self.train_indices) == len(self.dataset.sample_ids):
                logger.info(
                    "All samples have been selected. Stopping. train=%d total=%d",
                    len(self.train_indices),
                    len(self.dataset.sample_ids),
                )
                break
        return self.round_tracker.rounds

    def save_results(self, output_path: Path) -> None:
        """
        Save experiment results to CSV files.

        Args:
            output_path: Path to save results
        """
        self.round_tracker.save_to_csv(output_path=output_path)

    @property
    def unlabeled_indices(self) -> list[int]:
        """Compute the remaining unlabeled indices."""
        return [
            idx
            for idx in range(len(self.dataset.sample_ids))
            if idx not in set(self.train_indices)
        ]
