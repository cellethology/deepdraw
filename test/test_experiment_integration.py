"""
Integration tests for ActiveLearningExperiment without a test split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from core.experiment import ActiveLearningExperiment
from core.initial_selection_strategies import RandomInitialSelection
from core.query_strategies import Random, TopPredictions


class TestActiveLearningExperiment:
    """Integration tests covering the no-test active learning workflow."""

    def create_dataset(self, tmp_path, n_samples: int = 20, dim: int = 10):
        """Create paired embeddings and metadata files."""
        embeddings = np.random.randn(n_samples, dim).astype(np.float32)
        ids = np.arange(n_samples, dtype=np.int32)
        emb_path = tmp_path / "embeddings.npz"
        np.savez_compressed(emb_path, embeddings=embeddings, ids=ids)

        df = pd.DataFrame(
            {
                "Sequence": [f"ATGC{i}" for i in range(n_samples)],
                "Expression": np.linspace(1.0, n_samples, n_samples),
            }
        )
        csv_path = tmp_path / "metadata.csv"
        df.to_csv(csv_path, index=False)

        return str(emb_path), str(csv_path)

    def _build_experiment(
        self,
        embeddings_path: str,
        metadata_path: str,
        query_strategy,
        starting_batch_size: int,
        batch_size: int,
        initial_selection_strategy=None,
    ) -> ActiveLearningExperiment:
        initial_strategy = initial_selection_strategy or RandomInitialSelection(
            seed=0, starting_batch_size=starting_batch_size
        )
        return ActiveLearningExperiment(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            initial_selection_strategy=initial_strategy,
            query_strategy=query_strategy,
            predictor=LinearRegression(),
            starting_batch_size=starting_batch_size,
            batch_size=batch_size,
            feature_transforms=[("scaler", StandardScaler())],
            target_transforms=[("log", FunctionTransformer(np.log1p, np.expm1))],
            label_key="Expression",
        )

    def test_experiment_initialization(self, tmp_path):
        """Experiment initializes with random initial pool and no test indices."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=18)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=Random(seed=42),
            starting_batch_size=6,
            batch_size=3,
            initial_selection_strategy=RandomInitialSelection(
                seed=42, starting_batch_size=6
            ),
        )

        assert len(experiment.dataset.sample_ids) == 18
        assert experiment.dataset.embeddings is not None
        assert len(experiment.train_indices) == 6
        assert len(experiment.unlabeled_indices) == 12

    def test_run_experiment_multiple_rounds(self, tmp_path):
        """Running multiple rounds grows the training pool and produces metrics."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=25)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=TopPredictions(),
            starting_batch_size=5,
            batch_size=4,
            initial_selection_strategy=RandomInitialSelection(
                seed=42, starting_batch_size=5
            ),
        )

        experiment.run_experiment(max_rounds=3)

        expected_train_size = 5 + (4 * 3)
        assert len(experiment.train_indices) == expected_train_size
        # Initial training set plus one entry per round
        assert len(experiment.round_tracker.rounds) == 1 + 3

    def test_run_experiment_logs_training_and_acquisition_boundaries(
        self, tmp_path, caplog
    ):
        """Each model-based round logs training and acquisition boundaries."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=20)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=TopPredictions(),
            starting_batch_size=5,
            batch_size=3,
            initial_selection_strategy=RandomInitialSelection(
                seed=42, starting_batch_size=5
            ),
        )

        with caplog.at_level("INFO"):
            experiment.run_experiment(max_rounds=1)

        assert "Starting model training for round 1" in caplog.text
        assert "Starting acquisition selection for round 1" in caplog.text

    def test_save_results(self, tmp_path):
        """Saving results writes results, custom metrics, and selected variants."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=20)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=Random(seed=99),
            starting_batch_size=5,
            batch_size=5,
            initial_selection_strategy=RandomInitialSelection(
                seed=99, starting_batch_size=5
            ),
        )

        experiment.run_experiment(max_rounds=2)

        output_path = Path(tmp_path) / "results.csv"
        experiment.save_results(output_path)

        assert output_path.exists()

    def test_label_key_required(self, tmp_path):
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=5)
        with pytest.raises(ValueError):
            ActiveLearningExperiment(
                embeddings_path=emb_path,
                metadata_path=csv_path,
                initial_selection_strategy=RandomInitialSelection(
                    seed=0, starting_batch_size=3
                ),
                query_strategy=Random(seed=0),
                predictor=LinearRegression(),
                starting_batch_size=3,
                batch_size=2,
                feature_transforms=[("scaler", StandardScaler())],
                target_transforms=[("log", FunctionTransformer(np.log1p, np.expm1))],
                label_key=None,
            )

    def test_starting_batch_size_larger_than_total(self, tmp_path):
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=4)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=Random(seed=1),
            starting_batch_size=10,
            batch_size=2,
        )
        assert len(experiment.train_indices) == 4

    def test_run_experiment_stops_when_all_selected(self, tmp_path):
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=6)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=Random(seed=2),
            starting_batch_size=3,
            batch_size=6,
        )
        experiment.run_experiment(max_rounds=5)
        assert len(experiment.train_indices) == len(experiment.dataset.sample_ids)

    def test_backward_compatibility_properties(self, tmp_path):
        """Compatibility properties stay available even without a test split."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=12)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=Random(seed=1),
            starting_batch_size=4,
            batch_size=4,
            initial_selection_strategy=RandomInitialSelection(
                seed=1, starting_batch_size=4
            ),
        )

        assert isinstance(experiment.dataset.sample_ids, np.ndarray)
        assert isinstance(experiment.dataset.labels, np.ndarray)
        assert isinstance(experiment.train_indices, list)
        assert isinstance(experiment.unlabeled_indices, list)
        assert isinstance(experiment.round_tracker.rounds, list)

    def test_training_failure_stops_early_and_preserves_partial_results(
        self, tmp_path, monkeypatch
    ):
        """Training errors stop the loop but keep rounds already tracked."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=20)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=TopPredictions(),
            starting_batch_size=5,
            batch_size=3,
        )

        def fail_train(X_train, y_train):
            raise RuntimeError("all attempts to fit model have failed")

        monkeypatch.setattr(experiment.trainer, "train", fail_train)

        rounds = experiment.run_experiment(max_rounds=3)

        # Only the initial selection round should be present.
        assert len(rounds) == 1
        assert len(experiment.round_tracker.rounds) == 1
        assert experiment.failure_info is not None
        assert experiment.failure_info["stage"] == "train"
        assert experiment.failure_info["round"] == 1

        output_path = Path(tmp_path) / "partial_results.csv"
        experiment.save_results(output_path)
        saved = pd.read_csv(output_path)
        assert len(saved) == 1

    def test_selection_failure_stops_early_and_preserves_partial_results(
        self, tmp_path, monkeypatch
    ):
        """Selection errors stop the loop but keep rounds already tracked."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=20)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=TopPredictions(),
            starting_batch_size=5,
            batch_size=3,
        )

        def fail_select():
            raise RuntimeError("selection interrupted before next batch completed")

        monkeypatch.setattr(experiment, "_select_next_batch", fail_select)

        rounds = experiment.run_experiment(max_rounds=3)

        # Only the initial selection round should be present.
        assert len(rounds) == 1
        assert len(experiment.round_tracker.rounds) == 1
        assert experiment.failure_info is not None
        assert experiment.failure_info["stage"] == "select"
        assert experiment.failure_info["round"] == 1

        output_path = Path(tmp_path) / "partial_results.csv"
        experiment.save_results(output_path)
        saved = pd.read_csv(output_path)
        assert len(saved) == 1
