"""
Integration tests for ActiveLearningExperiment without a test split.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from benchmarks.noise_simulation import RetrospectiveExperiment
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
        assert "n_confirmed_top" not in pd.read_csv(output_path).columns
        assert not (Path(tmp_path) / "simulated_measurements.csv").exists()
        assert not (Path(tmp_path) / "training_observations.csv").exists()

    def test_replicate_measurements_are_recorded_and_used_for_training(self, tmp_path):
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=12)
        experiment = RetrospectiveExperiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            initial_selection_strategy=RandomInitialSelection(
                seed=0, starting_batch_size=4
            ),
            query_strategy=TopPredictions(),
            predictor=LinearRegression(),
            starting_batch_size=4,
            batch_size=2,
            feature_transforms=[("scaler", StandardScaler())],
            target_transforms=[("log", FunctionTransformer(np.log1p, np.expm1))],
            label_key="Expression",
            measurement_simulation={
                "enabled": True,
                "replicates_per_construct": 3,
                "noise_sigma_log10_expression": 0.0,
            },
        )

        experiment.run_experiment(max_rounds=1)
        output_path = Path(tmp_path) / "replicate_results.csv"
        experiment.save_results(output_path)

        measurements = pd.read_csv(Path(tmp_path) / "simulated_measurements.csv")
        training = pd.read_csv(Path(tmp_path) / "training_observations.csv")
        assert len(measurements) == 18
        assert set(measurements["replicate"]) == {0, 1, 2}
        assert len(training) == 4
        assert "n_confirmed_top" in pd.read_csv(output_path).columns

    def test_confirmed_top_early_stop_waits_for_minimum_and_saves_outputs(
        self, tmp_path
    ):
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=30)
        fixed_initial = type(
            "FixedInitial",
            (),
            {
                "name": "FIXED",
                "select": lambda self, dataset: [29, 0],
            },
        )()
        experiment = RetrospectiveExperiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            initial_selection_strategy=fixed_initial,
            query_strategy=TopPredictions(),
            predictor=LinearRegression(),
            starting_batch_size=2,
            batch_size=2,
            feature_transforms=[("scaler", StandardScaler())],
            target_transforms=[("log", FunctionTransformer(np.log1p, np.expm1))],
            label_key="Expression",
            measurement_simulation={
                "enabled": True,
                "replicates_per_construct": 2,
                "noise_sigma_log10_expression": 0.0,
                "confirmation_z": 1.96,
            },
            confirmed_top_stopping={
                "enabled": True,
                "confirmation_z": 0.0,
                "min_total_rounds": 3,
            },
        )

        rounds = experiment.run_experiment(max_rounds=9)

        assert len(rounds) == 3
        assert len(experiment.train_indices) == 6
        assert experiment.failure_info is None
        assert experiment.early_stop_info == {
            "reason": "confirmed_top",
            "round": 3,
            "first_confirmation_round": 1,
            "confirmation_z": 0.0,
            "min_total_rounds": 3,
            "n_confirmed_top_at_stop": 1,
        }

        output_path = Path(tmp_path) / "early_stop_results.csv"
        experiment.save_results(output_path)
        results = pd.read_csv(output_path)
        measurements = pd.read_csv(Path(tmp_path) / "simulated_measurements.csv")
        training = pd.read_csv(Path(tmp_path) / "training_observations.csv")

        assert len(results) == 3
        assert len(measurements) == 12
        assert measurements["round"].max() == 2
        assert len(training) == 6

    def test_confirmed_top_early_stop_requires_measurement_simulation(self, tmp_path):
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=10)

        with pytest.raises(ValueError, match="measurement_simulation.enabled"):
            RetrospectiveExperiment(
                embeddings_path=emb_path,
                metadata_path=csv_path,
                initial_selection_strategy=RandomInitialSelection(
                    seed=0, starting_batch_size=2
                ),
                query_strategy=Random(seed=0),
                predictor=LinearRegression(),
                starting_batch_size=2,
                batch_size=2,
                label_key="Expression",
                confirmed_top_stopping={"enabled": True},
            )

    def test_confirmed_top_after_minimum_stops_on_confirmation_round(self, tmp_path):
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=20)
        metadata = pd.read_csv(csv_path)
        metadata["Expression"] = 1.0
        metadata.loc[11, "Expression"] = 100.0
        metadata.to_csv(csv_path, index=False)

        fixed_initial = type(
            "FixedInitial",
            (),
            {"name": "FIXED", "select": lambda self, dataset: [0]},
        )()
        ordered_query = type(
            "OrderedQuery",
            (),
            {
                "name": "ORDERED",
                "requires_model": False,
                "select": lambda self, experiment: experiment.unlabeled_indices[
                    : experiment.batch_size
                ],
            },
        )()
        experiment = RetrospectiveExperiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            initial_selection_strategy=fixed_initial,
            query_strategy=ordered_query,
            predictor=LinearRegression(),
            starting_batch_size=1,
            batch_size=1,
            label_key="Expression",
            measurement_simulation={
                "enabled": True,
                "replicates_per_construct": 1,
                "noise_sigma_log10_expression": 0.0,
            },
            confirmed_top_stopping={
                "enabled": True,
                "confirmation_z": 0.0,
                "min_total_rounds": 10,
            },
        )

        rounds = experiment.run_experiment(max_rounds=19)

        assert len(rounds) == 12
        assert experiment.early_stop_info is not None
        assert experiment.early_stop_info["first_confirmation_round"] == 12
        assert experiment.early_stop_info["round"] == 12

    def test_zero_noise_expression_simulation_matches_disabled_behavior(self, tmp_path):
        rng = np.random.default_rng(17)
        n_samples = 30
        embeddings = rng.normal(size=(n_samples, 6)).astype(np.float32)
        ids = np.arange(n_samples, dtype=np.int32)
        embedding_path = tmp_path / "fc_embeddings.npz"
        np.savez_compressed(embedding_path, embeddings=embeddings, ids=ids)

        basal = np.linspace(100.0, 400.0, n_samples)
        fold_change = np.linspace(1.0, 12.0, n_samples)
        induced = basal * fold_change
        label_key = "Fold Change (Induced/Basal)"
        metadata_path = tmp_path / "fc_metadata.csv"
        pd.DataFrame(
            {
                "Sequence": [f"ATGC{i}" for i in range(n_samples)],
                "Basal Exp (GFP, au)": basal,
                "Induced Exp (GFP, au)": induced,
                label_key: fold_change,
            }
        ).to_csv(metadata_path, index=False)

        def build_experiment(measurement_simulation=None):
            experiment_class = ActiveLearningExperiment
            optional_settings = {}
            if measurement_simulation is not None:
                experiment_class = RetrospectiveExperiment
                optional_settings["measurement_simulation"] = measurement_simulation
            return experiment_class(
                embeddings_path=str(embedding_path),
                metadata_path=str(metadata_path),
                initial_selection_strategy=RandomInitialSelection(
                    seed=5, starting_batch_size=6
                ),
                query_strategy=TopPredictions(),
                predictor=LinearRegression(),
                starting_batch_size=6,
                batch_size=4,
                random_seed=5,
                feature_transforms=[("scaler", StandardScaler())],
                target_transforms=[("log", FunctionTransformer(np.log1p, np.expm1))],
                label_key=label_key,
                **optional_settings,
            )

        baseline = build_experiment()
        zero_noise = build_experiment(
            {
                "enabled": True,
                "replicates_per_construct": 1,
                "noise_sigma_log10_expression": 0.0,
                "expression_columns": (
                    "Basal Exp (GFP, au)",
                    "Induced Exp (GFP, au)",
                ),
                "score_mode": "induced_over_basal",
            }
        )

        baseline.run_experiment(max_rounds=3)
        zero_noise.run_experiment(max_rounds=3)

        assert baseline.train_indices == zero_noise.train_indices
        assert baseline.unlabeled_indices == zero_noise.unlabeled_indices
        baseline_rounds = pd.DataFrame(baseline.round_tracker.rounds)
        simulated_rounds = pd.DataFrame(zero_noise.round_tracker.rounds)
        assert set(simulated_rounds.columns) - set(baseline_rounds.columns) == {
            "n_confirmed_top"
        }
        pd.testing.assert_frame_equal(
            baseline_rounds, simulated_rounds[baseline_rounds.columns]
        )
        np.testing.assert_allclose(
            baseline.get_training_targets(baseline.train_indices),
            zero_noise.get_training_targets(zero_noise.train_indices),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            baseline.trainer.predict(embeddings),
            zero_noise.trainer.predict(embeddings),
            rtol=0.0,
            atol=1e-12,
        )
        cumulative_best = [
            record["best_true_score_found"]
            for record in zero_noise.round_tracker.rounds
        ]
        assert cumulative_best == sorted(cumulative_best)

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


def test_core_import_does_not_load_retrospective_tools():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import core.experiment; "
            "assert not any(name == 'benchmarks' or name.startswith('benchmarks.') "
            "for name in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
