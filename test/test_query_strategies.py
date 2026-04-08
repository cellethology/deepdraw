"""
Unit tests for query strategy implementations.
"""

import importlib
from dataclasses import dataclass

import numpy as np
import pytest

from core.query_strategies import (
    BoTorchAcquisition,
    PredStdHybrid,
    Random,
    TopPredictions,
)


@dataclass
class DummyDataset:
    embeddings: np.ndarray
    labels: np.ndarray


class DummyTrainer:
    def __init__(self, outputs: np.ndarray, stds: np.ndarray = None) -> None:
        self._outputs = outputs
        self._stds = stds if stds is not None else np.zeros_like(outputs)

    def predict(self, _, return_std: bool = False):
        if return_std:
            return self._outputs, self._stds
        return self._outputs


class DummyExperiment:
    def __init__(
        self,
        unlabeled_indices,
        batch_size,
        embeddings=None,
        labels=None,
        trainer=None,
        log_likelihoods=None,
        train_indices=None,
        random_seed=0,
    ):
        self.unlabeled_indices = unlabeled_indices
        self.batch_size = batch_size
        self.dataset = DummyDataset(
            embeddings=embeddings
            if embeddings is not None
            else np.zeros((len(unlabeled_indices), 1)),
            labels=labels if labels is not None else np.zeros(len(unlabeled_indices)),
        )
        self.trainer = trainer or DummyTrainer(
            np.zeros(len(unlabeled_indices), dtype=float)
        )
        self.all_log_likelihoods = log_likelihoods
        self.train_indices = train_indices or []
        self.random_seed = random_seed


class _TrainerWithModel:
    def __init__(self, model) -> None:
        self._model = model

    def get_model(self):
        return self._model


class _DummyPosteriorAcq:
    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, X):
        posterior = self.model.posterior(X)
        return posterior.mean.squeeze(-1)


def _import_botorch_models():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    return importlib.import_module("core.botorch_models")


class TestRandomStrategy:
    def test_random_sampling_is_reproducible(self):
        exp = DummyExperiment(unlabeled_indices=list(range(10)), batch_size=4)
        strategy = Random(seed=42)
        first = strategy.select(exp)
        second = strategy.select(exp)

        assert first == second
        assert len(first) == 4
        assert len(set(first)) == 4


class TestTopPredictionsStrategy:
    def test_top_predictions_selects_highest_scores(self):
        unlabeled = [0, 1, 2, 3]
        embeddings = np.eye(4)
        trainer = DummyTrainer(np.array([0.1, 0.9, 0.2, 0.8]))
        exp = DummyExperiment(
            unlabeled_indices=unlabeled,
            batch_size=2,
            embeddings=embeddings,
            trainer=trainer,
        )

        selected = TopPredictions().select(exp)
        assert sorted(selected) == [1, 3]

    def test_top_predictions_returns_all_when_pool_small(self):
        unlabeled = [0, 1]
        trainer = DummyTrainer(np.array([0.5, 0.4]))
        exp = DummyExperiment(
            unlabeled_indices=unlabeled, batch_size=5, trainer=trainer
        )

        assert TopPredictions().select(exp) == unlabeled


class TestPredStdHybridStrategy:
    def test_pred_std_hybrid_selects_based_on_weighted_score(self):
        unlabeled = [0, 1, 2, 3]
        embeddings = np.eye(4)
        # High pred, low std -> score = 0.75 * 0.9 + 0.25 * 0.1 = 0.7
        # Low pred, high std -> score = 0.75 * 0.1 + 0.25 * 0.9 = 0.3
        preds = np.array([0.1, 0.9, 0.2, 0.8])
        stds = np.array([0.9, 0.1, 0.8, 0.2])
        trainer = DummyTrainer(preds, stds)
        exp = DummyExperiment(
            unlabeled_indices=unlabeled,
            batch_size=2,
            embeddings=embeddings,
            trainer=trainer,
        )

        # With alpha=0.75, should prefer high predictions
        selected = PredStdHybrid(alpha=0.75).select(exp)
        assert sorted(selected) == [1, 3]

    def test_pred_std_hybrid_returns_all_when_pool_small(self):
        unlabeled = [0, 1]
        preds = np.array([0.5, 0.4])
        stds = np.array([0.1, 0.2])
        trainer = DummyTrainer(preds, stds)
        exp = DummyExperiment(
            unlabeled_indices=unlabeled, batch_size=5, trainer=trainer
        )

        assert PredStdHybrid(alpha=0.75).select(exp) == unlabeled


class TestBoTorchAcquisition:
    def test_resolve_qlog_ei_class(self):
        strategy = BoTorchAcquisition(acquisition="qlog_ei", discrete_optimizer="exact")
        acq_class = strategy._resolve_acquisition_class()
        assert acq_class.__name__ == "qLogExpectedImprovement"

    def test_resolve_q_ucb_class(self):
        strategy = BoTorchAcquisition(acquisition="q_ucb", discrete_optimizer="exact")
        acq_class = strategy._resolve_acquisition_class()
        assert acq_class.__name__ == "qUpperConfidenceBound"

    def test_exact_discrete_falls_back_to_greedy_for_x_pending_errors(
        self, monkeypatch
    ):
        strategy = BoTorchAcquisition(acquisition="log_ei", discrete_optimizer="exact")

        def _raise_x_pending(*args, **kwargs):
            raise AttributeError(
                "'LogExpectedImprovement' object has no attribute 'X_pending'"
            )

        monkeypatch.setattr("botorch.optim.optimize_acqf_discrete", _raise_x_pending)
        monkeypatch.setattr(
            strategy,
            "_greedy_indices",
            lambda acq, candidate_set, batch_size: [2, 0],
        )

        selected = strategy._optimize_discrete(
            torch=None,
            acq=object(),
            candidate_set=object(),
            batch_size=2,
        )
        assert selected == [2, 0]

    def test_exact_discrete_reraises_unrelated_errors(self, monkeypatch):
        strategy = BoTorchAcquisition(acquisition="log_ei", discrete_optimizer="exact")

        def _raise_unrelated(*args, **kwargs):
            raise AttributeError("some other acquisition error")

        monkeypatch.setattr("botorch.optim.optimize_acqf_discrete", _raise_unrelated)

        with pytest.raises(AttributeError, match="some other acquisition error"):
            strategy._optimize_discrete(
                torch=None,
                acq=object(),
                candidate_set=object(),
                batch_size=2,
            )

    def test_qlog_nei_accepts_linear_gp_model(self, monkeypatch):
        torch = pytest.importorskip("torch")
        pytest.importorskip("botorch")
        botorch_models = _import_botorch_models()
        monkeypatch.setattr(
            botorch_models, "fit_gpytorch_mll", lambda *args, **kwargs: None
        )

        X = np.array([[0.0], [1.0], [2.0]], dtype=float)
        y = np.array([0.0, 1.0, 0.5], dtype=float)
        reg = botorch_models.BoTorchGPRegressor(kernel="linear")
        reg.fit(X, y)

        strategy = BoTorchAcquisition(
            acquisition="qlog_nei", discrete_optimizer="greedy"
        )
        model = strategy._as_botorch_model(reg)
        X_train = strategy._to_model_tensor(torch, model, X).squeeze(-2)
        acq = strategy._build_acquisition(
            torch=torch,
            model=model,
            best_f=float(y.max()),
            train_X=X_train,
            candidate_set=X_train,
        )

        values = acq(X_train[:2].unsqueeze(-2))
        assert values.shape[0] == 2

    def test_qlog_nei_accepts_rf_ensemble_model(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("botorch")
        botorch_models = _import_botorch_models()

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0.0, 1.0, 0.5, 1.5], dtype=float)
        reg = botorch_models.BoTorchRandomForestEnsembleRegressor(
            n_estimators=5,
            max_depth=2,
            random_state=0,
        )
        reg.fit(X, y)

        strategy = BoTorchAcquisition(
            acquisition="qlog_nei", discrete_optimizer="greedy"
        )
        model = strategy._as_botorch_model(reg)
        X_train = strategy._to_model_tensor(torch, model, X).squeeze(-2)
        acq = strategy._build_acquisition(
            torch=torch,
            model=model,
            best_f=float(y.max()),
            train_X=X_train,
            candidate_set=X_train,
        )

        values = acq(X_train[:2].unsqueeze(-2))
        assert values.shape[0] == 2

    @pytest.mark.parametrize(
        ("factory",), [("linear_gp",), ("bnn_gp",), ("rf_ensemble",)]
    )
    def test_botorch_models_work_with_selection_pipeline(self, monkeypatch, factory):
        botorch_models = _import_botorch_models()
        if factory in {"linear_gp", "bnn_gp"}:
            monkeypatch.setattr(
                botorch_models, "fit_gpytorch_mll", lambda *args, **kwargs: None
            )

        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=float)
        y = np.array([0.1, 0.9, 0.2, 0.8, 0.3], dtype=float)
        if factory == "linear_gp":
            model = botorch_models.BoTorchGPRegressor(kernel="linear")
        elif factory == "bnn_gp":
            model = botorch_models.BoTorchGPRegressor(
                kernel="infinite_width_bnn", ard=True
            )
        elif factory == "rf_ensemble":
            model = botorch_models.BoTorchRandomForestEnsembleRegressor(
                n_estimators=5,
                max_depth=2,
                random_state=0,
            )
        model.fit(X[:3], y[:3])

        strategy = BoTorchAcquisition(
            acquisition="qlog_nei", discrete_optimizer="greedy"
        )
        monkeypatch.setattr(
            strategy,
            "_build_acquisition",
            lambda torch, model, best_f, train_X, candidate_set: _DummyPosteriorAcq(
                model
            ),
        )
        exp = DummyExperiment(
            unlabeled_indices=[3, 4],
            batch_size=1,
            embeddings=X,
            labels=y,
            trainer=_TrainerWithModel(model),
            train_indices=[0, 1, 2],
        )

        selected = strategy.select(exp)
        assert selected in ([3], [4])
