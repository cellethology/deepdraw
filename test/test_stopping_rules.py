"""Measured-threshold stopping and EI in objective units."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.models import SingleTaskGP
from scipy.stats import norm
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from benchmarks.noise_simulation import RetrospectiveExperiment
from core.botorch_models import BoTorchGPRegressor
from core.experiment import ActiveLearningExperiment
from core.predictor_trainer import PredictorTrainer
from core.query_strategies import Random, TopPredictions
from core.stopping import (
    StoppingConfig,
    ei_stopping_trainer,
    maximum_expected_improvement,
    stopping_decision,
)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"rule": "other"}, "stopping_rule"),
        ({"rule": "ei"}, "epsilon"),
        ({"rule": "ei", "epsilon": 0}, "epsilon"),
        ({"rule": "ei", "epsilon": np.nan}, "epsilon"),
        ({"rule": "threshold"}, "phenotype_threshold"),
        ({"rule": "threshold", "phenotype_threshold": np.inf}, "phenotype_threshold"),
    ],
)
def test_stopping_rule_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        StoppingConfig(**settings)


def _transformed_gp(logarithmic=True):
    x = np.array([[0.0], [0.4], [0.8]])
    scores = np.expm1([2.0, 6.0, 10.0])
    steps = [("log1p", FunctionTransformer(np.log1p))] if logarithmic else []
    transformer = Pipeline([*steps, ("scaler", StandardScaler())]).fit(scores[:, None])
    model = SingleTaskGP(
        torch.as_tensor(x, dtype=torch.double),
        torch.as_tensor(transformer.transform(scores[:, None]), dtype=torch.double),
    ).eval()
    wrapper = SimpleNamespace(get_botorch_model=lambda: model)
    fitted = TransformedTargetRegressor()
    fitted.regressor_ = wrapper
    fitted.transformer_ = transformer
    trainer = SimpleNamespace(get_model=lambda: fitted)
    return trainer, model, scores, transformer


def test_ei_undoes_external_scaling_once_and_retains_log_objective():
    trainer, model, scores, transformer = _transformed_gp()
    pool = np.array([[0.2], [1.0]])
    with torch.no_grad():
        posterior = model.posterior(torch.as_tensor(pool, dtype=torch.double))
    scaler = transformer.named_steps["scaler"]
    mean = posterior.mean.numpy().ravel() * scaler.scale_[0] + scaler.mean_[0]
    std = posterior.variance.sqrt().numpy().ravel() * scaler.scale_[0]
    improvement = mean - np.log1p(scores).max()
    z = improvement / std
    expected = np.max(improvement * norm.cdf(z) + std * norm.pdf(z))
    actual = maximum_expected_improvement(trainer, pool, scores)
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert stopping_decision(
        StoppingConfig(rule="ei", epsilon=actual * 1.1),
        trainer=trainer,
        pool_features=pool,
        observed_scores=scores,
    )["stopped"]
    assert not stopping_decision(
        StoppingConfig(rule="ei", epsilon=actual * 0.9),
        trainer=trainer,
        pool_features=pool,
        observed_scores=scores,
    )["stopped"]


def test_threshold_is_inclusive_and_needs_no_predictor():
    result = stopping_decision(
        StoppingConfig(rule="threshold", phenotype_threshold=5.0),
        trainer=None,
        pool_features=np.ones((3, 1)),
        observed_scores=np.array([2.0, 5.0]),
    )
    assert result["stopped"]
    assert result["reason"] == "phenotype_threshold"
    assert result["max_ei"] is None


def test_low_level_ei_requires_a_gaussian_posterior():
    fitted = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="Gaussian process"):
        maximum_expected_improvement(
            SimpleNamespace(get_model=lambda: fitted),
            np.array([[0.5]]),
            np.array([0.0, 1.0]),
        )


def test_ei_reuses_a_compatible_fitted_search_gp(monkeypatch):
    trainer, _, scores, _ = _transformed_gp()

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("A compatible fitted GP should be reused.")

    monkeypatch.setattr(BoTorchGPRegressor, "fit", unexpected_fit)
    assert (
        ei_stopping_trainer(
            trainer, train_features=np.ones((3, 1)), observed_scores=scores
        )
        is trainer
    )


def test_separate_stopping_gp_preserves_search_model_and_input_data(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(BoTorchGPRegressor, "_fit_mll", lambda self, mll: None)
    features = np.array([[0.0], [0.4], [0.8]])
    scores = np.array([1.0, 3.0, 9.0])
    original_features, original_scores = features.copy(), scores.copy()
    trainer = PredictorTrainer(
        Ridge(),
        feature_transform=[("features", StandardScaler(copy=False))],
        target_transform=[("targets", StandardScaler(copy=False))],
    )
    trainer.train(features.copy(), scores.copy())
    search_model = trainer.get_model()
    expected_predictions = trainer.predict(features.copy())

    stopping_trainer = ei_stopping_trainer(
        trainer, train_features=features, observed_scores=scores
    )

    assert stopping_trainer is not trainer
    assert isinstance(stopping_trainer.base_predictor, BoTorchGPRegressor)
    assert stopping_trainer.base_predictor.device == "cpu"
    fitted_gp = (
        stopping_trainer.get_model()
        .regressor_.named_steps["estimator"]
        .get_botorch_model()
    )
    assert fitted_gp.train_inputs[0].device.type == "cpu"
    assert trainer.get_model() is search_model
    np.testing.assert_array_equal(
        trainer.predict(features.copy()), expected_predictions
    )
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(scores, original_scores)
    assert np.isfinite(
        maximum_expected_improvement(stopping_trainer, np.array([[0.2]]), scores)
    )


def test_ei_does_not_mutate_inputs_with_in_place_transformers():
    trainer, _, scores, transformer = _transformed_gp(logarithmic=False)
    transformer.named_steps["scaler"].copy = False
    pool = np.array([[0.2], [1.0]])
    expected_scores = scores.copy()
    expected_pool = pool.copy()
    feature_scaler = StandardScaler(copy=False).fit([[0.0], [1.0]])
    fitted = trainer.get_model()
    fitted.regressor_ = Pipeline(
        [("features", feature_scaler), ("predictor", fitted.regressor_)]
    )
    maximum_expected_improvement(trainer, pool, scores)
    np.testing.assert_array_equal(scores, expected_scores)
    np.testing.assert_array_equal(pool, expected_pool)


def test_ei_rejects_unhandled_target_scaling():
    trainer, _, scores, _ = _transformed_gp()
    trainer.get_model().transformer_ = Pipeline([("minmax", MinMaxScaler())]).fit(
        scores[:, None]
    )
    with pytest.raises(ValueError, match="StandardScaler"):
        maximum_expected_improvement(trainer, np.array([[0.5]]), scores)


def _experiment(tmp_path, **kwargs):
    x = np.linspace(0, 1, 8)[:, None]
    np.savez(tmp_path / "features.npz", ids=np.arange(len(x)), embeddings=x)
    pd.DataFrame({"score": np.linspace(1.0, 8.0, len(x))}).to_csv(
        tmp_path / "scores.csv", index=False
    )
    initial = SimpleNamespace(name="FIXED", select=lambda dataset: [0, 1])
    settings = {
        "embeddings_path": str(tmp_path / "features.npz"),
        "metadata_path": str(tmp_path / "scores.csv"),
        "label_key": "score",
        "initial_selection_strategy": initial,
        "query_strategy": Random(seed=4),
        "predictor": BoTorchGPRegressor(),
        "starting_batch_size": 2,
        "batch_size": 2,
        "target_transforms": [
            ("log1p", FunctionTransformer(np.log1p, np.expm1)),
            ("scaler", StandardScaler()),
        ],
    }
    settings.update(kwargs)
    experiment_class = (
        RetrospectiveExperiment
        if "measurement_simulation" in settings or "confirmed_top_stopping" in settings
        else ActiveLearningExperiment
    )
    return experiment_class(**settings)


def test_ei_stops_after_initialization_even_with_random_query(tmp_path, monkeypatch):
    monkeypatch.setattr(BoTorchGPRegressor, "_fit_mll", lambda self, mll: None)
    experiment = _experiment(tmp_path, stopping_rule="ei", epsilon=1e6)
    records = experiment.run_experiment(max_rounds=2)
    assert len(records) == 1
    assert experiment.trainer.get_model() is not None
    assert experiment.early_stop_info["reason"] == "ei"
    assert experiment.early_stop_info["round"] == 1


@pytest.mark.parametrize("strategy", ["random", "topk"])
def test_ei_fits_each_measured_prefix_once_including_final(
    tmp_path, monkeypatch, strategy
):
    fits = []
    monkeypatch.setattr(
        BoTorchGPRegressor, "_fit_mll", lambda self, mll: fits.append(1)
    )
    experiment = _experiment(
        tmp_path,
        stopping_rule="ei",
        epsilon=1e-30,
        query_strategy=Random(seed=4) if strategy == "random" else TopPredictions(),
    )
    records = experiment.run_experiment(max_rounds=1)
    assert len(records) == 2
    assert len(fits) == 2
    assert experiment.stopping_result["max_ei"] > 1e-30
    assert experiment.early_stop_info is None


@pytest.mark.parametrize("observed, expected_stop", [(1.0, False), (5.0, True)])
def test_threshold_uses_noisy_observations_instead_of_reference(
    tmp_path, monkeypatch, observed, expected_stop
):
    experiment = _experiment(
        tmp_path,
        stopping_rule="threshold",
        phenotype_threshold=5.0,
        measurement_simulation={"enabled": True},
    )
    simulator = experiment.measurement_simulator

    def measure(indices, round_num):
        simulator.observed_means[indices] = observed
        simulator.replicate_counts[indices] = 1

    monkeypatch.setattr(simulator, "measure", measure)
    experiment.dataset.labels[:2] = 1000.0
    records = experiment.run_experiment(max_rounds=0)
    assert len(records) == 1
    assert experiment.stopping_result["best_score"] == observed
    assert experiment.stopping_result["stopped"] == expected_stop


@pytest.mark.parametrize("predictor_class", [BoTorchGPRegressor, Ridge])
def test_unmeasured_reference_scores_do_not_change_ei(
    tmp_path, monkeypatch, predictor_class
):
    monkeypatch.setattr(BoTorchGPRegressor, "_fit_mll", lambda self, mll: None)
    first = _experiment(
        tmp_path, stopping_rule="ei", epsilon=1e6, predictor=predictor_class()
    )
    first.run_experiment(max_rounds=0)
    second = _experiment(
        tmp_path, stopping_rule="ei", epsilon=1e6, predictor=predictor_class()
    )
    second.dataset.labels[2:] *= 10000
    second.run_experiment(max_rounds=0)
    assert first.stopping_result == second.stopping_result


@pytest.mark.parametrize("strategy", ["random", "topk"])
@pytest.mark.parametrize("epsilon, expected_rounds", [(1e6, 1), (1e-30, 2)])
def test_non_gp_search_uses_separate_gp_for_ei(
    tmp_path, monkeypatch, strategy, epsilon, expected_rounds
):
    fits = []
    monkeypatch.setattr(
        BoTorchGPRegressor, "_fit_mll", lambda self, mll: fits.append(1)
    )
    query = Random(seed=4) if strategy == "random" else TopPredictions()
    experiment = _experiment(
        tmp_path,
        predictor=Ridge(),
        query_strategy=query,
        stopping_rule="ei",
        epsilon=epsilon,
    )
    search_trainer = experiment.trainer
    used_for_selection = []
    original_select = query.select

    def record_selection(current):
        used_for_selection.append(current.trainer)
        return original_select(current)

    monkeypatch.setattr(query, "select", record_selection)
    records = experiment.run_experiment(max_rounds=1)

    assert len(records) == expected_rounds
    assert len(fits) == expected_rounds
    assert experiment.failure_info is None
    assert experiment.trainer is search_trainer
    assert isinstance(search_trainer.base_predictor, Ridge)
    assert experiment.stopping_trainer is not search_trainer
    assert experiment.stopping_result["model_source"] == "separate_gp"
    assert experiment.stopping_result["predictor"] == "BoTorchGPRegressor"
    assert all(trainer is search_trainer for trainer in used_for_selection)
    assert experiment.stopping_result["stopped"] is (expected_rounds == 1)


def test_separate_stopping_gp_receives_simulated_scores_and_variances(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(BoTorchGPRegressor, "_fit_mll", lambda self, mll: None)
    original_train = PredictorTrainer.train
    observed_training = []

    def record_train(self, X_train, y_train, y_var_train=None):
        if isinstance(self.base_predictor, BoTorchGPRegressor):
            observed_training.append((np.asarray(y_train).copy(), y_var_train.copy()))
        return original_train(self, X_train, y_train, y_var_train)

    monkeypatch.setattr(PredictorTrainer, "train", record_train)
    experiment = _experiment(
        tmp_path,
        predictor=Ridge(),
        stopping_rule="ei",
        epsilon=1e6,
        measurement_simulation={
            "enabled": True,
            "replicates_per_construct": 3,
            "noise_sigma_log10_expression": 0.2,
        },
    )
    experiment.run_experiment(max_rounds=0)

    assert len(observed_training) == 1
    scores, variances = observed_training[0]
    np.testing.assert_array_equal(
        scores, experiment.get_training_targets(experiment.train_indices)
    )
    np.testing.assert_array_equal(
        variances, experiment.get_training_y_var(experiment.train_indices)
    )
    assert not np.array_equal(
        scores, experiment.dataset.labels[experiment.train_indices]
    )


def test_separate_stopping_gp_failure_is_recorded(tmp_path, monkeypatch):
    def fail_fit(*args, **kwargs):
        raise RuntimeError("Stopping GP could not be fitted.")

    monkeypatch.setattr(BoTorchGPRegressor, "fit", fail_fit)
    experiment = _experiment(
        tmp_path, predictor=Ridge(), stopping_rule="ei", epsilon=1e6
    )
    records = experiment.run_experiment(max_rounds=2)

    assert len(records) == 1
    assert experiment.failure_info["stage"] == "stopping_train"
    assert experiment.failure_info["error_type"] == "RuntimeError"
    assert experiment.early_stop_info is None


def test_user_and_confirmed_top_stopping_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError, match="either stopping_rule"):
        _experiment(
            tmp_path,
            stopping_rule="threshold",
            phenotype_threshold=2.0,
            confirmed_top_stopping={"enabled": True},
            measurement_simulation={"enabled": True},
        )
