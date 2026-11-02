"""
Unit tests for BoTorch model wrappers.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _import_botorch_models():
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    return importlib.import_module("core.botorch_models")


class _DummySingleTaskGP:
    def __init__(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> None:
        self.train_inputs = [X]
        self.train_targets = y
        self.kwargs = kwargs
        self.likelihood = object()

    def train(self):
        return self

    def eval(self):
        return self


def test_botorch_gp_kernel_selection_and_dtype():
    botorch_models = _import_botorch_models()
    reg = botorch_models.BoTorchGPRegressor(kernel="rbf", ard=True, dtype="float32")
    X_tensor = torch.zeros(4, 3, dtype=torch.float)

    kernel = reg._build_kernel(X_tensor)
    assert kernel.base_kernel.__class__.__name__ == "StableRBFKernel"
    assert kernel.base_kernel.ard_num_dims == 3
    assert reg._resolve_dtype() == torch.float

    reg.kernel = "linear"
    kernel = reg._build_kernel(X_tensor)
    assert kernel.__class__.__name__ == "LinearKernel"
    assert kernel.ard_num_dims == 3


def test_botorch_gp_kernel_invalid_raises():
    botorch_models = _import_botorch_models()
    reg = botorch_models.BoTorchGPRegressor(kernel="invalid")
    with pytest.raises(ValueError, match="Unsupported kernel"):
        reg._build_kernel(torch.zeros(2, 1))


def test_botorch_gp_invalid_dtype_raises():
    botorch_models = _import_botorch_models()
    reg = botorch_models.BoTorchGPRegressor(dtype="int8")
    with pytest.raises(ValueError, match="Unsupported dtype"):
        reg._resolve_dtype()


def test_botorch_gp_predict_with_std(monkeypatch):
    botorch_models = _import_botorch_models()
    monkeypatch.setattr(
        botorch_models, "fit_gpytorch_mll", lambda *args, **kwargs: None
    )

    X = np.array([[0.0], [1.0], [2.0]], dtype=float)
    y = np.array([0.0, 1.0, 0.5], dtype=float)
    reg = botorch_models.BoTorchGPRegressor()
    reg.fit(X, y)

    mean, std = reg.predict_with_std(X)
    assert mean.shape == (len(X),)
    assert std.shape == (len(X),)
    assert np.all(std >= 0.0)


def test_botorch_gp_fit_passes_train_yvar(monkeypatch):
    botorch_models = _import_botorch_models()
    monkeypatch.setattr(botorch_models, "SingleTaskGP", _DummySingleTaskGP)
    monkeypatch.setattr(
        botorch_models, "fit_gpytorch_mll", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        botorch_models, "ExactMarginalLogLikelihood", lambda *args, **kwargs: object()
    )

    X = np.array([[0.0], [1.0], [2.0]], dtype=float)
    y = np.array([0.0, 1.0, 0.5], dtype=float)
    y_var = np.array([0.1, 0.2, 0.3], dtype=float)
    reg = botorch_models.BoTorchGPRegressor()
    reg.fit(X, y, y_var=y_var)

    assert "train_Yvar" in reg.model_.kwargs
    assert "likelihood" not in reg.model_.kwargs
    np.testing.assert_allclose(
        reg.model_.kwargs["train_Yvar"].detach().cpu().numpy().reshape(-1),
        y_var,
    )


def test_rf_ensemble_predict_with_std():
    botorch_models = _import_botorch_models()
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 0.5, 1.5], dtype=float)
    reg = botorch_models.BoTorchRandomForestEnsembleRegressor(
        n_estimators=5,
        max_depth=2,
        random_state=0,
    )
    reg.fit(X, y)

    mean, std = reg.predict_with_std(X)
    assert mean.shape == (len(X),)
    assert std.shape == (len(X),)
    assert np.all(std >= 0.0)

    posterior = reg.posterior(X)
    samples = posterior.rsample(sample_shape=torch.Size([3]))
    assert samples.shape == (3, len(X), 1)


def test_unfitted_model_raises():
    botorch_models = _import_botorch_models()
    reg = botorch_models.BoTorchGPRegressor()
    with pytest.raises(ValueError, match="fit must be called"):
        reg.predict(np.array([[0.0]]))
