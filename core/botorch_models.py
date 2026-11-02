"""
BoTorch model wrappers for sklearn-style integration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.ensemble import EnsembleModel
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import LinearKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor

from core.stable_rbf_kernel import StableRBFKernel


class _RandomForestEnsembleModel(EnsembleModel):
    """
    BoTorch ensemble wrapper around fitted sklearn tree ensembles.
    """

    def __init__(
        self,
        estimators: list[Any],
        train_X: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__(weights=weights)
        self.estimators = list(estimators)
        self.train_inputs = [train_X]
        self._num_outputs = 1

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        flat_X = X.detach().cpu().reshape(-1, X.shape[-1]).numpy()
        member_values: list[torch.Tensor] = []
        for estimator in self.estimators:
            pred = estimator.predict(flat_X)
            pred_tensor = torch.as_tensor(pred, device=X.device, dtype=X.dtype)
            pred_tensor = pred_tensor.reshape(*batch_shape, q, 1)
            member_values.append(pred_tensor)
        return torch.stack(member_values, dim=len(batch_shape))


class _BoTorchBaseRegressor(RegressorMixin):
    """
    Shared utilities for BoTorch sklearn-style wrappers.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        ard: bool = False,
        kernel: str = "rbf",
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.ard = ard
        self.kernel = kernel
        self.model_: Any | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "ard": self.ard,
            "kernel": self.kernel,
        }

    def set_params(self, **params: Any):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_std(X)
        return mean

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_tensor = self._to_tensor(X)
        with torch.no_grad():
            posterior = self.model_.posterior(X_tensor)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1).cpu().numpy()
        return mean, std

    def posterior(self, X: np.ndarray) -> Any:
        self._check_is_fitted()
        X_tensor = self._to_tensor(X)
        return self.model_.posterior(X_tensor)

    def get_botorch_model(self) -> Any:
        self._check_is_fitted()
        return self.model_

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _resolve_dtype(self) -> torch.dtype:
        if isinstance(self.dtype, torch.dtype):
            return self.dtype
        text = str(self.dtype).lower()
        if text in {"float64", "double", "torch.float64"}:
            return torch.double
        if text in {"float32", "float", "torch.float32"}:
            return torch.float
        raise ValueError(f"Unsupported dtype '{self.dtype}'.")

    def _build_kernel(self, X_tensor: torch.Tensor):
        ard_dims = X_tensor.shape[-1] if self.ard else None
        kernel_key = str(self.kernel).lower()
        if kernel_key in {"rbf", "se", "squared_exponential"}:
            base_kernel = StableRBFKernel(ard_num_dims=ard_dims)
            return ScaleKernel(base_kernel)
        elif kernel_key in {"linear"}:
            # LinearKernel already has its own variance parameter, so wrapping it
            # again in ScaleKernel adds a redundant output scale and makes fitting
            # less stable on tiny high-dimensional batches.
            return LinearKernel(ard_num_dims=ard_dims)
        else:
            raise ValueError(f"Unsupported kernel '{self.kernel}'. Use rbf or linear.")

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        device = self._resolve_device()
        dtype = self._resolve_dtype()
        return torch.as_tensor(array, device=device, dtype=dtype)

    def _check_is_fitted(self) -> None:
        if self.model_ is None:
            raise ValueError(
                f"{self.__class__.__name__}.fit must be called before predict."
            )

    def _fit_mll(self, mll: ExactMarginalLogLikelihood) -> None:
        fit_gpytorch_mll(mll)


class BoTorchGPRegressor(_BoTorchBaseRegressor):
    """
    Sklearn-compatible wrapper around a BoTorch SingleTaskGP.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        ard: bool = False,
        kernel: str = "rbf",
        noise_constraint: float | None = None,
        noise_prior: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(
            device=device,
            dtype=dtype,
            ard=ard,
            kernel=kernel,
        )
        self.noise_constraint = noise_constraint
        self.noise_prior = noise_prior

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params.update(
            {
                "noise_constraint": self.noise_constraint,
                "noise_prior": self.noise_prior,
            }
        )
        return params

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_var: np.ndarray | None = None,
    ) -> BoTorchGPRegressor:
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y).view(-1, 1)
        y_var_tensor = None
        if y_var is not None:
            y_var_tensor = self._to_tensor(y_var).view(-1, 1)
            y_var_tensor = y_var_tensor.clamp_min(1e-12)

        covar_module = self._build_kernel(X_tensor)
        model_kwargs = {
            "covar_module": covar_module.to(
                device=X_tensor.device,
                dtype=X_tensor.dtype,
            ),
        }
        if y_var_tensor is None:
            likelihood = self._build_likelihood()
            model_kwargs["likelihood"] = likelihood.to(
                device=X_tensor.device,
                dtype=X_tensor.dtype,
            )
        else:
            model_kwargs["train_Yvar"] = y_var_tensor

        self.model_ = SingleTaskGP(X_tensor, y_tensor, **model_kwargs)
        self.model_.train()
        mll = ExactMarginalLogLikelihood(self.model_.likelihood, self.model_)
        self._fit_mll(mll)
        self.model_.eval()
        return self

    def _build_likelihood(self) -> GaussianLikelihood:
        constraint = (
            GreaterThan(self.noise_constraint)
            if self.noise_constraint is not None
            else None
        )
        likelihood = (
            GaussianLikelihood(noise_constraint=constraint)
            if constraint is not None
            else GaussianLikelihood()
        )
        if self.noise_prior is not None:
            concentration, rate = self.noise_prior
            likelihood.noise_covar.register_prior(
                "noise_prior",
                GammaPrior(concentration, rate),
                "noise",
            )
        return likelihood


class BoTorchRandomForestEnsembleRegressor(_BoTorchBaseRegressor):
    """
    Sklearn-compatible random forest with a BoTorch ensemble posterior.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = 1.0,
        bootstrap: bool = True,
        random_state: int | None = None,
        n_jobs: int | None = None,
        dtype: str = "float64",
    ) -> None:
        super().__init__(device="cpu", dtype=dtype, ard=False, kernel="rbf")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimator_: RandomForestRegressor | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "dtype": self.dtype,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> BoTorchRandomForestEnsembleRegressor:
        X_np = np.asarray(X)
        y_np = np.asarray(y)
        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        estimator.fit(X_np, y_np)
        self.estimator_ = estimator
        train_X = torch.as_tensor(
            X_np, device=torch.device("cpu"), dtype=self._resolve_dtype()
        )
        self.model_ = _RandomForestEnsembleModel(
            estimators=list(estimator.estimators_),
            train_X=train_X,
        )
        self.model_.eval()
        return self
