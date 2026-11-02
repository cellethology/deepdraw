"""Stopping settings and decisions for active-learning experiments."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

logger = logging.getLogger(__name__)


def ei_stopping_trainer(
    search_trainer: Any,
    *,
    train_features: np.ndarray,
    observed_scores: np.ndarray,
    observation_variances: np.ndarray | None = None,
) -> Any:
    """Reuse a fitted GP or fit a separate GP for the stopping decision."""
    from botorch.models.gpytorch import GPyTorchModel

    from core.botorch_models import BoTorchGPRegressor
    from core.predictor_trainer import PredictorTrainer

    estimator = search_trainer.get_model()
    if isinstance(estimator, TransformedTargetRegressor):
        estimator = estimator.regressor_
    if isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1]
    model = (
        estimator.get_botorch_model()
        if hasattr(estimator, "get_botorch_model")
        else estimator
    )
    if isinstance(model, GPyTorchModel) and model.num_outputs == 1:
        return search_trainer

    logger.info("Fitting a separate Gaussian process for EI stopping.")
    trainer = PredictorTrainer(
        predictor=BoTorchGPRegressor(
            device="cpu",
            dtype="float64",
            ard=True,
            kernel="rbf",
            noise_constraint=1e-4,
        ),
        feature_transform=deepcopy(search_trainer.feature_transform),
        target_transform=deepcopy(search_trainer.target_transform),
    )
    trainer.train(
        X_train=np.asarray(train_features, dtype=float).copy(),
        y_train=np.asarray(observed_scores, dtype=float).copy(),
        y_var_train=(
            None
            if observation_variances is None
            else np.asarray(observation_variances, dtype=float).copy()
        ),
    )
    return trainer


@dataclass(frozen=True)
class StoppingConfig:
    rule: str = "none"
    epsilon: float | None = None
    phenotype_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.rule not in {"none", "ei", "threshold"}:
            raise ValueError("stopping_rule must be none, ei, or threshold.")
        if self.rule == "ei" and (
            self.epsilon is None or not np.isfinite(self.epsilon) or self.epsilon <= 0
        ):
            raise ValueError("EI stopping requires a positive finite epsilon.")
        if self.rule == "threshold" and (
            self.phenotype_threshold is None
            or not np.isfinite(self.phenotype_threshold)
        ):
            raise ValueError(
                "Threshold stopping requires a finite phenotype_threshold."
            )


def stopping_decision(
    config: StoppingConfig,
    *,
    trainer: Any,
    pool_features: np.ndarray,
    observed_scores: np.ndarray,
) -> dict[str, Any]:
    """Evaluate only measured scores and the remaining candidate features."""
    scores = np.asarray(observed_scores, dtype=float).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Stopping requires finite measured scores.")
    result = {
        "stopped": False,
        "rule": config.rule,
        "reason": None,
        "best_score": float(scores.max()),
        "max_ei": None,
    }
    if (
        config.rule == "threshold"
        and result["best_score"] >= config.phenotype_threshold
    ):
        result.update(stopped=True, reason="phenotype_threshold")
        return result
    if len(pool_features) == 0:
        result.update(stopped=True, reason="pool_exhausted")
        return result
    if config.rule == "ei":
        value = maximum_expected_improvement(trainer, pool_features, scores)
        result["max_ei"] = value
        if value <= config.epsilon:
            result.update(stopped=True, reason="ei")
    return result


def _objective_scale(transformer: Any) -> float:
    """Undo external standardization, retaining the preceding objective transform."""
    if transformer is None:
        return 1.0
    steps = (
        transformer.steps if isinstance(transformer, Pipeline) else [("y", transformer)]
    )
    scale = 1.0
    standardized = False
    for _, step in steps:
        if step is None or (isinstance(step, str) and step == "passthrough"):
            continue
        if isinstance(step, StandardScaler):
            if step.with_std:
                scale *= float(np.asarray(step.scale_).reshape(-1)[0])
            standardized = True
        elif isinstance(step, FunctionTransformer) and not standardized:
            continue
        else:
            raise ValueError(
                "EI stopping supports objective transforms followed by StandardScaler."
            )
    return scale


def maximum_expected_improvement(
    trainer: Any,
    pool_features: np.ndarray,
    observed_scores: np.ndarray,
) -> float:
    """Maximum latent EI in objective units before external standardization."""
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.models.gpytorch import GPyTorchModel

    estimator = trainer.get_model()
    if estimator is None:
        raise ValueError("EI stopping requires a fitted Gaussian-process predictor.")
    targets = np.asarray(observed_scores, dtype=float).reshape(-1, 1).copy()
    transformer = None
    if isinstance(estimator, TransformedTargetRegressor):
        transformer = estimator.transformer_
        targets = np.asarray(transformer.transform(targets))
        estimator = estimator.regressor_
    scale = _objective_scale(transformer)
    features = np.asarray(pool_features, dtype=float).copy()
    if isinstance(estimator, Pipeline):
        if len(estimator.steps) > 1:
            features = Pipeline(estimator.steps[:-1]).transform(features)
        estimator = estimator.steps[-1][1]
    model = (
        estimator.get_botorch_model()
        if hasattr(estimator, "get_botorch_model")
        else estimator
    )
    if not isinstance(model, GPyTorchModel) or model.num_outputs != 1:
        raise ValueError(
            "EI stopping requires a single-output BoTorch Gaussian process."
        )
    reference = model.train_inputs[0]
    best = torch.as_tensor(
        targets.max(), dtype=reference.dtype, device=reference.device
    )
    acquisition = LogExpectedImprovement(model, best_f=best)
    best_log_ei = -np.inf
    with torch.no_grad():
        for offset in range(0, len(features), 4096):
            points = torch.as_tensor(
                features[offset : offset + 4096],
                dtype=reference.dtype,
                device=reference.device,
            )
            value = float(acquisition(points.unsqueeze(-2)).max())
            if np.isnan(value) or value == np.inf:
                raise ValueError("The GP returned non-finite expected improvement.")
            best_log_ei = max(best_log_ei, value)
    return float(np.exp(best_log_ei) * scale)
