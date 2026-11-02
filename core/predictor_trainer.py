"""
Model training and evaluation utilities for active learning experiments.
"""

import inspect
import logging
from typing import Any

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from core.uncertainty import UncertaintyWrapper

logger = logging.getLogger(__name__)


class PredictorTrainer:
    """
    Handles model training and evaluation for active learning experiments.
    """

    def __init__(
        self,
        predictor: RegressorMixin,
        feature_transform: list[tuple[str, Any]] | None = None,
        target_transform: list[tuple[str, Any]] | None = None,
    ) -> None:
        """
        Initialize the predictor trainer.

        Args:
            predictor: Scikit-learn compatible regression predictor
            feature_transform: List of (name, transformer) steps to apply to the *features*
            target_transform: List of (name, transformer) steps to apply to the *targets*
        """
        self.base_predictor = predictor
        self.feature_transform = feature_transform
        self.target_transform = target_transform
        self.model_: Any | None = None

        if feature_transform:
            logger.info(
                f"PredictorTrainer initialized with feature_transform={self.feature_transform}"
            )
        if target_transform:
            logger.info(
                f"PredictorTrainer initialized with target_transform={self.target_transform}"
            )

    def _build_estimator(
        self,
        feature_transform: list[tuple[str, Any]] | None = None,
        target_transform: list[tuple[str, Any]] | None = None,
    ) -> Any:
        """
        Create a fresh estimator with optional feature and target transformers.

        Parameters
        ----------
        feature_transform :
            List of (name, transformer) steps to apply to the *features*
            before the base predictor, e.g.
            [("scaler", StandardScaler()), ("pca", PCA())].

        target_transform :
            List of (name, transformer) steps to apply to the *targets*
            via a Pipeline wrapped in TransformedTargetRegressor, e.g.
            [("log", FunctionTransformer(np.log1p, np.expm1))].
        """
        feature_transform = feature_transform or []
        target_transform = target_transform or []

        estimator = self._build_feature_estimator(feature_transform)

        if target_transform:
            y_pipeline = Pipeline(target_transform)

            estimator = TransformedTargetRegressor(
                regressor=estimator,
                transformer=y_pipeline,
            )

        return estimator

    def _build_feature_estimator(
        self,
        feature_transform: list[tuple[str, Any]] | None = None,
    ) -> Any:
        feature_transform = feature_transform or []
        if feature_transform:
            pipeline_steps = feature_transform + [
                ("estimator", clone(self.base_predictor))
            ]
            return Pipeline(pipeline_steps)
        return clone(self.base_predictor)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        y_var_train: np.ndarray | None = None,
    ) -> None:
        """
        Train a fresh model on training data, re-normalizing features/labels if specified.

        Args:
            X_train: Training features
            y_train: Training labels
            y_var_train: Optional observation variance for each training label
        """
        if len(X_train) == 0:
            raise ValueError("Training requires at least one sample.")

        logger.info(f"Total training samples: {len(X_train)}")

        if y_var_train is not None:
            self.model_ = self._fit_with_target_variance(
                X_train=X_train,
                y_train=y_train,
                y_var_train=y_var_train,
            )
            return

        estimator = self._build_estimator(
            feature_transform=self.feature_transform,
            target_transform=self.target_transform,
        )
        estimator.fit(X_train, y_train)
        self.model_ = estimator

    def _fit_with_target_variance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        y_var_train: np.ndarray,
    ) -> Any:
        estimator = self._build_feature_estimator(self.feature_transform)
        y_values = np.asarray(y_train, dtype=float).reshape(-1)
        y_var_values = np.asarray(y_var_train, dtype=float).reshape(-1)
        if y_var_values.shape != y_values.shape:
            raise ValueError("y_var_train must have the same shape as y_train.")

        if not self.target_transform:
            self._fit_estimator(estimator, X_train, y_values, y_var_values)
            return estimator

        transformer = Pipeline(self.target_transform)
        y_2d = y_values.reshape(-1, 1)
        transformer.fit(y_2d)
        y_transformed = np.asarray(transformer.transform(y_2d)).reshape(-1)
        y_var_transformed = self._transform_target_variance(
            y=y_values,
            y_var=y_var_values,
            transformer=transformer,
        )
        self._fit_estimator(estimator, X_train, y_transformed, y_var_transformed)

        wrapped = TransformedTargetRegressor(
            regressor=estimator,
            transformer=transformer,
        )
        wrapped.regressor_ = estimator
        wrapped.transformer_ = transformer
        wrapped._training_dim = 1
        return wrapped

    def _fit_estimator(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        y_var_train: np.ndarray | None,
    ) -> None:
        if y_var_train is None:
            estimator.fit(X_train, y_train)
            return

        if isinstance(estimator, Pipeline):
            final_name, final_estimator = estimator.steps[-1]
            if self._fit_accepts_y_var(final_estimator):
                estimator.fit(
                    X_train,
                    y_train,
                    **{f"{final_name}__y_var": y_var_train},
                )
                return
            estimator.fit(X_train, y_train)
            return

        if self._fit_accepts_y_var(estimator):
            estimator.fit(X_train, y_train, y_var=y_var_train)
            return
        estimator.fit(X_train, y_train)

    def _fit_accepts_y_var(self, estimator: Any) -> bool:
        try:
            params = inspect.signature(estimator.fit).parameters
        except (AttributeError, TypeError, ValueError):
            return False
        return "y_var" in params or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )

    def _transform_target_variance(
        self,
        *,
        y: np.ndarray,
        y_var: np.ndarray,
        transformer: Pipeline,
    ) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        y_var = np.asarray(y_var, dtype=float).reshape(-1)
        steps = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(y))
        upper = np.asarray(transformer.transform((y + steps).reshape(-1, 1))).reshape(
            -1
        )
        lower = np.asarray(transformer.transform((y - steps).reshape(-1, 1))).reshape(
            -1
        )
        slopes = (upper - lower) / (2.0 * steps)
        transformed = np.maximum((slopes**2) * y_var, 0.0)
        return transformed

    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Make predictions using the trained model.
        """
        if self.model_ is None:
            raise ValueError("PredictorTrainer.train must be called before predict.")

        preds = self.model_.predict(X)

        if not return_std:
            return preds

        stds = UncertaintyWrapper(self.model_).compute_std(X)
        return preds, stds

    def get_model(self) -> Any | None:
        """Return the underlying fitted model (pipeline or estimator)."""
        return self.model_
