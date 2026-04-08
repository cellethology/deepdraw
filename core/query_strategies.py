"""
Query strategy implementations.

This module provides a clean, extensible way to implement different
selection strategies for active learning experiments.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class QueryStrategyBase(ABC):
    """
    Abstract base class for query strategies.

    Each concrete strategy implements the `_select_batch` method to choose
    the next batch of samples based on its specific criteria.
    """

    requires_model: bool = True

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    def select(self, experiment: Any) -> list[int]:
        """
        Select the next batch of samples (template method).

        This method handles bounds checking and delegates to _select_batch()
        for the actual selection logic.

        Returns:
            List of selected indices from unlabeled_indices
        """
        unlabeled_pool = experiment.unlabeled_indices
        batch_size = experiment.batch_size

        # Early return if pool is smaller than batch size
        if len(unlabeled_pool) < batch_size:
            return unlabeled_pool

        # Delegate to strategy-specific selection logic
        selected_indices = self._select_batch(experiment, unlabeled_pool, batch_size)
        self._log_round(selected_indices)
        return selected_indices

    @abstractmethod
    def _select_batch(
        self, experiment: Any, unlabeled_pool: list[int], batch_size: int
    ) -> list[int]:
        """
        Strategy-specific batch selection logic.

        Args:
            experiment: The active learning experiment
            unlabeled_pool: List of unlabeled indices (guaranteed to have length >= batch_size)
            batch_size: Number of samples to select (guaranteed to be <= len(unlabeled_pool))

        Returns:
            List of selected indices from unlabeled_pool
        """
        pass

    def _log_round(
        self,
        selected_indices: list[int],
        extra_info: str | None = None,
    ) -> None:
        """
        Log information about the round.

        Args:
            selected_indices: Indices that were selected
            extra_info: Extra information to include in the log message
        """
        log_msg = f"Selected indices: {selected_indices}"
        if extra_info:
            log_msg += f" {extra_info}"
        logger.info(log_msg)


class Random(QueryStrategyBase):
    """Strategy that selects samples randomly."""

    def __init__(self, seed: int) -> None:
        super().__init__("RANDOM")
        self.seed = seed
        self.requires_model = False

    def _select_batch(
        self, experiment: Any, unlabeled_pool: list[int], batch_size: int
    ) -> list[int]:
        rng = np.random.default_rng(self.seed)
        selected_indices = rng.choice(
            unlabeled_pool, batch_size, replace=False
        ).tolist()
        return selected_indices


class TopPredictions(QueryStrategyBase):
    """Selects samples with highest k predicted label values."""

    def __init__(self) -> None:
        super().__init__("TOP_K_PRED")

    def _select_batch(
        self, experiment: Any, unlabeled_pool: list[int], batch_size: int
    ) -> list[int]:
        preds = experiment.trainer.predict(
            experiment.dataset.embeddings[unlabeled_pool, :]
        )

        # get indices of top k predictions (descending)
        top_k_local = np.argpartition(-preds, batch_size - 1)[:batch_size]

        # map to original indices
        selected_indices = [unlabeled_pool[i] for i in top_k_local]
        return selected_indices


class PredStdHybrid(QueryStrategyBase):
    """Selects samples with highest combined prediction and uncertainty."""

    def __init__(self, alpha: float) -> None:
        super().__init__("PRED_STD_HYBRID")
        self.alpha = alpha

    def _select_batch(
        self, experiment: Any, unlabeled_pool: list[int], batch_size: int
    ) -> list[int]:
        # get weighted sum of prediction and standard deviation of prediction
        preds, stds = experiment.trainer.predict(
            experiment.dataset.embeddings[unlabeled_pool, :],
            return_std=True,
        )
        weights = np.array([self.alpha, 1 - self.alpha])
        weighted_preds = preds * weights[0] + stds * weights[1]
        top_k_local = np.argpartition(-weighted_preds, batch_size - 1)[:batch_size]
        selected_indices = [unlabeled_pool[i] for i in top_k_local]
        return selected_indices


class BoTorchAcquisition(QueryStrategyBase):
    """Select samples by maximizing a BoTorch acquisition over the unlabeled pool."""

    def __init__(
        self,
        acquisition: str = "ei",
        beta: float = 2.0,
        maximize: bool = True,
        seed: int | None = None,
        num_mv_samples: int = 10,
        max_value_sampling: str = "gumbel",
        num_optima: int = 32,
        num_samples: int | None = None,
        discrete_optimizer: str = "exact",
    ) -> None:
        super().__init__(f"BOTORCH_{acquisition.upper()}")
        self.acquisition = acquisition.lower()
        self.beta = beta
        self.maximize = maximize
        self.seed = seed
        self.num_mv_samples = max(1, int(num_mv_samples))
        sampling = str(max_value_sampling).lower()
        if sampling not in {"gumbel", "thompson"}:
            raise ValueError("max_value_sampling must be 'gumbel' or 'thompson'.")
        self.use_gumbel = sampling == "gumbel"
        self.num_optima = max(1, int(num_optima))
        self.num_samples = max(1, int(num_samples)) if num_samples is not None else None
        optimizer = str(discrete_optimizer).lower()
        if optimizer not in {"exact", "greedy"}:
            raise ValueError("discrete_optimizer must be 'exact' or 'greedy'.")
        self.discrete_optimizer = optimizer

    def _select_batch(
        self, experiment: Any, unlabeled_pool: list[int], batch_size: int
    ) -> list[int]:
        torch = self._import_torch()
        seed = self.seed
        if seed is None:
            seed = getattr(experiment, "random_seed", None)
        self._seed_torch(torch, seed)
        model, feature_transformer, target_transformer = self._unwrap_estimator(
            experiment.trainer.get_model()
        )

        botorch_model = self._as_botorch_model(model)

        X_pool = experiment.dataset.embeddings[unlabeled_pool, :]
        if feature_transformer is not None:
            X_pool = feature_transformer.transform(X_pool)
        if not isinstance(X_pool, np.ndarray):
            X_pool = np.asarray(X_pool)

        X_train = experiment.dataset.embeddings[experiment.train_indices, :]
        if feature_transformer is not None:
            X_train = feature_transformer.transform(X_train)
        if not isinstance(X_train, np.ndarray):
            X_train = np.asarray(X_train)

        train_labels = experiment.dataset.labels[experiment.train_indices]
        train_labels = self._transform_targets(train_labels, target_transformer)
        best_f = train_labels.max() if self.maximize else train_labels.min()

        X_tensor = self._to_model_tensor(torch, botorch_model, X_pool)
        X_train_tensor = self._to_model_tensor(torch, botorch_model, X_train)

        acq = self._build_acquisition(
            torch=torch,
            model=botorch_model,
            best_f=float(best_f),
            train_X=X_train_tensor.squeeze(-2),
            candidate_set=X_tensor.squeeze(-2),
        )
        if self.acquisition == "ts":
            logger.info(
                "BOTORCH_TS: using direct posterior sampling for batch selection."
            )
            selected_local = self._select_thompson_batch(
                torch=torch,
                model=botorch_model,
                X_tensor=X_tensor,
                batch_size=batch_size,
            )
            return [unlabeled_pool[i] for i in selected_local]
        selected_local = self._optimize_discrete(
            torch=torch,
            acq=acq,
            candidate_set=X_tensor.squeeze(-2),
            batch_size=batch_size,
        )
        return [unlabeled_pool[i] for i in selected_local]

    def _import_torch(self):
        import torch

        return torch

    def _seed_torch(self, torch, seed: int | None) -> None:
        if seed is None:
            return
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _build_acquisition(self, torch, model, best_f, train_X, candidate_set):
        acq_class = self._resolve_acquisition_class()
        kwargs = {
            "model": model,
            "best_f": best_f,
            "maximize": self.maximize,
            "beta": self.beta,
            "X_baseline": train_X,
            "candidate_set": candidate_set,
        }
        if self.acquisition in {"mes", "gibbon"}:
            if self.num_mv_samples is not None:
                kwargs["num_mv_samples"] = self.num_mv_samples
            kwargs["use_gumbel"] = self.use_gumbel
        if self.acquisition in {"pes", "jes"}:
            optimal_inputs, optimal_outputs = self._sample_optima_from_candidates(
                torch=torch,
                model=model,
                candidate_set=candidate_set,
                num_optima=self.num_optima,
            )
            kwargs["optimal_inputs"] = optimal_inputs
            kwargs["optimal_outputs"] = optimal_outputs
            if self.num_samples is not None:
                kwargs["num_samples"] = self.num_samples
        return self._init_acquisition(acq_class, **kwargs)

    def _resolve_acquisition_class(self):
        if self.acquisition == "log_ei":
            from botorch.acquisition.analytic import LogExpectedImprovement

            return LogExpectedImprovement
        if self.acquisition == "log_pi":
            from botorch.acquisition.analytic import LogProbabilityOfImprovement

            return LogProbabilityOfImprovement
        if self.acquisition == "ucb":
            from botorch.acquisition.analytic import UpperConfidenceBound

            return UpperConfidenceBound
        if self.acquisition in {"q_ucb", "qucb"}:
            from botorch.acquisition.monte_carlo import qUpperConfidenceBound

            return qUpperConfidenceBound
        if self.acquisition == "ts":
            from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling

            return PathwiseThompsonSampling
        if self.acquisition == "qlog_ei":
            from botorch.acquisition.logei import qLogExpectedImprovement

            return qLogExpectedImprovement
        if self.acquisition in {"log_nei", "qlog_nei"}:
            from botorch.acquisition.logei import qLogNoisyExpectedImprovement

            return qLogNoisyExpectedImprovement
        if self.acquisition == "mes":
            from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

            return qMaxValueEntropy
        if self.acquisition == "gibbon":
            from botorch.acquisition.max_value_entropy_search import (
                qLowerBoundMaxValueEntropy,
            )

            return qLowerBoundMaxValueEntropy
        if self.acquisition == "pes":
            from botorch.acquisition.predictive_entropy_search import (
                qPredictiveEntropySearch,
            )

            return qPredictiveEntropySearch
        if self.acquisition == "jes":
            from botorch.acquisition.joint_entropy_search import qJointEntropySearch

            return qJointEntropySearch
        raise ValueError(
            f"Unsupported acquisition '{self.acquisition}'. Use one of: "
            "log_ei, log_pi, qlog_ei, qlog_nei, ucb, q_ucb, ts, mes, gibbon, pes, jes."
        )

    def _init_acquisition(self, acq_class, **kwargs):
        params = inspect.signature(acq_class).parameters
        filtered = {key: value for key, value in kwargs.items() if key in params}
        return acq_class(**filtered)

    def _optimize_discrete(self, torch, acq, candidate_set, batch_size: int):
        if self.discrete_optimizer == "greedy":
            return self._greedy_indices(acq, candidate_set, batch_size)
        from botorch.exceptions.errors import UnsupportedError
        from botorch.optim import optimize_acqf_discrete

        try:
            candidates, _ = optimize_acqf_discrete(
                acq_function=acq,
                choices=candidate_set,
                q=batch_size,
                unique=True,
            )
        except (AttributeError, UnsupportedError) as exc:
            if not self._is_x_pending_compat_error(exc):
                raise
            logger.warning(
                "BoTorch exact discrete optimization is incompatible with %s "
                "(%s). Falling back to greedy selection.",
                self.acquisition,
                exc,
            )
            return self._greedy_indices(acq, candidate_set, batch_size)
        return self._map_candidates_to_indices(torch, candidate_set, candidates)

    def _is_x_pending_compat_error(self, exc: Exception) -> bool:
        return "x_pending" in str(exc).lower()

    def _greedy_indices(self, acq, candidate_set, batch_size: int) -> list[int]:
        import torch

        with torch.no_grad():
            scores = acq(candidate_set.unsqueeze(-2)).detach().cpu().numpy().reshape(-1)
        rank_scores = scores if self.maximize else -scores
        top_k_local = np.argpartition(-rank_scores, batch_size - 1)[:batch_size]
        return [int(idx) for idx in top_k_local]

    def _select_thompson_batch(self, torch, model, X_tensor, batch_size: int):
        with torch.no_grad():
            posterior = model.posterior(X_tensor)
            samples = posterior.rsample(sample_shape=torch.Size([batch_size]))
        values = samples.squeeze(-1)
        if values.ndim > 2:
            values = values.squeeze(-1)
        if self.maximize:
            picks = values.argmax(dim=1)
        else:
            picks = values.argmin(dim=1)
        selected: list[int] = []
        for idx in picks.tolist():
            if idx not in selected:
                selected.append(int(idx))
        if len(selected) < batch_size:
            mean_scores = values.mean(dim=0)
            rank_scores = mean_scores if self.maximize else -mean_scores
            for idx in torch.argsort(rank_scores, descending=True).tolist():
                if idx not in selected:
                    selected.append(int(idx))
                    if len(selected) == batch_size:
                        break
        return selected[:batch_size]

    def _map_candidates_to_indices(self, torch, candidate_set, candidates):
        indices: list[int] = []
        for cand in candidates:
            matches = (candidate_set == cand).all(dim=-1)
            if matches.any():
                indices.append(int(matches.nonzero(as_tuple=True)[0][0]))
            else:
                distances = torch.cdist(cand.unsqueeze(0), candidate_set)
                indices.append(int(distances.argmin(dim=1)[0]))
        return indices

    def _sample_optima_from_candidates(
        self, torch, model, candidate_set, num_optima: int
    ):
        candidate = (
            candidate_set.squeeze(-2) if candidate_set.ndim == 3 else candidate_set
        )
        with torch.no_grad():
            posterior = model.posterior(candidate)
            samples = posterior.rsample(sample_shape=torch.Size([num_optima]))
        values = samples.squeeze(-1)
        if self.maximize:
            best_idx = values.argmax(dim=1)
        else:
            best_idx = values.argmin(dim=1)
        optimal_inputs = candidate.index_select(0, best_idx)
        row_idx = torch.arange(num_optima, device=best_idx.device)
        optimal_outputs = values[row_idx, best_idx].unsqueeze(-1)
        return optimal_inputs, optimal_outputs

    def _unwrap_estimator(self, estimator: Any):
        if estimator is None:
            raise ValueError("PredictorTrainer.train must be called before select.")

        target_transformer = None
        if isinstance(estimator, TransformedTargetRegressor):
            target_transformer = estimator.transformer_
            estimator = estimator.regressor_

        feature_transformer = None
        if isinstance(estimator, Pipeline):
            if len(estimator.steps) == 0:
                raise ValueError("Pipeline has no steps to unwrap.")
            if len(estimator.steps) > 1:
                feature_transformer = Pipeline(estimator.steps[:-1])
            estimator = estimator.steps[-1][1]

        return estimator, feature_transformer, target_transformer

    def _as_botorch_model(self, estimator: Any):
        if hasattr(estimator, "get_botorch_model"):
            return estimator.get_botorch_model()
        if hasattr(estimator, "posterior"):
            return estimator
        raise ValueError(
            "Predictor does not expose a BoTorch model. "
            "Use BoTorchGPRegressor or a compatible BoTorch model."
        )

    def _to_model_tensor(self, torch, model: Any, X: np.ndarray):
        train_input = None
        if hasattr(model, "train_inputs") and model.train_inputs:
            train_input = model.train_inputs[0]
        device = train_input.device if train_input is not None else torch.device("cpu")
        dtype = train_input.dtype if train_input is not None else torch.double
        X_tensor = torch.as_tensor(X, device=device, dtype=dtype)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(-2)
        return X_tensor

    def _transform_targets(self, targets: np.ndarray, transformer: Any) -> np.ndarray:
        if transformer is None:
            return np.asarray(targets)
        values = np.asarray(targets).reshape(-1, 1)
        transformed = transformer.transform(values)
        return np.asarray(transformed).reshape(-1)
