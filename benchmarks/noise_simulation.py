"""Simulated measurement records for retrospective active-learning runs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.experiment import ActiveLearningExperiment
from core.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)

_EXPRESSION_SCORE_MODES = {"induced_over_basal", "and_score", "or_score"}
_SCORE_MODES = {"label", *_EXPRESSION_SCORE_MODES}
_LN_10_SQUARED = float(np.log(10.0) ** 2)


@dataclass(frozen=True)
class MeasurementSimulationConfig:
    """Configuration for simulated assay measurements."""

    enabled: bool = False
    replicates_per_construct: int = 1
    noise_sigma_log10_expression: float = 0.0
    expression_columns: tuple[str, ...] = ()
    score_mode: str = "label"
    basal_column: str | None = None
    induced_column: str | None = None
    single_input_a_column: str | None = None
    single_input_b_column: str | None = None
    dual_input_column: str | None = None
    confirmation_z: float = 1.96
    train_yvar_floor: float = 1e-12

    @classmethod
    def from_config(cls, value: Any | None) -> MeasurementSimulationConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value

        allowed = {field.name for field in fields(cls)}
        data = {
            key: _get_config_value(value, key)
            for key in allowed
            if _has_config_value(value, key)
        }
        if "expression_columns" in data and data["expression_columns"] is not None:
            data["expression_columns"] = tuple(data["expression_columns"])
        return cls(**data)

    def __post_init__(self) -> None:
        if self.replicates_per_construct < 1:
            raise ValueError("replicates_per_construct must be at least 1.")
        if self.noise_sigma_log10_expression < 0:
            raise ValueError("noise_sigma_log10_expression must be non-negative.")
        if self.confirmation_z < 0:
            raise ValueError("confirmation_z must be non-negative.")
        if self.train_yvar_floor < 0:
            raise ValueError("train_yvar_floor must be non-negative.")
        if self.score_mode not in _SCORE_MODES:
            raise ValueError(
                "score_mode must be one of: " + ", ".join(sorted(_SCORE_MODES))
            )


class MeasurementSimulator:
    """Generate and retain noisy replicate observations for selected constructs."""

    def __init__(
        self,
        *,
        sample_ids: np.ndarray | list[Any],
        true_labels: np.ndarray,
        metadata: pd.DataFrame | None,
        label_key: str,
        config: MeasurementSimulationConfig | None = None,
        random_seed: int = 0,
    ) -> None:
        self.sample_ids = np.asarray(sample_ids)
        self.true_labels = np.asarray(true_labels, dtype=float)
        self.metadata = (
            metadata.reset_index(drop=True) if metadata is not None else None
        )
        self.label_key = label_key
        self.config = config or MeasurementSimulationConfig()
        self.rng = np.random.default_rng(random_seed)

        n_samples = len(self.true_labels)
        self.observed_means = np.full(n_samples, np.nan, dtype=float)
        self.replicate_score_means = np.full(n_samples, np.nan, dtype=float)
        self.observed_vars = np.full(n_samples, np.nan, dtype=float)
        self.replicate_counts = np.zeros(n_samples, dtype=int)
        self.selected_rounds = np.full(n_samples, -1, dtype=int)
        self.measurement_rows: list[dict[str, Any]] = []
        self.training_observation_rows: list[dict[str, Any]] = []

        self._validate_metadata()
        expression_columns = self._expression_columns_to_simulate()
        self.expression_replicates = {
            column: np.full(
                (n_samples, self.config.replicates_per_construct),
                np.nan,
                dtype=float,
            )
            for column in expression_columns
        }
        self.observed_expression_means = {
            column: np.full(n_samples, np.nan, dtype=float)
            for column in expression_columns
        }

    def measure(self, indices: list[int] | np.ndarray, round_num: int) -> None:
        """Simulate replicate measurements for newly selected constructs."""
        for sample_index in indices:
            sample_index = int(sample_index)
            if self.replicate_counts[sample_index] > 0:
                continue
            if self.selected_rounds[sample_index] < 0:
                self.selected_rounds[sample_index] = int(round_num)

            measurement_row_start = len(self.measurement_rows)
            replicate_scores = []
            for replicate_index in range(self.config.replicates_per_construct):
                score, row_values, expression_values = self._simulate_one_measurement(
                    sample_index
                )
                replicate_scores.append(score)
                if expression_values is not None:
                    for column, value in expression_values.items():
                        self.expression_replicates[column][
                            sample_index, replicate_index
                        ] = value
                self.measurement_rows.append(
                    {
                        "round": int(round_num),
                        "sample_index": sample_index,
                        "sample_id": self._sample_id(sample_index),
                        "replicate": replicate_index,
                        "replicates_per_construct": (
                            self.config.replicates_per_construct
                        ),
                        "noise_sigma_log10_expression": (
                            self.config.noise_sigma_log10_expression
                        ),
                        "score_mode": self.config.score_mode,
                        "true_score": float(self.true_labels[sample_index]),
                        "observed_score": float(score),
                        **row_values,
                    }
                )

            values = np.asarray(replicate_scores, dtype=float)
            self.replicate_score_means[sample_index] = float(np.mean(values))
            self.replicate_counts[sample_index] = int(values.size)
            if values.size >= 2:
                self.observed_vars[sample_index] = float(np.var(values, ddof=1))

            if self.config.score_mode in _EXPRESSION_SCORE_MODES:
                mean_expressions = {
                    column: float(
                        np.mean(self.expression_replicates[column][sample_index, :])
                    )
                    for column in self.expression_replicates
                }
                for column, value in mean_expressions.items():
                    self.observed_expression_means[column][sample_index] = value
                aggregate_derived_score = self._score_from_expression(mean_expressions)
                observed_score = self._calibrated_expression_score(
                    sample_index,
                    aggregate_derived_score,
                )
            else:
                mean_expressions = {}
                aggregate_derived_score = np.nan
                observed_score = self.replicate_score_means[sample_index]
            self.observed_means[sample_index] = observed_score

            aggregate_values = {
                "aggregate_observed_score": observed_score,
                "aggregate_expression_derived_score": aggregate_derived_score,
                "replicate_score_mean": self.replicate_score_means[sample_index],
                **{
                    f"mean_observed_expression_{column}": value
                    for column, value in mean_expressions.items()
                },
            }
            for row in self.measurement_rows[measurement_row_start:]:
                row.update(aggregate_values)

    def training_targets(self, indices: list[int] | np.ndarray) -> np.ndarray:
        """Return aggregate observed scores for measured training constructs."""
        values = self.observed_means[np.asarray(indices, dtype=int)]
        if np.any(~np.isfinite(values)):
            raise ValueError(
                "Training requested before all selected samples were measured."
            )
        return values

    def training_y_var(self, indices: list[int] | np.ndarray) -> np.ndarray | None:
        """Return replicate-estimated variance of each aggregate score."""
        index_array = np.asarray(indices, dtype=int)
        return self._score_y_var(index_array, apply_floor=True)

    def _score_y_var(
        self,
        index_array: np.ndarray,
        *,
        apply_floor: bool,
    ) -> np.ndarray | None:
        counts = self.replicate_counts[index_array]
        if np.any(counts < 1):
            raise ValueError("Training variance requested for unmeasured samples.")

        if self.config.score_mode in _EXPRESSION_SCORE_MODES:
            return self._expression_score_y_var(
                index_array,
                apply_floor=apply_floor,
            )

        pooled_var = self.pooled_within_construct_variance()
        if pooled_var is None:
            return None

        y_var = pooled_var / counts.astype(float)
        if apply_floor and self.config.train_yvar_floor > 0:
            y_var = np.maximum(y_var, self.config.train_yvar_floor)
        return y_var

    def record_training_observations(
        self,
        *,
        indices: list[int] | np.ndarray,
        training_round: int,
        train_y_var: np.ndarray | None,
    ) -> None:
        """Record the observations and variances used in one model fit."""
        index_array = np.asarray(indices, dtype=int)
        y_var_values = (
            np.full(len(index_array), np.nan, dtype=float)
            if train_y_var is None
            else np.asarray(train_y_var, dtype=float)
        )
        pooled_var = self.pooled_within_construct_variance()
        pooled_value = float(pooled_var) if pooled_var is not None else np.nan
        pooled_log10_vars = self.pooled_log10_expression_variances() or {}

        for sample_index, train_var in zip(index_array, y_var_values, strict=True):
            sample_index = int(sample_index)
            row = {
                "training_round": int(training_round),
                "sample_index": sample_index,
                "sample_id": self._sample_id(sample_index),
                "selected_round": int(self.selected_rounds[sample_index]),
                "replicate_count": int(self.replicate_counts[sample_index]),
                "aggregation_method": (
                    "score_from_mean_expression"
                    if self.config.score_mode in _EXPRESSION_SCORE_MODES
                    else "mean_replicate_score"
                ),
                "true_score": float(self.true_labels[sample_index]),
                "observed_score": float(self.observed_means[sample_index]),
                # Retained for compatibility with earlier simulation outputs.
                "observed_score_mean": float(self.observed_means[sample_index]),
                "replicate_score_mean": float(self.replicate_score_means[sample_index]),
                "replicate_score_var": _finite_or_nan(self.observed_vars[sample_index]),
                "observed_score_var": _finite_or_nan(self.observed_vars[sample_index]),
                "pooled_observed_score_var": pooled_value,
                "train_yvar": _finite_or_nan(train_var),
            }
            for column, means in self.observed_expression_means.items():
                row[f"mean_observed_expression_{column}"] = _finite_or_nan(
                    means[sample_index]
                )
                log10_var = pooled_log10_vars.get(column, np.nan)
                row[f"pooled_log10_expression_var_{column}"] = _finite_or_nan(log10_var)
                row[f"estimated_sigma_log10_expression_{column}"] = (
                    float(np.sqrt(log10_var)) if np.isfinite(log10_var) else np.nan
                )
            self.training_observation_rows.append(row)

    def confirmed_true_top_indices(
        self,
        indices: list[int] | np.ndarray,
        top_p: float,
        confirmation_z: float | None = None,
    ) -> np.ndarray:
        """Return selected indices that are truly top and confirmed by observations."""
        z_value = (
            self.config.confirmation_z
            if confirmation_z is None
            else float(confirmation_z)
        )
        if z_value < 0:
            raise ValueError("confirmation_z must be non-negative.")
        threshold = self.top_threshold(top_p)
        confirmed = []
        index_array = np.asarray(indices, dtype=int)
        y_var_values = self._score_y_var(index_array, apply_floor=False)

        for position, sample_index in enumerate(index_array):
            mean = self.observed_means[sample_index]
            if not np.isfinite(mean):
                continue
            true_top = self.true_labels[sample_index] >= threshold
            if self.replicate_counts[sample_index] <= 1:
                called_top = mean >= threshold
            else:
                var = y_var_values[position] if y_var_values is not None else np.nan
                if not np.isfinite(var):
                    var = 0.0
                se = float(np.sqrt(max(var, 0.0)))
                called_top = mean - z_value * se >= threshold
            if true_top and called_top:
                confirmed.append(int(sample_index))

        return np.asarray(confirmed, dtype=int)

    def top_threshold(self, top_p: float) -> float:
        """Return the true-label threshold for the top fraction."""
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        count = max(1, int(len(self.true_labels) * top_p))
        return float(np.sort(self.true_labels)[-count])

    def pooled_within_construct_variance(self) -> float | None:
        """Estimate shared replicate-score variance for label-mode simulations."""
        mask = (self.replicate_counts >= 2) & np.isfinite(self.observed_vars)
        if not np.any(mask):
            return None
        weights = self.replicate_counts[mask] - 1
        if np.sum(weights) <= 0:
            return None
        variance = np.sum(weights * self.observed_vars[mask]) / np.sum(weights)
        return float(max(variance, 0.0))

    def pooled_log10_expression_variances(self) -> dict[str, float] | None:
        """Estimate each state's log10 expression noise from measured replicates."""
        if not self.expression_replicates:
            return None

        estimates: dict[str, float] = {}
        total_sum_squares = 0.0
        total_degrees_of_freedom = 0
        for column, replicate_matrix in self.expression_replicates.items():
            sum_squares = 0.0
            degrees_of_freedom = 0
            for sample_index in np.flatnonzero(self.replicate_counts >= 2):
                count = int(self.replicate_counts[sample_index])
                values = replicate_matrix[sample_index, :count]
                if np.any(~np.isfinite(values)) or np.any(values <= 0):
                    continue
                log_values = np.log10(values)
                sum_squares += float(np.sum((log_values - np.mean(log_values)) ** 2))
                degrees_of_freedom += count - 1
            if degrees_of_freedom > 0:
                estimates[column] = max(sum_squares / degrees_of_freedom, 0.0)
                total_sum_squares += sum_squares
                total_degrees_of_freedom += degrees_of_freedom

        if total_degrees_of_freedom <= 0:
            return None
        global_variance = max(
            total_sum_squares / total_degrees_of_freedom,
            0.0,
        )
        return {
            column: estimates.get(column, global_variance)
            for column in self.expression_replicates
        }

    def save_outputs(self, output_dir: Path) -> None:
        """Write replicate-level and training-observation records."""
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.measurement_rows).to_csv(
            output_dir / "simulated_measurements.csv",
            index=False,
        )
        pd.DataFrame(self.training_observation_rows).to_csv(
            output_dir / "training_observations.csv",
            index=False,
        )

    def _simulate_one_measurement(
        self, sample_index: int
    ) -> tuple[float, dict[str, Any], dict[str, float] | None]:
        if self.config.score_mode == "induced_over_basal":
            return self._simulate_induced_over_basal(sample_index)
        if self.config.score_mode in {"and_score", "or_score"}:
            return self._simulate_multi_input_score(sample_index)
        return self._simulate_label(sample_index)

    def _simulate_label(self, sample_index: int) -> tuple[float, dict[str, Any], None]:
        true_score = float(self.true_labels[sample_index])
        if self.config.noise_sigma_log10_expression == 0:
            observed_score = true_score
            log10_noise = 0.0
        elif true_score > 0:
            log10_noise = float(
                self.rng.normal(0.0, self.config.noise_sigma_log10_expression)
            )
            observed_score = true_score * 10**log10_noise
        else:
            log10_noise = float(
                self.rng.normal(0.0, self.config.noise_sigma_log10_expression)
            )
            observed_score = true_score + log10_noise
        return observed_score, {"score_log10_noise": log10_noise}, None

    def _simulate_induced_over_basal(
        self, sample_index: int
    ) -> tuple[float, dict[str, Any], dict[str, float]]:
        noisy_by_column, values = self._simulate_expression_values(sample_index)
        derived_score = self._score_from_expression(noisy_by_column)
        score = self._calibrated_expression_score(sample_index, derived_score)
        values["expression_derived_score"] = derived_score
        values["historical_score_calibration_factor"] = (
            self._historical_score_calibration_factor(sample_index)
        )
        return score, values, noisy_by_column

    def _simulate_multi_input_score(
        self, sample_index: int
    ) -> tuple[float, dict[str, Any], dict[str, float]]:
        noisy_by_column, values = self._simulate_expression_values(sample_index)
        derived_score = self._score_from_expression(noisy_by_column)
        score = self._calibrated_expression_score(sample_index, derived_score)
        values["expression_derived_score"] = derived_score
        values["historical_score_calibration_factor"] = (
            self._historical_score_calibration_factor(sample_index)
        )
        return score, values, noisy_by_column

    def _calibrated_expression_score(
        self,
        sample_index: int,
        derived_score: float,
    ) -> float:
        if self.config.noise_sigma_log10_expression == 0:
            return float(self.true_labels[sample_index])
        return derived_score * self._historical_score_calibration_factor(sample_index)

    def _historical_score_calibration_factor(self, sample_index: int) -> float:
        if self.metadata is None:
            raise ValueError(
                f"metadata is required for {self.config.score_mode} simulation."
            )
        row = self.metadata.iloc[sample_index]
        true_expression = {
            column: float(row[column])
            for column in self._expression_columns_to_simulate()
        }
        expression_score = self._score_from_expression(true_expression)
        historical_score = float(self.true_labels[sample_index])
        if expression_score == 0:
            if historical_score == 0:
                return 1.0
            raise ValueError(
                "Cannot anchor a nonzero historical score to a zero expression-derived "
                f"score for sample {self._sample_id(sample_index)}."
            )
        factor = historical_score / expression_score
        if not np.isfinite(factor):
            raise ValueError(
                "Historical score calibration must be finite for sample "
                f"{self._sample_id(sample_index)}."
            )
        return float(factor)

    def _score_from_expression(self, expression: dict[str, float]) -> float:
        numerator_column, denominator_column = self._score_ratio_columns(expression)
        return _safe_ratio(
            expression[numerator_column],
            expression[denominator_column],
        )

    def _score_ratio_columns(self, expression: dict[str, float]) -> tuple[str, str]:
        if self.config.score_mode == "induced_over_basal":
            basal_column, induced_column = self._two_state_columns()
            return induced_column, basal_column

        basal_column, input_a_column, input_b_column, dual_column = (
            self._multi_input_columns()
        )
        if self.config.score_mode == "and_score":
            denominator_candidates = (
                basal_column,
                input_a_column,
                input_b_column,
            )
            denominator_column = max(
                denominator_candidates,
                key=lambda column: expression[column],
            )
            return dual_column, denominator_column

        numerator_candidates = (input_a_column, input_b_column, dual_column)
        numerator_column = min(
            numerator_candidates,
            key=lambda column: expression[column],
        )
        return numerator_column, basal_column

    def _expression_score_y_var(
        self,
        index_array: np.ndarray,
        *,
        apply_floor: bool,
    ) -> np.ndarray | None:
        pooled_variances = self.pooled_log10_expression_variances()
        if pooled_variances is None:
            return None

        y_var = np.empty(len(index_array), dtype=float)
        for position, sample_index in enumerate(index_array):
            count = int(self.replicate_counts[sample_index])
            expression = {
                column: means[sample_index]
                for column, means in self.observed_expression_means.items()
            }
            numerator_column, denominator_column = self._score_ratio_columns(expression)
            relative_variance = (
                np.expm1(_LN_10_SQUARED * pooled_variances[numerator_column]) / count
                + np.expm1(_LN_10_SQUARED * pooled_variances[denominator_column])
                / count
            )
            y_var[position] = self.observed_means[sample_index] ** 2 * float(
                relative_variance
            )

        if apply_floor and self.config.train_yvar_floor > 0:
            y_var = np.maximum(y_var, self.config.train_yvar_floor)
        return y_var

    def _simulate_expression_values(
        self, sample_index: int
    ) -> tuple[dict[str, float], dict[str, Any]]:
        if self.metadata is None:
            raise ValueError(
                f"metadata is required for {self.config.score_mode} simulation."
            )

        row = self.metadata.iloc[sample_index]
        values: dict[str, Any] = {}
        noisy_by_column: dict[str, float] = {}

        for column in self._expression_columns_to_simulate():
            true_expression = float(row[column])
            if true_expression < 0:
                raise ValueError(f"Expression column '{column}' must be non-negative.")
            log10_noise = (
                0.0
                if self.config.noise_sigma_log10_expression == 0
                else float(
                    self.rng.normal(0.0, self.config.noise_sigma_log10_expression)
                )
            )
            noisy_expression = true_expression * 10**log10_noise
            noisy_by_column[column] = noisy_expression
            values[f"true_expression_{column}"] = true_expression
            values[f"observed_expression_{column}"] = noisy_expression
            values[f"log10_noise_{column}"] = log10_noise

        return noisy_by_column, values

    def _validate_metadata(self) -> None:
        if self.config.score_mode not in _EXPRESSION_SCORE_MODES:
            return
        if self.metadata is None:
            raise ValueError(
                f"metadata is required for {self.config.score_mode} simulation."
            )
        self._score_columns()
        required = set(self._expression_columns_to_simulate())
        missing = sorted(required - set(self.metadata.columns))
        if missing:
            raise KeyError(f"Missing expression column(s): {missing}")

    def _score_columns(self) -> tuple[str, ...]:
        if self.config.score_mode == "induced_over_basal":
            return self._two_state_columns()
        if self.config.score_mode in {"and_score", "or_score"}:
            return self._multi_input_columns()
        return ()

    def _two_state_columns(self) -> tuple[str, str]:
        columns = self.config.expression_columns
        if self.config.basal_column is None and len(columns) < 1:
            raise ValueError(
                "induced_over_basal requires a basal_column or at least one "
                "expression column."
            )
        if self.config.induced_column is None and len(columns) < 2:
            raise ValueError(
                "induced_over_basal requires an induced_column or at least two "
                "expression columns."
            )
        return (
            self.config.basal_column or columns[0],
            self.config.induced_column or columns[1],
        )

    def _multi_input_columns(self) -> tuple[str, str, str, str]:
        columns = self.config.expression_columns
        named_columns = (
            self.config.basal_column,
            self.config.single_input_a_column,
            self.config.single_input_b_column,
            self.config.dual_input_column,
        )
        names = ("basal", "single_input_a", "single_input_b", "dual_input")
        resolved: list[str] = []
        for index, (name, configured) in enumerate(
            zip(names, named_columns, strict=True)
        ):
            if configured is not None:
                resolved.append(configured)
            elif len(columns) > index:
                resolved.append(columns[index])
            else:
                raise ValueError(
                    f"{self.config.score_mode} requires {name}_column or at least "
                    "four expression columns ordered as basal, single input A, "
                    "single input B, and dual input."
                )
        return resolved[0], resolved[1], resolved[2], resolved[3]

    def _expression_columns_to_simulate(self) -> tuple[str, ...]:
        if self.config.score_mode not in _EXPRESSION_SCORE_MODES:
            return ()
        return tuple(
            dict.fromkeys((*self.config.expression_columns, *self._score_columns()))
        )

    def _sample_id(self, sample_index: int) -> Any:
        value = self.sample_ids[sample_index]
        if hasattr(value, "item"):
            return value.item()
        return value


def _has_config_value(config: Any, key: str) -> bool:
    if isinstance(config, dict):
        return key in config
    try:
        return hasattr(config, key)
    except Exception:
        return False


def _get_config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.divide(np.float64(numerator), np.float64(denominator)))


@dataclass(frozen=True)
class ConfirmedTopStoppingConfig:
    """Benchmark-only stopping after a true-top confirmation and minimum rounds."""

    enabled: bool = False
    min_total_rounds: int = 10
    confirmation_z: float = 0.0

    @classmethod
    def from_config(cls, value: Any | None) -> ConfirmedTopStoppingConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value

        allowed = {field.name for field in fields(cls)}
        if isinstance(value, Mapping):
            data = {key: value[key] for key in allowed if key in value}
        else:
            data = {key: getattr(value, key) for key in allowed if hasattr(value, key)}
        return cls(**data)

    def __post_init__(self) -> None:
        if self.min_total_rounds < 1:
            raise ValueError("min_total_rounds must be at least 1.")
        if self.confirmation_z < 0:
            raise ValueError("confirmation_z must be non-negative.")


class _SimulationMetricsCalculator(MetricsCalculator):
    def __init__(self, labels: np.ndarray, simulator: MeasurementSimulator) -> None:
        super().__init__(labels)
        self.simulator = simulator

    def compute_metrics_for_round(
        self,
        selected_indices: np.ndarray,
        train_indices: np.ndarray,
        train_predictions: np.ndarray | None,
        pool_indices: np.ndarray,
        pool_predictions: np.ndarray | None,
        top_p: float = 0.01,
    ) -> dict[str, float]:
        metrics = super().compute_metrics_for_round(
            selected_indices=selected_indices,
            train_indices=train_indices,
            train_predictions=train_predictions,
            pool_indices=pool_indices,
            pool_predictions=pool_predictions,
            top_p=top_p,
        )
        confirmed = self.simulator.confirmed_true_top_indices(
            selected_indices, top_p=top_p
        )
        n_top = metrics.pop("n_top")
        return {
            "n_top": n_top,
            "n_confirmed_top": int(np.intersect1d(selected_indices, confirmed).size),
            **metrics,
        }


class RetrospectiveExperiment(ActiveLearningExperiment):
    """Optional simulated observations and confirmed-top stopping."""

    def __init__(
        self,
        *args,
        measurement_simulation: dict[str, Any]
        | MeasurementSimulationConfig
        | None = None,
        confirmed_top_stopping: dict[str, Any]
        | ConfirmedTopStoppingConfig
        | None = None,
        **kwargs,
    ) -> None:
        self.measurement_simulation_config = MeasurementSimulationConfig.from_config(
            measurement_simulation
        )
        self.confirmed_top_stopping_config = ConfirmedTopStoppingConfig.from_config(
            confirmed_top_stopping
        )
        self.confirmed_top_stopping_first_confirmation_round: int | None = None
        if (
            self.confirmed_top_stopping_config.enabled
            and kwargs.get("stopping_rule", "none") != "none"
        ):
            raise ValueError("Choose either stopping_rule or confirmed_top_stopping.")
        if (
            self.confirmed_top_stopping_config.enabled
            and not self.measurement_simulation_config.enabled
        ):
            raise ValueError(
                "Confirmed-top early stopping requires measurement_simulation.enabled=true."
            )
        super().__init__(*args, **kwargs)
        self.measurement_simulator: MeasurementSimulator | None = None
        if self.measurement_simulation_config.enabled:
            self.measurement_simulator = MeasurementSimulator(
                sample_ids=self.dataset.sample_ids,
                true_labels=self.dataset.labels,
                metadata=self.dataset.metadata,
                label_key=self.label_key,
                config=self.measurement_simulation_config,
                random_seed=self.random_seed,
            )
            self.metrics_calculator = _SimulationMetricsCalculator(
                self.dataset.labels, self.measurement_simulator
            )
            self.round_tracker.summary_metric_rules["rounds_to_confirm_top"] = (
                "rounds_to_top",
                "n_confirmed_top",
            )

    def _observe_batch(self, indices: list[int], round_num: int) -> None:
        if self.measurement_simulator is not None:
            self.measurement_simulator.measure(indices, round_num=round_num)

    def get_training_targets(self, indices: list[int] | np.ndarray) -> np.ndarray:
        if self.measurement_simulator is not None:
            return self.measurement_simulator.training_targets(indices)
        return super().get_training_targets(indices)

    def get_training_y_var(self, indices: list[int] | np.ndarray) -> np.ndarray | None:
        if self.measurement_simulator is not None:
            return self.measurement_simulator.training_y_var(indices)
        return super().get_training_y_var(indices)

    def _before_training(self, round_num: int, y_var_train: np.ndarray | None) -> None:
        if self.measurement_simulator is not None:
            self.measurement_simulator.record_training_observations(
                indices=self.train_indices,
                training_round=round_num,
                train_y_var=y_var_train,
            )

    def save_results(self, output_path: Path) -> None:
        super().save_results(output_path)
        if self.measurement_simulator is not None:
            self.measurement_simulator.save_outputs(Path(output_path).parent)

    def _should_stop_after_batch(
        self,
        *,
        top_p: float,
        has_remaining_rounds: bool,
    ) -> bool:
        """Update confirmation state and decide whether to stop intentionally."""
        config = self.confirmed_top_stopping_config
        if not config.enabled:
            return False
        if self.measurement_simulator is None:
            raise RuntimeError(
                "Confirmed-top early stopping requires a measurement simulator."
            )

        measured_indices = np.flatnonzero(
            self.measurement_simulator.replicate_counts > 0
        )
        confirmed_indices = self.measurement_simulator.confirmed_true_top_indices(
            measured_indices,
            top_p=top_p,
            confirmation_z=config.confirmation_z,
        )
        total_rounds = len(self.round_tracker.rounds)
        if (
            confirmed_indices.size > 0
            and self.confirmed_top_stopping_first_confirmation_round is None
        ):
            self.confirmed_top_stopping_first_confirmation_round = total_rounds

        should_stop = (
            has_remaining_rounds
            and confirmed_indices.size > 0
            and self.confirmed_top_stopping_first_confirmation_round is not None
            and total_rounds >= config.min_total_rounds
        )
        if not should_stop:
            return False

        self.early_stop_info = {
            "reason": "confirmed_top",
            "round": total_rounds,
            "first_confirmation_round": (
                self.confirmed_top_stopping_first_confirmation_round
            ),
            "confirmation_z": config.confirmation_z,
            "min_total_rounds": config.min_total_rounds,
            "n_confirmed_top_at_stop": int(confirmed_indices.size),
        }
        logger.info(
            "Stopping intentionally after round %d: a true top was first "
            "confirmed at round %d with z=%.3g and min_total_rounds=%d.",
            total_rounds,
            self.confirmed_top_stopping_first_confirmation_round,
            config.confirmation_z,
            config.min_total_rounds,
        )
        return True
