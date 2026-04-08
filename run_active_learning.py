"""
Active Learning Loop for genetic circuit design.

This script implements an active learning approach to design circuit with specific function.
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import ListConfig, OmegaConf

from core.experiment import ActiveLearningExperiment

logger = logging.getLogger(__name__)


def run_one_experiment(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Run a single experiment with given configuration.

    Args:
        cfg: Dictionary containing all experiment parameters

    Returns:
        Dictionary containing the results summary
    """

    # Extract active learning settings
    al_settings = cfg.al_settings
    output_dir = al_settings.get("output_dir", None)
    if output_dir is None:
        raise ValueError("al_settings.output_dir must be provided in the config.")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    dataset_name = getattr(cfg, "dataset_name", "")
    embedding_model_name = getattr(cfg, "embedding_model", "")
    seed = al_settings.get("seed", 0)
    logger.info(
        "RUN_CONTEXT dataset_name=%s seed=%s output_dir=%s",
        dataset_name,
        seed,
        output_dir_path,
    )

    try:
        # Instantiate components
        query_strategy = instantiate(cfg.query_strategy)
        predictor = instantiate(cfg.predictor)
        initial_selection_strategy = instantiate(cfg.initial_selection_strategy)
        feature_transforms = make_steps(cfg.feature_transforms.steps)
        target_transforms = make_steps(cfg.target_transforms.steps)

        # Extract active learning settings
        embeddings_path = cfg.embedding_path
        metadata_path = cfg.metadata_path
        subset_ids_path = getattr(cfg, "subset_ids_path", None)
        batch_size = al_settings.get("batch_size")
        starting_batch_size = al_settings.get("starting_batch_size", batch_size)
        max_rounds = al_settings.get("max_rounds")
        label_key = al_settings.get("label_key", None)

        # Check embedding path matches embedding model
        if not embeddings_path.endswith(f"{embedding_model_name}.npz"):
            raise ValueError(
                f"Embedding path {embeddings_path} does not match embedding model {embedding_model_name}"
            )

        # Create experiment
        experiment = ActiveLearningExperiment(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            query_strategy=query_strategy,
            predictor=predictor,
            starting_batch_size=starting_batch_size,
            batch_size=batch_size,
            random_seed=seed,
            feature_transforms=feature_transforms,
            target_transforms=target_transforms,
            label_key=label_key,
            initial_selection_strategy=initial_selection_strategy,
            subset_ids_path=subset_ids_path,
        )

        # Run experiment
        experiment.run_experiment(max_rounds=max_rounds)

        # Save individual results
        experiment.save_results(output_path=output_dir_path / "results.csv")

        # Compute summary metrics
        summary_metrics = experiment.round_tracker.compute_summary_metrics()
        summary_metrics_history = (
            experiment.round_tracker.compute_summary_metrics_history()
        )
    except Exception:
        error_path = output_dir_path / "error.txt"
        error_details = [
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"dataset_name: {dataset_name}",
            f"seed: {seed}",
            "",
            traceback.format_exc(),
        ]
        error_path.write_text("\n".join(error_details))
        raise

    # Summarize results
    feature_transforms_names = (
        [item[0] for item in feature_transforms] if feature_transforms else []
    )
    target_transforms_names = (
        [item[0] for item in target_transforms] if target_transforms else []
    )
    is_random_strategy = query_strategy.name.upper() == "RANDOM"
    reported_embedding_model = "None" if is_random_strategy else embedding_model_name

    overrides = _collect_hydra_overrides(cfg)
    override_map = _build_override_map(overrides)
    strategy_label = _format_component_label(
        query_strategy.name, "query_strategy", override_map
    )
    predictor_label = "None"
    if not is_random_strategy:
        predictor_label = _format_component_label(
            predictor.__class__.__name__, "predictor", override_map
        )

    summary = {
        "dataset_name": dataset_name,
        "embedding_model": reported_embedding_model,
        "query_strategy": strategy_label,
        "predictor": predictor_label,
        "initial_selection": initial_selection_strategy.name,
        "feature_transforms": feature_transforms_names,
        "target_transforms": target_transforms_names,
        "seed": seed,
        "auc_true": summary_metrics["auc_true"],
        "avg_top": summary_metrics["avg_top"],
        "rounds_to_top": summary_metrics["rounds_to_top"],
        "overall_true": summary_metrics["overall_true"],
        "max_train_spearman": summary_metrics["max_train_spearman"],
        "max_extreme_value_auc": summary_metrics["max_extreme_value_auc"],
        "summary_by_round": summary_metrics_history,
        "completed_rounds": len(experiment.round_tracker.rounds),
        "stopped_early": experiment.failure_info is not None,
        "failure_info": experiment.failure_info,
        "hydra_overrides": overrides,
    }

    # Persist summary for downstream aggregation
    summary_path = output_dir_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def make_steps(steps_cfg: ListConfig) -> list[tuple[str, Any]]:
    """
    Make a list of (name, transformer) steps from a list of step configurations.

    Args:
        steps_cfg: List of step configurations

    Returns:
        List of (name, transformer) steps
    """
    steps: list[tuple[str, Any]] = []
    for step_cfg in steps_cfg:
        step_dict = OmegaConf.to_container(step_cfg, resolve=True)
        name = step_dict.pop("id")
        transformer = instantiate(step_dict)
        steps.append((name, transformer))
    return steps


def _collect_hydra_overrides(cfg: dict[str, Any]) -> list[str]:
    """Return the Hydra override strings recorded for this task."""
    stored_overrides = None
    try:
        stored_overrides = OmegaConf.select(cfg, "hydra_overrides", default=None)
    except Exception:  # pragma: no cover - defensive
        stored_overrides = None
    if stored_overrides is not None:
        return _normalize_override_values(stored_overrides)
    if HydraConfig.initialized():
        try:
            task_overrides = HydraConfig.get().overrides.task
        except Exception:  # pragma: no cover - defensive
            task_overrides = None
        if task_overrides:
            return [str(item) for item in task_overrides]
    return []


def _build_override_map(overrides: list[str]) -> dict[str, str]:
    """Create a key/value lookup from override strings."""
    items: dict[str, str] = {}
    for entry in overrides:
        text = str(entry).strip()
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            continue
        items[key] = value.strip()
    return items


def _format_component_label(
    base_label: str, prefix: str, override_map: dict[str, str]
) -> str:
    """Append override parameters to a label when that component was swept."""
    prefix_with_dot = f"{prefix}."
    details: list[str] = []
    for key in sorted(override_map):
        if key.startswith(prefix_with_dot):
            param = key[len(prefix_with_dot) :]
            details.append(f"{param}={override_map[key]}")
    if not details:
        return base_label
    detail_str = ", ".join(details)
    return f"{base_label}[{detail_str}]"


def _normalize_override_values(value: Any) -> list[str]:
    """Normalize Hydra override containers to a list of strings."""
    if value is None:
        return []
    if isinstance(value, ListConfig):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
