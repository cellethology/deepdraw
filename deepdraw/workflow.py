"""Production Deepdraw workflow for real experimental active learning.

The retrospective runner evaluates methods when every sequence already has a
known label. This module handles the real experimental loop instead: start from
an unlabeled design pool, choose a first batch from embeddings only, then keep
suggesting new batches as measured labels arrive.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from core.data_loader import Dataset
from core.predictor_trainer import PredictorTrainer

logger = logging.getLogger(__name__)

STATE_FILENAME = "deepdraw_state.json"
LATEST_RECOMMENDATIONS_FILENAME = "latest_recommendations.csv"
SELECTION_HISTORY_FILENAME = "selection_history.csv"

DEEPDRAW_ID_COLUMN = "deepdraw_id"
DEEPDRAW_POOL_INDEX_COLUMN = "deepdraw_pool_index"
DEEPDRAW_ROUND_COLUMN = "deepdraw_round"
DEEPDRAW_STAGE_COLUMN = "deepdraw_stage"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _REPO_ROOT / "job_sub" / "conf"
_SEQUENCE_COLUMN_CANDIDATES = (
    "sequence",
    "Sequence",
    "seq",
    "Seq",
    "dna_sequence",
    "DNA_sequence",
)


@dataclass
class DeepdrawState:
    """Serializable state for a real-world Deepdraw run."""

    pool_csv: str
    embeddings_path: str
    sequence_column: str
    id_column: str | None
    output_dir: str
    batch_size: int
    starting_batch_size: int
    seed: int
    predictor: str
    query_strategy: str
    initial_selection_strategy: str
    feature_transforms: str
    target_transforms: str
    label_column: str | None = None
    rounds: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def state_path(self) -> Path:
        return self.output_path / STATE_FILENAME


def initialize_run(
    *,
    pool_csv: str | Path,
    embeddings_path: str | Path,
    output_dir: str | Path = "deepdraw_run",
    sequence_column: str | None = None,
    id_column: str | None = None,
    starting_batch_size: int = 12,
    batch_size: int = 12,
    seed: int = 0,
    predictor_name: str = "botorch_gp",
    query_strategy_name: str = "botorch_mes",
    initial_selection_strategy_name: str = "probcover_euclidean",
    feature_transforms_name: str = "standardize",
    target_transforms_name: str = "log_standardize",
    force: bool = False,
) -> DeepdrawState:
    """Create a Deepdraw run and write the initial batch to measure."""

    output_path = Path(output_dir).expanduser().resolve()
    state_path = output_path / STATE_FILENAME
    if state_path.exists() and not force:
        raise FileExistsError(
            f"{state_path} already exists. Use force=True to overwrite this run."
        )

    pool_df, pool_ids, resolved_sequence_column = _load_pool(
        pool_csv=pool_csv,
        sequence_column=sequence_column,
        id_column=id_column,
    )
    embeddings = _load_aligned_embeddings(
        embeddings_path=embeddings_path,
        pool_ids=pool_ids,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    al_settings = _build_al_settings(
        seed=seed,
        starting_batch_size=starting_batch_size,
        batch_size=batch_size,
    )
    initial_selection = _instantiate_component(
        kind="initial_selection_strategy",
        name=initial_selection_strategy_name,
        al_settings=al_settings,
    )

    dataset = Dataset(
        sample_ids=pool_ids.tolist(),
        labels=np.full(len(pool_ids), np.nan),
        embeddings=embeddings,
    )
    selected_indices = initial_selection.select(dataset)

    state = DeepdrawState(
        pool_csv=str(Path(pool_csv).expanduser().resolve()),
        embeddings_path=str(Path(embeddings_path).expanduser().resolve()),
        sequence_column=resolved_sequence_column,
        id_column=id_column,
        output_dir=str(output_path),
        batch_size=batch_size,
        starting_batch_size=starting_batch_size,
        seed=seed,
        predictor=predictor_name,
        query_strategy=query_strategy_name,
        initial_selection_strategy=initial_selection_strategy_name,
        feature_transforms=feature_transforms_name,
        target_transforms=target_transforms_name,
    )
    _append_round(
        state=state,
        round_num=0,
        stage="initial",
        selected_indices=selected_indices,
        pool_ids=pool_ids,
    )
    save_state(state)
    _write_recommendation_outputs(
        state=state,
        pool_df=pool_df,
        pool_ids=pool_ids,
        selected_indices=selected_indices,
        round_num=0,
        stage="initial",
    )
    _write_selection_history(state=state, pool_df=pool_df, pool_ids=pool_ids)
    return state


def suggest_next_batch(
    *,
    run_dir: str | Path,
    measurements_csv: str | Path,
    label_column: str | None = None,
    measurement_id_column: str | None = None,
) -> DeepdrawState:
    """Train on measured designs and write the next batch to measure."""

    state = load_state(run_dir)
    if label_column is None:
        label_column = state.label_column
    if label_column is None:
        raise ValueError(
            "label_column is required the first time you call suggest_next_batch."
        )

    pool_df, pool_ids, _ = _load_pool(
        pool_csv=state.pool_csv,
        sequence_column=state.sequence_column,
        id_column=state.id_column,
    )
    embeddings = _load_aligned_embeddings(
        embeddings_path=state.embeddings_path,
        pool_ids=pool_ids,
    )
    measured_labels = _load_measurements(
        measurements_csv=measurements_csv,
        label_column=label_column,
        measurement_id_column=measurement_id_column,
        state=state,
        pool_df=pool_df,
        pool_ids=pool_ids,
    )
    if not measured_labels:
        raise ValueError(
            f"No non-empty labels were found in {measurements_csv} column "
            f"'{label_column}'."
        )

    _require_previous_selections_measured(
        state=state,
        measured_indices=set(measured_labels),
        pool_ids=pool_ids,
    )

    labels = np.full(len(pool_ids), np.nan)
    train_indices = sorted(measured_labels)
    labels[train_indices] = [measured_labels[idx] for idx in train_indices]

    dataset = Dataset(
        sample_ids=pool_ids.tolist(),
        labels=labels,
        embeddings=embeddings,
    )
    feature_transforms = _make_transform_steps(
        name=state.feature_transforms,
        al_settings=_build_al_settings_for_state(state),
    )
    if feature_transforms:
        feature_pipeline = Pipeline(feature_transforms)
        dataset.embeddings = feature_pipeline.fit_transform(dataset.embeddings)

    al_settings = _build_al_settings_for_state(state)
    predictor = _instantiate_component(
        kind="predictor",
        name=state.predictor,
        al_settings=al_settings,
    )
    query_strategy = _instantiate_component(
        kind="query_strategy",
        name=state.query_strategy,
        al_settings=al_settings,
    )
    target_transforms = _make_transform_steps(
        name=state.target_transforms,
        al_settings=al_settings,
    )
    trainer = PredictorTrainer(
        predictor=predictor,
        feature_transform=None,
        target_transform=target_transforms,
    )
    trainer.train(
        X_train=dataset.embeddings[train_indices, :],
        y_train=labels[train_indices],
    )

    experiment_view = _ProductionExperimentView(
        dataset=dataset,
        trainer=trainer,
        query_strategy=query_strategy,
        train_indices=train_indices,
        batch_size=state.batch_size,
        random_seed=state.seed,
    )
    selected_indices = query_strategy.select(experiment_view)
    round_num = _next_round_number(state)

    state.label_column = label_column
    _append_round(
        state=state,
        round_num=round_num,
        stage="acquisition",
        selected_indices=selected_indices,
        pool_ids=pool_ids,
    )
    save_state(state)
    _write_recommendation_outputs(
        state=state,
        pool_df=pool_df,
        pool_ids=pool_ids,
        selected_indices=selected_indices,
        round_num=round_num,
        stage="acquisition",
    )
    _write_selection_history(state=state, pool_df=pool_df, pool_ids=pool_ids)
    return state


def load_state(run_dir: str | Path) -> DeepdrawState:
    """Load Deepdraw run state from a run directory."""

    state_path = Path(run_dir).expanduser().resolve() / STATE_FILENAME
    data = json.loads(state_path.read_text())
    return DeepdrawState(**data)


def save_state(state: DeepdrawState) -> None:
    """Persist Deepdraw run state."""

    state.output_path.mkdir(parents=True, exist_ok=True)
    state.state_path.write_text(json.dumps(asdict(state), indent=2))


class _ProductionExperimentView:
    """Small adapter that lets existing query strategies operate on measured data."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        trainer: PredictorTrainer,
        query_strategy: Any,
        train_indices: list[int],
        batch_size: int,
        random_seed: int,
    ) -> None:
        self.dataset = dataset
        self.trainer = trainer
        self.query_strategy = query_strategy
        self.train_indices = train_indices
        self.batch_size = batch_size
        self.random_seed = random_seed
        self._measured = set(train_indices)

    @property
    def unlabeled_indices(self) -> list[int]:
        return [
            idx
            for idx in range(len(self.dataset.sample_ids))
            if idx not in self._measured
        ]


def _load_pool(
    *,
    pool_csv: str | Path,
    sequence_column: str | None,
    id_column: str | None,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    pool_path = Path(pool_csv).expanduser().resolve()
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool CSV does not exist: {pool_path}")

    pool_df = pd.read_csv(pool_path)
    resolved_sequence_column = _resolve_sequence_column(pool_df, sequence_column)
    if id_column is not None and id_column not in pool_df.columns:
        raise ValueError(f"id_column '{id_column}' was not found in {pool_path}.")

    if id_column is None:
        pool_ids = np.asarray([str(idx) for idx in range(len(pool_df))], dtype=object)
    else:
        pool_ids = pool_df[id_column].map(_stringify_id).to_numpy(dtype=object)
    _ensure_unique_ids(pool_ids, "pool")
    return pool_df, pool_ids, resolved_sequence_column


def _resolve_sequence_column(
    pool_df: pd.DataFrame,
    requested: str | None,
) -> str:
    if requested:
        if requested not in pool_df.columns:
            raise ValueError(f"sequence_column '{requested}' was not found.")
        return requested
    for candidate in _SEQUENCE_COLUMN_CANDIDATES:
        if candidate in pool_df.columns:
            return candidate
    raise ValueError(
        "Could not infer the sequence column. Pass --sequence-column explicitly."
    )


def _load_aligned_embeddings(
    *,
    embeddings_path: str | Path,
    pool_ids: np.ndarray,
) -> np.ndarray:
    path = Path(embeddings_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Embeddings NPZ does not exist: {path}")

    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data:
        raise ValueError(
            f"'embeddings' array not found in {path}. Available keys: {list(data.keys())}"
        )
    embeddings = np.asarray(data["embeddings"])
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D; got shape {embeddings.shape}.")

    embedding_ids = _extract_embedding_ids(data)
    if embedding_ids is None:
        if embeddings.shape[0] != len(pool_ids):
            raise ValueError(
                "Embeddings have no ids/sample_ids array and row count does not "
                f"match pool size ({embeddings.shape[0]} vs {len(pool_ids)})."
            )
        return embeddings

    if len(embedding_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Embedding ids length ({len(embedding_ids)}) does not match "
            f"embedding rows ({embeddings.shape[0]})."
        )

    row_indices = _as_complete_row_indices(embedding_ids, len(pool_ids))
    if row_indices is not None:
        aligned = np.empty((len(pool_ids), embeddings.shape[1]), dtype=embeddings.dtype)
        aligned[row_indices] = embeddings
        return aligned

    embedding_id_strings = np.asarray(
        [_stringify_id(value) for value in embedding_ids], dtype=object
    )
    _ensure_unique_ids(embedding_id_strings, "embedding")
    id_to_embedding_row = {
        str(sample_id): idx for idx, sample_id in enumerate(embedding_id_strings)
    }
    missing = [
        sample_id for sample_id in pool_ids if sample_id not in id_to_embedding_row
    ]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(
            f"Embeddings are missing {len(missing)} pool ids. First missing ids: {preview}"
        )
    order = [id_to_embedding_row[str(sample_id)] for sample_id in pool_ids]
    return embeddings[order]


def _extract_embedding_ids(data: Any) -> np.ndarray | None:
    if "ids" in data:
        return np.asarray(data["ids"])
    if "sample_ids" in data:
        return np.asarray(data["sample_ids"])
    return None


def _as_complete_row_indices(ids: np.ndarray, pool_size: int) -> np.ndarray | None:
    try:
        numeric = np.asarray(ids, dtype=int)
    except (TypeError, ValueError):
        return None
    if numeric.shape[0] != pool_size:
        return None
    if sorted(numeric.tolist()) != list(range(pool_size)):
        return None
    return numeric


def _load_measurements(
    *,
    measurements_csv: str | Path,
    label_column: str,
    measurement_id_column: str | None,
    state: DeepdrawState,
    pool_df: pd.DataFrame,
    pool_ids: np.ndarray,
) -> dict[int, float]:
    path = Path(measurements_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Measurements CSV does not exist: {path}")
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise ValueError(f"label_column '{label_column}' was not found in {path}.")

    id_column = _resolve_measurement_id_column(
        measurements_df=df,
        measurement_id_column=measurement_id_column,
        state=state,
    )
    id_to_index = {str(sample_id): idx for idx, sample_id in enumerate(pool_ids)}
    sequence_to_index = _unique_sequence_mapping(pool_df, state.sequence_column)

    measured: dict[int, float] = {}
    for row_number, row in df.iterrows():
        raw_label = row[label_column]
        if pd.isna(raw_label):
            continue
        label = float(raw_label)
        pool_index = _measurement_row_to_pool_index(
            row=row,
            row_number=int(row_number) + 2,
            id_column=id_column,
            id_to_index=id_to_index,
            sequence_to_index=sequence_to_index,
            pool_size=len(pool_ids),
        )
        existing = measured.get(pool_index)
        if existing is not None and not np.isclose(existing, label):
            raise ValueError(
                f"Conflicting labels for pool index {pool_index}: {existing} vs {label}."
            )
        measured[pool_index] = label
    return measured


def _resolve_measurement_id_column(
    *,
    measurements_df: pd.DataFrame,
    measurement_id_column: str | None,
    state: DeepdrawState,
) -> str:
    if measurement_id_column:
        if measurement_id_column not in measurements_df.columns:
            raise ValueError(
                f"measurement_id_column '{measurement_id_column}' was not found."
            )
        return measurement_id_column

    candidates = [
        state.id_column,
        DEEPDRAW_ID_COLUMN,
        DEEPDRAW_POOL_INDEX_COLUMN,
        state.sequence_column,
    ]
    for candidate in candidates:
        if candidate and candidate in measurements_df.columns:
            return candidate
    raise ValueError(
        "Could not find an identifier column in the measurements CSV. Include "
        f"'{DEEPDRAW_ID_COLUMN}', '{DEEPDRAW_POOL_INDEX_COLUMN}', the original id "
        "column, or pass --measurement-id-column."
    )


def _measurement_row_to_pool_index(
    *,
    row: pd.Series,
    row_number: int,
    id_column: str,
    id_to_index: dict[str, int],
    sequence_to_index: dict[str, int],
    pool_size: int,
) -> int:
    if id_column == DEEPDRAW_POOL_INDEX_COLUMN:
        pool_index = int(row[id_column])
        if pool_index < 0 or pool_index >= pool_size:
            raise ValueError(
                f"Measurement row {row_number} has out-of-range pool index {pool_index}."
            )
        return pool_index

    raw_id = _stringify_id(row[id_column])
    if raw_id in id_to_index:
        return id_to_index[raw_id]
    if raw_id in sequence_to_index:
        return sequence_to_index[raw_id]
    raise ValueError(
        f"Measurement row {row_number} references unknown design id/sequence '{raw_id}'."
    )


def _unique_sequence_mapping(
    pool_df: pd.DataFrame,
    sequence_column: str,
) -> dict[str, int]:
    values = pool_df[sequence_column].map(_stringify_id)
    if values.duplicated().any():
        return {}
    return {value: idx for idx, value in enumerate(values)}


def _require_previous_selections_measured(
    *,
    state: DeepdrawState,
    measured_indices: set[int],
    pool_ids: np.ndarray,
) -> None:
    required = _previously_selected_indices(state)
    missing = sorted(required - measured_indices)
    if not missing:
        return
    missing_ids = [str(pool_ids[idx]) for idx in missing[:10]]
    raise ValueError(
        "Measurements are missing labels for previously recommended designs. "
        f"First missing ids: {', '.join(missing_ids)}"
    )


def _previously_selected_indices(state: DeepdrawState) -> set[int]:
    selected: set[int] = set()
    for round_record in state.rounds:
        selected.update(int(idx) for idx in round_record["selected_pool_indices"])
    return selected


def _append_round(
    *,
    state: DeepdrawState,
    round_num: int,
    stage: str,
    selected_indices: list[int],
    pool_ids: np.ndarray,
) -> None:
    state.rounds.append(
        {
            "round": int(round_num),
            "stage": stage,
            "size": len(selected_indices),
            "selected_pool_indices": [int(idx) for idx in selected_indices],
            "selected_ids": [str(pool_ids[idx]) for idx in selected_indices],
        }
    )


def _next_round_number(state: DeepdrawState) -> int:
    if not state.rounds:
        return 0
    return max(int(round_record["round"]) for round_record in state.rounds) + 1


def _write_recommendation_outputs(
    *,
    state: DeepdrawState,
    pool_df: pd.DataFrame,
    pool_ids: np.ndarray,
    selected_indices: list[int],
    round_num: int,
    stage: str,
) -> Path:
    frame = _build_recommendation_frame(
        pool_df=pool_df,
        pool_ids=pool_ids,
        selected_indices=selected_indices,
        round_num=round_num,
        stage=stage,
        include_internal_columns=False,
    )
    output_path = state.output_path
    round_path = output_path / f"round_{round_num:03d}_to_measure.csv"
    latest_path = output_path / LATEST_RECOMMENDATIONS_FILENAME
    frame.to_csv(round_path, index=False)
    frame.to_csv(latest_path, index=False)
    logger.info("Wrote Deepdraw recommendations to %s", round_path)
    return round_path


def _write_selection_history(
    *,
    state: DeepdrawState,
    pool_df: pd.DataFrame,
    pool_ids: np.ndarray,
) -> None:
    frames = []
    for round_record in state.rounds:
        frames.append(
            _build_recommendation_frame(
                pool_df=pool_df,
                pool_ids=pool_ids,
                selected_indices=[
                    int(idx) for idx in round_record["selected_pool_indices"]
                ],
                round_num=int(round_record["round"]),
                stage=str(round_record["stage"]),
                include_internal_columns=True,
            )
        )
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    history.to_csv(state.output_path / SELECTION_HISTORY_FILENAME, index=False)


def _build_recommendation_frame(
    *,
    pool_df: pd.DataFrame,
    pool_ids: np.ndarray,
    selected_indices: list[int],
    round_num: int,
    stage: str,
    include_internal_columns: bool,
) -> pd.DataFrame:
    selected_df = pool_df.iloc[selected_indices].copy()
    selected_df = selected_df.drop(
        columns=[
            column
            for column in (
                DEEPDRAW_ROUND_COLUMN,
                DEEPDRAW_STAGE_COLUMN,
                DEEPDRAW_POOL_INDEX_COLUMN,
                DEEPDRAW_ID_COLUMN,
            )
            if column in selected_df.columns
        ],
        errors="ignore",
    )
    if not include_internal_columns:
        return selected_df

    selected_df.insert(
        0, DEEPDRAW_ID_COLUMN, [str(pool_ids[idx]) for idx in selected_indices]
    )
    selected_df.insert(
        0, DEEPDRAW_POOL_INDEX_COLUMN, [int(idx) for idx in selected_indices]
    )
    selected_df.insert(0, DEEPDRAW_STAGE_COLUMN, stage)
    selected_df.insert(0, DEEPDRAW_ROUND_COLUMN, round_num)
    return selected_df


def _instantiate_component(
    *,
    kind: str,
    name: str,
    al_settings: dict[str, Any],
) -> Any:
    cfg = _load_named_config(kind=kind, name=name, al_settings=al_settings)
    return instantiate(cfg)


def _make_transform_steps(
    *,
    name: str,
    al_settings: dict[str, Any],
) -> list[tuple[str, Any]]:
    cfg = _load_named_config(kind="transforms", name=name, al_settings=al_settings)
    steps: list[tuple[str, Any]] = []
    for step_cfg in cfg.steps:
        step_dict = OmegaConf.to_container(step_cfg, resolve=True)
        step_name = step_dict.pop("id")
        transformer = instantiate(step_dict)
        steps.append((step_name, transformer))
    return steps


def _load_named_config(
    *,
    kind: str,
    name: str,
    al_settings: dict[str, Any],
) -> Any:
    path = _CONFIG_ROOT / kind / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown {kind} config '{name}' ({path}).")
    component_cfg = OmegaConf.load(path)
    root = OmegaConf.create(
        {
            "al_settings": al_settings,
            "component": component_cfg,
        }
    )
    OmegaConf.resolve(root)
    return root.component


def _build_al_settings(
    *,
    seed: int,
    starting_batch_size: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "starting_batch_size": int(starting_batch_size),
        "batch_size": int(batch_size),
    }


def _build_al_settings_for_state(state: DeepdrawState) -> dict[str, Any]:
    return _build_al_settings(
        seed=state.seed,
        starting_batch_size=state.starting_batch_size,
        batch_size=state.batch_size,
    )


def _ensure_unique_ids(ids: np.ndarray, source: str) -> None:
    series = pd.Series(ids)
    duplicated = series[series.duplicated()].unique()
    if len(duplicated) > 0:
        preview = ", ".join(map(str, duplicated[:10]))
        raise ValueError(f"{source} ids must be unique. Duplicate ids: {preview}")


def _stringify_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.integer):
        return str(int(value))
    if isinstance(value, np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)
