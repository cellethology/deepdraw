import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepdraw import workflow
from deepdraw.workflow import (
    _training_indices_for_state,
    initialize_run,
    load_state,
    suggest_next_batch,
)


def _write_pool_and_embeddings(tmp_path: Path, n_samples: int = 12, dim: int = 4):
    pool = pd.DataFrame(
        {
            "variant_id": [f"variant_{idx}" for idx in range(n_samples)],
            "sequence": [f"ATGC{idx:02d}" for idx in range(n_samples)],
            "notes": [f"design {idx}" for idx in range(n_samples)],
        }
    )
    pool_path = tmp_path / "design_pool.csv"
    pool.to_csv(pool_path, index=False)

    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(n_samples, dim)).astype(np.float32)
    embeddings[:, 0] = np.linspace(0.0, 1.0, n_samples)
    embeddings_path = tmp_path / "embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        ids=np.arange(n_samples, dtype=np.int32),
    )
    return pool_path, embeddings_path


def _initialize_small_run(tmp_path: Path, starting_batch_size: int = 4):
    pool_path, embeddings_path = _write_pool_and_embeddings(tmp_path)
    return initialize_run(
        pool_csv=pool_path,
        embeddings_path=embeddings_path,
        output_dir=tmp_path / "deepdraw_run",
        sequence_column="sequence",
        id_column="variant_id",
        starting_batch_size=starting_batch_size,
        batch_size=3,
        seed=11,
        initial_selection_strategy_name="random",
        predictor_name="ridge_regressor",
        query_strategy_name="topk",
        feature_transforms_name="none",
        target_transforms_name="none",
    )


def test_initialize_run_writes_initial_batch_without_labels(tmp_path):
    state = _initialize_small_run(tmp_path)

    run_dir = Path(state.output_dir)
    first_batch = pd.read_csv(run_dir / "round_000_to_measure.csv")
    saved_state = json.loads((run_dir / "deepdraw_state.json").read_text())

    assert state.rounds[0]["stage"] == "initial"
    assert len(first_batch) == 4
    assert list(first_batch.columns) == ["variant_id", "sequence", "notes"]
    assert saved_state["rounds"][0]["size"] == 4
    assert (run_dir / "latest_recommendations.csv").exists()
    assert (run_dir / "selection_history.csv").exists()


def test_initialize_run_defaults_to_mes_query_strategy(tmp_path):
    pool_path, embeddings_path = _write_pool_and_embeddings(tmp_path)

    state = initialize_run(
        pool_csv=pool_path,
        embeddings_path=embeddings_path,
        output_dir=tmp_path / "default_query_run",
        sequence_column="sequence",
        id_column="variant_id",
        starting_batch_size=3,
        batch_size=2,
        initial_selection_strategy_name="random",
    )

    assert state.predictor == "gp"
    assert state.query_strategy == "mes"


def test_predictor_short_alias_resolves_legacy_config_file() -> None:
    path = workflow._resolve_named_config_path(kind="predictor", name="gp")

    assert path.name == "botorch_gp.yaml"
    assert workflow._display_component_name("predictor", "botorch_gp") == "gp"


def test_predictor_accepts_legacy_botorch_prefix(tmp_path):
    pool_path, embeddings_path = _write_pool_and_embeddings(tmp_path)

    state = initialize_run(
        pool_csv=pool_path,
        embeddings_path=embeddings_path,
        output_dir=tmp_path / "legacy_predictor_run",
        sequence_column="sequence",
        id_column="variant_id",
        starting_batch_size=3,
        batch_size=2,
        initial_selection_strategy_name="random",
        predictor_name="botorch_gp",
    )
    predictor = workflow._instantiate_component(
        kind="predictor",
        name=state.predictor,
        al_settings={"seed": 0, "starting_batch_size": 3, "batch_size": 2},
    )

    assert state.predictor == "gp"
    assert predictor.__class__.__name__ == "BoTorchGPRegressor"


def test_query_strategy_accepts_legacy_botorch_prefix(tmp_path):
    pool_path, embeddings_path = _write_pool_and_embeddings(tmp_path)

    state = initialize_run(
        pool_csv=pool_path,
        embeddings_path=embeddings_path,
        output_dir=tmp_path / "legacy_query_run",
        sequence_column="sequence",
        id_column="variant_id",
        starting_batch_size=3,
        batch_size=2,
        initial_selection_strategy_name="random",
        query_strategy_name="botorch_mes",
    )
    strategy = workflow._instantiate_component(
        kind="query_strategy",
        name=state.query_strategy,
        al_settings={"seed": 0, "starting_batch_size": 3, "batch_size": 2},
    )

    assert state.query_strategy == "mes"
    assert strategy.acquisition == "mes"
    assert strategy.name == "MES"


def test_query_strategy_short_alias_resolves_legacy_config_file() -> None:
    path = workflow._resolve_named_config_path(kind="query_strategy", name="qlog_nei")

    assert path.name == "botorch_qlog_nei.yaml"
    assert workflow._display_query_strategy_name("botorch_qlog_nei") == "qlog_nei"


def test_suggest_next_batch_uses_measurements_and_excludes_measured(tmp_path):
    state = _initialize_small_run(tmp_path)
    run_dir = Path(state.output_dir)

    initial_batch = pd.read_csv(run_dir / "round_000_to_measure.csv")
    initial_batch["Expression"] = np.linspace(1.0, 4.0, len(initial_batch))
    measurements_path = tmp_path / "measurements.csv"
    initial_batch.to_csv(measurements_path, index=False)

    updated_state = suggest_next_batch(
        run_dir=run_dir,
        measurements_csv=measurements_path,
        label_column="Expression",
    )

    next_batch = pd.read_csv(run_dir / "round_001_to_measure.csv")
    history = pd.read_csv(run_dir / "selection_history.csv")
    reloaded = load_state(run_dir)

    assert len(updated_state.rounds) == 2
    assert updated_state.rounds[1]["stage"] == "acquisition"
    assert len(next_batch) == 3
    assert set(next_batch["variant_id"]).isdisjoint(set(initial_batch["variant_id"]))
    assert len(history) == len(initial_batch) + len(next_batch)
    assert list(history.columns) == [
        "deepdraw_round",
        "variant_id",
        "sequence",
        "notes",
    ]
    assert reloaded.label_column == "Expression"


def test_suggest_requires_labels_for_previous_recommendations(tmp_path):
    state = _initialize_small_run(tmp_path, starting_batch_size=3)
    run_dir = Path(state.output_dir)

    incomplete = pd.read_csv(run_dir / "round_000_to_measure.csv").iloc[:2].copy()
    incomplete["Expression"] = [1.0, 2.0]
    measurements_path = tmp_path / "incomplete_measurements.csv"
    incomplete.to_csv(measurements_path, index=False)

    with pytest.raises(ValueError, match="missing labels"):
        suggest_next_batch(
            run_dir=run_dir,
            measurements_csv=measurements_path,
            label_column="Expression",
        )


def test_training_indices_preserve_deepdraw_round_order(tmp_path):
    state = _initialize_small_run(tmp_path)
    selected = [int(idx) for idx in state.rounds[0]["selected_pool_indices"]]
    extras = [idx for idx in range(12) if idx not in selected][:2]
    measured_labels = {idx: float(idx) for idx in [*reversed(selected), *extras]}

    assert _training_indices_for_state(state, measured_labels) == [
        *selected,
        *sorted(extras),
    ]


def test_random_query_strategy_skips_predictor_training(tmp_path, monkeypatch):
    pool_path, embeddings_path = _write_pool_and_embeddings(tmp_path)
    state = initialize_run(
        pool_csv=pool_path,
        embeddings_path=embeddings_path,
        output_dir=tmp_path / "random_query_run",
        sequence_column="sequence",
        id_column="variant_id",
        starting_batch_size=4,
        batch_size=3,
        seed=11,
        initial_selection_strategy_name="random",
        predictor_name="ridge_regressor",
        query_strategy_name="random",
        feature_transforms_name="none",
        target_transforms_name="none",
    )
    run_dir = Path(state.output_dir)
    measurements = pd.read_csv(run_dir / "round_000_to_measure.csv")
    measurements["Expression"] = np.linspace(1.0, 4.0, len(measurements))
    measurements_path = tmp_path / "measurements.csv"
    measurements.to_csv(measurements_path, index=False)

    def fail_train(self, X_train, y_train):
        raise AssertionError("random acquisition should not train a predictor")

    monkeypatch.setattr(workflow.PredictorTrainer, "train", fail_train)

    updated_state = suggest_next_batch(
        run_dir=run_dir,
        measurements_csv=measurements_path,
        label_column="Expression",
    )

    assert len(updated_state.rounds) == 2


def test_dummy_example_files_drive_workflow(tmp_path):
    example_dir = Path(__file__).resolve().parents[1] / "examples" / "deepdraw_dummy"
    run_dir = tmp_path / "dummy_run"

    state = initialize_run(
        pool_csv=example_dir / "design_pool.csv",
        embeddings_path=example_dir / "embeddings.npz",
        output_dir=run_dir,
        sequence_column="sequence",
        id_column="variant_id",
    )
    first_batch = pd.read_csv(run_dir / "round_000_to_measure.csv")
    measurements = pd.read_csv(example_dir / "measurements.csv")

    updated_state = suggest_next_batch(
        run_dir=run_dir,
        measurements_csv=example_dir / "measurements.csv",
        label_column="Expression",
    )
    next_batch = pd.read_csv(run_dir / "round_001_to_measure.csv")

    assert state.initial_selection_strategy == "probcover_euclidean"
    assert state.predictor == "gp"
    assert state.query_strategy == "mes"
    assert len(first_batch) == 12
    assert list(first_batch["variant_id"]) == list(measurements["variant_id"])
    assert len(updated_state.rounds) == 2
    assert len(next_batch) == 12
