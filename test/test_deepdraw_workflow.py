import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from core.botorch_models import BoTorchGPRegressor
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
        predictor_name="gradboost_regressor",
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
        predictor_name="gradboost_regressor",
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

    assert state.initial_selection_strategy == "probcover"
    assert state.predictor == "gp"
    assert state.query_strategy == "mes"
    assert len(first_batch) == 12
    assert list(first_batch["variant_id"]) == list(measurements["variant_id"])
    assert len(updated_state.rounds) == 2
    assert len(next_batch) == 12


def _initialize_stopping_run(tmp_path, **overrides):
    pool_path, embeddings_path = _write_pool_and_embeddings(tmp_path)
    options = {
        "pool_csv": pool_path,
        "embeddings_path": embeddings_path,
        "output_dir": tmp_path / "stopping_run",
        "sequence_column": "sequence",
        "id_column": "variant_id",
        "starting_batch_size": 4,
        "batch_size": 3,
        "seed": 11,
        "initial_selection_strategy_name": "random",
        "predictor_name": "gp",
        "query_strategy_name": "random",
    }
    options.update(overrides)
    return initialize_run(**options)


def _write_initial_measurements(state, tmp_path, values):
    measured = pd.read_csv(state.output_path / "round_000_to_measure.csv")
    measured["Expression"] = values
    path = tmp_path / "measurements.csv"
    measured.to_csv(path, index=False)
    return path


def test_threshold_stops_on_raw_measured_value_without_training(tmp_path, monkeypatch):
    state = _initialize_stopping_run(
        tmp_path, stopping_rule="threshold", phenotype_threshold=8.0
    )
    measurements = _write_initial_measurements(state, tmp_path, [1.0, 2.0, 3.0, 8.0])

    def fail_train(*args, **kwargs):
        raise AssertionError("A reached phenotype threshold does not need a model.")

    monkeypatch.setattr(workflow.PredictorTrainer, "train", fail_train)
    stopped = suggest_next_batch(
        run_dir=state.output_dir,
        measurements_csv=measurements,
        label_column="Expression",
    )

    assert stopped.stopped
    assert stopped.stopping_result["reason"] == "phenotype_threshold"
    assert stopped.stopping_result["best_score"] == 8.0
    assert stopped.label_column == "Expression"
    assert len(stopped.rounds) == 1
    assert not (state.output_path / "round_001_to_measure.csv").exists()
    assert not (state.output_path / "latest_recommendations.csv").exists()
    assert (state.output_path / "round_000_to_measure.csv").exists()
    assert load_state(state.output_dir).stopped

    repeated = suggest_next_batch(
        run_dir=state.output_dir,
        measurements_csv=tmp_path / "unused.csv",
    )
    assert repeated == stopped


def test_ei_stopping_trains_random_query_and_uses_only_measured_scores(
    tmp_path, monkeypatch
):
    state = _initialize_stopping_run(tmp_path, stopping_rule="ei", epsilon=1e6)
    pool = pd.read_csv(state.pool_csv)
    pool["Expression"] = 1e9
    pool.to_csv(state.pool_csv, index=False)
    measurements = _write_initial_measurements(state, tmp_path, [1.0, 2.0, 3.0, 4.0])
    original_train = workflow.PredictorTrainer.train
    observed_training_values = []

    def record_train(self, X_train, y_train):
        observed_training_values.append(np.asarray(y_train).copy())
        return original_train(self, X_train, y_train)

    monkeypatch.setattr(workflow.PredictorTrainer, "train", record_train)
    stopped = suggest_next_batch(
        run_dir=state.output_dir,
        measurements_csv=measurements,
        label_column="Expression",
    )

    assert len(observed_training_values) == 1
    np.testing.assert_array_equal(observed_training_values[0], [1.0, 2.0, 3.0, 4.0])
    assert stopped.stopped
    assert stopped.stopping_result["reason"] == "ei"
    assert 0 <= stopped.stopping_result["max_ei"] <= 1e6
    assert stopped.stopping_result["best_score"] == 4.0
    assert len(stopped.rounds) == 1
    assert not (state.output_path / "latest_recommendations.csv").exists()


def test_stopping_waits_for_all_previous_measurements(tmp_path):
    state = _initialize_stopping_run(
        tmp_path, stopping_rule="threshold", phenotype_threshold=1.0
    )
    measured = pd.read_csv(state.output_path / "round_000_to_measure.csv").iloc[:1]
    measured["Expression"] = 100.0
    path = tmp_path / "partial_measurements.csv"
    measured.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing labels"):
        suggest_next_batch(
            run_dir=state.output_dir,
            measurements_csv=path,
            label_column="Expression",
        )
    assert not load_state(state.output_dir).stopped


def test_threshold_reports_pool_exhaustion_when_unmet(tmp_path):
    state = _initialize_stopping_run(
        tmp_path,
        stopping_rule="threshold",
        phenotype_threshold=100.0,
        starting_batch_size=12,
    )
    measurements = _write_initial_measurements(state, tmp_path, np.arange(1.0, 13.0))
    stopped = suggest_next_batch(
        run_dir=state.output_dir,
        measurements_csv=measurements,
        label_column="Expression",
    )

    assert stopped.stopped
    assert stopped.stopping_result["reason"] == "pool_exhausted"
    assert len(stopped.rounds) == 1


def test_missing_epsilon_fails_before_creating_run(tmp_path):
    with pytest.raises(ValueError, match="epsilon"):
        initialize_run(
            pool_csv=tmp_path / "missing.csv",
            embeddings_path=tmp_path / "missing.npz",
            output_dir=tmp_path / "invalid_run",
            stopping_rule="ei",
        )
    assert not (tmp_path / "invalid_run").exists()


def test_old_state_loads_without_stopping_fields(tmp_path):
    state = _initialize_small_run(tmp_path)
    data = json.loads(state.state_path.read_text())
    for key in ("stopping_rule", "epsilon", "phenotype_threshold", "stopping_result"):
        data.pop(key)
    data["version"] = 1
    state.state_path.write_text(json.dumps(data))

    restored = load_state(state.output_dir)

    assert restored.version == 1
    assert restored.stopping_rule == "none"
    assert restored.epsilon is None
    assert restored.phenotype_threshold is None
    assert not restored.stopped


@pytest.mark.parametrize("epsilon, expected_stop", [(1e6, True), (1e-30, False)])
def test_ei_fallback_preserves_candidate_selection(
    tmp_path, monkeypatch, epsilon, expected_stop
):
    monkeypatch.setattr(BoTorchGPRegressor, "_fit_mll", lambda self, mll: None)
    original_train = workflow.PredictorTrainer.train
    trained_predictors = []

    def record_train(self, X_train, y_train, y_var_train=None):
        trained_predictors.append(type(self.base_predictor))
        return original_train(self, X_train, y_train, y_var_train)

    monkeypatch.setattr(workflow.PredictorTrainer, "train", record_train)
    baseline_dir, ei_dir = tmp_path / "baseline", tmp_path / "ei"
    baseline_dir.mkdir()
    ei_dir.mkdir()
    baseline = _initialize_stopping_run(
        baseline_dir, predictor_name="gradboost_regressor", query_strategy_name="topk"
    )
    adaptive = _initialize_stopping_run(
        ei_dir,
        predictor_name="gradboost_regressor",
        query_strategy_name="topk",
        stopping_rule="ei",
        epsilon=epsilon,
    )
    baseline_measurements = _write_initial_measurements(
        baseline, baseline_dir, [1.0, 2.0, 3.0, 4.0]
    )
    adaptive_measurements = _write_initial_measurements(
        adaptive, ei_dir, [1.0, 2.0, 3.0, 4.0]
    )
    suggest_next_batch(
        run_dir=baseline.output_dir,
        measurements_csv=baseline_measurements,
        label_column="Expression",
    )
    updated = suggest_next_batch(
        run_dir=adaptive.output_dir,
        measurements_csv=adaptive_measurements,
        label_column="Expression",
    )

    assert trained_predictors == [
        GradientBoostingRegressor,
        GradientBoostingRegressor,
        BoTorchGPRegressor,
    ]
    assert updated.predictor == "gradboost_regressor"
    assert updated.stopped is expected_stop
    assert updated.stopping_result["model_source"] == "separate_gp"
    assert updated.stopping_result["predictor"] == "BoTorchGPRegressor"
    assert load_state(updated.output_dir).stopping_result == updated.stopping_result
    if expected_stop:
        assert len(updated.rounds) == 1
        assert not (updated.output_path / "round_001_to_measure.csv").exists()
    else:
        expected = pd.read_csv(baseline.output_path / "round_001_to_measure.csv")
        actual = pd.read_csv(updated.output_path / "round_001_to_measure.csv")
        assert actual["variant_id"].tolist() == expected["variant_id"].tolist()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"starting_batch_size": 0}, "starting_batch_size"),
        ({"starting_batch_size": -1}, "starting_batch_size"),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": -1}, "batch_size"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"predictor_name": "unknown"}, "Unknown predictor"),
        ({"query_strategy_name": "unknown"}, "Unknown query_strategy"),
        (
            {"predictor_name": "gradboost_regressor", "query_strategy_name": "mes"},
            "does not support",
        ),
        *[
            (
                {"predictor_name": "rf_ensemble", "query_strategy_name": query},
                "does not support",
            )
            for query in ["mes", "pes", "gibbon"]
        ],
    ],
)
def test_invalid_user_settings_fail_before_creating_run(tmp_path, overrides, message):
    pool, embeddings = _write_pool_and_embeddings(tmp_path)
    output = tmp_path / "invalid_run"
    options = {
        "pool_csv": pool,
        "embeddings_path": embeddings,
        "output_dir": output,
        "id_column": "variant_id",
        "sequence_column": "sequence",
    }
    options.update(overrides)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        initialize_run(**options)
    assert not output.exists()


@pytest.mark.parametrize("n_samples", [1, 3])
@pytest.mark.parametrize(
    "initial",
    [
        "core_set",
        "density_core_set",
        "kmeans",
        "probcover",
        "probcover_cosine",
        "typiclust",
        "random",
    ],
)
def test_small_pools_finish_without_empty_batches(tmp_path, n_samples, initial):
    pool, embeddings = _write_pool_and_embeddings(tmp_path, n_samples=n_samples)
    state = initialize_run(
        pool_csv=pool,
        embeddings_path=embeddings,
        output_dir=tmp_path / "small_pool",
        id_column="variant_id",
        sequence_column="sequence",
        initial_selection_strategy_name=initial,
    )
    batch = pd.read_csv(state.output_path / "round_000_to_measure.csv")
    assert len(batch) == n_samples
    assert batch.variant_id.is_unique
    batch["Expression"] = np.arange(n_samples) + 1.0
    measurements = tmp_path / "measurements.csv"
    batch.to_csv(measurements, index=False)
    final = suggest_next_batch(
        run_dir=state.output_path,
        measurements_csv=measurements,
        label_column="Expression",
    )
    assert final.stopped
    assert final.stopping_result["reason"] == "pool_exhausted"
    assert len(final.rounds) == 1
    assert not (state.output_path / "round_001_to_measure.csv").exists()
    assert not (state.output_path / "latest_recommendations.csv").exists()


@pytest.mark.parametrize(
    "score, message",
    [(-1.0, "greater than -1"), (-2.0, "greater than -1"), (np.inf, "finite")],
)
def test_invalid_measurement_scores_preserve_run_state(tmp_path, score, message):
    state = _initialize_stopping_run(tmp_path)
    measurements = _write_initial_measurements(state, tmp_path, [score, 1.0, 2.0, 3.0])
    saved = state.state_path.read_bytes()
    with pytest.raises(ValueError, match=message):
        suggest_next_batch(
            run_dir=state.output_path,
            measurements_csv=measurements,
            label_column="Expression",
        )
    assert state.state_path.read_bytes() == saved
    assert not (state.output_path / "round_001_to_measure.csv").exists()


def test_run_seed_makes_forest_predictions_reproducible():
    settings = {"seed": 11, "starting_batch_size": 8, "batch_size": 5}
    models = [
        workflow._instantiate_component(
            kind="predictor", name="rf_ensemble", al_settings=settings
        )
        for _ in range(2)
    ]
    rng = np.random.default_rng(3)
    features = rng.normal(size=(20, 4))
    labels = rng.normal(size=20)
    for model in models:
        model.fit(features, labels)
    np.testing.assert_array_equal(
        models[0].predict(features), models[1].predict(features)
    )


@pytest.mark.parametrize("dtype", [int, float])
def test_shuffled_numeric_design_ids_align_embeddings_by_id(tmp_path, dtype):
    pool = pd.DataFrame(
        {
            "variant_id": np.array([2, 0, 1], dtype=dtype),
            "sequence": ["ATGC", "ATGT", "ATGA"],
        }
    )
    pool_path = tmp_path / "numeric_pool.csv"
    pool.to_csv(pool_path, index=False)
    _, ids, _ = workflow._load_pool(
        pool_csv=pool_path, sequence_column="sequence", id_column="variant_id"
    )
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    path = tmp_path / "numeric_embeddings.npz"
    np.savez_compressed(path, embeddings=features, ids=np.arange(3, dtype=dtype))
    actual = workflow._load_aligned_embeddings(embeddings_path=path, pool_ids=ids)
    np.testing.assert_array_equal(actual, features[[2, 0, 1]])


def test_fractional_embedding_ids_are_not_truncated_to_row_positions(tmp_path):
    path = tmp_path / "embeddings.npz"
    np.savez_compressed(path, embeddings=np.ones((2, 4)), ids=np.array([0.25, 1.25]))
    with pytest.raises(ValueError, match="missing"):
        workflow._load_aligned_embeddings(
            embeddings_path=path, pool_ids=np.array(["variant_0", "variant_1"])
        )
