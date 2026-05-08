"""
Unit tests for baseline_scores helpers.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from utils import baseline_scores
from utils.baseline_scores import (
    DatasetSpec,
    build_top_mask,
    compute_random_summary_metrics_history,
    draw_random_rounds,
    load_dataset_specs,
    load_labels,
    load_subset_ids,
)


def test_rounds_to_top_all_nan_when_no_hits():
    labels = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    top_mask = np.zeros(len(labels), dtype=bool)

    history = compute_random_summary_metrics_history(
        labels=labels,
        top_mask=top_mask,
        max_label=float(labels.max()),
        num_rounds=2,
        num_samples_per_round=2,
        seed=0,
    )

    assert len(history) == 2
    for record in history:
        assert np.isnan(record["rounds_to_top"])


def test_rounds_to_top_first_hit_persists():
    labels = np.linspace(1.0, 6.0, 6, dtype=float)
    num_rounds = 3
    num_samples_per_round = 2
    seed = 7

    rng = np.random.default_rng(seed)
    rounds = draw_random_rounds(
        num_samples=len(labels),
        num_rounds=num_rounds,
        num_samples_per_round=num_samples_per_round,
        rng=rng,
    )

    top_mask = np.zeros(len(labels), dtype=bool)
    top_mask[int(rounds[1][0])] = True

    history = compute_random_summary_metrics_history(
        labels=labels,
        top_mask=top_mask,
        max_label=float(labels.max()),
        num_rounds=num_rounds,
        num_samples_per_round=num_samples_per_round,
        seed=seed,
    )

    assert np.isnan(history[0]["rounds_to_top"])
    assert pytest.approx(2.0) == history[1]["rounds_to_top"]
    assert pytest.approx(2.0) == history[2]["rounds_to_top"]


def test_load_dataset_specs_resolves_relative_paths(tmp_path):
    (tmp_path / "metadata.csv").write_text("label\n1\n")
    (tmp_path / "subset.txt").write_text("0\n")
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(
        "datasets:\n"
        "  - name: toy\n"
        "    metadata_path: metadata.csv\n"
        "    subset_ids_path: subset.txt\n"
    )

    specs = load_dataset_specs(yaml_path)

    assert specs == [
        DatasetSpec(
            name="toy",
            metadata_path=(tmp_path / "metadata.csv").resolve(),
            subset_ids_path=(tmp_path / "subset.txt").resolve(),
        )
    ]


def test_load_dataset_specs_validates_required_fields(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_dataset_specs(missing)

    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text("datasets: []\n")
    with pytest.raises(ValueError, match="No datasets"):
        load_dataset_specs(yaml_path)

    yaml_path.write_text("datasets:\n  - metadata_path: labels.csv\n")
    with pytest.raises(ValueError, match="missing name"):
        load_dataset_specs(yaml_path)

    yaml_path.write_text("datasets:\n  - name: toy\n")
    with pytest.raises(ValueError, match="missing metadata_path"):
        load_dataset_specs(yaml_path)


def test_load_subset_ids_skips_blanks_and_rejects_bad_values(tmp_path):
    subset_path = tmp_path / "subset.txt"
    subset_path.write_text("\n1\n 3 \n")
    np.testing.assert_array_equal(load_subset_ids(subset_path), np.array([1, 3]))

    subset_path.write_text("a\n")
    with pytest.raises(ValueError, match="Invalid sample id"):
        load_subset_ids(subset_path)

    subset_path.write_text("\n")
    with pytest.raises(ValueError, match="did not contain any ids"):
        load_subset_ids(subset_path)


def test_load_labels_applies_subset_and_filters_nonfinite(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"label": [1.0, "bad", 3.5, np.nan]}).to_csv(
        metadata_path, index=False
    )
    subset_path = tmp_path / "subset.txt"
    subset_path.write_text("0\n1\n2\n")
    dataset = DatasetSpec("toy", metadata_path, subset_path)

    labels, sample_ids = load_labels(dataset, "label", {}, {})

    np.testing.assert_allclose(labels, np.array([1.0, 3.5]))
    np.testing.assert_array_equal(sample_ids, np.array([0, 2]))


def test_load_labels_rejects_missing_label_out_of_bounds_and_empty(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"label": [np.nan, np.nan]}).to_csv(metadata_path, index=False)
    dataset = DatasetSpec("toy", metadata_path)

    with pytest.raises(ValueError, match="Label key"):
        load_labels(dataset, "missing", {}, {})

    subset_path = tmp_path / "subset.txt"
    subset_path.write_text("2\n")
    with pytest.raises(ValueError, match="out of bounds"):
        load_labels(DatasetSpec("toy", metadata_path, subset_path), "label", {}, {})

    with pytest.raises(ValueError, match="No finite labels"):
        load_labels(dataset, "label", {}, {})


def test_build_top_mask_marks_highest_values() -> None:
    labels = np.array([1.0, 5.0, 3.0, 4.0])

    np.testing.assert_array_equal(
        build_top_mask(labels, 0.25), [False, True, False, False]
    )
    np.testing.assert_array_equal(build_top_mask(labels, 1.0), [True, True, True, True])


def test_draw_random_rounds_validates_sample_counts() -> None:
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="must be > 0"):
        draw_random_rounds(10, 0, 2, rng)
    with pytest.raises(ValueError, match="exceed dataset size"):
        draw_random_rounds(3, 2, 2, rng)


def test_main_writes_random_baseline_csv(tmp_path, monkeypatch):
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"label": [1.0, 2.0, 3.0, 4.0]}).to_csv(metadata_path, index=False)
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(
        f"datasets:\n  - name: toy\n    metadata_path: {metadata_path}\n"
    )
    output_csv = tmp_path / "baseline.csv"
    monkeypatch.setattr(
        baseline_scores,
        "parse_args",
        lambda: SimpleNamespace(
            datasets_yaml=str(yaml_path),
            output_csv=str(output_csv),
            label_key="label",
            num_experiments=2,
            num_rounds=2,
            num_samples_per_round=1,
            top_p=0.25,
            dataset=["toy"],
        ),
    )
    monkeypatch.setattr(baseline_scores, "tqdm", lambda iterable: iterable)

    baseline_scores.main()

    frame = pd.read_csv(output_csv)
    assert len(frame) == 4
    assert set(frame["dataset_name"]) == {"toy"}
    assert set(frame["seed"]) == {0, 1}


def test_main_validates_arguments_and_dataset_filter(tmp_path, monkeypatch):
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"label": [1.0, 2.0]}).to_csv(metadata_path, index=False)
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(
        f"datasets:\n  - name: toy\n    metadata_path: {metadata_path}\n"
    )

    def set_args(**updates):
        base = {
            "datasets_yaml": str(yaml_path),
            "output_csv": str(tmp_path / "out.csv"),
            "label_key": "label",
            "num_experiments": 1,
            "num_rounds": 1,
            "num_samples_per_round": 1,
            "top_p": 0.25,
            "dataset": [],
        }
        base.update(updates)
        monkeypatch.setattr(
            baseline_scores, "parse_args", lambda: SimpleNamespace(**base)
        )

    set_args(num_experiments=0)
    with pytest.raises(ValueError, match="num_experiments"):
        baseline_scores.main()

    set_args(top_p=2.0)
    with pytest.raises(ValueError, match="top_p"):
        baseline_scores.main()

    set_args(dataset=["missing"])
    with pytest.raises(ValueError, match="Requested datasets not found"):
        baseline_scores.main()
