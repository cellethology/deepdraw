"""Unit tests for recomputing n_top columns in results CSVs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import add_n_top


def _write_npz(path: Path, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=np.ones((len(ids), 2)), ids=ids)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_parse_and_normalize_selected_ids() -> None:
    assert add_n_top._parse_selected_ids([1, "2"]) == [1, "2"]
    assert add_n_top._parse_selected_ids("[1, 'x']") == [1, "x"]
    assert add_n_top._parse_selected_ids("a,b,, c") == ["a", "b", " c"]
    assert add_n_top._parse_selected_ids(None) == []

    assert add_n_top._normalize_id(3) == "3"
    assert add_n_top._normalize_id(np.int64(4)) == "4"
    assert add_n_top._normalize_id("5.0") == "5"
    assert add_n_top._normalize_id("variant_a") == "variant_a"
    assert add_n_top._normalize_id("  ") == ""


def test_resolve_embedding_model_prefers_summary_then_overrides() -> None:
    assert (
        add_n_top._resolve_embedding_model({"embedding_model": "model_a"}) == "model_a"
    )
    assert (
        add_n_top._resolve_embedding_model(
            {"embedding_model": "NONE", "hydra_overrides": ["+embedding_model=model_b"]}
        )
        == "model_b"
    )
    assert (
        add_n_top._resolve_embedding_model(
            {"hydra_overrides": ["embedding_model: model_c"]}
        )
        == "model_c"
    )
    assert add_n_top._resolve_embedding_model({}) is None


def test_load_dataset_map_resolves_paths_and_validates_entries(tmp_path: Path) -> None:
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(
        "datasets:\n"
        "  - name: toy\n"
        "    metadata_path: labels.csv\n"
        "    embedding_dir: embeddings\n"
        "    subset_ids_path: subset.txt\n"
    )

    dataset_map = add_n_top._load_dataset_map(yaml_path)

    assert dataset_map["toy"] == {
        "metadata_path": (tmp_path / "labels.csv").resolve(),
        "embedding_dir": (tmp_path / "embeddings").resolve(),
        "subset_ids_path": (tmp_path / "subset.txt").resolve(),
    }

    with pytest.raises(FileNotFoundError):
        add_n_top._load_dataset_map(tmp_path / "missing.yaml")

    yaml_path.write_text("datasets: []\n")
    with pytest.raises(ValueError, match="No datasets"):
        add_n_top._load_dataset_map(yaml_path)

    yaml_path.write_text("datasets:\n  - name: toy\n    metadata_path: labels.csv\n")
    with pytest.raises(ValueError, match="missing embedding_dir"):
        add_n_top._load_dataset_map(yaml_path)


def test_compute_top_id_set_uses_embeddings_metadata_and_subset(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "embeddings" / "toy_model.npz"
    metadata_path = tmp_path / "labels.csv"
    subset_path = tmp_path / "subset.txt"
    _write_npz(embeddings_path, np.array([0, 1, 2, 3]))
    pd.DataFrame({"label": [0.1, 0.9, 0.3, 0.8]}).to_csv(metadata_path, index=False)
    subset_path.write_text("0\n2\n3\n")

    assert add_n_top._compute_top_id_set(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        label_key="label",
        subset_ids_path=None,
        top_p=0.5,
    ) == {"1", "3"}
    assert add_n_top._compute_top_id_set(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        label_key="label",
        subset_ids_path=subset_path,
        top_p=0.34,
    ) == {"3"}

    subset_path.write_text("99\n")
    with pytest.raises(ValueError, match="removed all samples"):
        add_n_top._compute_top_id_set(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            label_key="label",
            subset_ids_path=subset_path,
            top_p=0.5,
        )


def test_load_helpers_validate_inputs(tmp_path: Path) -> None:
    subset_path = tmp_path / "subset.txt"
    subset_path.write_text("bad\n")
    with pytest.raises(ValueError, match="Invalid sample id"):
        add_n_top._load_subset_ids(subset_path)

    subset_path.write_text("\n")
    with pytest.raises(ValueError, match="did not contain"):
        add_n_top._load_subset_ids(subset_path)

    embeddings_path = tmp_path / "no_ids.npz"
    np.savez(embeddings_path, embeddings=np.ones((2, 2)))
    with pytest.raises(ValueError, match="ids"):
        add_n_top._load_sample_ids(embeddings_path)

    with pytest.raises(FileNotFoundError):
        add_n_top._load_summary(tmp_path / "missing_summary.json")


def test_update_results_csv_writes_counts_and_respects_overwrite(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        'round,selected_sample_ids,n_top\n0,"[1, 3, 5]",old\n1,"[2, 4]",old\n'
    )

    assert not add_n_top._update_results_csv(
        results_path, top_ids={"1", "3"}, column_name="n_top", overwrite=False
    )

    assert add_n_top._update_results_csv(
        results_path, top_ids={"1", "3"}, column_name="n_top", overwrite=True
    )
    rows = _read_rows(results_path)
    assert [row["n_top"] for row in rows] == ["2", "0"]

    assert add_n_top._update_results_csv(
        results_path, top_ids={"2", "4"}, column_name="n_top_50", overwrite=False
    )
    rows = _read_rows(results_path)
    assert rows[0]["n_top_50"] == "0"
    assert rows[1]["n_top_50"] == "2"


def test_update_results_csv_handles_empty_and_missing_columns(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("selected_sample_ids\n")
    assert not add_n_top._update_results_csv(
        empty_path, top_ids={"1"}, column_name="n_top", overwrite=False
    )

    missing_path = tmp_path / "missing.csv"
    missing_path.write_text("round\n0\n")
    with pytest.raises(ValueError, match="selected_sample_ids"):
        add_n_top._update_results_csv(
            missing_path, top_ids={"1"}, column_name="n_top", overwrite=False
        )


def test_main_updates_results_from_dataset_yaml(tmp_path: Path, monkeypatch) -> None:
    metadata_path = tmp_path / "labels.csv"
    embeddings_dir = tmp_path / "embeddings"
    datasets_yaml = tmp_path / "datasets.yaml"
    run_dir = tmp_path / "runs" / "run_0"
    results_path = run_dir / "results.csv"
    run_dir.mkdir(parents=True)
    pd.DataFrame({"label": [0.1, 0.9, 0.3, 0.8]}).to_csv(metadata_path, index=False)
    _write_npz(embeddings_dir / "toy_model.npz", np.array([0, 1, 2, 3]))
    datasets_yaml.write_text(
        "datasets:\n"
        "  - name: toy\n"
        f"    metadata_path: {metadata_path}\n"
        f"    embedding_dir: {embeddings_dir}\n"
    )
    (run_dir / "summary.json").write_text(
        '{"dataset_name": "toy", "embedding_model": "toy_model"}'
    )
    results_path.write_text('round,selected_sample_ids\n0,"[1, 3]"\n1,"[0, 2]"\n')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "add_n_top.py",
            str(tmp_path / "runs"),
            "--datasets-yaml",
            str(datasets_yaml),
            "--top-p",
            "0.5",
            "--column-name",
            "n_top",
            "--label-key",
            "label",
        ],
    )

    assert add_n_top.main() == 0

    rows = _read_rows(results_path)
    assert [row["n_top"] for row in rows] == ["2", "0"]


def test_main_rejects_invalid_arguments(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["add_n_top.py", str(tmp_path), "--top-p", "2"])
    with pytest.raises(SystemExit, match="top_p"):
        add_n_top.main()

    missing = tmp_path / "missing"
    monkeypatch.setattr(sys, "argv", ["add_n_top.py", str(missing)])
    with pytest.raises(SystemExit, match="Root dir not found"):
        add_n_top.main()
