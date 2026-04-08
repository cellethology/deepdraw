"""Unit tests for config utility helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_sub.utils import config_utils as cu


@pytest.fixture(autouse=True)
def clear_datasets_cache():
    cu._datasets_cache.clear()
    yield
    cu._datasets_cache.clear()


def _write_datasets_yaml(
    path: Path, embedding_dir: Path, subset_rel: str = "ids.txt"
) -> None:
    path.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: ds_a",
                "    metadata_path: meta.csv",
                f"    embedding_dir: {embedding_dir}",
                f"    subset_ids_path: {subset_rel}",
                "  - name: ds_b",
                "    metadata_path: meta_b.csv",
                f"    embedding_dir: {embedding_dir}",
            ]
        )
    )


def test_parse_override_value_handles_plain_and_plus() -> None:
    argv = ["query_strategy=botorch_mes", "+dataset_index=3", "foo=bar"]
    assert cu.parse_override_value(argv, "query_strategy") == "botorch_mes"
    assert cu.parse_override_value(argv, "dataset_index") == "3"
    assert cu.parse_override_value(argv, "missing") is None


def test_get_datasets_file_setting_prefers_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("datasets_file: datasets/default.yaml\n")

    assert (
        cu.get_datasets_file_setting(
            ["datasets_file=datasets/override.yaml"], config_path
        )
        == "datasets/override.yaml"
    )
    assert cu.get_datasets_file_setting([], config_path) == "datasets/default.yaml"


def test_load_dataset_configs_resolves_paths_and_uses_cache(tmp_path: Path) -> None:
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    subset_dir = tmp_path / "subsets"
    subset_dir.mkdir()
    (subset_dir / "ids.txt").write_text("0\n1\n")
    datasets_file = tmp_path / "datasets.yaml"
    _write_datasets_yaml(datasets_file, embedding_dir, subset_rel="subsets/ids.txt")

    first = cu.load_dataset_configs(datasets_file)
    second = cu.load_dataset_configs(datasets_file)

    assert first is second
    assert len(first) == 2
    assert first[0].name == "ds_a"
    assert first[0].embedding_dir == str(embedding_dir.resolve())
    assert first[0].subset_ids_path == str((subset_dir / "ids.txt").resolve())
    assert first[1].subset_ids_path is None


def test_load_dataset_configs_validates_required_fields(tmp_path: Path) -> None:
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()

    missing_metadata = tmp_path / "missing_metadata.yaml"
    missing_metadata.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: ds_a",
                "    metadata_path: ''",
                f"    embedding_dir: {embedding_dir}",
            ]
        )
    )
    with pytest.raises(ValueError, match="missing metadata_path"):
        cu.load_dataset_configs(missing_metadata)

    missing_embedding = tmp_path / "missing_embedding.yaml"
    missing_embedding.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: ds_a",
                "    metadata_path: meta.csv",
            ]
        )
    )
    with pytest.raises(ValueError, match="missing embedding_dir"):
        cu.load_dataset_configs(missing_embedding)

    nonexistent_embedding = tmp_path / "missing_dir.yaml"
    nonexistent_embedding.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: ds_a",
                "    metadata_path: meta.csv",
                f"    embedding_dir: {tmp_path / 'does_not_exist'}",
            ]
        )
    )
    with pytest.raises(ValueError, match="does not exist"):
        cu.load_dataset_configs(nonexistent_embedding)


def test_resolve_dataset_field_handles_defaults_and_errors(tmp_path: Path) -> None:
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    datasets_file = tmp_path / "datasets.yaml"
    _write_datasets_yaml(datasets_file, embedding_dir)

    assert cu._resolve_dataset_field(str(datasets_file), 0, "name") == "ds_a"
    assert (
        cu._resolve_dataset_field(str(datasets_file), 10, "name", default="fallback")
        == "fallback"
    )
    assert (
        cu._resolve_dataset_field(str(datasets_file), 10, "name", default="null")
        is None
    )

    with pytest.raises(ValueError, match="Invalid dataset index"):
        cu._resolve_dataset_field(str(datasets_file), "abc", "name")
    with pytest.raises(ValueError, match="Unknown dataset field"):
        cu._resolve_dataset_field(str(datasets_file), 0, "unknown")
    with pytest.raises(IndexError, match="out of range"):
        cu._resolve_dataset_field(str(datasets_file), 10, "name")


def test_load_datasets_or_raise_when_empty(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("datasets_file: datasets/does_not_exist.yaml\n")

    with pytest.raises(RuntimeError, match="No datasets configured"):
        cu.load_datasets_or_raise([], config_path)


def test_seed_env_from_datasets_seeds_and_respects_guards(monkeypatch) -> None:
    datasets = [
        cu.DatasetConfig(
            name="dataset_a",
            metadata_path="meta.csv",
            embedding_dir="/tmp/embeds",
            subset_ids_path="/tmp/subset.txt",
        )
    ]

    # Normal path: set env vars.
    for key in ("TEST_CHILD", "TEST_DATASET", "TEST_META", "TEST_EMBED", "TEST_SUBSET"):
        monkeypatch.delenv(key, raising=False)
    cu.seed_env_from_datasets(
        datasets,
        hydra_child_env="TEST_CHILD",
        dataset_env="TEST_DATASET",
        metadata_env="TEST_META",
        embedding_env="TEST_EMBED",
        subset_env="TEST_SUBSET",
    )
    assert cu.os.environ["TEST_DATASET"] == "dataset_a"
    assert cu.os.environ["TEST_META"] == "meta.csv"
    assert cu.os.environ["TEST_EMBED"] == "/tmp/embeds"
    assert cu.os.environ["TEST_SUBSET"] == "/tmp/subset.txt"

    # Guard: hydra child should short-circuit.
    for key in ("TEST_DATASET2", "TEST_META2", "TEST_EMBED2", "TEST_SUBSET2"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TEST_CHILD2", "1")
    cu.seed_env_from_datasets(
        datasets,
        hydra_child_env="TEST_CHILD2",
        dataset_env="TEST_DATASET2",
        metadata_env="TEST_META2",
        embedding_env="TEST_EMBED2",
        subset_env="TEST_SUBSET2",
    )
    assert "TEST_DATASET2" not in cu.os.environ

    # Guard: pre-set dataset env should short-circuit.
    monkeypatch.setenv("TEST_DATASET3", "preset")
    for key in ("TEST_META3", "TEST_EMBED3", "TEST_SUBSET3"):
        monkeypatch.delenv(key, raising=False)
    cu.seed_env_from_datasets(
        datasets,
        hydra_child_env="TEST_CHILD3",
        dataset_env="TEST_DATASET3",
        metadata_env="TEST_META3",
        embedding_env="TEST_EMBED3",
        subset_env="TEST_SUBSET3",
    )
    assert cu.os.environ["TEST_DATASET3"] == "preset"
    assert "TEST_META3" not in cu.os.environ
