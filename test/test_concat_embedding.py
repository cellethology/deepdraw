"""Unit tests for embedding concatenation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils import concat_embedding


def _write_npz(path: Path, embeddings: np.ndarray, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=embeddings, ids=ids)


def test_concat_embeddings_matches_ids_in_first_file_order(tmp_path: Path) -> None:
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    output = tmp_path / "nested" / "out.npz"
    _write_npz(
        path_a,
        np.array([[2.0, 0.0], [0.0, 4.0], [1.0, 1.0]]),
        np.array([2, 1, 3]),
    )
    _write_npz(
        path_b,
        np.array([[0.0, 5.0], [6.0, 0.0], [7.0, 7.0]]),
        np.array([3, 2, 4]),
    )

    concat_embedding.concat_embeddings(path_a, path_b, output)

    data = np.load(output)
    np.testing.assert_array_equal(data["ids"], np.array([2, 3]))
    np.testing.assert_allclose(
        data["embeddings"],
        np.array([[2.0, 0.0, 6.0, 0.0], [1.0, 1.0, 0.0, 5.0]]),
    )


def test_concat_embeddings_can_normalize_before_concatenation(tmp_path: Path) -> None:
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    output = tmp_path / "out.npz"
    _write_npz(path_a, np.array([[3.0, 4.0], [0.0, 0.0]]), np.array([1, 2]))
    _write_npz(path_b, np.array([[0.0, 5.0], [8.0, 6.0]]), np.array([1, 2]))

    concat_embedding.concat_embeddings(path_a, path_b, output, normalize=True)

    data = np.load(output)
    np.testing.assert_allclose(
        data["embeddings"],
        np.array([[0.6, 0.8, 0.0, 1.0], [0.0, 0.0, 0.8, 0.6]]),
    )


def test_concat_embeddings_rejects_duplicate_ids(tmp_path: Path) -> None:
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    _write_npz(path_a, np.ones((2, 1)), np.array([1, 1]))
    _write_npz(path_b, np.ones((2, 1)), np.array([1, 2]))

    with pytest.raises(ValueError, match="duplicates"):
        concat_embedding.concat_embeddings(path_a, path_b, tmp_path / "out.npz")


def test_concat_embeddings_rejects_no_overlap(tmp_path: Path) -> None:
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    _write_npz(path_a, np.ones((1, 1)), np.array([1]))
    _write_npz(path_b, np.ones((1, 1)), np.array([2]))

    with pytest.raises(ValueError, match="No overlapping ids"):
        concat_embedding.concat_embeddings(path_a, path_b, tmp_path / "out.npz")


def test_load_npz_validates_required_arrays_and_shapes(tmp_path: Path) -> None:
    missing = tmp_path / "missing_keys.npz"
    mismatch = tmp_path / "mismatch.npz"
    np.savez(missing, embeddings=np.ones((2, 2)))
    _write_npz(mismatch, np.ones((2, 2)), np.array([1]))

    with pytest.raises(ValueError, match="must contain"):
        concat_embedding._load_npz(missing)
    with pytest.raises(ValueError, match="length mismatch"):
        concat_embedding._load_npz(mismatch)


def test_apply_pca_variance_validates_target_and_handles_degenerate_data() -> None:
    with pytest.raises(ValueError, match="target_variance"):
        concat_embedding._apply_pca_variance(np.ones((2, 2)), 0.0)

    projected, n_components = concat_embedding._apply_pca_variance(np.ones((3, 2)), 0.9)

    assert n_components == 1
    np.testing.assert_allclose(projected, np.zeros((3, 2)))
