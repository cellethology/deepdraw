"""
Unit tests for DataLoader and Dataset classes.

Embeddings are stored in a compressed NPZ file with ids and embeddings arrays.
"""

import numpy as np
import pandas as pd
import pytest

from core.data_loader import DataLoader, Dataset


class TestDataset:
    """Dataset dataclass validation."""

    def test_length_mismatch_raises(self):
        sample_ids = ["sample_0", "sample_1"]
        labels = np.array([1.0])
        embeddings = np.random.randn(2, 3)

        with pytest.raises(ValueError):
            Dataset(
                sample_ids=sample_ids,
                labels=labels,
                embeddings=embeddings,
            )


class TestDataLoader:
    """Test loading paired embeddings/metadata files."""

    def _create_embeddings_file(self, tmp_path, n_samples=4, dim=3):
        embeddings = np.random.randn(n_samples, dim).astype(np.float32)
        ids = np.arange(n_samples, dtype=np.int32)
        path = tmp_path / "embeddings.npz"
        np.savez_compressed(path, embeddings=embeddings, ids=ids)
        return str(path), embeddings

    def _create_metadata_csv(self, tmp_path, n_samples=4, target_col="Expression"):
        expressions = np.linspace(0.0, 1.0, n_samples)
        data = {target_col: expressions}

        path = tmp_path / "metadata.csv"
        pd.DataFrame(data).to_csv(path, index=False)
        return str(path), data

    def test_load_paired_files(self, tmp_path):
        emb_path, embeddings = self._create_embeddings_file(
            tmp_path, n_samples=5, dim=6
        )
        csv_path, data = self._create_metadata_csv(tmp_path, n_samples=5)

        loader = DataLoader(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            label_key="Expression",
        )

        dataset = loader.load()

        assert dataset.embeddings.shape == embeddings.shape
        assert dataset.labels.shape[0] == embeddings.shape[0]
        expected_ids = np.arange(embeddings.shape[0], dtype=np.int32)
        np.testing.assert_array_equal(dataset.sample_ids, expected_ids)
        assert np.allclose(dataset.labels, data["Expression"])

    def test_missing_target_column(self, tmp_path):
        emb_path, _ = self._create_embeddings_file(tmp_path, n_samples=3)
        csv_path = tmp_path / "metadata.csv"
        pd.DataFrame({"Sequence": ["A", "B", "C"]}).to_csv(csv_path, index=False)

        loader = DataLoader(
            embeddings_path=emb_path,
            metadata_path=str(csv_path),
            label_key="Expression",
        )

        with pytest.raises(KeyError, match="Expression"):
            loader.load()

    def test_mismatched_lengths(self, tmp_path):
        emb_path, _ = self._create_embeddings_file(tmp_path, n_samples=4)
        csv_path, _ = self._create_metadata_csv(tmp_path, n_samples=3)

        loader = DataLoader(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            label_key="Expression",
        )

        with pytest.raises(IndexError):
            loader.load()

    def test_missing_embeddings_key_raises(self, tmp_path):
        path = tmp_path / "embeddings.npz"
        np.savez_compressed(path, ids=np.arange(2, dtype=np.int32))
        csv_path, _ = self._create_metadata_csv(tmp_path, n_samples=2)

        loader = DataLoader(
            embeddings_path=str(path),
            metadata_path=csv_path,
            label_key="Expression",
        )

        with pytest.raises(ValueError, match="embeddings"):
            loader.load()

    def test_sample_ids_out_of_bounds(self, tmp_path):
        # Create embeddings with ids that reference rows outside the CSV
        embeddings = np.random.randn(3, 4).astype(np.float32)
        ids = np.array([0, 5, 6], dtype=np.int32)
        emb_path = tmp_path / "embeddings.npz"
        np.savez_compressed(emb_path, embeddings=embeddings, ids=ids)

        csv_path, _ = self._create_metadata_csv(tmp_path, n_samples=3)

        loader = DataLoader(
            embeddings_path=str(emb_path),
            metadata_path=csv_path,
            label_key="Expression",
        )

        with pytest.raises(IndexError):
            loader.load()

    def test_subset_ids_filters_embeddings_and_labels(self, tmp_path):
        emb_path, embeddings = self._create_embeddings_file(tmp_path, n_samples=6)
        csv_path, data = self._create_metadata_csv(tmp_path, n_samples=6)
        subset_ids = [1, 4, 5]
        subset_path = tmp_path / "subset.txt"
        subset_path.write_text("\n".join(str(i) for i in subset_ids))

        loader = DataLoader(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            label_key="Expression",
            subset_ids_path=str(subset_path),
        )

        dataset = loader.load()

        np.testing.assert_array_equal(dataset.sample_ids, np.array(subset_ids))
        assert dataset.embeddings.shape[0] == len(subset_ids)
        expected_labels = np.array(data["Expression"])[subset_ids]
        assert np.allclose(dataset.labels, expected_labels)

    def test_subset_ids_with_no_overlap_errors(self, tmp_path):
        emb_path, _ = self._create_embeddings_file(tmp_path, n_samples=3)
        csv_path, _ = self._create_metadata_csv(tmp_path, n_samples=3)
        subset_path = tmp_path / "subset.txt"
        subset_path.write_text("10\n11\n")

        loader = DataLoader(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            label_key="Expression",
            subset_ids_path=str(subset_path),
        )

        with pytest.raises(ValueError, match="Subset id filtering removed all samples"):
            loader.load()
