"""
Data loading utilities for active learning experiments.

This loader pairs an embeddings file with a CSV containing
labels (and optional sample identifiers), ensuring row alignment between the two.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """
    Container for loaded dataset with all necessary components.

    Attributes:
        sample_ids: Stable identifiers aligned with embeddings/labels
        labels: Array of target label values (e.g., expression)
        embeddings: Pre-computed embeddings (required)
    """

    sample_ids: list[str]
    labels: np.ndarray
    embeddings: np.ndarray

    def __post_init__(self) -> None:
        """Validate dataset after initialization."""
        if len(self.sample_ids) != len(self.labels):
            raise ValueError(
                f"Sample IDs ({len(self.sample_ids)}) and labels "
                f"({len(self.labels)}) must have the same length"
            )
        if len(self.sample_ids) != len(self.embeddings):
            raise ValueError(
                f"Sample IDs ({len(self.sample_ids)}) and embeddings "
                f"({len(self.embeddings)}) must have the same length"
            )


class DataLoader:
    """
    Load embeddings from NPZ and labels from a paired CSV.
    """

    def __init__(
        self,
        embeddings_path: str,
        metadata_path: str,
        label_key: str,
        subset_ids_path: str | None = None,
    ) -> None:
        """
        Initialize the data loader.

        Args:
            embeddings_path: Path to npz file containing embeddings.
            metadata_path: CSV with labels aligned to embeddings.
            label_key: Column in the CSV to use as the training label.
            subset_ids_path: Optional path to a newline-delimited file of sample IDs to keep.
        """
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.label_key = label_key
        self.subset_ids_path = subset_ids_path
        self.dataset: Dataset | None = None

    def load(self) -> Dataset:
        """
        Load paired embeddings/metadata and return a Dataset.
        """
        logger.info(
            f"Loading embeddings from {self.embeddings_path} "
            f"and metadata from {self.metadata_path}"
        )

        embeddings, sample_ids = self._load_embeddings()
        embeddings, sample_ids = self._apply_subset_if_needed(embeddings, sample_ids)
        labels = self._load_metadata(sample_ids)

        self.dataset = Dataset(
            sample_ids=sample_ids,
            labels=labels,
            embeddings=embeddings,
        )

        logger.info(
            f"Loaded dataset with {len(self.dataset.sample_ids)} samples. "
            f"Embeddings shape: {self.dataset.embeddings.shape}"
        )

        return self.dataset

    def _load_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(self.embeddings_path, allow_pickle=True)
        if "embeddings" not in data:
            raise ValueError(
                f"'embeddings' array not found in {self.embeddings_path}. "
                f"Available keys: {list(data.keys())}"
            )
        embeddings = data["embeddings"]
        sample_ids = data["ids"].astype(
            np.int32, copy=False
        )  # sample_ids is row index of csv, so we need to convert it to integer
        return embeddings, sample_ids

    def _load_metadata(self, sample_ids: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.metadata_path)
        df = df.iloc[sample_ids]
        labels = df[self.label_key].to_numpy()
        return labels

    def _apply_subset_if_needed(
        self, embeddings: np.ndarray, sample_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        subset_ids = self._load_subset_ids(sample_ids.dtype)
        if subset_ids is None:
            return embeddings, sample_ids

        mask = np.isin(sample_ids, subset_ids)
        if not np.any(mask):
            raise ValueError(
                "Subset id filtering removed all samples. "
                "Ensure the subset ids match those stored in the embeddings file."
            )

        filtered_embeddings = embeddings[mask]
        filtered_sample_ids = sample_ids[mask]

        missing = set(subset_ids.tolist()) - set(filtered_sample_ids.tolist())
        if missing:
            logger.warning(
                "Subset ids file contained %d ids not present in embeddings; ignoring smallest few: %s",
                len(missing),
                sorted(missing)[:5],
            )

        logger.info(
            "Subset filtering retained %d / %d samples.",
            len(filtered_sample_ids),
            len(sample_ids),
        )
        return filtered_embeddings, filtered_sample_ids

    def _load_subset_ids(self, dtype) -> np.ndarray | None:
        if not self.subset_ids_path:
            return None
        subset_path = Path(self.subset_ids_path)
        if not subset_path.exists():
            raise FileNotFoundError(f"Subset ids file {subset_path} does not exist.")
        subset_ids = []
        for line in subset_path.read_text().splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                subset_ids.append(int(text))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid sample id '{text}' in subset file {subset_path}"
                ) from exc
        if not subset_ids:
            raise ValueError(
                f"Subset ids file {subset_path} did not contain any sample ids."
            )
        return np.asarray(subset_ids, dtype=dtype)
