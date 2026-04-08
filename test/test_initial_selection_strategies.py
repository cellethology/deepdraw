"""
Tests for initial selection strategies.
"""

import numpy as np

from core.data_loader import Dataset
from core.initial_selection_strategies import (
    CoreSetInitialSelection,
    DensityWeightedCoreSetInitialSelection,
    KMeansInitialSelection,
    RandomInitialSelection,
    TypiClustInitialSelection,
)


def _create_dataset(n_samples: int, embedding_dim: int = 4) -> Dataset:
    sample_ids = [f"sample_{i}" for i in range(n_samples)]
    labels = np.linspace(0, 1, n_samples)
    embeddings = np.random.randn(n_samples, embedding_dim)
    return Dataset(sample_ids=sample_ids, labels=labels, embeddings=embeddings)


def _dataset_from_embeddings(embeddings) -> Dataset:
    embeddings = np.asarray(embeddings, dtype=float)
    sample_ids = [f"sample_{i}" for i in range(len(embeddings))]
    labels = np.linspace(0, 1, len(embeddings))
    return Dataset(sample_ids=sample_ids, labels=labels, embeddings=embeddings)


def test_random_initial_selection_basic():
    dataset = _create_dataset(20)
    strategy = RandomInitialSelection(seed=123, starting_batch_size=5)

    indices = strategy.select(dataset=dataset)

    assert len(indices) == 5
    assert len(set(indices)) == 5


def test_kmeans_initial_selection_returns_expected_count():
    dataset = _create_dataset(15, embedding_dim=3)
    strategy = KMeansInitialSelection(seed=42, starting_batch_size=6)

    indices = strategy.select(dataset=dataset)

    assert len(indices) == 6
    assert len(set(indices)) == 6


def test_core_set_initial_selection_prefers_dense_seed():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.05, 0.0],
            [0.0, 0.05],
            [5.0, 5.0],
            [6.0, 6.0],
        ],
        dtype=float,
    )
    dataset = _dataset_from_embeddings(embeddings)
    # Use a batch size equal to the number of unique clusters to avoid duplicates
    strategy = CoreSetInitialSelection(
        seed=0, starting_batch_size=2, density_neighbors=3
    )

    indices = strategy.select(dataset=dataset)

    density_scores = strategy._estimate_density_scores(dataset.embeddings)
    assert density_scores is not None
    expected_first = int(np.argmin(density_scores))

    assert indices[0] == expected_first
    assert len(indices) == 2
    assert len(set(indices)) == 2


def test_core_set_initial_selection_handles_empty_dataset():
    dataset = Dataset(sample_ids=[], labels=np.array([]), embeddings=np.empty((0, 2)))
    strategy = CoreSetInitialSelection(seed=0, starting_batch_size=3)

    indices = strategy.select(dataset)

    assert indices == []


def test_core_set_initial_selection_handles_density_none(monkeypatch):
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [5.0, 5.0],
        ]
    )
    dataset = _dataset_from_embeddings(embeddings)
    strategy = CoreSetInitialSelection(
        seed=0, starting_batch_size=2, density_neighbors=1
    )

    class DummyRNG:
        def integers(self, low, high=None, size=None, dtype=None):
            return low + 1

    strategy._rng = DummyRNG()

    indices = strategy.select(dataset)

    assert indices[0] == 1
    assert len(indices) == 2
    assert len(set(indices)) == 2


def test_density_weighted_core_set_applies_weights(monkeypatch):
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [0.9, 0.0],
        ],
        dtype=float,
    )
    dataset = _dataset_from_embeddings(embeddings)
    strategy = DensityWeightedCoreSetInitialSelection(
        seed=0, starting_batch_size=2, density_scale=1.0
    )

    def fake_density_scores(self, _embeddings):
        return np.array([0.0, 1.0, 2.0], dtype=float)

    def fake_density_weights(self, _scores, num_samples):
        assert num_samples == 3
        return np.array([1.0, 50.0, 1.0], dtype=float)

    monkeypatch.setattr(
        DensityWeightedCoreSetInitialSelection,
        "_estimate_density_scores",
        fake_density_scores,
    )
    monkeypatch.setattr(
        DensityWeightedCoreSetInitialSelection,
        "_build_density_weights",
        fake_density_weights,
    )

    indices = strategy.select(dataset=dataset)

    assert indices == [0, 1]


def test_density_weighted_core_set_runs_multiple_steps():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    dataset = _dataset_from_embeddings(embeddings)
    strategy = DensityWeightedCoreSetInitialSelection(
        seed=0, starting_batch_size=2, density_scale=0.5
    )

    indices = strategy.select(dataset)

    assert len(indices) == 2
    assert len(set(indices)) == 2


def test_density_weighted_build_density_weights_handles_zero_scale():
    strategy = DensityWeightedCoreSetInitialSelection(
        seed=0, starting_batch_size=2, density_scale=0.0
    )
    weights = strategy._build_density_weights(
        density_scores=np.array([0.5, 0.2]), num_samples=2
    )
    assert np.allclose(weights, np.ones(2))


def test_density_weighted_build_density_weights_scales_inverse_density():
    strategy = DensityWeightedCoreSetInitialSelection(
        seed=0, starting_batch_size=2, density_scale=2.0
    )
    weights = strategy._build_density_weights(
        density_scores=np.array([0.5, 1.0]), num_samples=2
    )
    assert weights[0] > weights[1]
    expected = 1.0 + strategy.density_scale * np.array([1.0, 0.5])
    assert np.allclose(weights, expected)


def test_typiclust_initial_selection_returns_unique_indices():
    rng = np.random.default_rng(0)
    embeddings = np.vstack(
        [
            rng.normal(loc=(-2.0, -2.0), scale=0.2, size=(20, 2)),
            rng.normal(loc=(2.0, 2.0), scale=0.2, size=(20, 2)),
        ]
    )
    dataset = _dataset_from_embeddings(embeddings)
    strategy = TypiClustInitialSelection(seed=123, starting_batch_size=8)

    indices = strategy.select(dataset)

    assert len(indices) == 8
    assert len(set(indices)) == 8
    assert all(0 <= idx < len(dataset.sample_ids) for idx in indices)


def test_typiclust_initial_selection_is_deterministic_for_seed():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ],
        dtype=float,
    )
    dataset = _dataset_from_embeddings(embeddings)

    strategy_a = TypiClustInitialSelection(seed=42, starting_batch_size=4)
    strategy_b = TypiClustInitialSelection(seed=42, starting_batch_size=4)

    indices_a = strategy_a.select(dataset)
    indices_b = strategy_b.select(dataset)

    assert indices_a == indices_b


def test_typiclust_initial_selection_caps_to_dataset_size():
    dataset = _dataset_from_embeddings([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    strategy = TypiClustInitialSelection(seed=1, starting_batch_size=10)

    indices = strategy.select(dataset)

    assert len(indices) == 3
    assert len(set(indices)) == 3


def test_typiclust_initial_selection_falls_back_when_clusters_filtered():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.1, 0.0],
            [4.0, 4.0],
            [4.1, 4.0],
            [4.0, 4.1],
            [8.0, 8.0],
            [8.1, 8.0],
        ],
        dtype=float,
    )
    dataset = _dataset_from_embeddings(embeddings)
    strategy = TypiClustInitialSelection(
        seed=7,
        starting_batch_size=5,
        min_cluster_size=1000,
    )

    indices = strategy.select(dataset)

    assert len(indices) == 5
    assert len(set(indices)) == 5
