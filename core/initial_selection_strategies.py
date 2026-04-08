"""
Initial selection strategies for choosing the seed batch.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

from core.data_loader import Dataset

logger = logging.getLogger(__name__)


class InitialSelectionStrategy(ABC):
    """Interface for selecting the initial labeled pool."""

    def __init__(self, name: str | None = None, starting_batch_size: int = 8) -> None:
        self.name = name or self.__class__.__name__
        self.starting_batch_size = starting_batch_size

    @abstractmethod
    def select(
        self,
        dataset: Dataset,
    ) -> list[int]:
        """Return indices for the initial training pool."""


class RandomInitialSelection(InitialSelectionStrategy):
    """Randomly sample the initial training points."""

    def __init__(self, seed: int, starting_batch_size: int) -> None:
        super().__init__("RANDOM", starting_batch_size=starting_batch_size)
        self.seed = seed

    def select(
        self,
        dataset: Dataset,
    ) -> list[int]:
        rng = np.random.default_rng(self.seed)
        return rng.choice(
            len(dataset.sample_ids), self.starting_batch_size, replace=False
        ).tolist()


class KMeansInitialSelection(InitialSelectionStrategy):
    """Select initial samples using K-means clustering on sequence features."""

    def __init__(self, seed: int, starting_batch_size: int) -> None:
        super().__init__("KMEANS", starting_batch_size=starting_batch_size)
        self.seed = seed

    def select(
        self,
        dataset: Dataset,
    ) -> list[int]:
        selected = self.kmeans_initial_selection(embeddings=dataset.embeddings)

        labels = dataset.labels[selected]
        logger.info(
            "KMEANS_INITIAL: selected %d sequences. Labels: [%s]",
            len(selected),
            ", ".join(f"{val:.3f}" for val in labels),
        )

        return selected

    def kmeans_initial_selection(
        self,
        embeddings: np.ndarray,
    ) -> list[int]:
        kmeans = KMeans(
            n_clusters=self.starting_batch_size, random_state=self.seed
        ).fit(embeddings)
        closest_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, embeddings
        )
        return closest_indices.tolist()


class CoreSetInitialSelection(InitialSelectionStrategy):
    """Select initial samples via k-center greedy (core-set) coverage."""

    def __init__(
        self,
        seed: int,
        starting_batch_size: int,
        metric: str = "cosine",
        density_neighbors: int = 32,
    ) -> None:
        super().__init__("CORESET", starting_batch_size=starting_batch_size)
        self.seed = seed
        self.metric = metric
        self.density_neighbors = max(1, density_neighbors)
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        dataset: Dataset,
    ) -> list[int]:
        selected = self.k_center_greedy(dataset.embeddings)

        labels = dataset.labels[selected]
        logger.info(
            "%s_INITIAL: selected %d sequences. Labels: [%s]",
            self.name.upper(),
            len(selected),
            ", ".join(f"{val:.3f}" for val in labels),
        )

        return selected

    def k_center_greedy(self, embeddings: np.ndarray) -> list[int]:
        num_samples = embeddings.shape[0]
        if num_samples == 0 or self.starting_batch_size == 0:
            return []

        target = min(self.starting_batch_size, num_samples)
        density_scores = self._estimate_density_scores(embeddings)
        first_idx = self._select_initial_index(num_samples, density_scores)
        selected = [first_idx]

        min_distances = pairwise_distances(
            embeddings, embeddings[[first_idx]], metric=self.metric
        ).reshape(-1)

        while len(selected) < target:
            farthest_idx = int(np.argmax(min_distances))
            selected.append(farthest_idx)
            if len(selected) == target:
                break
            new_distances = pairwise_distances(
                embeddings, embeddings[[farthest_idx]], metric=self.metric
            ).reshape(-1)
            min_distances = np.minimum(min_distances, new_distances)

        return selected

    def _estimate_density_scores(self, embeddings: np.ndarray) -> np.ndarray | None:
        """Estimate local density via average distance to k nearest neighbors."""
        num_samples = embeddings.shape[0]
        if num_samples <= 1 or self.density_neighbors <= 1:
            return None
        n_neighbors = min(self.density_neighbors, num_samples)
        try:
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
            nn.fit(embeddings)
            distances, _ = nn.kneighbors(embeddings)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Failed to compute density scores for CORE-SET (metric=%s): %s",
                self.metric,
                exc,
            )
            return None

        if distances.shape[1] > 1:
            local_density = np.mean(distances[:, 1:], axis=1)
        else:
            local_density = distances[:, 0]
        return local_density

    def _select_initial_index(
        self, num_samples: int, density_scores: np.ndarray | None
    ) -> int:
        if density_scores is None or not np.all(np.isfinite(density_scores)):
            return int(self._rng.integers(0, num_samples))
        return int(np.argmin(density_scores))


class DensityWeightedCoreSetInitialSelection(CoreSetInitialSelection):
    """K-center greedy that upweights dense regions when selecting new points."""

    def __init__(
        self,
        seed: int,
        starting_batch_size: int,
        metric: str = "cosine",
        density_neighbors: int = 32,
        density_scale: float = 1.0,
    ) -> None:
        super().__init__(
            seed=seed,
            starting_batch_size=starting_batch_size,
            metric=metric,
            density_neighbors=density_neighbors,
        )
        self.name = "CORESET_DENSITY"
        self.density_scale = max(0.0, density_scale)

    def k_center_greedy(self, embeddings: np.ndarray) -> list[int]:
        num_samples = embeddings.shape[0]
        if num_samples == 0 or self.starting_batch_size == 0:
            return []

        target = min(self.starting_batch_size, num_samples)
        density_scores = self._estimate_density_scores(embeddings)
        weights = self._build_density_weights(density_scores, num_samples)
        first_idx = self._select_initial_index(num_samples, density_scores)
        selected = [first_idx]

        min_distances = pairwise_distances(
            embeddings, embeddings[[first_idx]], metric=self.metric
        ).reshape(-1)

        while len(selected) < target:
            weighted_scores = min_distances * weights
            farthest_idx = int(np.argmax(weighted_scores))
            selected.append(farthest_idx)
            if len(selected) == target:
                break
            new_distances = pairwise_distances(
                embeddings, embeddings[[farthest_idx]], metric=self.metric
            ).reshape(-1)
            min_distances = np.minimum(min_distances, new_distances)

        return selected

    def _build_density_weights(
        self, density_scores: np.ndarray | None, num_samples: int
    ) -> np.ndarray:
        if density_scores is None or self.density_scale == 0.0:
            return np.ones(num_samples)

        eps = 1e-8
        inv_density = 1.0 / (density_scores + eps)
        max_val = np.max(inv_density)
        if max_val > 0:
            inv_density = inv_density / max_val
        return 1.0 + self.density_scale * inv_density


class TypiClustInitialSelection(InitialSelectionStrategy):
    """Select initial samples via TypiClust (cluster + local typicality)."""

    def __init__(
        self,
        seed: int,
        starting_batch_size: int,
        metric: str = "euclidean",
        min_cluster_size: int = 5,
        max_num_clusters: int = 500,
        k_nn: int = 20,
        mini_batch_threshold: int = 50,
        mini_batch_size: int = 5000,
    ) -> None:
        super().__init__("TYPICLUST", starting_batch_size=starting_batch_size)
        self.seed = seed
        self.metric = metric
        self.min_cluster_size = max(1, min_cluster_size)
        self.max_num_clusters = max(1, max_num_clusters)
        self.k_nn = max(1, k_nn)
        self.mini_batch_threshold = max(2, mini_batch_threshold)
        self.mini_batch_size = max(10, mini_batch_size)
        self._rng = np.random.default_rng(seed)

    def select(self, dataset: Dataset) -> list[int]:
        embeddings = np.asarray(dataset.embeddings)
        num_samples = embeddings.shape[0]
        target = min(self.starting_batch_size, num_samples)
        if target == 0:
            return []

        num_clusters = min(target, self.max_num_clusters, num_samples)
        labels = self._cluster_embeddings(embeddings, num_clusters)
        cluster_order = self._build_cluster_order(labels)

        available = np.ones(num_samples, dtype=bool)
        selected: list[int] = []

        for i in range(target):
            cluster = self._next_non_empty_cluster(
                labels=labels,
                available=available,
                cluster_order=cluster_order,
                start_offset=i,
            )
            if cluster is None:
                break

            cluster_indices = np.flatnonzero((labels == cluster) & available)
            cluster_feats = embeddings[cluster_indices]
            num_neighbors = min(self.k_nn, max(1, len(cluster_indices) // 2))
            typicality = self._calculate_typicality(cluster_feats, num_neighbors)
            chosen_idx = int(cluster_indices[int(np.argmax(typicality))])
            selected.append(chosen_idx)
            available[chosen_idx] = False

        if len(selected) < target:
            selected.extend(
                self._sample_fallback_indices(available, target - len(selected))
            )

        labels_selected = dataset.labels[selected]
        logger.info(
            "%s_INITIAL: selected %d sequences. Labels: [%s]",
            self.name.upper(),
            len(selected),
            ", ".join(f"{val:.3f}" for val in labels_selected),
        )
        return selected

    def _cluster_embeddings(
        self, embeddings: np.ndarray, num_clusters: int
    ) -> np.ndarray:
        if num_clusters <= self.mini_batch_threshold:
            model = KMeans(n_clusters=num_clusters, random_state=self.seed, n_init=10)
            return model.fit_predict(embeddings)
        model = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=self.seed,
            batch_size=self.mini_batch_size,
            n_init=3,
        )
        return model.fit_predict(embeddings)

    def _build_cluster_order(self, labels: np.ndarray) -> list[int]:
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_stats = sorted(
            zip(cluster_ids.tolist(), cluster_sizes.tolist(), strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
        filtered = [
            int(cluster_id)
            for cluster_id, cluster_size in cluster_stats
            if cluster_size > self.min_cluster_size
        ]
        if filtered:
            return filtered
        return [int(cluster_id) for cluster_id, _ in cluster_stats]

    def _next_non_empty_cluster(
        self,
        labels: np.ndarray,
        available: np.ndarray,
        cluster_order: list[int],
        start_offset: int,
    ) -> int | None:
        if not cluster_order:
            return None
        n_clusters = len(cluster_order)
        for j in range(n_clusters):
            cluster = cluster_order[(start_offset + j) % n_clusters]
            if np.any((labels == cluster) & available):
                return cluster
        return None

    def _calculate_typicality(
        self, features: np.ndarray, num_neighbors: int
    ) -> np.ndarray:
        num_samples = features.shape[0]
        if num_samples == 0:
            return np.array([], dtype=float)
        if num_samples == 1:
            return np.array([1.0], dtype=float)

        n_neighbors = min(num_samples, max(2, num_neighbors + 1))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        nn.fit(features)
        distances, _ = nn.kneighbors(features)
        mean_distance = distances[:, 1:].mean(axis=1)
        return 1.0 / (mean_distance + 1e-5)

    def _sample_fallback_indices(self, available: np.ndarray, count: int) -> list[int]:
        if count <= 0:
            return []
        remaining = np.flatnonzero(available)
        if remaining.size == 0:
            return []
        if remaining.size <= count:
            return remaining.tolist()
        sampled = self._rng.choice(remaining, size=count, replace=False)
        return sampled.tolist()


class ProbCoverInitialSelection(InitialSelectionStrategy):
    """Select initial samples via ProbCover (delta-neighborhood cover)."""

    def __init__(
        self,
        seed: int,
        starting_batch_size: int,
        delta: float | None = None,
        batch_size: int = 512,
        device: str = "cpu",
        metric: str = "cosine",
        auto_delta: bool = False,
        alpha: float = 0.95,
        kmeans_clusters: int | None = None,
        delta_candidates: int = 25,
        delta_sample_size: int = 1000,
        pair_sample_size: int = 20000,
    ) -> None:
        super().__init__("PROBCOVER", starting_batch_size=starting_batch_size)
        self.seed = seed
        self.delta = delta
        self.batch_size = max(1, batch_size)
        self.device = device
        self.metric = metric
        self.auto_delta = auto_delta
        self.alpha = alpha
        self.kmeans_clusters = kmeans_clusters
        self.delta_candidates = max(5, delta_candidates)
        self.delta_sample_size = max(50, delta_sample_size)
        self.pair_sample_size = max(1000, pair_sample_size)
        self._rng = np.random.default_rng(seed)

    def select(self, dataset: Dataset) -> list[int]:
        embeddings = np.asarray(dataset.embeddings)
        num_samples = embeddings.shape[0]
        target = min(self.starting_batch_size, num_samples)
        if target == 0:
            return []

        if self.auto_delta or self.delta is None:
            self.delta = self._estimate_delta(embeddings)
            logger.info("PROBCOVER: auto-selected delta=%.4f", self.delta)

        x_edges, y_edges = self._construct_graph(embeddings)
        selected = self._greedy_cover(x_edges, y_edges, num_samples, target)

        labels = dataset.labels[selected]
        logger.info(
            "%s_INITIAL: selected %d sequences. Labels: [%s]",
            self.name.upper(),
            len(selected),
            ", ".join(f"{val:.3f}" for val in labels),
        )
        return selected

    def _estimate_delta(self, embeddings: np.ndarray) -> float:
        num_samples = embeddings.shape[0]
        num_clusters = self._resolve_kmeans_clusters(num_samples)
        pseudo_labels = self._compute_pseudo_labels(embeddings, num_clusters)

        sample_size = min(self.delta_sample_size, num_samples)
        sample_idx = self._rng.choice(num_samples, size=sample_size, replace=False)
        sample_emb = embeddings[sample_idx]
        sample_labels = pseudo_labels[sample_idx]

        candidates = self._candidate_deltas(embeddings)
        if not candidates.size:
            return float(self.delta) if self.delta is not None else 0.5
        max_delta = float(candidates[-1])

        nn = NearestNeighbors(radius=max_delta, metric=self.metric)
        nn.fit(embeddings)
        distances, neighbors = nn.radius_neighbors(sample_emb, return_distance=True)

        purity_scores = []
        for delta in candidates:
            purity_sum = 0.0
            count = 0
            for dist, idx, label in zip(
                distances, neighbors, sample_labels, strict=False
            ):
                if dist.size == 0:
                    continue
                within = dist <= delta
                if not np.any(within):
                    continue
                labels = pseudo_labels[idx[within]]
                purity_sum += float(np.mean(labels == label))
                count += 1
            avg_purity = purity_sum / count if count > 0 else 0.0
            purity_scores.append(avg_purity)

        best_delta = None
        for delta, purity in zip(candidates, purity_scores, strict=False):
            if purity >= self.alpha:
                best_delta = float(delta)
        if best_delta is None:
            best_delta = float(candidates[0])
        return best_delta

    def _resolve_kmeans_clusters(self, num_samples: int) -> int:
        if self.kmeans_clusters is not None:
            return max(2, min(self.kmeans_clusters, num_samples))
        return max(2, min(10, num_samples))

    def _compute_pseudo_labels(
        self, embeddings: np.ndarray, num_clusters: int
    ) -> np.ndarray:
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=self.seed,
            batch_size=1024,
        )
        return kmeans.fit_predict(embeddings)

    def _candidate_deltas(self, embeddings: np.ndarray) -> np.ndarray:
        num_samples = embeddings.shape[0]
        pair_count = min(self.pair_sample_size, num_samples * num_samples)
        idx1 = self._rng.integers(0, num_samples, size=pair_count)
        idx2 = self._rng.integers(0, num_samples, size=pair_count)
        mask = idx1 != idx2
        if not np.any(mask):
            return np.array([], dtype=float)
        if self.metric == "cosine":
            left = embeddings[idx1[mask]]
            right = embeddings[idx2[mask]]
            left_norm = np.linalg.norm(left, axis=1, keepdims=True)
            right_norm = np.linalg.norm(right, axis=1, keepdims=True)
            left_unit = left / (left_norm + 1e-12)
            right_unit = right / (right_norm + 1e-12)
            distances = 1.0 - np.sum(left_unit * right_unit, axis=1)
        else:
            diffs = embeddings[idx1[mask]] - embeddings[idx2[mask]]
            distances = np.linalg.norm(diffs, axis=1)
        quantiles = np.linspace(0.1, 0.99, self.delta_candidates)
        candidates = np.unique(np.quantile(distances, quantiles))
        candidates = candidates[candidates > 0]
        return candidates

    def _construct_graph(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_samples = embeddings.shape[0]
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        logger.info(
            "PROBCOVER: constructing graph with delta=%.4f (n=%d, batch=%d)",
            self.delta,
            num_samples,
            self.batch_size,
        )

        if self.device != "cpu":
            try:
                import torch
            except ImportError:
                logger.warning("Torch not available; falling back to CPU distances.")
                return self._construct_graph_cpu(embeddings)
            device = self._resolve_torch_device(torch)
            feats = torch.as_tensor(embeddings, device=device, dtype=torch.float32)
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                cur = feats[start:end]
                dist = torch.cdist(cur, feats)
                mask = dist < self.delta
                x_idx, y_idx = mask.nonzero(as_tuple=True)
                xs.append((x_idx + start).cpu().numpy())
                ys.append(y_idx.cpu().numpy())
        else:
            return self._construct_graph_cpu(embeddings)

        x_edges = np.concatenate(xs) if xs else np.array([], dtype=int)
        y_edges = np.concatenate(ys) if ys else np.array([], dtype=int)
        logger.info("PROBCOVER: graph has %d edges", len(x_edges))
        return x_edges, y_edges

    def _construct_graph_cpu(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        num_samples = embeddings.shape[0]
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for start in range(0, num_samples, self.batch_size):
            end = min(start + self.batch_size, num_samples)
            cur = embeddings[start:end]
            dist = pairwise_distances(cur, embeddings, metric=self.metric)
            mask = dist < self.delta
            x_idx, y_idx = np.nonzero(mask)
            xs.append(x_idx + start)
            ys.append(y_idx)
        x_edges = np.concatenate(xs) if xs else np.array([], dtype=int)
        y_edges = np.concatenate(ys) if ys else np.array([], dtype=int)
        logger.info("PROBCOVER: graph has %d edges", len(x_edges))
        return x_edges, y_edges

    def _resolve_torch_device(self, torch):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _greedy_cover(
        self,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        num_samples: int,
        target: int,
    ) -> list[int]:
        selected: list[int] = []
        covered = np.zeros(num_samples, dtype=bool)
        cur_x = x_edges
        cur_y = y_edges

        for i in range(target):
            if cur_x.size == 0:
                break
            degrees = np.bincount(cur_x, minlength=num_samples)
            cur = int(degrees.argmax())
            selected.append(cur)
            new_covered = cur_y[cur_x == cur]
            covered[new_covered] = True
            keep = ~covered[cur_y]
            cur_x = cur_x[keep]
            cur_y = cur_y[keep]
            coverage = float(np.mean(covered)) if covered.size else 0.0
            logger.info(
                "PROBCOVER: iter=%d max_degree=%d coverage=%.3f",
                i,
                int(degrees.max()) if degrees.size else 0,
                coverage,
            )

        if len(selected) < target:
            remaining = np.setdiff1d(
                np.arange(num_samples), np.asarray(selected), assume_unique=False
            )
            if remaining.size:
                need = target - len(selected)
                extra = (
                    self._rng.choice(
                        remaining,
                        size=min(need, remaining.size),
                        replace=False,
                    )
                    if remaining.size > need
                    else remaining
                )
                selected.extend(int(idx) for idx in extra)
        return selected
