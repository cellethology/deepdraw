"""Merzbacher categorical Hyperopt TPE and batch selection."""

import numpy as np
from hyperopt.tpe import (
    DEFAULT_LF,
    ap_split_trials,
    linear_forgetting_weights,
    tpe_cat_pseudocounts,
)


class CategoricalTPE:
    def __init__(self, cardinalities, gamma=15.0, prior_weight=1.0):
        self.cardinalities = np.asarray(cardinalities, dtype=int)
        if np.any(self.cardinalities < 1) or len(self.cardinalities) == 0:
            raise ValueError("At least one nonempty categorical variable is required")
        if not np.isfinite(gamma) or gamma <= 0 or prior_weight <= 0:
            raise ValueError("Invalid gamma or prior weight")
        self.gamma = gamma
        self.prior_weight = prior_weight

    def fit(self, observed_components, observed_phenotypes):
        x = self._validate(observed_components)
        y = np.asarray(observed_phenotypes, dtype=float)
        if y.shape != (len(x),) or len(x) < 2 or not np.isfinite(y).all():
            raise ValueError("Need at least two finite observed phenotypes")
        tids = np.arange(len(y))
        below, above = ap_split_trials(tids, tids, tids, -y, self.gamma)
        below = np.asarray(below, dtype=int)
        above = np.asarray(above, dtype=int)
        self.below_indices_ = below.copy()
        self.above_indices_ = above.copy()
        self.log_ratios_ = []
        for col, levels in enumerate(self.cardinalities):
            prior = np.full(levels, 1.0 / levels)
            densities = []
            for group in (below, above):
                weights = linear_forgetting_weights(len(group), DEFAULT_LF)
                counts = np.bincount(x[group, col], weights=weights, minlength=levels)
                densities.append(
                    tpe_cat_pseudocounts(counts, self.prior_weight, prior, (1,))
                )
            self.log_ratios_.append(np.log(densities[0]) - np.log(densities[1]))
        return self

    def score(self, candidate_components):
        x = self._validate(candidate_components)
        return sum(ratio[x[:, col]] for col, ratio in enumerate(self.log_ratios_))

    def select(self, candidate_components, batch_size, rng):
        scores = self.score(candidate_components)
        order = np.lexsort((rng.random(len(scores)), -scores))
        return order[:batch_size], scores[order[:batch_size]]

    def _validate(self, components):
        x = np.asarray(components)
        if x.ndim != 2 or x.shape[1] != len(self.cardinalities):
            raise ValueError("Component matrix has the wrong shape")
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("Components must be integer categorical codes")
        if np.any(x < 0) or np.any(x >= self.cardinalities):
            raise ValueError("Component code outside categorical domain")
        return x


def select_batch(pool, observed, available, seed, round_number, tie_rng):
    observed = np.asarray(observed, dtype=int)
    available = np.asarray(available, dtype=int)
    model = CategoricalTPE(np.max(pool.components, axis=0) + 1)
    model.fit(pool.components[observed], pool.scores[observed])
    choice, _ = model.select(
        pool.components[available], min(12, len(available)), tie_rng
    )
    return available[choice]
