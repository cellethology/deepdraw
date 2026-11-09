"""NUTS fitting and compiled latent gate predictions."""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .constants import model_metadata
from .fitting import build_model, parameter_layout, predict_symbolic
from .model import Circuit
from .types import Posterior

EXPRESSION_COLUMNS = (
    "Basal Exp (GFP, au)",
    "4-OHT Induced Exp (GFP, au)",
    "GZV Induced Exp (GFP, au)",
    "Dual Induced Exp (GFP, au)",
)
SAMPLER = {
    "chains": 3,
    "cores": 3,
    "draws": 1000,
    "tune": 500,
    "target_accept": 0.65,
    "init": "jitter+adapt_full",
}


@dataclass
class CandidatePool:
    ids: list
    features: list[dict]

    def fingerprint(self):
        value = json.dumps(
            {"ids": self.ids, "features": self.features},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(value.encode()).hexdigest()


@dataclass
class LogicModel:
    pool: CandidatePool
    metadata: dict
    circuits: list[Circuit]
    compiled_scores: object = field(default=None, repr=False)

    @property
    def ndim(self):
        return parameter_layout(self.metadata)["dimension"]

    def compile(self):
        if self.compiled_scores is None:
            import pytensor
            import pytensor.tensor as pt

            theta = pt.dvector("theta")
            values = predict_symbolic(
                theta, self.circuits, self.metadata, return_log=True
            )
            scores = pt.stack(
                (
                    values[:, 3] - pt.max(values[:, :3], axis=-1),
                    pt.min(values[:, 1:], axis=-1) - values[:, 0],
                ),
                axis=-1,
            )
            self.compiled_scores = pytensor.function([theta], scores, mode="FAST_RUN")
        return self.compiled_scores


class CachedGateModel:
    def __init__(self, model, theta, draw_ids, cache, task):
        self.pool = model.pool
        self.ndim = model.ndim
        self.gate = {"AND": 0, "OR": 1}[task]
        self.rows = {
            theta[int(index)].tobytes(): row for row, index in enumerate(draw_ids)
        }
        self.cache = cache

    def log_fold_change(self, z, indices):
        rows = np.asarray(
            [self.rows[np.asarray(t).tobytes()] for t in z], dtype=np.int64
        )
        indices = np.asarray(indices, dtype=np.int64)
        return self.cache[rows[:, None], indices[None, :], self.gate]


def create_model(pool):
    features = [dict(row) for row in pool.circuits]
    candidate_pool = CandidatePool(ids=np.asarray(pool.ids).tolist(), features=features)
    return LogicModel(
        candidate_pool,
        model_metadata(),
        [Circuit(**row) for row in features],
    )


def fit(model, observed, fluorescence, seed, settings=None):
    import pymc as pm

    sampler = dict(SAMPLER)
    if settings:
        sampler.update({k: settings[k] for k in sampler if k in settings})
        if "warmup" in settings:
            sampler["tune"] = settings["warmup"]
    observed = np.asarray(observed, dtype=np.int64)
    fluorescence = np.asarray(fluorescence, dtype=float)
    if fluorescence.shape != (len(observed), 4):
        raise ValueError("Expected four fluorescence conditions per observation")
    if not np.isfinite(fluorescence).all() or np.any(fluorescence <= 0):
        raise ValueError("Fluorescence must be finite and positive")
    with build_model(
        [model.circuits[i] for i in observed], fluorescence, model.metadata
    ):
        inference = pm.sample(
            **sampler,
            random_seed=int(seed),
            progressbar=False,
            compute_convergence_checks=True,
            idata_kwargs={"log_likelihood": False},
        )
    theta = np.asarray(inference.posterior.theta).copy()
    if not np.isfinite(theta).all():
        raise FloatingPointError("Nonfinite posterior draws")
    return Posterior(theta, {"candidate_fingerprint": model.pool.fingerprint()})


def prepare_cache(model, posterior, config, seed, task):
    theta = posterior.flat()
    order = np.random.default_rng(seed).permutation(len(theta))
    draw_ids = order[: config.proposal_draws]
    if len(draw_ids) != config.proposal_draws:
        raise ValueError("Insufficient posterior draws for proposals")
    predictor = model.compile()
    cache = np.stack([predictor(theta[int(i)]) for i in draw_ids])
    if not np.isfinite(cache).all():
        raise FloatingPointError("Nonfinite gate predictions")
    return CachedGateModel(model, theta, draw_ids, cache, task)
