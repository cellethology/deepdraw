"""Single-input thermodynamic fluorescence model."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import helmert
from scipy.special import gammaln, logsumexp

from .config import Priors
from .data import CandidatePool, Observations


@dataclass(frozen=True)
class Block:
    name: str
    start: int
    stop: int

    @property
    def slice(self):
        return slice(self.start, self.stop)


def contrast_basis(n: int) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 0))
    return helmert(n, full=False).T * np.sqrt(n / (n - 1))


def nuclear_fraction(log_ratio, log_induction_multiplier=0.0):
    return np.exp(-np.logaddexp(0.0, np.asarray(log_ratio) + log_induction_multiplier))


def printed_activation(p, n: int, cooperativity):
    from math import comb

    p = np.asarray(p, dtype=float)
    if n < 1 or np.any(p < 0) or np.any(p > 1) or np.any(np.asarray(cooperativity) < 0):
        raise ValueError("Invalid occupancy, binding count, or cooperativity")
    result = p.copy()
    for k in range(2, n + 1):
        result = result + comb(n, k) * (k / n) * p ** (k + cooperativity)
    return result


class RaiModel:
    def __init__(self, pool: CandidatePool, priors: Priors | None = None):
        self.pool = pool
        self.priors = priors or Priors()
        self.counts = {k: len(v) for k, v in pool.labels.items()}
        d = self.counts
        self.table_sizes = {
            "production": d["syn_promoter"] * d["orientation"],
            "degradation": d["syn_terminator"] * d["syn_spacer"],
            "reporter_degradation": d["reporter_spacer"],
            "kd_effect": d["zf"],
        }
        sizes = {
            "production": self.table_sizes["production"] - 1,
            "degradation": self.table_sizes["degradation"] - 1,
            "reporter_degradation": self.table_sizes["reporter_degradation"] - 1,
            "kd_location": 1,
            "kd_effect": d["zf"] - 1,
            "partition": d["idp"],
            "induction": 1,
            "baseline": d["core_promoter"],
            "amplitude": d["core_promoter"] * d["ad"],
            "cooperativity": d["ad"] * d["idp"],
            "noise": 2,
        }
        self.blocks = []
        self.by_name = {}
        start = 0
        for name, size in sizes.items():
            block = Block(name, start, start + size)
            self.by_name[name] = block
            if size:
                self.blocks.append(block)
            start += size
        self.ndim = start
        self.bases = {k: contrast_basis(n) for k, n in self.table_sizes.items()}
        c = pool.codes
        self.indices = {
            "production": c["syn_promoter"] * d["orientation"] + c["orientation"],
            "degradation": c["syn_terminator"] * d["syn_spacer"] + c["syn_spacer"],
            "reporter_degradation": c["reporter_spacer"],
            "kd_effect": c["zf"],
            "partition": c["idp"],
            "baseline": c["core_promoter"],
            "amplitude": c["core_promoter"] * d["ad"] + c["ad"],
            "cooperativity": c["ad"] * d["idp"] + c["idp"],
        }
        n = pool.binding_sites[:, None]
        self.k = np.arange(1, int(n.max()) + 1, dtype=float)
        k = self.k[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            coeff = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + np.log(k / n)
        self.log_coeff = np.where(k <= n, coeff, -np.inf)

    def _parameters(self, z, xp=np):
        p = self.priors

        def take(name):
            return z[:, self.by_name[name].slice]

        effect = {}
        for name, basis in self.bases.items():
            if basis.shape[1]:
                effect[name] = p.part_scale * xp.dot(take(name), basis.T)
            else:
                effect[name] = xp.zeros((z.shape[0], 1))
        return {
            **effect,
            "kd_location": p.kd_location_sd * take("kd_location"),
            "partition": p.partition_location + p.partition_sd * take("partition"),
            "log_alpha": -xp.logaddexp(
                0.0,
                -(
                    p.induction_logit_location
                    + p.induction_logit_sd * take("induction")
                ),
            ),
            "baseline": p.log_baseline_location + p.log_baseline_sd * take("baseline"),
            "amplitude": p.log_amplitude_location
            + p.log_amplitude_sd * take("amplitude"),
            "cooperativity": xp.exp(
                p.log_cooperativity_location
                + p.log_cooperativity_sd * take("cooperativity")
            ),
            "log_sigma": p.log_noise_location + p.log_noise_sd * take("noise"),
        }

    def log_mean(self, z, indices=None, xp=np):
        if xp is np:
            z = np.atleast_2d(np.asarray(z, dtype=float))
            if z.shape[1] != self.ndim or not np.all(np.isfinite(z)):
                raise ValueError("Wrong latent shape or nonfinite parameters")
        idx = (
            np.arange(len(self.pool.ids))
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        params = self._parameters(z, xp)

        def pick(key):
            return params[key][:, self.indices[key][idx]]

        log_s = pick("production") - pick("degradation")
        log_kd = params["kd_location"][:, 0][:, None] + pick("kd_effect")
        log_ratio = pick("partition")
        conditions = np.asarray([0.0, 1.0])
        log_nuclear = log_s[:, :, None] - xp.logaddexp(
            0.0,
            log_ratio[:, :, None]
            + params["log_alpha"][:, 0][:, None, None] * conditions,
        )
        log_p = -xp.logaddexp(0.0, log_kd[:, :, None] - log_nuclear)
        coop = pick("cooperativity")
        exponent = self.k + coop[:, :, None] * (self.k >= 2)
        log_terms = (
            self.log_coeff[idx][None, :, None, :]
            + exponent[:, :, None, :] * log_p[:, :, :, None]
        )
        log_g = (
            logsumexp(log_terms, axis=-1)
            if xp is np
            else xp.logsumexp(log_terms, axis=-1)
        )
        return (
            xp.logaddexp(
                pick("baseline")[:, :, None], pick("amplitude")[:, :, None] + log_g
            )
            - pick("reporter_degradation")[:, :, None]
        )

    def log_fold_change(self, z, indices=None):
        pred = self.log_mean(z, indices)
        return pred[:, :, 1] - pred[:, :, 0]

    def log_likelihood(self, z, observations: Observations):
        with np.errstate(
            over="ignore", under="ignore", invalid="ignore", divide="ignore"
        ):
            mu = self.log_mean(z, observations.indices)[0]
            log_sigma = self._parameters(np.asarray(z)[None, :])["log_sigma"][0]
            residual = (observations.log_fluorescence - mu) * np.exp(-log_sigma)
            val = -0.5 * np.sum(residual**2 + 2 * log_sigma + np.log(2 * np.pi))
        return float(val) if np.isfinite(val) else -np.inf
