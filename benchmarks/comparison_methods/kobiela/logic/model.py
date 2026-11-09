"""Dual-input thermodynamic fluorescence model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln, logsumexp

MODEL_VARIANT = "rai_dual_printed_sum_no_reporter_spacer_v031"
CONDITIONS = ((0, 0), (1, 0), (0, 1), (1, 1))
CONDITION_LABELS = ("neither", "OHT_only", "GZV_only", "both")


def _count(n: int, name: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(n)


def expand_pattern(pattern: str, total_sites: int) -> str:
    total_sites = _count(total_sites, "total_sites")
    if total_sites < 2 or total_sites % 2:
        raise ValueError("Dual arrays require a positive even total >=2")
    m = total_sites // 2
    patterns = {
        "alternating_AB": "AB" * m,
        "alternating_BA": "BA" * m,
        "clustered_AB": "A" * m + "B" * m,
        "clustered_BA": "B" * m + "A" * m,
    }
    if pattern not in patterns:
        raise ValueError(f"Unknown pattern {pattern!r}; use an explicit named pattern")
    return patterns[pattern]


def motif_geometry(motifs: str, rho: float) -> tuple[float, float, float]:
    if not motifs or set(motifs) - {"A", "B"}:
        raise ValueError("motifs must be a nonempty A/B string, distal to proximal")
    if not np.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("rho must be in (0,1]")
    log_rho = math.log(rho)
    reach = []
    n = len(motifs)
    for name in "AB":
        dist = np.asarray(
            [n - 1 - i for i, c in enumerate(motifs) if c == name], dtype=float
        )
        m = len(dist)
        reach.append(
            0.0
            if m == 0
            else float(
                np.exp(logsumexp(log_rho * dist) - logsumexp(log_rho * np.arange(m)))
            )
        )
    mix = (
        sum((a != b for a, b in zip(motifs, motifs[1:], strict=False))) / (n - 1)
        if n > 1
        else 0.0
    )
    return (reach[0], reach[1], float(mix))


def joint_occupancy(
    log_z_a: ArrayLike, log_z_b: ArrayLike, n_a: int, n_b: int, coupling: ArrayLike
):
    n_a, n_b = (_count(n_a, "n_a"), _count(n_b, "n_b"))
    a, b, j = np.broadcast_arrays(
        np.asarray(log_z_a, dtype=float),
        np.asarray(log_z_b, dtype=float),
        np.asarray(coupling, dtype=float),
    )
    if (
        np.any(np.isnan(a) | np.isposinf(a) | np.isnan(b) | np.isposinf(b))
        or not np.isfinite(j).all()
    ):
        raise ValueError("Activities must be finite or -inf; coupling must be finite")
    k = np.arange(n_a + 1, dtype=float)
    ell = np.arange(n_b + 1, dtype=float)
    lc_a = gammaln(n_a + 1) - gammaln(k + 1) - gammaln(n_a - k + 1)
    lc_b = gammaln(n_b + 1) - gammaln(ell + 1) - gammaln(n_b - ell + 1)
    ta = np.zeros(a.shape + (n_a + 1,))
    tb = np.zeros(b.shape + (n_b + 1,))
    ta[..., 1:] = a[..., None] * k[1:]
    tb[..., 1:] = b[..., None] * ell[1:]
    log_w = (
        lc_a[:, None]
        + lc_b[None, :]
        + ta[..., :, None]
        + tb[..., None, :]
        + j[..., None, None] * k[:, None] * ell[None, :]
    )
    probs = np.exp(log_w - logsumexp(log_w, axis=(-2, -1), keepdims=True))
    pa = np.sum(probs * k[:, None], axis=(-2, -1)) / n_a if n_a else np.zeros(a.shape)
    pb = np.sum(probs * ell[None, :], axis=(-2, -1)) / n_b if n_b else np.zeros(a.shape)
    return (pa, pb, probs)


def printed_response(p: ArrayLike, n: int, h: float) -> np.ndarray:
    n = _count(n, "n")
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any(p < -1e-14) or np.any(p > 1 + 1e-14):
        raise ValueError("p must be in [0,1]")
    if not np.isfinite(h) or h < 0:
        raise ValueError("h must be nonnegative (prototype support choice)")
    if n == 0:
        return np.zeros_like(p)
    p = np.clip(p, 0, 1)
    with np.errstate(divide="ignore"):
        logp = np.log(p)
    k = np.arange(1, n + 1, dtype=float)
    coeff = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + np.log(k / n)
    exponents = k + h * (k >= 2)
    return np.exp(logsumexp(coeff + logp[..., None] * exponents, axis=-1))


@dataclass(frozen=True)
class Parameters:
    log_abundance_a: float
    log_abundance_b_on: float
    log_kd_a: float
    log_kd_b: float
    log_partition_a: float
    alpha_a: float
    leak_b: float
    baseline: float
    amplitude_a: float
    amplitude_b: float
    h_a: float
    h_b: float
    coupling_idp_pair: float
    coupling_mixing: float
    rho: float

    def validate(self):
        if not all(np.isfinite(v) for v in self.__dict__.values()):
            raise ValueError("Parameters must be finite")
        if not (0 < self.alpha_a <= 1 and 0 < self.leak_b <= 1 and (0 < self.rho <= 1)):
            raise ValueError("alpha_a, leak_b and rho must be in (0,1]")
        if (
            self.baseline <= 0
            or min(self.amplitude_a, self.amplitude_b, self.h_a, self.h_b) < 0
        ):
            raise ValueError("baseline>0, amplitudes>=0 and h>=0 required")


def predict(motifs: str, p: Parameters) -> dict:
    p.validate()
    ga, gb, mix = motif_geometry(motifs, p.rho)
    n_a, n_b = (motifs.count("A"), motifs.count("B"))
    c = np.asarray(CONDITIONS, dtype=float)
    log_z_a = (
        p.log_abundance_a
        - p.log_kd_a
        - np.logaddexp(0, p.log_partition_a + c[:, 0] * math.log(p.alpha_a))
    )
    log_z_b = p.log_abundance_b_on - p.log_kd_b + (1 - c[:, 1]) * math.log(p.leak_b)
    j = p.coupling_idp_pair + p.coupling_mixing * mix
    pa, pb, probability = joint_occupancy(log_z_a, log_z_b, n_a, n_b, j)
    fa = p.amplitude_a * ga * printed_response(pa, n_a, p.h_a)
    fb = p.amplitude_b * gb * printed_response(pb, n_b, p.h_b)
    fluorescence = p.baseline + fa + fb
    if not np.isfinite(fluorescence).all():
        raise FloatingPointError("Nonfinite model output")
    return {
        "conditions": CONDITION_LABELS,
        "fluorescence": fluorescence,
        "contribution_a": fa,
        "contribution_b": fb,
        "p_a": pa,
        "p_b": pb,
        "log_activity_a": log_z_a,
        "log_activity_b": log_z_b,
        "coupling": j,
        "reach_a": ga,
        "reach_b": gb,
        "mixing": mix,
        "joint_probabilities": probability,
    }


@dataclass(frozen=True)
class Circuit:
    promoter_a: str
    promoter_b: str
    ad_a: str
    ad_b: str
    idp_a: str
    idp_b: str
    zf_a: str
    zf_b: str
    spacer_a: str
    spacer_b: str
    reporter_spacer: str
    core_promoter: str
    orientation: str
    pattern: str
    total_sites: int


@dataclass(frozen=True)
class PartTables:
    log_production_a: dict[tuple[str, str], float]
    log_production_b: dict[tuple[str, str], float]
    log_degradation_a: dict[str, float]
    log_degradation_b: dict[str, float]
    log_kd_a: dict[str, float]
    log_kd_b: dict[str, float]
    log_partition_a: dict[str, float]
    log_baseline: dict[str, float]
    log_amplitude: dict[tuple[str, str], float]
    h: dict[tuple[str, str], float]
    coupling_idp_pair: dict[tuple[str, str], float]
    alpha_a: float
    leak_b: float
    coupling_mixing: float
    rho: float

    def for_circuit(self, c: Circuit) -> Parameters:
        try:
            return Parameters(
                log_abundance_a=self.log_production_a[c.promoter_a, c.orientation]
                - self.log_degradation_a[c.spacer_a],
                log_abundance_b_on=self.log_production_b[c.promoter_b, c.orientation]
                - self.log_degradation_b[c.spacer_b],
                log_kd_a=self.log_kd_a[c.zf_a],
                log_kd_b=self.log_kd_b[c.zf_b],
                log_partition_a=self.log_partition_a[c.idp_a],
                alpha_a=self.alpha_a,
                leak_b=self.leak_b,
                baseline=math.exp(self.log_baseline[c.core_promoter]),
                amplitude_a=math.exp(self.log_amplitude[c.core_promoter, c.ad_a]),
                amplitude_b=math.exp(self.log_amplitude[c.core_promoter, c.ad_b]),
                h_a=self.h[c.ad_a, c.idp_a],
                h_b=self.h[c.ad_b, c.idp_b],
                coupling_idp_pair=self.coupling_idp_pair[c.idp_a, c.idp_b],
                coupling_mixing=self.coupling_mixing,
                rho=self.rho,
            )
        except KeyError as exc:
            raise KeyError(f"Missing shared part-table entry: {exc.args[0]!r}") from exc

    def predict_circuit(self, c: Circuit) -> dict:
        params = self.for_circuit(c)
        out = predict(expand_pattern(c.pattern, c.total_sites), params)
        out["model_variant"] = MODEL_VARIANT
        out["unmodeled_features"] = ["reporter_spacer"]
        out["reporter_spacer_mapping"] = "not modeled"
        out["effective_baseline"] = params.baseline
        return out
