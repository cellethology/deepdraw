"""Bayesian parameterization and fluorescence likelihood."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from scipy.special import expit, logsumexp

from .contrasts import _scaled_contrasts
from .model import MODEL_VARIANT, Circuit, PartTables, expand_pattern

CONDITION_CODES = ("00", "10", "01", "11")
INPUTS = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), dtype=float)
DEFAULT_PRIORS = {
    "production_sd": 1.0,
    "degradation_sd": 1.0,
    "kd_effect_sd": 1.0,
    "kd_intercept_mean": 0.0,
    "kd_intercept_sd": 2.0,
    "partition_mean": 0.0,
    "partition_sd": 2.0,
    "alpha_logit_mean": -2.0,
    "alpha_logit_sd": 2.0,
    "leak_logit_mean": -2.0,
    "leak_logit_sd": 2.0,
    "log_h_mean": 0.0,
    "log_h_sd": 1.0,
    "log_baseline_mean": math.log(1000.0),
    "log_baseline_sd": 2.0,
    "log_amplitude_mean": math.log(1000.0),
    "log_amplitude_sd": 2.0,
    "coupling_sd": 0.1,
    "mixing_sd": 0.1,
    "rho_logit_mean": 2.0,
    "rho_logit_sd": 1.0,
    "log_sigma_mean": math.log(0.5),
    "log_sigma_sd": 1.0,
}
FEATURE_FIELDS = tuple(f.name for f in dataclasses.fields(Circuit))


def as_circuits(circuits: Sequence) -> list[Circuit]:
    result = []
    for c in circuits:
        if isinstance(c, Circuit):
            result.append(c)
        elif isinstance(c, dict):
            result.append(Circuit(**{name: c[name] for name in FEATURE_FIELDS}))
        else:
            raise TypeError(
                "Circuits must be Circuit instances or feature dictionaries"
            )
    if not result:
        raise ValueError("At least one circuit is required")
    for c in result:
        expand_pattern(c.pattern, c.total_sites)
        for name in FEATURE_FIELDS:
            if name != "total_sites" and (not isinstance(getattr(c, name), str)):
                raise TypeError(f"{name} labels must be explicitly decoded strings")
    return result


def build_metadata(circuits: Sequence, priors: dict | None = None) -> dict:
    cs = as_circuits(circuits)
    p = dict(DEFAULT_PRIORS)
    if priors is not None:
        if set(priors) - set(p):
            raise ValueError(f"Unknown prior keys: {set(priors) - set(p)}")
        p.update(priors)
    if any(not np.isfinite(v) for v in p.values()):
        raise ValueError("Prior values must be finite")
    if any((v <= 0 for k, v in p.items() if k.endswith("_sd"))):
        raise ValueError("Prior SD values must be positive")

    def levels(*fields):
        return sorted({getattr(c, f) for c in cs for f in fields})

    vocab = {
        name: levels(name)
        for name in FEATURE_FIELDS
        if name not in ("pattern", "total_sites", "reporter_spacer")
    }
    vocab["ad"] = levels("ad_a", "ad_b")
    vocab["idp"] = levels("idp_a", "idp_b")
    return {
        "adapter_version": "dual_input",
        "model_variant": MODEL_VARIANT,
        "condition_codes": list(CONDITION_CODES),
        "vocabulary_source": "candidate_features_only",
        "feature_fields": list(FEATURE_FIELDS),
        "vocab": vocab,
        "priors": p,
        "patterns": levels("pattern"),
        "total_sites": sorted({c.total_sites for c in cs}),
        "reporter_treatment": "Ignored identity field; no reporter parameter block, degradation or orientation effect",
        "reporter_spacer_levels_audited": levels("reporter_spacer"),
        "beta_fixed": 0.0,
        "alpha0_fixed": 1.0,
        "baseline_policy": "Original effective B_core absorbs fluorescence scale/background; no independent offset; fresh prior, no old posterior",
        "likelihood": "Independent Normal on natural-log raw fluorescence; one sigma per condition",
        "reference_constraints": {
            "production": "Separate A/B flattened promoter x orientation zero-sum tables",
            "degradation": "Separate A/B spacer zero-sum tables",
            "kd": "Separate A/B ZF zero-sum effects plus free arm-specific log Kd intercept",
        },
        "phenotypes_used_for_metadata": False,
        "inherited_posterior_used": False,
    }


def parameter_layout(metadata: dict) -> dict:
    v = metadata["vocab"]
    o = len(v["orientation"])
    shapes = {
        "production_a": (len(v["promoter_a"]) * o - 1,),
        "production_b": (len(v["promoter_b"]) * o - 1,),
        "degradation_a": (len(v["spacer_a"]) - 1,),
        "degradation_b": (len(v["spacer_b"]) - 1,),
        "kd_a": (len(v["zf_a"]) - 1,),
        "kd_b": (len(v["zf_b"]) - 1,),
        "kd_intercept": (2,),
        "partition_a": (len(v["idp_a"]),),
        "alpha": (1,),
        "leak": (1,),
        "h": (len(v["ad"]), len(v["idp"])),
        "baseline": (len(v["core_promoter"]),),
        "amplitude": (len(v["core_promoter"]), len(v["ad"])),
        "coupling": (len(v["idp_a"]), len(v["idp_b"])),
        "mixing": (1,),
        "rho": (1,),
        "sigma": (4,),
    }
    blocks = {}
    offset = 0
    for name, shape in shapes.items():
        size = math.prod(shape)
        blocks[name] = {"start": offset, "stop": offset + size, "shape": list(shape)}
        offset += size
    return {"dimension": offset, "blocks": blocks}


layout = parameter_layout


def transform(theta, metadata: dict, symbolic: bool = False) -> dict:
    v, p = (metadata["vocab"], metadata["priors"])
    lay = parameter_layout(metadata)
    if symbolic:
        import pytensor.tensor as pt

        xp = pt
    else:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (lay["dimension"],) or not np.isfinite(theta).all():
            raise ValueError(
                f"Expected {lay['dimension']} finite standard-normal coordinates"
            )
        xp = np

    def z(name):
        b = lay["blocks"][name]
        return theta[b["start"] : b["stop"]].reshape(tuple(b["shape"]))

    def centered(name, n, sd):
        if n == 1:
            return xp.zeros((1,), dtype="float64")
        return sd * (
            xp.as_tensor_variable(_scaled_contrasts(n)) @ z(name)
            if symbolic
            else _scaled_contrasts(n) @ z(name)
        )

    def logistic(x):
        return xp.sigmoid(x) if symbolic else expit(x)

    o = len(v["orientation"])
    prod_a = centered("production_a", len(v["promoter_a"]) * o, p["production_sd"])
    prod_b = centered("production_b", len(v["promoter_b"]) * o, p["production_sd"])
    deg_a = centered("degradation_a", len(v["spacer_a"]), p["degradation_sd"])
    deg_b = centered("degradation_b", len(v["spacer_b"]), p["degradation_sd"])
    kd0 = p["kd_intercept_mean"] + p["kd_intercept_sd"] * z("kd_intercept")
    kd_a = kd0[0] + centered("kd_a", len(v["zf_a"]), p["kd_effect_sd"])
    kd_b = kd0[1] + centered("kd_b", len(v["zf_b"]), p["kd_effect_sd"])

    def log_logistic(x):
        return -xp.softplus(-x) if symbolic else -np.logaddexp(0.0, -x)

    out = {
        "log_production_a": prod_a.reshape((len(v["promoter_a"]), o)),
        "log_production_b": prod_b.reshape((len(v["promoter_b"]), o)),
        "log_degradation_a": deg_a,
        "log_degradation_b": deg_b,
        "log_kd_a": kd_a,
        "log_kd_b": kd_b,
        "log_partition_a": p["partition_mean"] + p["partition_sd"] * z("partition_a"),
        "log_alpha_a": log_logistic(
            p["alpha_logit_mean"] + p["alpha_logit_sd"] * z("alpha")[0]
        ),
        "log_leak_b": log_logistic(
            p["leak_logit_mean"] + p["leak_logit_sd"] * z("leak")[0]
        ),
        "h": xp.exp(p["log_h_mean"] + p["log_h_sd"] * z("h")),
        "log_baseline": p["log_baseline_mean"] + p["log_baseline_sd"] * z("baseline"),
        "log_amplitude": p["log_amplitude_mean"]
        + p["log_amplitude_sd"] * z("amplitude"),
        "coupling": p["coupling_sd"] * z("coupling"),
        "mixing": p["mixing_sd"] * z("mixing")[0],
        "log_rho": log_logistic(p["rho_logit_mean"] + p["rho_logit_sd"] * z("rho")[0]),
        "log_sigma": p["log_sigma_mean"] + p["log_sigma_sd"] * z("sigma"),
    }
    return out


def _indices(circuits, metadata):
    cs, v = (as_circuits(circuits), metadata["vocab"])
    maps = {name: {x: i for i, x in enumerate(values)} for name, values in v.items()}
    result = {}
    for name in v:
        if name in ("ad", "idp"):
            continue
        try:
            result[name] = np.asarray(
                [maps[name][getattr(c, name)] for c in cs], dtype=np.int64
            )
        except KeyError as e:
            raise KeyError(
                f"Circuit category absent from frozen {name} vocabulary: {e.args[0]}"
            ) from e
    for family in ("ad", "idp"):
        for arm in ("a", "b"):
            result[f"{family}_{arm}_shared"] = np.asarray(
                [maps[family][getattr(c, f"{family}_{arm}")] for c in cs],
                dtype=np.int64,
            )
    return (cs, result)


def _predict(theta, circuits, metadata, symbolic, return_log):
    cs, ix = _indices(circuits, metadata)
    t = transform(theta, metadata, symbolic)
    if symbolic:
        import pytensor.tensor as pt

        xp = pt
        from pytensor.gradient import disconnected_grad

        def lse(x, axis=-1, keepdims=False):
            maximum = disconnected_grad(pt.max(x, axis=axis, keepdims=True))
            shifted = pt.exp(x - maximum)
            reduced = maximum + pt.log(pt.sum(shifted, axis=axis, keepdims=True))
            return reduced if keepdims else pt.squeeze(reduced, axis=axis)
    else:
        xp = np
        lse = logsumexp
    oa = ix["orientation"]
    logsa = (
        t["log_production_a"][ix["promoter_a"], oa]
        - t["log_degradation_a"][ix["spacer_a"]]
    )
    logsb = (
        t["log_production_b"][ix["promoter_b"], oa]
        - t["log_degradation_b"][ix["spacer_b"]]
    )
    partition = (
        t["log_partition_a"][ix["idp_a"], None] + INPUTS[None, :, 0] * t["log_alpha_a"]
    )
    log_partition_denominator = (
        xp.softplus(partition) if symbolic else np.logaddexp(0.0, partition)
    )
    la = logsa[:, None] - t["log_kd_a"][ix["zf_a"], None] - log_partition_denominator
    lb = (
        logsb[:, None]
        - t["log_kd_b"][ix["zf_b"], None]
        + (1.0 - INPUTS[None, :, 1]) * t["log_leak_b"]
    )
    ha = t["h"][ix["ad_a_shared"], ix["idp_a_shared"]]
    hb = t["h"][ix["ad_b_shared"], ix["idp_b_shared"]]
    aa = t["log_amplitude"][ix["core_promoter"], ix["ad_a_shared"]]
    ab = t["log_amplitude"][ix["core_promoter"], ix["ad_b_shared"]]
    base = t["log_baseline"][ix["core_promoter"]]
    result = xp.zeros((len(cs), 4), dtype="float64")
    for total in sorted({c.total_sites for c in cs}):
        group = np.asarray(
            [i for i, c in enumerate(cs) if c.total_sites == total], dtype=np.int64
        )
        m = total // 2
        k, ell = (np.arange(m + 1, dtype=float), np.arange(m + 1, dtype=float))
        lc = np.asarray([math.log(math.comb(m, int(x))) for x in k])
        motifs = [expand_pattern(cs[i].pattern, total) for i in group]
        dista = np.asarray(
            [[total - 1 - j for j, x in enumerate(s) if x == "A"] for s in motifs]
        )
        distb = np.asarray(
            [[total - 1 - j for j, x in enumerate(s) if x == "B"] for s in motifs]
        )
        mix = np.asarray(
            [
                sum((a != b for a, b in zip(s, s[1:], strict=False))) / (total - 1)
                for s in motifs
            ]
        )
        den = lse(t["log_rho"] * np.arange(m))
        lga = lse(t["log_rho"] * dista, axis=1) - den
        lgb = lse(t["log_rho"] * distb, axis=1) - den
        j = t["coupling"][ix["idp_a"][group], ix["idp_b"][group]] + t["mixing"] * mix
        lw = (
            lc[None, None, :, None]
            + lc[None, None, None, :]
            + la[group, :, None, None] * k[None, None, :, None]
            + lb[group, :, None, None] * ell[None, None, None, :]
            + j[:, None, None, None] * (k[:, None] * ell[None, :])[None, None, :, :]
        )
        norm = lse(lw.reshape((len(group), 4, (m + 1) ** 2)), axis=-1)
        terms_a = lw[:, :, 1:, :] + np.log(k[1:] / m)[None, None, :, None]
        terms_b = lw[:, :, :, 1:] + np.log(ell[1:] / m)[None, None, None, :]
        logpa = lse(terms_a.reshape((len(group), 4, m * (m + 1))), axis=-1) - norm
        logpb = lse(terms_b.reshape((len(group), 4, m * (m + 1))), axis=-1) - norm

        def log_h(lp, h, m=m):
            if m == 1:
                return lp
            kk = np.arange(2, m + 1, dtype=float)
            coeff = np.asarray([math.log(math.comb(m, int(x)) * x / m) for x in kk])
            more = coeff[None, None, :] + lp[:, :, None] * (
                kk[None, None, :] + h[:, None, None]
            )
            return lse(xp.concatenate((lp[:, :, None], more), axis=-1), axis=-1)

        a_contrib = aa[group, None] + lga[:, None] + log_h(logpa, ha[group])
        b_contrib = ab[group, None] + lgb[:, None] + log_h(logpb, hb[group])
        bcast_base = base[group, None] + xp.zeros((len(group), 4), dtype="float64")
        output = lse(xp.stack((bcast_base, a_contrib, b_contrib), axis=-1), axis=-1)
        if symbolic:
            result = xp.set_subtensor(result[group], output)
        else:
            result[group] = output
    return result if return_log else xp.exp(result)


def predict_numpy(theta, circuits, metadata, return_log=False):
    return _predict(theta, circuits, metadata, False, return_log)


def predict_symbolic(theta, circuits, metadata, return_log=False):
    return _predict(theta, circuits, metadata, True, return_log)


def build_model(circuits, observed, metadata):
    import pymc as pm

    observed = np.asarray(observed, dtype=float)
    cs = as_circuits(circuits)
    if (
        observed.shape != (len(cs), 4)
        or not np.isfinite(observed).all()
        or np.any(observed <= 0)
    ):
        raise ValueError(
            "Observed training fluorescence must be positive finite circuit x 4"
        )
    lay = parameter_layout(metadata)
    with pm.Model(
        coords={
            "condition": CONDITION_CODES,
            "circuit": np.arange(len(cs)),
            "latent": np.arange(lay["dimension"]),
        }
    ) as model:
        from ..priors import packed_latent

        noise = lay["blocks"]["sigma"]
        assert noise["stop"] - noise["start"] == 4
        theta = packed_latent(
            "theta",
            lay["dimension"],
            noise["start"],
            noise["stop"],
            metadata["priors"]["log_sigma_mean"],
            metadata["priors"]["log_sigma_sd"],
            latent_dims="latent",
            condition_dims="condition",
        )
        tables = transform(theta, metadata, symbolic=True)
        mean_log = predict_symbolic(theta, cs, metadata, return_log=True)
        pm.Deterministic(
            "mean_log_fluorescence", mean_log, dims=("circuit", "condition")
        )
        sigma = pm.Deterministic(
            "sigma", pm.math.exp(tables["log_sigma"]), dims="condition"
        )
        pm.Normal(
            "log_fluorescence",
            mu=mean_log,
            sigma=sigma[None, :],
            observed=np.log(observed),
            dims=("circuit", "condition"),
        )
    return model


def to_part_tables(theta, metadata) -> PartTables:
    t, v = (transform(theta, metadata), metadata["vocab"])

    def table1(key, values):
        return dict(zip(values, map(float, t[key]), strict=False))

    def table2(key, rows, columns):
        return {
            (r, c): float(t[key][i, j])
            for i, r in enumerate(rows)
            for j, c in enumerate(columns)
        }

    return PartTables(
        log_production_a=table2("log_production_a", v["promoter_a"], v["orientation"]),
        log_production_b=table2("log_production_b", v["promoter_b"], v["orientation"]),
        log_degradation_a=table1("log_degradation_a", v["spacer_a"]),
        log_degradation_b=table1("log_degradation_b", v["spacer_b"]),
        log_kd_a=table1("log_kd_a", v["zf_a"]),
        log_kd_b=table1("log_kd_b", v["zf_b"]),
        log_partition_a=table1("log_partition_a", v["idp_a"]),
        log_baseline=table1("log_baseline", v["core_promoter"]),
        log_amplitude=table2("log_amplitude", v["core_promoter"], v["ad"]),
        h=table2("h", v["ad"], v["idp"]),
        coupling_idp_pair=table2("coupling", v["idp_a"], v["idp_b"]),
        alpha_a=float(np.exp(t["log_alpha_a"])),
        leak_b=float(np.exp(t["log_leak_b"])),
        coupling_mixing=float(t["mixing"]),
        rho=float(np.exp(t["log_rho"])),
    )
