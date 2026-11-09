"""Kobiela-style Bayesian optimization of single-input circuits."""

from functools import lru_cache

import numpy as np

from .config import Config, Priors
from .data import CandidatePool, Observations
from .inference import fit
from .model import RaiModel
from .selection import select_batch as select_posterior_batch

PARTS = {
    "SynTF Prom.": "syn_promoter",
    "AD": "ad",
    "IDR": "idp",
    "ZF Aff": "zf",
    "pMin": "core_promoter",
    "Rep. Term. Sp.": "reporter_spacer",
    "Orientation": "orientation",
}


@lru_cache(maxsize=3)
def create_model(pool, priors=None):
    if pool.task != "IS":
        raise ValueError("The single-input model requires an IS pool")
    names = pool.component_names
    rows = []
    for circuit_id, values in zip(
        pool.ids, pool.features.to_dict("records"), strict=True
    ):
        row = {"circuit_id": str(int(circuit_id))}
        for column, target in PARTS.items():
            row[target] = names[column][str(int(values[column]))]
        combined = int(values["SynTF Term & Sp."]) - 1
        row["syn_terminator"] = names["SynTF Term"][str(combined // 3 + 1)]
        row["syn_spacer"] = names["Syn. Term. Sp."][str(combined % 3 + 1)]
        row["binding_sites"] = names["No. of BS"][str(int(values["No. of BS"]))]
        rows.append(row)
    return RaiModel(CandidatePool.from_rows(rows), priors)


def fit_and_select(
    model,
    observed_indices,
    raw_fluorescence,
    inference_seed,
    selection_seed,
    config=None,
):
    config = config or Config()
    values = np.asarray(raw_fluorescence, dtype=float)
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError("Fluorescence must be positive and finite")
    observations = Observations(
        np.asarray(observed_indices, dtype=np.int64), np.log(values)
    ).validate(model.pool)
    posterior = fit(model, observations, config, inference_seed)
    return select_posterior_batch(
        model, posterior, observations.indices, config, selection_seed
    )


def select_batch(pool, observed, available, seed, round_number, tie_rng, settings=None):
    settings = {} if settings is None else dict(settings)
    inference_seed = settings.pop("inference_seed", seed)
    selection_seed = settings.pop("selection_seed", seed)
    if isinstance(settings.get("priors"), dict):
        settings["priors"] = Priors(**settings["priors"])
    config = Config(**settings).validate()
    model = create_model(pool, config.priors)
    raw = pool.features.iloc[observed][
        ["Basal Exp (GFP, au)", "Induced Exp (GFP, au)"]
    ].to_numpy(dtype=float)
    result = fit_and_select(
        model, observed, raw, inference_seed, selection_seed, config
    )
    if not np.isin(result.indices, available).all():
        raise ValueError("Selection contains an unavailable circuit")
    return result.indices


__all__ = ["Config", "Priors", "create_model", "fit_and_select", "select_batch"]
