"""Bayesian thermodynamic optimization for OR and AND circuits."""

from dataclasses import fields

import numpy as np

from .acquisition import select_batch as select_posterior_batch
from .inference import EXPRESSION_COLUMNS, create_model, fit, prepare_cache
from .types import Config


def select_batch(pool, observed, available, seed, round_number, tie_rng, settings=None):
    settings = {} if settings is None else settings
    model = settings.get("model")
    if model is None:
        model = getattr(pool, "_kobiela_logic_model", None)
    if model is None:
        model = create_model(pool)
        try:
            pool._kobiela_logic_model = model
        except (AttributeError, TypeError):
            pass
    config = Config(
        **{f.name: settings[f.name] for f in fields(Config) if f.name in settings}
    ).validate()
    observed = np.asarray(observed, dtype=np.int64)
    available = np.asarray(available, dtype=np.int64)
    expected = np.flatnonzero(~np.isin(np.arange(len(pool.ids)), observed))
    if not np.array_equal(available, expected):
        raise ValueError("Available candidates must align with the observed pool")
    inference_seed = int(settings.get("inference_seed", seed))
    selection_seed = int(settings.get("selection_seed", seed))
    fluorescence = pool.features.iloc[observed][list(EXPRESSION_COLUMNS)].to_numpy(
        dtype=float
    )
    posterior = fit(model, observed, fluorescence, inference_seed, settings)
    cached_model = prepare_cache(model, posterior, config, selection_seed, pool.task)
    return select_posterior_batch(
        cached_model, posterior, observed, config, selection_seed
    ).indices


__all__ = ["Config", "create_model", "fit", "select_batch"]
