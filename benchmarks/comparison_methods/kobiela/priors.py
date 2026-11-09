"""Condition-specific residual standard deviations for the biochemical models."""

import math

import numpy as np


def packed_latent(
    name,
    dimension,
    noise_start,
    noise_stop,
    location,
    scale,
    latent_dims=None,
    condition_dims=None,
):
    import pymc as pm
    import pytensor.tensor as pt

    if not (
        0 <= noise_start < noise_stop <= dimension
        and math.isfinite(location)
        and math.isfinite(scale)
        and scale > 0
    ):
        raise ValueError("Invalid residual parameter layout")
    count = noise_stop - noise_start
    nonnoise = pm.Normal(name + "_nonnoise", mu=0.0, sigma=1.0, shape=dimension - count)
    sigma = pm.InverseGamma(
        "sigma_log",
        alpha=np.float64(2.0),
        beta=np.float64(3.0),
        shape=count,
        dims=condition_dims,
    )
    encoded = (pt.log(sigma) - location) / scale
    packed = pt.concatenate([nonnoise[:noise_start], encoded, nonnoise[noise_start:]])
    return pm.Deterministic(name, packed, dims=latent_dims)
