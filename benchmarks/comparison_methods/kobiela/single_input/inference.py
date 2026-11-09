"""NUTS inference with condition-specific InverseGamma residual scales."""

from dataclasses import dataclass, field

import numpy as np

from ..priors import packed_latent
from .config import Config
from .data import Observations
from .model import RaiModel


@dataclass
class Posterior:
    z: np.ndarray
    log_likelihood: np.ndarray
    metadata: dict
    sampler_stats: dict[str, np.ndarray] = field(default_factory=dict)

    def flat(self):
        return self.z.reshape(-1, self.z.shape[-1])


def build_nuts_model(model: RaiModel, observations: Observations):
    import pymc as pm
    import pytensor.tensor as pt

    observations.validate(model.pool)
    noise = model.by_name["noise"]
    with pm.Model() as probabilistic_model:
        latent = packed_latent(
            "z",
            model.ndim,
            noise.start,
            noise.stop,
            model.priors.log_noise_location,
            model.priors.log_noise_sd,
        )
        mean = model.log_mean(latent[None, :], observations.indices, xp=pt)[0]
        log_sigma = model._parameters(latent[None, :], xp=pt)["log_sigma"][0]
        pm.Normal(
            "observed_log_fluorescence",
            mu=mean,
            sigma=pt.exp(log_sigma),
            observed=observations.log_fluorescence,
        )
    return probabilistic_model


def fit(model: RaiModel, observations: Observations, config: Config, seed: int):
    import pymc as pm

    observations.validate(model.pool)
    config.validate()
    with build_nuts_model(model, observations):
        trace = pm.sample(
            draws=config.draws,
            tune=config.warmup,
            chains=config.chains,
            cores=min(config.chains, config.nuts_cores),
            random_seed=seed,
            target_accept=config.nuts_target_accept,
            init=config.nuts_init,
            return_inferencedata=False,
            compute_convergence_checks=False,
            progressbar=False,
        )
    draws = np.stack([trace.get_values("z", chains=chain) for chain in trace.chains])
    if draws.shape != (config.chains, config.draws, model.ndim):
        raise RuntimeError("Unexpected posterior shape")
    statistics = {
        name: np.stack(
            [trace.get_sampler_stats(name, chains=chain) for chain in trace.chains]
        )
        for name in (
            "diverging",
            "energy",
            "depth",
            "tree_size",
            "step_size_bar",
            "mean_tree_accept",
            "reached_max_treedepth",
            "model_logp",
        )
        if name in trace.stat_names
    }
    likelihood = np.asarray(
        [
            [model.log_likelihood(value, observations) for value in chain]
            for chain in draws
        ]
    )
    return Posterior(
        draws,
        likelihood,
        {"candidate_fingerprint": model.pool.fingerprint()},
        statistics,
    )
