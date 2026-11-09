"""Thompson proposals and posterior-loss batch selection."""

from dataclasses import dataclass

import numpy as np

from ..selection import final_selection, occurrence_counts
from .config import Config
from .inference import Posterior
from .model import RaiModel


@dataclass
class Selection:
    indices: np.ndarray
    median_loss: np.ndarray
    upper_loss: np.ndarray
    audit: dict
    proposal_indices: np.ndarray | None = None
    proposal_draw_indices: np.ndarray | None = None
    risk_draw_indices: np.ndarray | None = None
    proposal_losses: np.ndarray | None = None
    proposal_median_loss: np.ndarray | None = None
    proposal_upper_loss: np.ndarray | None = None
    tie_keys: np.ndarray | None = None


def _thompson_rankings(model, z, available, depth, rng, cfg):
    rankings = []
    for s in range(0, len(z), cfg.posterior_chunk_size):
        zz = z[s : s + cfg.posterior_chunk_size]
        scores = np.full((len(zz), depth), -np.inf)
        ties = np.full_like(scores, -np.inf)
        ids = np.full(scores.shape, -1, dtype=np.int64)
        for start in range(0, len(available), cfg.candidate_chunk_size):
            idx = available[start : start + cfg.candidate_chunk_size]
            pred = model.log_fold_change(zz, idx)
            if not np.all(np.isfinite(pred)):
                raise FloatingPointError(
                    "Nonfinite candidate predictions; refusing arbitrary selection"
                )
            both_scores = np.concatenate([scores, pred], axis=1)
            both_ties = np.concatenate([ties, rng.random(pred.shape)], axis=1)
            both_ids = np.concatenate([ids, np.broadcast_to(idx, pred.shape)], axis=1)
            order = np.lexsort((-both_ties, -both_scores), axis=1)[:, :depth]
            scores = np.take_along_axis(both_scores, order, axis=1)
            ties = np.take_along_axis(both_ties, order, axis=1)
            ids = np.take_along_axis(both_ids, order, axis=1)
        rankings.append(ids)
    return np.concatenate(rankings, axis=0)


def select_batch(
    model: RaiModel, posterior: Posterior, observed_indices, cfg: Config, seed: int
):
    cfg.validate()
    fingerprint = posterior.metadata.get("candidate_fingerprint")
    if fingerprint != model.pool.fingerprint():
        raise ValueError("Posterior and candidate pool fingerprints differ")
    old = np.asarray(observed_indices, dtype=np.int64)
    if (
        old.ndim != 1
        or len(np.unique(old)) != len(old)
        or np.any(old < 0)
        or np.any(old >= len(model.pool.ids))
    ):
        raise ValueError("Invalid observed indices")
    mask = np.ones(len(model.pool.ids), dtype=bool)
    mask[old] = False
    available = np.flatnonzero(mask)
    if not len(available):
        raise ValueError("No unmeasured candidates remain")
    size = min(cfg.batch_size, len(available))
    rng = np.random.default_rng(seed)
    samples = posterior.flat()
    if samples.shape[1] != model.ndim or not np.all(np.isfinite(samples)):
        raise ValueError("Invalid posterior")
    order = rng.permutation(len(samples))
    proposal_ids = order[: min(cfg.proposal_draws, len(samples))]
    if cfg.proposal_risk_draw_relationship == "proposal_prefix":
        if len(proposal_ids) != cfg.proposal_draws:
            raise ValueError(
                "Insufficient posterior samples for complete proposal prefix"
            )
        risk_ids = proposal_ids[: cfg.risk_draws].copy()
    else:
        remaining = order[len(proposal_ids) :]
        if len(remaining) >= cfg.risk_draws:
            risk_ids = remaining[: cfg.risk_draws]
        else:
            risk_ids = rng.choice(
                len(samples), min(cfg.risk_draws, len(samples)), replace=False
            )
    rankings = _thompson_rankings(
        model, samples[proposal_ids], available, size, rng, cfg
    )
    depth = 1
    candidates = np.unique(rankings[:, :depth])
    while len(candidates) < size:
        depth += 1
        candidates = np.unique(rankings[:, :depth])
    losses = []
    for i in range(0, len(risk_ids), cfg.posterior_chunk_size):
        losses.append(
            -model.log_fold_change(
                samples[risk_ids[i : i + cfg.posterior_chunk_size]], candidates
            )
        )
    losses = np.concatenate(losses, axis=0)
    if not np.all(np.isfinite(losses)):
        raise FloatingPointError("Nonfinite risk evaluation")
    median = np.quantile(losses, 0.5, axis=0, method="linear")
    upper = np.quantile(losses, cfg.risk_quantile, axis=0, method="linear")
    tie_keys = rng.random(len(candidates))
    first_counts, expanded_counts = occurrence_counts(rankings, candidates, depth)
    geometry, fields = ([], ())
    selected, policy_audit = final_selection(
        candidates,
        median,
        upper,
        tie_keys,
        size,
        first_counts,
        expanded_counts,
        depth,
        geometry,
        fields,
        policy=cfg.selection_policy,
        multiplier=cfg.centroid_shortlist_multiplier,
    )
    return Selection(
        candidates[selected],
        median[selected],
        upper[selected],
        {
            "policy": "thompson_proposals_then_minimax_loss_ranks"
            if cfg.selection_policy == "minimax_loss_rank"
            else "thompson_proposals_then_upper_quantile_then_median",
            "loss": "-log(induced_latent_fluorescence/basal_latent_fluorescence)",
            "risk_quantile": cfg.risk_quantile,
            "proposal_count": len(candidates),
            "unique_first_choice_thompson_proposals": int(
                len(np.unique(rankings[:, 0]))
            ),
            "fallback_ranking_depth": depth,
            "proposal_draws_used": len(proposal_ids),
            "risk_draws_used": len(risk_ids),
            "proposal_risk_draw_overlap": int(
                len(np.intersect1d(proposal_ids, risk_ids))
            ),
            "proposal_risk_draw_relationship": cfg.proposal_risk_draw_relationship,
            "seed": seed,
            "unmeasured_count": len(available),
            "noisy_observation_risk_included": False,
            **policy_audit,
        },
        proposal_indices=candidates,
        proposal_draw_indices=proposal_ids,
        risk_draw_indices=risk_ids,
        proposal_losses=losses,
        proposal_median_loss=median,
        proposal_upper_loss=upper,
        tie_keys=tie_keys,
    )
