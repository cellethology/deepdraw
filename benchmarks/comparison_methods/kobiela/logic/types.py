"""Acquisition settings and posterior draws."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    batch_size: int = 12
    proposal_draws: int = 100
    risk_draws: int = 99
    proposal_risk_draw_relationship: str = "proposal_prefix"
    risk_quantile: float = 0.75
    selection_policy: str = "minimax_loss_rank"
    centroid_shortlist_multiplier: int = 4
    candidate_chunk_size: int = 512
    posterior_chunk_size: int = 32

    def validate(self):
        for name in (
            "batch_size",
            "proposal_draws",
            "risk_draws",
            "candidate_chunk_size",
            "posterior_chunk_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name)
        if self.selection_policy != "minimax_loss_rank":
            raise ValueError("Unknown selection policy")
        if self.proposal_risk_draw_relationship != "proposal_prefix":
            raise ValueError("Risk draws must reuse the proposal prefix")
        if self.risk_draws > self.proposal_draws:
            raise ValueError("Risk prefix exceeds proposal draws")
        if not 0.5 <= self.risk_quantile < 1:
            raise ValueError("risk_quantile")
        return self


@dataclass
class Posterior:
    z: np.ndarray
    metadata: dict

    def flat(self):
        return self.z.reshape(-1, self.z.shape[-1])
