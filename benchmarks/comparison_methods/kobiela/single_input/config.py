"""Settings for single-input Bayesian fitting and selection."""

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Priors:
    part_scale: float = 1.0
    kd_location_sd: float = 2.0
    partition_location: float = 0.0
    partition_sd: float = 2.0
    induction_logit_location: float = -2.0
    induction_logit_sd: float = 2.0
    log_cooperativity_location: float = 0.0
    log_cooperativity_sd: float = 1.0
    log_baseline_location: float = math.log(1000.0)
    log_baseline_sd: float = 2.0
    log_amplitude_location: float = math.log(1000.0)
    log_amplitude_sd: float = 2.0
    log_noise_location: float = math.log(0.5)
    log_noise_sd: float = 1.0


@dataclass(frozen=True)
class Config:
    chains: int = 3
    warmup: int = 500
    draws: int = 1000
    nuts_target_accept: float = 0.65
    nuts_cores: int = 3
    nuts_init: str = "jitter+adapt_full"
    batch_size: int = 12
    proposal_draws: int = 100
    risk_draws: int = 99
    proposal_risk_draw_relationship: str = "proposal_prefix"
    risk_quantile: float = 0.75
    selection_policy: str = "minimax_loss_rank"
    centroid_shortlist_multiplier: int = 4
    candidate_chunk_size: int = 512
    posterior_chunk_size: int = 32
    priors: Priors = field(default_factory=Priors)

    def validate(self):
        for name in (
            "chains",
            "draws",
            "nuts_cores",
            "batch_size",
            "proposal_draws",
            "risk_draws",
            "candidate_chunk_size",
            "posterior_chunk_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.warmup < 0 or not 0 < self.nuts_target_accept < 1:
            raise ValueError("Invalid sampler settings")
        if not 0.5 <= self.risk_quantile < 1:
            raise ValueError("Invalid risk quantile")
        if self.proposal_risk_draw_relationship != "proposal_prefix":
            raise ValueError("Risk evaluation uses the proposal prefix")
        if self.risk_draws > self.proposal_draws:
            raise ValueError("Risk prefix exceeds proposal draws")
        if self.selection_policy != "minimax_loss_rank":
            raise ValueError("Unsupported selection policy")
        for name, value in asdict(self.priors).items():
            positive = name.endswith("_sd") or name.endswith("_scale")
            if not math.isfinite(value) or (positive and value <= 0):
                raise ValueError(f"Invalid prior {name}")
        return self

    def to_dict(self):
        return asdict(self)
