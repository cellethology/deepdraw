"""Shared component vocabulary for the dual-input library."""

from copy import deepcopy

from .fitting import DEFAULT_PRIORS

VOCABULARY = {
    "promoter_a": ["hEF1a v1", "hEF1a v2", "hPGK"],
    "promoter_b": ["hEF1a v1", "hEF1a v2", "hPGK"],
    "ad_a": ["NFZ", "VP64", "VPR", "p65"],
    "ad_b": ["NFZ", "VP64", "VPR", "p65"],
    "idp_a": ["FusN", "None"],
    "idp_b": ["FusN", "None"],
    "zf_a": ["A:WT (High)", "A:∆C (Med)", "A:∆C 6x (Low)", "A:∆C2 (Very low)"],
    "zf_b": ["B:WT (High)", "B:∆C (Med)", "B:∆C 6x (Low)", "B:∆C2 (Very low)"],
    "spacer_a": ["1000 bp"],
    "spacer_b": ["1000 bp"],
    "core_promoter": ["YB-TATA", "miniCMV"],
    "orientation": [
        "A -> B -> R",
        "A -> R -> B",
        "B -> A -> R",
        "B -> R -> A",
        "R -> A -> B",
        "R -> B -> A",
    ],
    "ad": ["NFZ", "VP64", "VPR", "p65"],
    "idp": ["FusN", "None"],
}


def model_metadata():
    return {
        "vocab": deepcopy(VOCABULARY),
        "priors": dict(DEFAULT_PRIORS),
        "beta_fixed": 0.0,
    }
