"""Joint median and upper-quartile ranking of posterior proposals."""

from collections import Counter

import numpy as np


def occurrence_counts(rankings, candidates, depth):
    rankings = np.asarray(rankings)
    if rankings.ndim != 2 or not 1 <= depth <= rankings.shape[1]:
        raise ValueError("Invalid proposal rank depth")
    if rankings.dtype.kind not in "iu" or any(
        len(set(row)) != len(row) for row in rankings
    ):
        raise ValueError("Per-draw rankings must contain distinct integer IDs")
    first = Counter(map(int, rankings[:, 0]))
    expanded = Counter(map(int, rankings[:, :depth].ravel()))
    if set(expanded) != set(map(int, candidates)):
        raise ValueError("Proposal support and included rankings differ")
    return [first[int(i)] for i in candidates], [expanded[int(i)] for i in candidates]


def competition_ranks(values):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty losses required")
    return np.searchsorted(np.sort(values), values, side="left") + 1


def joint_rank_order(candidates, median, upper, tie_keys):
    candidates = np.asarray(candidates)
    median, upper, tie_keys = (np.asarray(x) for x in (median, upper, tie_keys))
    count = len(candidates)
    if (
        candidates.ndim != 1
        or candidates.dtype.kind not in "iu"
        or len(set(map(int, candidates))) != count
    ):
        raise ValueError("Distinct proposal identities required")
    if any(
        v.shape != (count,) or not np.isfinite(v).all()
        for v in (median, upper, tie_keys)
    ):
        raise ValueError("Finite aligned rank inputs required")
    r50, r75 = competition_ranks(median), competition_ranks(upper)
    joint, rank_sum = np.maximum(r50, r75), r50 + r75
    canonical = np.argsort(candidates, kind="stable")
    order = canonical[
        np.lexsort(
            (
                tie_keys[canonical],
                upper[canonical],
                rank_sum[canonical],
                joint[canonical],
            )
        )
    ]
    return order, r50, r75, joint, rank_sum


def final_selection(
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
    *,
    policy="minimax_loss_rank",
    multiplier=4,
):
    if policy != "minimax_loss_rank":
        raise ValueError("Expected the joint loss-rank policy")
    candidates = np.asarray(candidates)
    if (
        isinstance(size, bool)
        or not isinstance(size, int | np.integer)
        or not 1 <= size <= len(candidates)
    ):
        raise ValueError("Invalid batch size")
    median, upper, tie_keys = (np.asarray(v) for v in (median, upper, tie_keys))
    order, r50, r75, joint, rank_sum = joint_rank_order(
        candidates, median, upper, tie_keys
    )
    selected = order[:size]
    audit = {
        "selection_policy": policy,
        "selection_policy_version": "minimax_loss_rank_v1",
        "loss_rank_method": "competition_min",
        "median_loss_ranks": r50.tolist(),
        "upper_loss_ranks": r75.tolist(),
        "joint_max_rank": joint.tolist(),
        "joint_rank_sum": rank_sum.tolist(),
        "proposal_indices": candidates.tolist(),
        "proposal_median_loss": median.tolist(),
        "proposal_upper_loss": upper.tolist(),
        "tie_keys": tie_keys.tolist(),
        "first_choice_occurrence_counts": first_counts,
        "expanded_occurrence_counts": expanded_counts,
        "rank_depth": depth,
        "final_ranking_indices": candidates[order].tolist(),
        "selected_indices": candidates[selected].tolist(),
    }
    return selected, audit
