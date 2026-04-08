import ast

import numpy as np
from scipy import stats


def strip_overrides(hydra_overrides):
    value = hydra_overrides
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return hydra_overrides

    if isinstance(value, list):
        filtered = [
            x
            for x in value
            if not (
                isinstance(x, str)
                and (
                    x.startswith("al_settings.seed=")
                    or x.startswith("dataset_index=")
                    or x.startswith("single_array_across_datasets=")
                )
            )
        ]
        ordered = sorted(filtered, key=_override_sort_key)
        values_only = []
        for override in ordered:
            if isinstance(override, str):
                _, sep, val = override.partition("=")
                values_only.append(val if sep else override)
            else:
                values_only.append(str(override))
        return "|".join(values_only)
    return hydra_overrides


def _override_sort_key(override):
    if not isinstance(override, str):
        return (1, str(override), "")
    key, _, value = override.partition("=")
    return (0, key, value)


def mannwhitney_p(
    values,
    baseline_values,
    alternative="two-sided",
    return_effect=False,
    method="auto",
):
    if len(values) == 0 or len(baseline_values) == 0:
        return (np.nan, np.nan) if return_effect else np.nan
    try:
        res = stats.mannwhitneyu(
            values,
            baseline_values,
            alternative=alternative,
            method=method,
        )
        p_value = float(res.pvalue)
    except ValueError:
        return (np.nan, np.nan) if return_effect else np.nan

    if not return_effect:
        return p_value

    # Probability of superiority (ties count as 0.5).
    n1 = len(values)
    n2 = len(baseline_values)
    combined = np.concatenate([values, baseline_values])
    ranks = stats.rankdata(combined)
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    ps = U1 / (n1 * n2)

    return p_value, float(ps)


def fold_change(strategy_median, baseline_median):
    return strategy_median / baseline_median
