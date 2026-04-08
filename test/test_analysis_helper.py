import numpy as np
from scipy import stats

from job_sub.utils.analysis_helper import mannwhitney_p


def test_mannwhitney_p_passes_through_method() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = np.array([5.0, 6.0, 7.0, 8.0])

    exact = mannwhitney_p(values, baseline, method="exact")
    asymptotic = mannwhitney_p(values, baseline, method="asymptotic")

    expected_exact = stats.mannwhitneyu(
        values, baseline, alternative="two-sided", method="exact"
    ).pvalue
    expected_asymptotic = stats.mannwhitneyu(
        values, baseline, alternative="two-sided", method="asymptotic"
    ).pvalue

    assert exact == expected_exact
    assert asymptotic == expected_asymptotic
    assert exact != asymptotic


def test_mannwhitney_p_returns_effect_with_method() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = np.array([5.0, 6.0, 7.0, 8.0])

    p_value, ps = mannwhitney_p(values, baseline, method="exact", return_effect=True)

    expected = stats.mannwhitneyu(
        values, baseline, alternative="two-sided", method="exact"
    ).pvalue

    assert p_value == expected
    assert ps == 0.0
