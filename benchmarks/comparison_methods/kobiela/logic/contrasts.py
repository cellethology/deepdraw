"""Zero-sum categorical contrasts."""

import math

import numpy as np


def _scaled_contrasts(n):
    if n < 1:
        raise ValueError("At least one category required")
    c = np.zeros((n, n - 1))
    for j in range(n - 1):
        norm = math.sqrt((j + 1) * (j + 2))
        c[: j + 1, j] = 1 / norm
        c[j + 1, j] = -(j + 1) / norm
    if n > 1:
        c *= math.sqrt(n / (n - 1))
    return c
