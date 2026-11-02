"""RBF covariance with direct Euclidean distances in lengthscale coordinates."""

import torch
from gpytorch.kernels import RBFKernel


class StableRBFKernel(RBFKernel):
    """Preserve RBF parameters while avoiding the quadratic distance expansion.

    The non-matrix-multiplication cdist implementation subtracts coordinates
    directly and does not materialize an N x M x D tensor. Use it consistently
    for training, cross covariance, and fantasy batches. The diagonal path
    avoids constructing a full covariance matrix for marginal variances.
    """

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if diag:
            squared = ((x1 - x2) / self.lengthscale).square()
            if last_dim_is_batch:
                return (-0.5 * squared).exp().transpose(-1, -2)
            return (-0.5 * squared.sum(dim=-1)).exp()

        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale
        if last_dim_is_batch:
            x1_scaled = x1_scaled.transpose(-1, -2).unsqueeze(-1)
            x2_scaled = x2_scaled.transpose(-1, -2).unsqueeze(-1)
        distance = torch.cdist(
            x1_scaled,
            x2_scaled,
            p=2,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        return (-0.5 * distance.square()).exp()
