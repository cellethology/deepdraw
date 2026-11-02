"""Check numerical stability without changing the RBF model or its parameters."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")
from gpytorch.kernels import RBFKernel  # noqa: E402

from core.stable_rbf_kernel import StableRBFKernel  # noqa: E402


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("ard", [False, True])
def test_values_parameters_and_gradients_match_rbf(batch_shape, ard):
    gen = torch.Generator().manual_seed(71)
    x = torch.randn(*batch_shape, 7, 3, generator=gen, dtype=torch.double)
    y = torch.randn(*batch_shape, 5, 3, generator=gen, dtype=torch.double)
    old = RBFKernel(ard_num_dims=3 if ard else None).double()
    new = StableRBFKernel(ard_num_dims=3 if ard else None).double()
    new.load_state_dict(old.state_dict(), strict=True)
    assert old.state_dict().keys() == new.state_dict().keys()
    for last_dim in [False, True]:
        a = old(x, y, last_dim_is_batch=last_dim).to_dense()
        b = new(x, y, last_dim_is_batch=last_dim).to_dense()
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    old_grad = torch.autograd.grad(old(x, y).to_dense().sum(), old.raw_lengthscale)[0]
    new_grad = torch.autograd.grad(new(x, y).to_dense().sum(), new.raw_lengthscale)[0]
    torch.testing.assert_close(old_grad, new_grad, atol=1e-11, rtol=1e-9)
    for last_dim in [False, True]:
        torch.testing.assert_close(
            new(x, x, diag=True, last_dim_is_batch=last_dim),
            new(x, x, last_dim_is_batch=last_dim).to_dense().diagonal(dim1=-2, dim2=-1),
        )


def test_tiny_lengthscales_match_direct_reference_and_allow_cholesky():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(8, 3, generator=gen, dtype=torch.double)
    x[1:4] = x[0] + torch.randn(3, 3, generator=gen, dtype=torch.double) * 1e-8
    kernel = StableRBFKernel(ard_num_dims=3).double()
    kernel.lengthscale = torch.full((1, 3), 1e-8, dtype=torch.double)
    actual = kernel(x).to_dense()
    expected = (
        -0.5 * ((x[:, None, :] - x[None, :, :]) / kernel.lengthscale).square().sum(-1)
    ).exp()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    torch.linalg.cholesky(actual + 1e-4 * torch.eye(len(x), dtype=x.dtype))
    actual.sum().backward()
    assert torch.isfinite(kernel.raw_lengthscale.grad).all()


def test_duplicate_inputs_have_unit_covariance_and_finite_gradients():
    x = torch.tensor(
        [[0.2, 0.4], [0.2, 0.4], [0.6, -0.1]], dtype=torch.double, requires_grad=True
    )
    kernel = StableRBFKernel(ard_num_dims=2).double()
    covariance = kernel(x).to_dense()
    assert covariance[0, 1].item() == 1.0
    covariance.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(kernel.raw_lengthscale.grad).all()
