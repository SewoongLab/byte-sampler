import torch

from byte_sampling.utils import scatter_logsumexp


def test_scatter_logsumexp_all_neg_inf_grad_nan_by_default():
    src = torch.tensor([-float("inf"), -float("inf")], requires_grad=True)
    index = torch.tensor([0, 0], dtype=torch.int64)

    out = scatter_logsumexp(src, index, dim_size=1)
    out.sum().backward()

    assert torch.isnan(src.grad).any(), "Expected NaN grads for all -inf bucket"


def test_scatter_logsumexp_all_neg_inf_grad_safe():
    src = torch.tensor([-float("inf"), -float("inf")], requires_grad=True)
    index = torch.tensor([0, 0], dtype=torch.int64)

    out = scatter_logsumexp(src, index, dim_size=1, safe_all_neg_inf=True)
    out.sum().backward()

    assert torch.isfinite(src.grad).all(), "Expected finite grads with safe_all_neg_inf"
    assert torch.all(src.grad == 0), "Expected zero grads with safe_all_neg_inf"
