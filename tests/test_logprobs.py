import math
import pytest

import torch

from byte_sampling.utils import (
    logprobs_from_logits,
)  # adjust if your module path is different

# === helpers ===


@pytest.fixture
def random_logits():
    torch.manual_seed(1234)
    batch = 4
    vocab = 10
    logits = torch.randn(batch, vocab) * 3.0
    logits[0, -1] = -1e9  # extreme low value
    return logits


@pytest.mark.parametrize(
    "temperature, top_k, top_p",
    [
        (1.0, None, None),
        (0.5, 5, None),
        (1.2, None, 0.9),
        (0.8, 3, 0.5),
        (1.0, 1, 1.0),
        (1.0, None, 0.02),  # aggressive nucleus
    ],
)
def test_distribution_sanity(random_logits, temperature, top_k, top_p):
    lp = logprobs_from_logits(
        random_logits, temperature=temperature, top_k=top_k, top_p=top_p
    )

    # No NaNs
    assert not torch.isnan(lp).any(), "Logprobs contain NaNs"

    # Valid probability distribution
    probs = lp.exp()
    sums = probs.sum(dim=-1)
    assert torch.allclose(
        sums, torch.ones_like(sums), atol=1e-6
    ), f"Prob sums not 1: {sums}"


def test_tiny_top_p_keeps_at_least_one(random_logits):
    lp = logprobs_from_logits(random_logits, top_p=1e-7)
    probs = lp.exp()
    nonzero = (probs > 0).sum(dim=-1)
    assert (nonzero >= 1).all(), "top_p eliminated all tokens in some batch entries"


def test_low_temperature_approximates_argmax(random_logits):
    low_temp = 1e-5
    lp = logprobs_from_logits(random_logits, temperature=low_temp)
    predicted = lp.argmax(dim=-1)
    expected = random_logits.argmax(dim=-1)
    assert torch.equal(
        predicted, expected
    ), "Low temperature did not collapse to argmax"


def test_invalid_args_raise():
    logits = torch.randn(2, 6)
    # with pytest.raises(ValueError):
    #     logprobs_from_logits(logits, temperature=0)
    with pytest.raises(ValueError):
        logprobs_from_logits(logits, top_p=-0.1)
    with pytest.raises(ValueError):
        logprobs_from_logits(logits, top_p=1.5)
    with pytest.raises(ValueError):
        logprobs_from_logits(logits, top_k=0)
