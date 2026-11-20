"""Checks of the transition matrices, the reverse posterior and the forward samplers."""

import math

import pytest
import torch

import diffusion as D


def test_closed_form_matches_product_of_steps():
    product = torch.eye(D.K, dtype=torch.float64)
    for step in range(1, D.T + 1):
        product = product @ D.step_matrix(step)
        assert torch.allclose(product, D.cumulative_matrix(step), atol=1e-12), step
    assert torch.allclose(product, torch.full((D.K, D.K), 1 / D.K, dtype=torch.float64), atol=1e-12)


def test_matrices_are_row_stochastic():
    for step in (1, 37, 100):
        for matrix in (D.step_matrix(step), D.cumulative_matrix(step)):
            assert (matrix >= 0).all()
            assert torch.allclose(matrix.sum(1), torch.ones(D.K, dtype=torch.float64))


def brute_force_reverse(f, x_t, step):
    """Explicit Bayes: q(x_{t-1}=i | x_t=j, x_0=k) from the joint, mixed by f(k)."""
    q_step, q_prev = D.step_matrix(step), D.cumulative_matrix(step - 1)
    out = torch.zeros(f.shape, dtype=torch.float64)
    for b in range(f.shape[0]):
        for position in range(f.shape[1]):
            j = int(x_t[b, position])
            joint = q_prev * q_step[:, j][None, :]           # [k, i] = p(x_{t-1}=i | k) p(x_t=j | i)
            posterior = joint / joint.sum(1, keepdim=True)   # normalise over i for each k
            out[b, position] = (f[b, position, :, None].double() * posterior).sum(0)
    return out


@pytest.mark.parametrize("step", [1, 2, 10, 25, 50, 75, 90, 99, 100])
def test_reverse_probs_match_bayes_and_sum_to_one(step):
    generator = torch.Generator().manual_seed(step)
    logits = torch.randn(2, 6, D.K, generator=generator) * 3
    x_t = torch.randint(D.K, (2, 6), generator=generator)
    fast = D.reverse_probs(logits, x_t, step).double()
    slow = brute_force_reverse(logits.double().softmax(-1), x_t, step)
    assert torch.allclose(fast, slow, atol=1e-6)
    assert torch.allclose(fast.sum(-1), torch.ones(2, 6, dtype=torch.float64), atol=1e-6)
    assert (fast >= 0).all()


def test_reverse_formula_quotient_form():
    """The task's quotient Q_t[i,j] Q̄_{t-1}[k,i] / Q̄_t[k,j] equals the normalised Bayes posterior."""
    for step in (1, 30, 100):
        q_step, q_prev, q_cum = D.step_matrix(step), D.cumulative_matrix(step - 1), D.cumulative_matrix(step)
        for j in (0, 1, 7):
            quotient = q_step[:, j][None, :] * q_prev / q_cum[:, j][:, None]   # [k, i]
            joint = q_prev * q_step[:, j][None, :]
            assert torch.allclose(quotient, joint / joint.sum(1, keepdim=True), atol=1e-12)


def chi_square_critical(df, alpha):
    """Wilson-Hilferty approximation of the upper chi-square quantile."""
    z = {1e-3: 3.090232, 1e-4: 3.719016}[alpha]
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


@pytest.mark.parametrize("step", [10, 50, 90])
def test_closed_form_and_stepwise_sampling_agree(step):
    n = 200_000
    x_0 = torch.full((n, 1), 7, dtype=torch.long)
    one_shot = D.corrupt(x_0, torch.full((n,), step), torch.Generator().manual_seed(1))
    g = torch.Generator().manual_seed(2)
    x = x_0
    for s in range(1, step + 1):
        x = D.forward_step(x, s, g)
    a = torch.bincount(one_shot.flatten(), minlength=D.K).double()
    b = torch.bincount(x.flatten(), minlength=D.K).double()
    # two-sample chi-square on the 51 categories (equal sample sizes)
    statistic = (((a - b) ** 2) / (a + b).clamp_min(1)).sum().item()
    assert statistic < chi_square_critical(D.K - 1, 1e-3)
    # both agree with the closed-form marginal Q̄_t[7, .]
    expected = D.cumulative_matrix(step)[7] * n
    for counts in (a, b):
        assert (((counts - expected) ** 2) / expected).sum().item() < chi_square_critical(D.K - 1, 1e-3)


def test_sample_categorical_matches_probabilities():
    probs = torch.tensor([0.1, 0.0, 0.6, 0.3])
    draws = D.sample_categorical(probs.expand(100_000, 4), torch.Generator().manual_seed(0))
    freq = torch.bincount(draws, minlength=4).double() / len(draws)
    assert freq[1] == 0
    assert torch.allclose(freq, probs.double(), atol=0.01)
