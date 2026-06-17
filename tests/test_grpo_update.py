"""Model-free tests for the GRPO update math: group advantages, the clipped+KL
loss, and the per-token logprob indexing. These pin down the equations a
reviewer will want to see, with no GPU/model needed."""

import math

import torch

from src.grpo import compute_advantages, grpo_loss, per_token_logprobs


# --------------------------------------------------------------------------- #
# compute_advantages
# --------------------------------------------------------------------------- #
def test_advantages_zero_mean_unit_scale_single_group():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    gids = torch.tensor([0, 0, 0, 0])
    adv = compute_advantages(rewards, gids, eps=1e-8)
    assert adv.mean().abs() < 1e-5                 # baseline subtracted
    assert adv.std(unbiased=False) == pytest_approx(1.0)
    # higher reward -> positive advantage
    assert (adv[:2] > 0).all() and (adv[2:] < 0).all()


def test_advantages_per_group_independent():
    rewards = torch.tensor([1.0, 0.0, 5.0, 5.0])
    gids = torch.tensor([0, 0, 1, 1])
    adv = compute_advantages(rewards, gids, eps=1e-8)
    # group 0 has spread -> nonzero; group 1 is constant -> ~0 (nothing to learn)
    assert adv[0] > 0 and adv[1] < 0
    assert abs(adv[2]) < 1e-3 and abs(adv[3]) < 1e-3


def test_advantages_degenerate_group_is_finite():
    rewards = torch.tensor([2.0, 2.0, 2.0])
    gids = torch.tensor([0, 0, 0])
    adv = compute_advantages(rewards, gids, eps=1e-4)
    assert torch.isfinite(adv).all()
    assert adv.abs().max() < 1e-3


# --------------------------------------------------------------------------- #
# grpo_loss
# --------------------------------------------------------------------------- #
def _ones(*shape):
    return torch.ones(*shape)


def test_loss_on_policy_reduces_to_neg_advantage():
    # ratio == 1 (new == old), no KL -> per-token loss = -A, averaged over seqs
    new = torch.zeros(2, 1)
    old = torch.zeros(2, 1)
    ref = torch.zeros(2, 1)
    adv = torch.tensor([2.0, -1.0])
    loss, m = grpo_loss(new, old, ref, adv, _ones(2, 1), clip_eps=0.2, kl_coef=0.0)
    assert loss.item() == pytest_approx((-2.0 + 1.0) / 2)   # = -0.5
    assert m["clip_frac"] == 0.0


def test_loss_ppo_clip_caps_positive_advantage():
    # new-old = 0.5 -> ratio = e^0.5 ~ 1.649 > 1.2; positive adv -> clipped to 1.2
    new = torch.tensor([[0.5]])
    old = torch.tensor([[0.0]])
    ref = torch.tensor([[0.5]])           # diff 0 -> no KL
    adv = torch.tensor([1.0])
    loss, m = grpo_loss(new, old, ref, adv, _ones(1, 1), clip_eps=0.2, kl_coef=0.0)
    assert loss.item() == pytest_approx(-1.2)
    assert m["clip_frac"] == 1.0


def test_loss_kl_term_matches_k3_estimator():
    # ratio 1, adv 0 -> only KL contributes. ref-new = 0.5
    new = torch.tensor([[0.0]])
    old = torch.tensor([[0.0]])
    ref = torch.tensor([[0.5]])
    adv = torch.tensor([0.0])
    kl_coef = 0.04
    loss, m = grpo_loss(new, old, ref, adv, _ones(1, 1), clip_eps=0.2, kl_coef=kl_coef)
    expected_kl = math.exp(0.5) - 0.5 - 1.0
    assert m["kl"] == pytest_approx(expected_kl)
    assert loss.item() == pytest_approx(kl_coef * expected_kl)


def test_loss_respects_completion_mask():
    # token 1 carries a huge KL but is masked out -> must not affect the loss
    new = torch.tensor([[0.0, 0.0]])
    old = torch.tensor([[0.0, 0.0]])
    ref = torch.tensor([[0.0, 10.0]])     # token1 KL is enormous
    adv = torch.tensor([0.0])
    mask = torch.tensor([[1.0, 0.0]])
    loss, _ = grpo_loss(new, old, ref, adv, mask, clip_eps=0.2, kl_coef=1.0)
    assert loss.item() == pytest_approx(0.0)


# --------------------------------------------------------------------------- #
# per_token_logprobs (indexing) — tiny fake model returning fixed logits
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, logits):
        self.logits = logits


class _FakeLM:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids=None, attention_mask=None):
        return _Out(self._logits)


def test_per_token_logprobs_picks_completion_positions():
    torch.manual_seed(0)
    B, L, V = 1, 4, 6
    logits = torch.randn(B, L, V)
    full = torch.tensor([[1, 2, 3, 4]])   # P=2 prompt, C=2 completion
    prompt_ids, completion_ids = full[:, :2], full[:, 2:]
    ones = torch.ones(B, 2)

    got = per_token_logprobs(_FakeLM(logits), prompt_ids, ones, completion_ids, ones)

    # reference: standard log_softmax + gather, then slice the completion tokens
    ref_lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    gathered = ref_lp.gather(-1, full[:, 1:].unsqueeze(-1)).squeeze(-1)  # [1, 3]
    expected = gathered[:, 1:]                                          # last C=2
    assert torch.allclose(got, expected, atol=1e-5)
    assert got.shape == (1, 2)


# small local approx helper so we don't depend on pytest.approx import style
def pytest_approx(x, tol=1e-5):
    class _A:
        def __eq__(self, other):
            return abs(other - x) <= tol + tol * abs(x)
    return _A()
