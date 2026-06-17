"""Model-free tests for the rollout tensor logic. The completion mask decides
which tokens count toward the loss/KL, so a bug here silently corrupts training."""

import torch

from src.grpo import make_completion_mask

EOS = 0


def test_mask_eos_in_middle():
    ids = torch.tensor([[5, 6, EOS, 9, 9]])
    assert make_completion_mask(ids, EOS).tolist() == [[1, 1, 1, 0, 0]]


def test_mask_no_eos_all_valid():
    # row hit max_new_tokens with no EOS -> every token is real
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    assert make_completion_mask(ids, EOS).tolist() == [[1, 1, 1, 1, 1]]


def test_mask_eos_first_token():
    ids = torch.tensor([[EOS, 9, 9, 9, 9]])
    assert make_completion_mask(ids, EOS).tolist() == [[1, 0, 0, 0, 0]]


def test_mask_eos_last_token():
    ids = torch.tensor([[1, 2, 3, 4, EOS]])
    assert make_completion_mask(ids, EOS).tolist() == [[1, 1, 1, 1, 1]]


def test_mask_only_first_eos_counts():
    # when pad_id == eos_id, trailing pads are EOS too; only the first ends it
    ids = torch.tensor([[7, EOS, EOS, EOS, EOS]])
    assert make_completion_mask(ids, EOS).tolist() == [[1, 1, 0, 0, 0]]


def test_mask_batched_rows_independent():
    ids = torch.tensor([
        [5, 6, EOS, 9, 9],   # eos mid
        [1, 2, 3, 4, 5],     # no eos
        [EOS, 1, 1, 1, 1],   # eos first
    ])
    expected = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
    ]
    assert make_completion_mask(ids, EOS).tolist() == expected
