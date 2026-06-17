"""Unit tests for the RLVR verifier — the piece a skeptical reviewer will poke
at first, so we cover the tricky parsing cases explicitly."""

import pytest

from src.data import extract_gold_answer
from src.verifier import (
    CORRECT_REWARD,
    FORMAT_REWARD,
    compute_reward,
    extract_pred_answer,
    has_answer_format,
    is_correct,
    normalize_number,
    score_completion,
)


# --------------------------------------------------------------------------- #
# normalize_number — decoration, signs, fractions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("42", 42.0),
        ("-5", -5.0),
        ("1,000", 1000.0),
        ("1,000,000", 1_000_000.0),
        ("$3.50", 3.5),
        ("50%", 50.0),
        ("3.0", 3.0),
        ("42.", 42.0),        # trailing sentence period
        ("3/4", 0.75),        # simple fraction
        ("-2/3", -2 / 3),
        (" 7 ", 7.0),
        ("", None),
        ("abc", None),
        (None, None),
    ],
)
def test_normalize_number(raw, expected):
    got = normalize_number(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# extract_pred_answer — priority and tricky text
# --------------------------------------------------------------------------- #
def test_extract_after_hash():
    assert extract_pred_answer("blah blah\n#### 18") == "18"


def test_extract_negative_after_hash():
    assert extract_pred_answer("... so the change is #### -5") == "-5"


def test_extract_comma_number():
    assert normalize_number(extract_pred_answer("Total: #### 1,000")) == 1000.0


def test_extract_trailing_text_after_number():
    # number followed by units/words on the same line — take the number
    assert extract_pred_answer("#### 42 apples in the basket") == "42"


def test_extract_takes_last_hash():
    # intermediate "####" noise should not win; the final marker does
    text = "first guess #### 7\nrecheck...\n#### 9"
    assert extract_pred_answer(text) == "9"


def test_extract_boxed_fallback():
    assert extract_pred_answer("The result is \\boxed{36}.") == "36"


def test_hash_preferred_over_boxed():
    text = "\\boxed{7}\nFinal: #### 12"
    assert extract_pred_answer(text) == "12"


def test_extract_last_number_fallback():
    # no #### and no \boxed -> fall back to the last number in the text
    assert extract_pred_answer("I think the answer is 24 in total") == "24"


def test_extract_none_when_no_number():
    assert extract_pred_answer("no numbers here at all") is None


# --------------------------------------------------------------------------- #
# format detection
# --------------------------------------------------------------------------- #
def test_format_true_for_hash():
    assert has_answer_format("#### 10") is True


def test_format_true_for_boxed():
    assert has_answer_format("\\boxed{10}") is True


def test_format_false_for_plain_text():
    assert has_answer_format("the answer is 10") is False


# --------------------------------------------------------------------------- #
# is_correct — tolerance, decimals, fractions
# --------------------------------------------------------------------------- #
def test_correct_integer():
    assert is_correct("18", "18") is True


def test_correct_with_commas():
    assert is_correct("1,000", "1000") is True


def test_correct_decimal_within_tolerance():
    assert is_correct("3.0", "3") is True


def test_correct_fraction_equivalent():
    assert is_correct("3/4", "0.75") is True


def test_incorrect_number():
    assert is_correct("19", "18") is False


def test_incorrect_when_none():
    assert is_correct(None, "18") is False


# --------------------------------------------------------------------------- #
# compute_reward / score_completion — the full RLVR signal
# --------------------------------------------------------------------------- #
def test_reward_correct_and_formatted():
    # right answer in the expected format -> 1.0 + 0.1
    assert compute_reward("reasoning...\n#### 42", "42") == pytest.approx(
        CORRECT_REWARD + FORMAT_REWARD
    )


def test_reward_wrong_but_formatted():
    # wrong answer but well-formatted -> only the 0.1 format bonus
    assert compute_reward("#### 99", "42") == pytest.approx(FORMAT_REWARD)


def test_reward_correct_but_unformatted():
    # correct via fallback extraction, but NOT in #### / boxed format -> 1.0 only
    out = score_completion("the answer is 42", "42")
    assert out["correct"] is True
    assert out["format"] is False
    assert out["reward"] == pytest.approx(CORRECT_REWARD)


def test_reward_no_answer():
    assert compute_reward("I am not sure how to solve this.", "42") == pytest.approx(0.0)


def test_score_breakdown_keys():
    out = score_completion("#### 42", "42")
    assert set(out) == {"pred", "correct", "format", "reward"}
    assert out["pred"] == "42"


# --------------------------------------------------------------------------- #
# gold extraction from the GSM8K answer field
# --------------------------------------------------------------------------- #
def test_extract_gold_answer():
    answer_field = "Natalia sold 48/2 = 24 clips in May.\nIn total she sold 72.\n#### 72"
    assert extract_gold_answer(answer_field) == "72"


def test_extract_gold_answer_strips_commas():
    assert extract_gold_answer("... so the total is 1,000.\n#### 1,000") == "1000"
