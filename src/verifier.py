"""RLVR verifier: a *deterministic* reward for GSM8K.

This is the heart of "RL from Verifiable Rewards": the reward is NOT a learned
model. We parse the model's final numeric answer out of its completion and
compare it to the gold answer with exact (tolerance-aware) numeric equality.

Reward shaping (matches the build spec):
    reward = 1.0 if the final answer is correct, else 0.0
           + 0.1 format bonus if the answer was emitted in the expected
             `#### <number>` or `\\boxed{<number>}` format.

The format bonus is independent of correctness: a well-formatted-but-wrong
answer still earns the 0.1, and a correct answer found only via the fallback
"last number in text" heuristic earns the 1.0 but NOT the format bonus. This
nudges the policy toward the parseable output format we want while keeping the
correctness signal honest.
"""

from __future__ import annotations

import re
from typing import Optional

CORRECT_REWARD = 1.0
FORMAT_REWARD = 0.1
# Tolerance for float comparison (relative + absolute). GSM8K gold answers are
# integers, but model predictions can be decimals/fractions, so we compare loosely.
_TOL = 1e-4

# A "number token": optional sign, optional $, digits (with optional thousands
# commas), optional decimal part, optionally followed by a `/ d+` fraction tail.
# Examples matched: 42  -5  $3.50  1,000  1,000.25  .5  3/4  -2/3
_NUMBER_RE = re.compile(r"[-+]?\$?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+)?")


# --------------------------------------------------------------------------- #
# Low-level parsing helpers
# --------------------------------------------------------------------------- #
def normalize_number(s: Optional[str]) -> Optional[float]:
    """Turn a raw numeric token into a float, or None if it isn't a number.

    Strips currency/grouping/percent decoration and evaluates simple fractions.
    """
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.strip().rstrip(".").strip()  # drop a trailing sentence period, e.g. "42."
    if not s:
        return None
    if "/" in s:  # simple fraction like "3/4" or "-2/3"
        try:
            num, den = s.split("/")
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _find_number(text: str, last: bool = False) -> Optional[str]:
    """Return the first (or last) number-like token in `text`, or None."""
    matches = [m for m in _NUMBER_RE.findall(text) if any(c.isdigit() for c in m)]
    if not matches:
        return None
    return (matches[-1] if last else matches[0]).strip()


def _last_hash_chunk(text: str) -> Optional[str]:
    """Text on the same line right after the LAST `####`, or None.

    Taking the last occurrence is deliberate: chain-of-thought may mention
    intermediate `####`-like noise; the final answer marker is what counts.
    """
    idx = text.rfind("####")
    if idx == -1:
        return None
    return text[idx + 4:].split("\n")[0]


def _last_boxed(text: str) -> Optional[str]:
    """Contents of the last `\\boxed{...}`, or None."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1] if matches else None


# --------------------------------------------------------------------------- #
# Public verifier API
# --------------------------------------------------------------------------- #
def extract_pred_answer(text: str) -> Optional[str]:
    """Extract the model's final answer token as a string (None if absent).

    Priority: number after the last `####`  ->  inside the last `\\boxed{}`  ->
    fallback to the last number anywhere in the text. The fallback lets us still
    grade unformatted-but-correct answers, while `has_answer_format` separately
    decides whether the format bonus is earned.
    """
    chunk = _last_hash_chunk(text)
    if chunk is not None:
        tok = _find_number(chunk)
        if tok is not None:
            return tok
    boxed = _last_boxed(text)
    if boxed is not None:
        tok = _find_number(boxed)
        if tok is not None:
            return tok
    return _find_number(text, last=True)


def has_answer_format(text: str) -> bool:
    """True iff a parseable number appears after `####` or inside `\\boxed{}`."""
    chunk = _last_hash_chunk(text)
    if chunk is not None and _find_number(chunk) is not None:
        return True
    boxed = _last_boxed(text)
    if boxed is not None and _find_number(boxed) is not None:
        return True
    return False


def is_correct(pred: Optional[str], gold: str) -> bool:
    """Tolerance-aware numeric equality between a predicted and gold answer."""
    p = normalize_number(pred)
    g = normalize_number(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= _TOL + _TOL * abs(g)


def score_completion(completion: str, gold: str) -> dict:
    """Full breakdown for one completion — handy for training/eval logging.

    Returns {pred, correct, format, reward}.
    """
    pred = extract_pred_answer(completion)
    correct = is_correct(pred, gold)
    formatted = has_answer_format(completion)
    reward = (CORRECT_REWARD if correct else 0.0) + (FORMAT_REWARD if formatted else 0.0)
    return {"pred": pred, "correct": correct, "format": formatted, "reward": reward}


def compute_reward(completion: str, gold: str) -> float:
    """Scalar RLVR reward for one completion against its gold answer."""
    return score_completion(completion, gold)["reward"]
