"""GSM8K loading + prompt construction.

Kept deliberately model-free so it runs on the Mac with no GPU and no LLM
download: we only fetch the (small) GSM8K dataset and build chat messages. The
actual prompt string is rendered with the policy tokenizer's chat template at
rollout time (pass `tokenizer=` to `load_gsm8k`, or call `render_prompt`).
"""

from __future__ import annotations

from typing import Optional

# GSM8K gold solutions are free-form reasoning ending in a literal "#### <n>".
GSM8K_HF_PATH = "openai/gsm8k"
GSM8K_CONFIG = "main"

SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step, then give "
    "the final answer."
)

# Appended to every question. Pins the output format the verifier rewards.
ANSWER_INSTRUCTION = (
    "Show your reasoning, then on the final line write the answer as "
    "`#### <number>` (digits only, no units)."
)


def extract_gold_answer(answer_field: str) -> str:
    """Pull the canonical gold answer (the number after `####`) from a GSM8K
    `answer` field, returned as a normalized string (commas stripped)."""
    after = answer_field.split("####")[-1].strip()
    return after.replace(",", "")


def build_messages(question: str) -> list[dict]:
    """Build the chat-format prompt (list of role/content dicts) for a question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip() + "\n\n" + ANSWER_INSTRUCTION},
    ]


def render_prompt(messages: list[dict], tokenizer) -> str:
    """Render chat messages to a single prompt string via the model's chat
    template, with the generation prompt appended (ready for `generate`)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_gsm8k(split: str = "test", n: Optional[int] = None, tokenizer=None) -> list[dict]:
    """Load GSM8K and return a list of records.

    Each record: {question, gold, messages}, plus a rendered `prompt` string if
    a `tokenizer` is provided. `n` caps the count (for tiny overfit subsets).
    """
    from datasets import load_dataset  # imported lazily so importing this module is cheap

    ds = load_dataset(GSM8K_HF_PATH, GSM8K_CONFIG, split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    records = []
    for ex in ds:
        messages = build_messages(ex["question"])
        rec = {
            "question": ex["question"],
            "gold": extract_gold_answer(ex["answer"]),
            "messages": messages,
        }
        if tokenizer is not None:
            rec["prompt"] = render_prompt(messages, tokenizer)
        records.append(rec)
    return records
