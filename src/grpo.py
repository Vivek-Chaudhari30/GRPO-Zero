"""Core GRPO machinery. Milestone 3 covers the ROLLOUT half: sample a group of
G completions per prompt and score each with the verifier, returning the token
tensors the policy update (M4) will consume.

GRPO recap (built out across M3-M4):
  - For each prompt, sample G completions  -> this file (rollout).
  - Score each with the verifiable reward   -> this file (verifier).
  - Group-relative advantage A_i = (r_i - mean_g(r)) / (std_g(r) + eps).
  - PPO-style clipped policy loss using A_i, + KL penalty to a frozen reference.
The group mean is GRPO's baseline, so there is no separate value/critic network.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import render_prompt
from .verifier import score_completion


@dataclass
class RolloutBatch:
    """One rollout, flattened to N*G rows (prompt-major: rows i*G .. i*G+G-1
    are the G samples for prompt i)."""
    prompt_ids: torch.Tensor       # [N*G, P]  left-padded prompt tokens
    prompt_mask: torch.Tensor      # [N*G, P]
    completion_ids: torch.Tensor   # [N*G, C]  generated tokens
    completion_mask: torch.Tensor  # [N*G, C]  1 up to & incl. first EOS, else 0
    rewards: torch.Tensor          # [N*G]     verifier reward per completion
    correct: torch.Tensor          # [N*G] bool
    group_ids: torch.Tensor        # [N*G]     prompt index each row belongs to
    completions: list[str]         # decoded completion text (len N*G)
    golds: list[str]               # gold answer per row (len N*G)

    def __len__(self) -> int:
        return int(self.rewards.shape[0])


def make_completion_mask(completion_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Mask of real completion tokens: 1 up to and including the first EOS in
    each row, 0 afterwards. Rows with no EOS (hit max_new_tokens) are all 1.

    This is what stops padding (and post-EOS junk) from contaminating the
    per-token loss / KL in the update step.
    """
    is_eos = completion_ids == eos_id
    B, C = completion_ids.shape
    has_eos = is_eos.any(dim=1)
    # argmax returns the FIRST max (first True); 0 when a row has no EOS, so guard it.
    first_eos = torch.argmax(is_eos.to(torch.int), dim=1)
    end = torch.where(has_eos, first_eos, torch.full_like(first_eos, C - 1))
    idx = torch.arange(C, device=completion_ids.device).unsqueeze(0)  # [1, C]
    return (idx <= end.unsqueeze(1)).to(torch.long)


@torch.no_grad()
def generate_rollouts(model, tokenizer, records, *, group_size, max_new_tokens,
                      temperature, top_p, device) -> RolloutBatch:
    """Sample `group_size` completions per record, score them, return a RolloutBatch."""
    prompts = [render_prompt(r["messages"], tokenizer) for r in records]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=1024).to(device)

    out = model.generate(
        **enc,
        do_sample=True, temperature=temperature, top_p=top_p,
        num_return_sequences=group_size, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    P = enc["input_ids"].shape[1]
    # generate() with num_return_sequences keeps prompt-major order, so repeat
    # each prompt G times to line the prompt tensors up with the G completions.
    prompt_ids = enc["input_ids"].repeat_interleave(group_size, dim=0)
    prompt_mask = enc["attention_mask"].repeat_interleave(group_size, dim=0)
    completion_ids = out[:, P:].contiguous()
    completion_mask = make_completion_mask(completion_ids, tokenizer.eos_token_id)

    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    rewards, correct, golds, group_ids = [], [], [], []
    for i, rec in enumerate(records):
        for j in range(group_size):
            s = score_completion(texts[i * group_size + j], rec["gold"])
            rewards.append(s["reward"])
            correct.append(s["correct"])
            golds.append(rec["gold"])
            group_ids.append(i)

    return RolloutBatch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        rewards=torch.tensor(rewards, dtype=torch.float32),
        correct=torch.tensor(correct, dtype=torch.bool),
        group_ids=torch.tensor(group_ids, dtype=torch.long),
        completions=texts,
        golds=golds,
    )


def group_reward_summary(batch: RolloutBatch) -> list[dict]:
    """Per-group reward stats — useful for logging and a preview of the
    group-relative baseline the advantage will subtract in M4."""
    rows = []
    for g in batch.group_ids.unique().tolist():
        rg = batch.rewards[batch.group_ids == g]
        cg = batch.correct[batch.group_ids == g]
        rows.append({
            "group": g,
            "n": int(rg.numel()),
            "reward_mean": round(rg.mean().item(), 4),
            "reward_std": round(rg.std(unbiased=False).item(), 4),
            "reward_min": round(rg.min().item(), 4),
            "reward_max": round(rg.max().item(), 4),
            "n_correct": int(cg.sum().item()),
        })
    return rows


def _demo():
    """Tiny end-to-end rollout demo: `python -m src.grpo --n 2 --group-size 8`."""
    import argparse

    from .data import load_gsm8k
    from .model_utils import load_config, load_model_and_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo_gsm8k.yaml")
    ap.add_argument("--n", type=int, default=2, help="# of prompts to roll out")
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rc = cfg["rollout"]
    G = args.group_size or rc["group_size"]
    mnt = args.max_new_tokens or rc["max_new_tokens"]

    model, tokenizer, device = load_model_and_tokenizer(
        cfg["model"]["id"], dtype_name=cfg["model"].get("dtype", "auto"))
    print(f"Device: {device} | model: {cfg['model']['id']} | G={G} | max_new_tokens={mnt}")

    records = load_gsm8k(args.split, n=args.n)
    batch = generate_rollouts(
        model, tokenizer, records, group_size=G, max_new_tokens=mnt,
        temperature=rc["temperature"], top_p=rc["top_p"], device=device)

    print(f"\nRollout: {len(records)} prompts x G={G} = {len(batch)} completions")
    print(f"shapes: prompt_ids={tuple(batch.prompt_ids.shape)} "
          f"completion_ids={tuple(batch.completion_ids.shape)} "
          f"completion_mask sum/row≈{batch.completion_mask.sum(1).float().mean():.0f} tok")
    for row in group_reward_summary(batch):
        print(f"  group {row['group']}: rewards mean={row['reward_mean']} "
              f"std={row['reward_std']} [{row['reward_min']}..{row['reward_max']}] "
              f"correct={row['n_correct']}/{row['n']}")
        rg = batch.rewards[batch.group_ids == row["group"]].tolist()
        print(f"    per-sample rewards: {[round(x,1) for x in rg]}")


if __name__ == "__main__":
    _demo()
