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

import math
from contextlib import nullcontext
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
        use_cache=True,   # explicit: training sets config.use_cache=False for the grad forwards
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


# =========================================================================== #
# GRPO update: advantages -> per-token logprobs -> clipped loss + KL
# =========================================================================== #
def compute_advantages(rewards: torch.Tensor, group_ids: torch.Tensor,
                       eps: float = 1e-4) -> torch.Tensor:
    """Group-relative advantage A_i = (r_i - mean_g(r)) / (std_g(r) + eps).

    The group mean is GRPO's baseline (no critic). Dividing by the group std
    whitens the signal across prompts of differing difficulty. When a whole
    group has identical reward (std=0) the advantage collapses to ~0 — there is
    nothing to learn from that prompt this step, which is the correct behavior.
    """
    advantages = torch.empty_like(rewards)
    for g in torch.unique(group_ids):
        m = group_ids == g
        rg = rewards[m]
        advantages[m] = (rg - rg.mean()) / (rg.std(unbiased=False) + eps)
    return advantages


def per_token_logprobs(model, prompt_ids, prompt_mask, completion_ids, completion_mask):
    """Log-prob the model assigns to each *completion* token (teacher forcing).

    Returns [B, C]. We concat prompt+completion, run one forward, and read off
    logπ(token_t | token_<t) for the completion positions. Uses
    `selected - logsumexp` instead of a full log_softmax to avoid materializing
    a [B, L, vocab] tensor.
    """
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attn = torch.cat([prompt_mask, completion_mask], dim=1)
    prompt_len = prompt_ids.shape[1]

    logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, L, V]
    logits = logits[:, :-1, :].float()      # position t predicts token t+1
    targets = input_ids[:, 1:]              # [B, L-1]
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    token_logprobs = selected - torch.logsumexp(logits, dim=-1)      # [B, L-1]
    # completion tokens live at absolute positions prompt_len .. L-1; their
    # predicting logits are at prompt_len-1 .. L-2 -> slice [prompt_len-1:].
    return token_logprobs[:, prompt_len - 1:]                         # [B, C]


@torch.no_grad()
def batched_logprobs(model, batch: "RolloutBatch", *, micro_batch: int,
                     disable_adapter: bool = False) -> torch.Tensor:
    """per_token_logprobs over a whole RolloutBatch, chunked to cap memory.

    `disable_adapter=True` turns the LoRA policy back into the frozen base model,
    giving the reference logprobs without holding a second copy in memory.
    """
    ctx = model.disable_adapter() if disable_adapter else nullcontext()
    chunks = []
    with ctx:
        for s in range(0, len(batch), micro_batch):
            sl = slice(s, s + micro_batch)
            chunks.append(per_token_logprobs(
                model, batch.prompt_ids[sl], batch.prompt_mask[sl],
                batch.completion_ids[sl], batch.completion_mask[sl]))
    return torch.cat(chunks, dim=0)


def grpo_loss(new_logprobs, old_logprobs, ref_logprobs, advantages,
              completion_mask, *, clip_eps: float, kl_coef: float):
    """GRPO objective for one (micro-)batch. Returns (scalar loss, metrics).

    Per token:
        ratio  = exp(logπ_new - logπ_old)
        L_pol  = -min(ratio * A, clip(ratio, 1±eps) * A)          # PPO clip
        KL     = exp(logπ_ref - logπ_new) - (logπ_ref - logπ_new) - 1   # k3, unbiased >=0
        L      = L_pol + kl_coef * KL
    Tokens are averaged within each sequence, then sequences are averaged
    (the DeepSeekMath GRPO normalization). The advantage A_i is the same for
    every token in completion i.
    """
    mask = completion_mask.float()
    adv = advantages.unsqueeze(1)                       # [B, 1] -> broadcast over tokens

    ratio = torch.exp(new_logprobs - old_logprobs)
    policy_per_token = -torch.min(
        ratio * adv,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
    )

    diff = ref_logprobs - new_logprobs
    kl_per_token = torch.exp(diff) - diff - 1.0         # k3 estimator

    per_token = policy_per_token + kl_coef * kl_per_token
    seq_lens = mask.sum(dim=1).clamp(min=1.0)
    seq_loss = (per_token * mask).sum(dim=1) / seq_lens  # [B] mean over each seq
    loss = seq_loss.mean()

    with torch.no_grad():
        tok = mask.sum().clamp(min=1.0)
        metrics = {
            "loss": loss.item(),
            "policy_loss": ((policy_per_token * mask).sum() / tok).item(),
            "kl": ((kl_per_token * mask).sum() / tok).item(),
            # fraction of tokens where the ratio left the trust region
            "clip_frac": (((ratio - 1.0).abs() > clip_eps).float() * mask).sum().item()
            / tok.item(),
        }
    return loss, metrics


def train_step(policy, tokenizer, records, optimizer, *, rollout_cfg, train_cfg, device):
    """One full GRPO step: rollout -> advantages -> (old, ref) logprobs ->
    clipped+KL update with gradient accumulation. Returns a metrics dict."""
    micro = train_cfg["micro_batch"]

    # 1) Rollout: sample G completions/prompt with the current policy (no grad).
    policy.eval()
    batch = generate_rollouts(
        policy, tokenizer, records,
        group_size=rollout_cfg["group_size"], max_new_tokens=rollout_cfg["max_new_tokens"],
        temperature=rollout_cfg["temperature"], top_p=rollout_cfg["top_p"], device=device)

    # 2) Group-relative advantages from the verifier rewards.
    advantages = compute_advantages(
        batch.rewards.to(device), batch.group_ids.to(device), eps=train_cfg["adv_eps"])

    # 3) Reference logprobs (frozen base = adapter disabled) and old policy logprobs.
    ref_logprobs = batched_logprobs(policy, batch, micro_batch=micro, disable_adapter=True)
    old_logprobs = batched_logprobs(policy, batch, micro_batch=micro)

    # 4) Update. ppo_epochs>1 reuses this rollout (off-policy); =1 is pure on-policy.
    policy.train()
    B = len(batch)
    n_micro = math.ceil(B / micro)
    agg = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "clip_frac": 0.0}
    for _ in range(train_cfg["ppo_epochs"]):
        optimizer.zero_grad()
        agg = {k: 0.0 for k in agg}
        for s in range(0, B, micro):
            sl = slice(s, s + micro)
            new_logprobs = per_token_logprobs(
                policy, batch.prompt_ids[sl], batch.prompt_mask[sl],
                batch.completion_ids[sl], batch.completion_mask[sl])
            loss, m = grpo_loss(
                new_logprobs, old_logprobs[sl], ref_logprobs[sl], advantages[sl],
                batch.completion_mask[sl], clip_eps=train_cfg["clip_eps"],
                kl_coef=train_cfg["kl_coef"])
            (loss / n_micro).backward()   # accumulate; scale so it's a mean over micro-batches
            for k in agg:
                agg[k] += m[k] / n_micro
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in policy.parameters() if p.requires_grad), train_cfg["grad_clip"])
        optimizer.step()

    agg.update({
        "reward_mean": batch.rewards.mean().item(),
        "reward_std": batch.rewards.std(unbiased=False).item(),
        "frac_correct": batch.correct.float().mean().item(),
        "grad_norm": float(grad_norm),
        "mean_completion_len": batch.completion_mask.sum(1).float().mean().item(),
        "n_completions": B,
    })
    return agg


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
