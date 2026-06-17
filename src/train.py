"""GRPO training loop with logging.

In M4 this is used for the overfit sanity check (fix a tiny prompt set, repeat,
watch reward climb). In M5 it scales to the full GSM8K train split and plots the
reward/KL curves. The actual GRPO update lives in grpo.train_step.

Usage:
    # M4 overfit sanity check (tiny, runnable on the Mac):
    python -m src.train --overfit 2 --steps 10 --group-size 8 --lr 2e-5
    # M5 full run (GPU):
    python -m src.train --config configs/grpo_gsm8k.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .data import load_gsm8k
from .grpo import train_step
from .model_utils import get_device, load_config, load_policy_for_training

RESULTS_DIR = "results"
STATE_FILE = "training_state.pt"   # optimizer + step + history, for --resume


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(save_dir, model, optimizer, step, history):
    """Save the LoRA adapter (for eval) plus resumable training state so a run
    interrupted by a GPU-quota disconnect can pick up where it left off."""
    from peft import get_peft_model_state_dict

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)  # adapter_model.safetensors — used by eval --adapter
    torch.save(
        {"step": step, "model": get_peft_model_state_dict(model),
         "optimizer": optimizer.state_dict(), "history": history},
        os.path.join(save_dir, STATE_FILE),
    )


def maybe_resume(save_dir, model, optimizer, device):
    """If a training_state.pt exists in save_dir, restore adapter + optimizer +
    history and return (start_step, history). Otherwise start fresh."""
    from peft import set_peft_model_state_dict

    path = os.path.join(save_dir, STATE_FILE)
    if not os.path.exists(path):
        return 0, []
    ckpt = torch.load(path, map_location=device, weights_only=False)
    set_peft_model_state_dict(model, ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start = ckpt["step"] + 1
    print(f"Resuming from {path}: continuing at step {start}")
    return start, ckpt["history"]


def get_prompt_batch(pool, step, prompts_per_step, overfit):
    """Pick the prompts for this step. Overfit mode keeps reusing the fixed pool;
    otherwise we stride through the train split."""
    if overfit:
        # cycle the fixed tiny set so every step trains on the same prompts
        idx = [(step * prompts_per_step + i) % len(pool) for i in range(prompts_per_step)]
        return [pool[i] for i in idx]
    start = (step * prompts_per_step) % max(1, len(pool) - prompts_per_step + 1)
    return pool[start:start + prompts_per_step]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo_gsm8k.yaml")
    ap.add_argument("--overfit", type=int, default=0,
                    help="N>0: overfit this many fixed train prompts (M4 sanity check)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--prompts-per-step", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="override train.micro_batch (lower if you hit CUDA OOM)")
    ap.add_argument("--log", default=os.path.join(RESULTS_DIR, "train_log.json"))
    ap.add_argument("--save-dir", default="checkpoints/grpo_lora",
                    help="where to save the trained LoRA adapter")
    ap.add_argument("--resume", action="store_true",
                    help="resume from save_dir/training_state.pt if present")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rcfg, tcfg = cfg["rollout"], cfg["train"]

    # CLI overrides (handy for the cheap overfit demo without editing the yaml)
    if args.group_size:       rcfg["group_size"] = args.group_size
    if args.max_new_tokens:   rcfg["max_new_tokens"] = args.max_new_tokens
    if args.prompts_per_step: tcfg["prompts_per_step"] = args.prompts_per_step
    if args.lr:               tcfg["lr"] = args.lr
    if args.micro_batch:      tcfg["micro_batch"] = args.micro_batch
    steps = args.steps if args.steps is not None else tcfg["steps"]

    set_seed(tcfg["seed"])
    device = get_device()
    print(f"Device: {device} | model: {cfg['model']['id']}")
    print(f"G={rcfg['group_size']} prompts/step={tcfg['prompts_per_step']} "
          f"max_new_tokens={rcfg['max_new_tokens']} lr={tcfg['lr']} "
          f"kl_coef={tcfg['kl_coef']} steps={steps}"
          + (f" | OVERFIT n={args.overfit}" if args.overfit else ""))

    model, tokenizer, device = load_policy_for_training(
        cfg["model"]["id"], lora_cfg=tcfg["lora"], dtype_name=cfg["model"].get("dtype", "auto"),
        device=device, grad_checkpoint=tcfg.get("grad_checkpoint", False))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=tcfg["lr"])

    start_step, history = (maybe_resume(args.save_dir, model, optimizer, device)
                           if args.resume else (0, []))

    n_pool = args.overfit if args.overfit else max(steps * tcfg["prompts_per_step"], 256)
    pool = load_gsm8k("train", n=n_pool)
    print(f"Prompt pool: {len(pool)} problems\n")

    t0 = time.time()
    for step in range(start_step, steps):
        records = get_prompt_batch(pool, step, tcfg["prompts_per_step"], args.overfit)
        m = train_step(model, tokenizer, records, optimizer,
                       rollout_cfg=rcfg, train_cfg=tcfg, device=device)
        m["step"] = step
        m["elapsed_s"] = round(time.time() - t0, 1)
        history.append(m)
        print(f"step {step:3d} | reward {m['reward_mean']:.3f} "
              f"(correct {m['frac_correct']:.2f}) | loss {m['loss']:+.4f} "
              f"| KL {m['kl']:.4f} | clip {m['clip_frac']:.2f} "
              f"| |g| {m['grad_norm']:.2f} | len {m['mean_completion_len']:.0f} "
              f"| {m['elapsed_s']:.0f}s", flush=True)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(args.log, "w") as f:
            json.dump(history, f, indent=2)

        # periodic checkpoint so a long run survives a GPU-quota disconnect (resume with --resume)
        if tcfg.get("eval_every") and (step + 1) % tcfg["eval_every"] == 0:
            save_checkpoint(args.save_dir, model, optimizer, step, history)
            print(f"  (checkpoint -> {args.save_dir})", flush=True)

    save_checkpoint(args.save_dir, model, optimizer, steps - 1, history)
    print(f"\nDone. {steps} steps in {time.time()-t0:.0f}s. Log -> {args.log}")
    print(f"Adapter saved -> {args.save_dir}")
    if history:
        first = sum(h["reward_mean"] for h in history[:3]) / min(3, len(history))
        last = sum(h["reward_mean"] for h in history[-3:]) / min(3, len(history))
        print(f"mean reward: first3={first:.3f} -> last3={last:.3f} "
              f"(delta {last-first:+.3f})")


if __name__ == "__main__":
    main()
