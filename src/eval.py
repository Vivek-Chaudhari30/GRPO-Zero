"""Baseline / post-training evaluation: pass@1 (greedy) and pass@k (sampled) on
GSM8K, scored by the RLVR verifier. Writes the real number to results/metrics.json.

Usage:
    python -m src.eval --config configs/grpo_gsm8k.yaml          # pass@1 on cfg.eval.n
    python -m src.eval --config configs/grpo_gsm8k.yaml --n 50   # override count
    python -m src.eval --config configs/grpo_gsm8k.yaml --pass-k 4 --tag baseline_passk
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .data import load_gsm8k, render_prompt
from .model_utils import load_config, load_model_and_tokenizer
from .verifier import score_completion

RESULTS_DIR = "results"
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")


@torch.no_grad()
def generate(model, tokenizer, prompts, *, max_new_tokens, do_sample, temperature,
             top_p, num_return_sequences, device):
    """Batched generation. Returns, per prompt, a list of `num_return_sequences`
    decoded completion strings (prompt tokens stripped)."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=1024).to(device)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                  num_return_sequences=num_return_sequences)
    if do_sample:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False)
    out = model.generate(**enc, **kwargs)
    # left-padding makes every prompt the same length, so slice off that prefix.
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    k = num_return_sequences
    return [texts[i * k:(i + 1) * k] for i in range(len(prompts))]


def evaluate(model, tokenizer, records, *, device, batch_size, max_new_tokens,
             k=1, temperature=0.8, top_p=0.95, verbose=True):
    """pass@k over `records`. k=1 -> greedy pass@1; k>1 -> sampled, correct if ANY
    of the k samples is correct. Returns (accuracy, details list)."""
    do_sample = k > 1
    n_correct = 0
    details = []
    t0 = time.time()
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        prompts = [render_prompt(r["messages"], tokenizer) for r in batch]
        per_prompt = generate(
            model, tokenizer, prompts,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p,
            num_return_sequences=k, device=device,
        )
        for rec, comps in zip(batch, per_prompt):
            scored = [score_completion(c, rec["gold"]) for c in comps]
            hit = any(s["correct"] for s in scored)
            n_correct += int(hit)
            details.append({
                "question": rec["question"],
                "gold": rec["gold"],
                "pred": scored[0]["pred"],
                "correct": hit,
                "completion": comps[0],
            })
        if verbose:
            done = min(start + batch_size, len(records))
            print(f"  [{done}/{len(records)}] running pass@{k} acc="
                  f"{n_correct / done:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return n_correct / len(records), details


def save_metric(tag: str, payload: dict):
    """Merge one tagged result into results/metrics.json (keeps prior tags)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            data = json.load(f)
    data[tag] = payload
    with open(METRICS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return METRICS_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo_gsm8k.yaml")
    ap.add_argument("--n", type=int, default=None, help="override eval.n")
    ap.add_argument("--pass-k", type=int, default=None, help="override eval.pass_k")
    ap.add_argument("--model", default=None, help="override model.id")
    ap.add_argument("--tag", default="baseline", help="key under which to save the metric")
    ap.add_argument("--save-completions", default=None,
                    help="optional path to dump per-example completions as JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ecfg = cfg["eval"]
    model_id = args.model or cfg["model"]["id"]
    n = args.n if args.n is not None else ecfg["n"]
    k = args.pass_k if args.pass_k is not None else ecfg["pass_k"]

    torch.manual_seed(ecfg.get("seed", 0))

    print(f"Loading {model_id} ...", flush=True)
    model, tokenizer, device = load_model_and_tokenizer(
        model_id, dtype_name=cfg["model"].get("dtype", "auto"))
    print(f"Device: {device}, dtype: {next(model.parameters()).dtype}", flush=True)

    records = load_gsm8k(ecfg["split"], n=n)
    print(f"Evaluating pass@{k} on {len(records)} GSM8K {ecfg['split']} problems "
          f"(max_new_tokens={ecfg['max_new_tokens']})", flush=True)

    acc, details = evaluate(
        model, tokenizer, records, device=device,
        batch_size=ecfg["batch_size"], max_new_tokens=ecfg["max_new_tokens"],
        k=k, temperature=ecfg["temperature"], top_p=ecfg["top_p"],
    )

    metric = f"pass@{k}"
    payload = {
        "model": model_id,
        "split": ecfg["split"],
        "n": len(records),
        "metric": metric,
        "accuracy": round(acc, 4),
        "max_new_tokens": ecfg["max_new_tokens"],
        "device": str(device),
        "full_test_set": n is None,
    }
    path = save_metric(args.tag, payload)
    print(f"\n==> {metric} = {acc:.4f} ({int(acc * len(records))}/{len(records)})")
    print(f"==> saved to {path} under tag '{args.tag}'")

    if args.save_completions:
        os.makedirs(os.path.dirname(args.save_completions) or ".", exist_ok=True)
        with open(args.save_completions, "w") as f:
            json.dump(details, f, indent=2)
        print(f"==> completions dumped to {args.save_completions}")


if __name__ == "__main__":
    main()
