"""Make the README figures for a finished run: before/after accuracy and training curves.

    python scripts/make_report_figures.py [results_dir]

Reads metrics.json (baseline_full vs trained), train_log_lr1e5.json and the two
full-test-set completions files; writes results/figures/{before_after,training_curves}.png
and prints the per-problem comparison used in results/RUN_2026-10-29.md.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless (Colab / remote box)
import matplotlib.pyplot as plt  # noqa: E402

BLUE, GRAY, INK, MUTED, SURFACE = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d5d0")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def rolling(xs, w):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]


def before_after(metrics, out):
    base, trained = metrics["baseline_full"], metrics["trained"]
    n = base["n"]
    labels = ["Baseline\n(untrained)", "After GRPO\n(200 steps, LoRA)"]
    accs = [base["accuracy"], trained["accuracy"]]
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    style(ax)
    bars = ax.bar(labels, [a * 100 for a in accs], color=[GRAY, BLUE], width=0.5)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a * 100 + 1, f"{a * 100:.2f}%\n({round(a * n)}/{n})",
                ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_ylim(0, 60)
    ax.set_ylabel("pass@1 (%)", color=MUTED)
    delta = (accs[1] - accs[0]) * 100
    ax.set_title(f"Qwen2.5-0.5B-Instruct on full GSM8K test (n={n}): +{delta:.1f} pts",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)


def training_curves(hist, out, w=20):
    steps = [h["step"] for h in hist]
    correct = [h["frac_correct"] * 100 for h in hist]
    kl = [h["kl"] for h in hist]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [3, 2]})
    for ax in (a1, a2):
        style(ax)
    a1.plot(steps, correct, color=BLUE, alpha=0.25, linewidth=1)
    a1.plot(steps, rolling(correct, w), color=BLUE, linewidth=2)
    a1.set_ylabel("rollouts correct (%)", color=MUTED)
    a1.set_title(f"Training: share of the 64 rollouts per step that are correct (thin = per step, "
                 f"bold = {w}-step mean)", color=INK, fontsize=10, loc="left")
    a2.plot(steps, kl, color=BLUE, linewidth=2)
    a2.set_ylabel("KL to reference", color=MUTED)
    a2.set_xlabel("GRPO step", color=MUTED)
    a2.set_title("KL divergence from the frozen base model", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)


def compare(base, trained):
    assert len(base) == len(trained) and all(b["question"] == t["question"] for b, t in zip(base, trained))
    fixed = sum(not b["correct"] and t["correct"] for b, t in zip(base, trained))
    broke = sum(b["correct"] and not t["correct"] for b, t in zip(base, trained))
    both = sum(b["correct"] and t["correct"] for b, t in zip(base, trained))
    print(f"both correct {both} | fixed by GRPO {fixed} | broken by GRPO {broke} | "
          f"both wrong {len(base) - both - fixed - broke}")


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "results"
    fdir = os.path.join(rdir, "figures")
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(rdir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(rdir, "train_log_lr1e5.json")) as f:
        hist = json.load(f)
    before_after(metrics, os.path.join(fdir, "before_after.png"))
    training_curves(hist, os.path.join(fdir, "training_curves.png"))
    with open(os.path.join(rdir, "baseline_completions_full.json")) as f:
        base = json.load(f)
    with open(os.path.join(rdir, "trained_completions.json")) as f:
        trained = json.load(f)
    compare(base, trained)
    print(f"saved figures to {fdir}")


if __name__ == "__main__":
    main()
