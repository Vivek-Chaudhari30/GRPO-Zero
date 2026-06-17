"""Plot the GRPO training curves (mean reward + KL) from results/train_log.json.

    python scripts/plot_curves.py [train_log.json] [out.png]
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")  # headless (Colab / remote box)
import matplotlib.pyplot as plt  # noqa: E402


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "results/train_log.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results/reward_curve.png"

    with open(log_path) as f:
        hist = json.load(f)
    if not hist:
        raise SystemExit(f"{log_path} is empty")

    steps = [h["step"] for h in hist]
    reward = [h["reward_mean"] for h in hist]
    kl = [h.get("kl", 0.0) for h in hist]
    frac = [h.get("frac_correct", 0.0) for h in hist]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(steps, reward, color="C0", label="mean reward")
    ax1.plot(steps, frac, color="C2", alpha=0.5, linestyle="--", label="frac correct")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("reward / accuracy", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(steps, kl, color="C1", alpha=0.6, label="KL to ref")
    ax2.set_ylabel("KL to reference", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=9)
    plt.title("GRPO on GSM8K (RLVR): reward and KL vs. step")
    fig.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
