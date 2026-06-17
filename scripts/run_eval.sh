#!/usr/bin/env bash
# Baseline (or post-training) eval on GSM8K. Run from the repo root.
#   ./scripts/run_eval.sh                       # uses configs/grpo_gsm8k.yaml
#   ./scripts/run_eval.sh --n 1319 --tag baseline_full   # full test set (GPU)
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/grpo_gsm8k.yaml}"

exec "$PY" -m src.eval --config "$CONFIG" "$@"
