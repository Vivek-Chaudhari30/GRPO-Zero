#!/usr/bin/env bash
# GRPO training run. Run from the repo root.
#   ./scripts/run_train.sh                                  # full run from the config
#   ./scripts/run_train.sh --overfit 16 --steps 30 --lr 5e-5   # overfit sanity check
# On Colab/GPU use the system python:  PYTHON=python ./scripts/run_train.sh
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/grpo_gsm8k.yaml}"

exec "$PY" -m src.train --config "$CONFIG" "$@"
