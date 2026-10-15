# Kaggle T4 run, 2026-10-15

Real numbers from a Kaggle notebook kernel (`vivekkchaudhari30/grpo-zero-run`, private). Config: `configs/grpo_gsm8k.yaml`.

| Item | Value |
|---|---|
| GPU | 1x Tesla T4 (15 GB; Kaggle exposes 2, `CUDA_VISIBLE_DEVICES=0`) |
| Software | Python 3.13.15, torch 2.11.0+cu128 |
| dtype | bfloat16 (`auto`; the T4 has no native bf16, so it is emulated and slow) |
| Model | Qwen/Qwen2.5-0.5B-Instruct (same id for baseline and trained) |
| Eval | full GSM8K test split, n=1319, greedy pass@1, max_new_tokens=512 |
| Tests | 53 passed |

## Results

| | pass@1 | correct |
|---|---|---|
| Baseline | **0.4397** | 580/1319 |
| GRPO-trained (70 steps) | **0.4329** | 571/1319 |
| Delta | -0.0068 | -9 |

The trained adapter did **not** improve on the baseline. The difference (-9 problems out of 1319) is small and not distinguishable from noise.
Per-problem: 90 baseline-wrong problems became correct, 99 baseline-correct problems became wrong.

## Training was cut short (200 steps planned, 70 used)

- The training run had a 4.5 h wall-clock cap (`timeout 16200`) and reached step 75 of 200 (~215 s/step).
- The adapter evaluated here is the last checkpoint, saved after step 70 (checkpoints every 10 steps). Steps 71-75 are in `train_log.json` but not in the evaluated adapter.
- Mean reward moved only slightly: steps 0-9 = 0.300, steps 66-75 = 0.324 (frac correct 0.234 -> 0.258). KL to reference stayed ~0.001.
- Conclusion: 70 steps at lr 1e-6 is too little training to show a measured gain; this run does not say whether the full 200 steps would help.

## Notes

- The baseline was measured in a first kernel run (Kaggle log time ~66 min) whose training stage failed with `Permission denied` on `scripts/run_train.sh` (not executable). The second run skipped the baseline and trained/evaluated using `bash scripts/run_train.sh`. Both runs used the same code, config, model id, dtype and GPU type.
- The overfit sanity check (16 prompts, 30 steps, lr 5e-5): mean reward first3 0.327 -> last3 0.363. See `overfit_log.json`.
- Shared Kaggle GPU: accuracy is valid evidence, timings are not.
- Files: `metrics.json`, `train_log.json`, `reward_curve.png`, `*_completions.json`, `adapter/` (LoRA), `logs/`.
