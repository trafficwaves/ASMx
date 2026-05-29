#!/usr/bin/env bash
# Stage 4 — Calibrate the six ASM parameters (tau, delta, c_cong, c_free, v_thr, v_delta).
#
# Reads:  data/processed_data/rds/lane{1..4}/{YYYY-MM-DD}.npy  (model input)
#         data/processed_data/motion/lane{1..4}/{YYYY-MM-DD}.npy  (ground truth)
# Writes: model/{RUNID}/best_model_lane{1..4}.pt
#         model/{RUNID}/{date_id}/best_model_lane{1..4}.pt  (day-to-day)
#         logs/calibration/{RUNID}/...                       (per-epoch JSON + log)
#         calibration/sensitivity_results.csv                (seed sweep)
#
# RUNID is generated from the local wall clock at launch (YYYYMMDD_HHMMSS).
#
# Run from anywhere; environment is managed by `uv`:
#   bash end-to-end/04_calibrate.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" && uv sync
cd "${REPO_ROOT}/calibration"

echo "[calibration] single-date training (train.py)"
uv run --project "${REPO_ROOT}" python -u train.py

echo "[calibration] day-to-day training (train_day_to_day.py)"
uv run --project "${REPO_ROOT}" python -u train_day_to_day.py

echo "[calibration] seed sensitivity sweep (seeds.py)"
uv run --project "${REPO_ROOT}" python -u seeds.py
