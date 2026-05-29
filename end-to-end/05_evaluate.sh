#!/usr/bin/env bash
# Stage 5 — Quantitative evaluation: IoU of congested contours vs. ground truth.
#
# Note: evaluation/IOU.ipynb auto-discovers the most recent calibrated model
# by globbing ../model/*/best_model_lane{lane}.pt and picking the
# lexicographically latest match (timestamp-named dirs sort correctly).
# Run stage 4 (calibration) before this stage so a model is available.
#
# Run from anywhere; environment is managed by `uv`:
#   bash end-to-end/05_evaluate.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" && uv sync
cd "${REPO_ROOT}/evaluation"
uv run --project "${REPO_ROOT}" jupyter nbconvert --to notebook --execute IOU.ipynb \
  --output IOU.executed.ipynb --ExecutePreprocessor.timeout=600
echo "[evaluate] wrote evaluation/IOU.executed.ipynb"
