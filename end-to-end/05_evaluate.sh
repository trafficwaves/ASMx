#!/usr/bin/env bash
# Stage 5 — Quantitative evaluation: IoU of congested contours vs. ground truth.
#
# Note: evaluation/IOU.ipynb hard-codes a specific best-model path
# (model/20250607_221107/best_model_lane{lane}.pt). If you want to evaluate
# the RUN_ID you just produced, edit that path in the notebook first or
# pass --execute with a parameterised papermill run.
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
