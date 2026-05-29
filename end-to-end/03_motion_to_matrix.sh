#!/usr/bin/env bash
# Stage 3 — Convert the motion CSVs to ASM-smoothed ground-truth matrices.
#
# Reads:  data/raw_data/motion/{YYYY-MM-DD}/lane_{1..4}_speed_matrix.csv
# Writes: data/processed_data/motion/lane{1..4}/{YYYY-MM-DD}.npy  (200, 3600) float32 (mph)
#
# Run from anywhere; environment is managed by `uv`:
#   bash end-to-end/03_motion_to_matrix.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" && uv sync
uv run --project "${REPO_ROOT}" python -u production/ASMxMOTION.py
