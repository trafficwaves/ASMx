#!/usr/bin/env bash
# Stage 2 — Build the (space, time) RDS speed matrices used by calibration.
#
# Reads:  data/raw_data/rds/{YYYY-MM-DD}.csv
# Writes: data/processed_data/rds/lane{1..4}/{YYYY-MM-DD}.npy  (200, 3600) float32
#
# Run from anywhere; environment is managed by `uv`:
#   bash end-to-end/02_rds_clean_to_matrix.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" && uv sync
cd "${REPO_ROOT}/preprocessing"
uv run --project "${REPO_ROOT}" python -u preprocessing.py
