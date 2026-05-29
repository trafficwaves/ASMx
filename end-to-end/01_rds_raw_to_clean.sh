#!/usr/bin/env bash
# Stage 1 — Clean and lane-split the RDS raw_record into raw_data/rds/*.csv
#
# Reads:  data/raw_record/rds/{YYYY-MM-DD}.csv
# Writes: data/raw_data/rds/{YYYY-MM-DD}.csv
#
# Run from anywhere; environment is managed by `uv`:
#   bash end-to-end/01_rds_raw_to_clean.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" && uv sync
cd "${REPO_ROOT}/preprocessing"
uv run --project "${REPO_ROOT}" python -u rds_fast_processing.py
