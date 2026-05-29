#!/usr/bin/env bash
# Stage 6 — Reproduce every paper figure by executing the visualization notebooks.
#
# Outputs are written next to each notebook (PDFs / PNGs / *.executed.ipynb).
#
# Run from anywhere; environment is managed by `uv`:
#   bash end-to-end/06_figures.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" && uv sync
cd "${REPO_ROOT}/visualization"
for nb in figure1.ipynb figure2-3.ipynb figure4.ipynb figure5.ipynb \
          figure6-7-8.ipynb figure9.ipynb figure10.ipynb; do
  if [[ -f "${nb}" ]]; then
    echo "[figures] executing ${nb}"
    uv run --project "${REPO_ROOT}" jupyter nbconvert --to notebook --execute "${nb}" \
      --output "${nb%.ipynb}.executed.ipynb" \
      --ExecutePreprocessor.timeout=900
  fi
done
