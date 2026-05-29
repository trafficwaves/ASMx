#!/usr/bin/env bash
# ============================================================================
# ASMx end-to-end pipeline
# ----------------------------------------------------------------------------
# Runs every stage from raw_record → processed_data → calibration → evaluation
# → figures, in the same order described in end-to-end/README.md.
#
# Usage (from repository root):
#   bash end-to-end/run_pipeline.sh                  # run every stage
#   bash end-to-end/run_pipeline.sh init             # only ensure .venv is in sync
#   bash end-to-end/run_pipeline.sh preprocess       # only stages 1-3
#   bash end-to-end/run_pipeline.sh calibrate        # only stage 4
#   bash end-to-end/run_pipeline.sh evaluate         # only stage 5
#   bash end-to-end/run_pipeline.sh figures          # only stage 6
#
# Every stage's stdout+stderr is teed to:
#   logs/end-to-end/{RUNID}/{NN_stage_name}.log
# and the combined transcript to:
#   logs/end-to-end/{RUNID}/00_pipeline.log
#
# Override RUNID by exporting PIPELINE_RUNID before invoking.
#
# Environment is managed by `uv` (https://docs.astral.sh/uv/).
# `uv sync` is invoked automatically at the start of every run to materialize
# .venv/ from pyproject.toml + uv.lock (idempotent — no-op if already in sync).
# Every Python invocation goes through `uv run python` so the local .venv is
# used regardless of the caller's shell state.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${1:-all}"
RUNID="${PIPELINE_RUNID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/logs/end-to-end/${RUNID}"
mkdir -p "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/00_pipeline.log"
# Redirect all subsequent output (including from `log` calls and child commands)
# through tee so we get both a live terminal view and a permanent transcript.
exec > >(tee -a "${MASTER_LOG}") 2>&1

log() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

# Ensure `uv` is on PATH (Astral's installer typically drops it in ~/.local/bin
# or ~/.cargo/bin); fall back to those locations if necessary.
if ! command -v uv >/dev/null 2>&1; then
  for candidate in "${HOME}/.local/bin" "${HOME}/.cargo/bin"; do
    if [[ -x "${candidate}/uv" ]]; then
      export PATH="${candidate}:${PATH}"
      break
    fi
  done
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 127
fi

# UV is the canonical way to launch a Python interpreter inside this repo.
# Using an array lets `set -u` stay happy when the command is expanded.
UV_RUN=(uv run --project "${REPO_ROOT}" python -u)

# run_stage <slug> <command...>
#   Executes the command, tees its output to logs/end-to-end/{RUNID}/{slug}.log,
#   and records the wall-clock duration + exit code in the master log.
run_stage() {
  local slug="$1"; shift
  local stage_log="${LOG_DIR}/${slug}.log"
  local started ended status
  started=$(date +%s)
  log "▶ START ${slug} — logging to ${stage_log#${REPO_ROOT}/}"
  set +e
  ( "$@" ) 2>&1 | tee "${stage_log}"
  status=${PIPESTATUS[0]}
  set -e
  ended=$(date +%s)
  if [[ ${status} -eq 0 ]]; then
    log "✔ DONE  ${slug} ($((ended - started))s)"
  else
    log "✘ FAIL  ${slug} (exit ${status}, $((ended - started))s) — see ${stage_log}"
    exit ${status}
  fi
}

log "ASMx pipeline RUNID=${RUNID}   stage=${STAGE}   uv=$(uv --version 2>&1 | head -1)"
log "Master log: ${MASTER_LOG}"

# ----------------------------------------------------------------------------
# Stage 0 — Ensure .venv is materialized from pyproject.toml + uv.lock.
# `uv sync` is idempotent: it's a no-op if everything already matches.
# Runs before every other stage so individual stage invocations also self-heal.
# ----------------------------------------------------------------------------
stage_init() {
  run_stage "00_uv_sync" \
    bash -c "cd '${REPO_ROOT}' && uv sync"
}

# ----------------------------------------------------------------------------
# Stage 1 — raw_record/rds → raw_data/rds
# ----------------------------------------------------------------------------
stage_rds_clean() {
  run_stage "01_rds_raw_to_clean" \
    bash -c "cd '${REPO_ROOT}/preprocessing' && ${UV_RUN[*]} rds_fast_processing.py"
}

# ----------------------------------------------------------------------------
# Stage 2 — raw_data/rds → processed_data/rds
# ----------------------------------------------------------------------------
stage_rds_matrix() {
  run_stage "02_rds_clean_to_matrix" \
    bash -c "cd '${REPO_ROOT}/preprocessing' && ${UV_RUN[*]} preprocessing.py"
}

# ----------------------------------------------------------------------------
# Stage 3 — raw_data/motion → processed_data/motion
# ----------------------------------------------------------------------------
stage_motion_matrix() {
  run_stage "03_motion_to_matrix" \
    bash -c "cd '${REPO_ROOT}' && ${UV_RUN[*]} production/ASMxMOTION.py"
}

# ----------------------------------------------------------------------------
# Stage 4 — Calibration (single-date + day-to-day + seed sensitivity)
# ----------------------------------------------------------------------------
stage_calibrate() {
  run_stage "04a_train_single_date" \
    bash -c "cd '${REPO_ROOT}/calibration' && ${UV_RUN[*]} train.py"
  run_stage "04b_train_day_to_day" \
    bash -c "cd '${REPO_ROOT}/calibration' && ${UV_RUN[*]} train_day_to_day.py"
  run_stage "04c_seed_sensitivity" \
    bash -c "cd '${REPO_ROOT}/calibration' && ${UV_RUN[*]} seeds.py"
}

# ----------------------------------------------------------------------------
# Stage 5 — Evaluation (IoU notebook)
# ----------------------------------------------------------------------------
stage_evaluate() {
  run_stage "05_evaluate_iou" \
    bash -c "cd '${REPO_ROOT}/evaluation' && \
      uv run --project '${REPO_ROOT}' jupyter nbconvert --to notebook --execute IOU.ipynb \
        --output IOU.executed.ipynb --ExecutePreprocessor.timeout=600"
}

# ----------------------------------------------------------------------------
# Stage 6 — Figures (every visualization/figure*.ipynb)
# ----------------------------------------------------------------------------
stage_figures() {
  local nbs=(figure1.ipynb figure2-3.ipynb figure4.ipynb figure5.ipynb \
             figure6-7-8.ipynb figure9.ipynb figure10.ipynb)
  for nb in "${nbs[@]}"; do
    if [[ -f "${REPO_ROOT}/visualization/${nb}" ]]; then
      local slug="06_$(echo "${nb%.ipynb}" | tr -c '[:alnum:]' '_')"
      run_stage "${slug}" \
        bash -c "cd '${REPO_ROOT}/visualization' && \
          uv run --project '${REPO_ROOT}' jupyter nbconvert --to notebook --execute '${nb}' \
            --output '${nb%.ipynb}.executed.ipynb' \
            --ExecutePreprocessor.timeout=900"
    fi
  done
}

case "${STAGE}" in
  init)
    stage_init ;;
  preprocess)
    stage_init; stage_rds_clean; stage_rds_matrix; stage_motion_matrix ;;
  calibrate)
    stage_init; stage_calibrate ;;
  evaluate)
    stage_init; stage_evaluate ;;
  figures)
    stage_init; stage_figures ;;
  all)
    stage_init
    stage_rds_clean
    stage_rds_matrix
    stage_motion_matrix
    stage_calibrate
    stage_evaluate
    stage_figures
    ;;
  smoke)
    # Smoke test: verify the logging infrastructure + that uv is wired up.
    stage_init
    run_stage "00_smoke_echo"   bash -c "echo 'hello from stage A'; echo 'line 2'"
    run_stage "00_smoke_python" bash -c "${UV_RUN[*]} -c 'import sys; print(\"py stdout\"); print(\"py stderr\", file=sys.stderr)'"
    run_stage "00_smoke_dates"  bash -c "cd '${REPO_ROOT}' && ${UV_RUN[*]} -c 'import pandas as pd; print(pd.read_csv(\"dates.csv\"))'"
    run_stage "00_smoke_imports" bash -c "${UV_RUN[*]} -c 'import numpy, pandas, matplotlib, scipy, torch, thop, pytz; print(\"all imports OK; torch=\", torch.__version__)'"
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Valid stages: init | preprocess | calibrate | evaluate | figures | all | smoke" >&2
    exit 2 ;;
esac

log "Pipeline stage '${STAGE}' completed. Logs in ${LOG_DIR#${REPO_ROOT}/}/"
