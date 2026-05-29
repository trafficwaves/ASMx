# End-to-End Reproducibility Guide

This folder ships an **explicit, ordered, copy-pasteable** workflow for reproducing every result in the ASMx paper — from the raw detector dumps to the final figures. It exists because the high-level README in the repository root does not enumerate the exact commands or the order in which each stage of the pipeline must be executed.

If you only want to run things, jump to [§3 Quickstart](#3-quickstart). If you want to understand what each stage consumes and produces, read [§4 Pipeline stages](#4-pipeline-stages).

---

## 1. Pipeline overview

```
                   ┌────────────────────────────────────────────────────────────────┐
                   │                       ASMx pipeline                            │
                   └────────────────────────────────────────────────────────────────┘

  data/raw_record/rds/*.csv                            data/raw_record/motion/*/lane_*.csv
            │                                                          │
            │ Stage 1                                                   │ (pass-through)
            │ preprocessing/rds_fast_processing.py                      │
            ▼                                                           ▼
  data/raw_data/rds/*.csv                              data/raw_data/motion/*/lane_*.csv
            │                                                          │
            │ Stage 2                                                   │ Stage 3
            │ preprocessing/preprocessing.py                            │ production/ASMxMOTION.py
            ▼                                                           ▼
  data/processed_data/rds/lane{1..4}/*.npy             data/processed_data/motion/lane{1..4}/*.npy
            │                                                          │
            └────────────────────────────┬─────────────────────────────┘
                                         │ Stage 4
                                         │ calibration/train.py
                                         │ calibration/train_day_to_day.py
                                         │ calibration/seeds.py
                                         ▼
                              model/{RUNID}/best_model_lane{1..4}.pt
                              logs/calibration/{RUNID}/...
                                         │
                          ┌──────────────┴──────────────┐
                          │                             │
                          │ Stage 5                     │ Stage 6
                          │ evaluation/IOU.ipynb        │ visualization/figure*.ipynb
                          ▼                             ▼
                  IoU tables & curves              Paper figures (PDF/PNG)
```

The pipeline has **6 stages**. Stages 1–3 produce the on-disk inputs that calibration consumes; stage 4 calibrates the six ASM parameters; stages 5–6 are evaluation and figure reproduction. The shipped repository already includes the outputs of every stage, so you can re-enter the pipeline at any point.

---

## 2. Environment

Environment is managed by [`uv`](https://docs.astral.sh/uv/). Dependencies and the Python version are pinned in [`pyproject.toml`](../pyproject.toml) (plus the auto-generated `uv.lock` after the first sync).

```bash
# one-time: install uv (skip if `uv --version` works)
curl -LsSf https://astral.sh/uv/install.sh | sh

# from repository root: create .venv and install everything pinned in pyproject.toml + uv.lock
uv sync
```

`run_pipeline.sh` invokes `uv sync` automatically before every stage, so you can skip the manual sync if you only ever launch through the pipeline. To run a one-off Python entry point inside the project env:

```bash
uv run python production/ASMxRDS.py
uv run jupyter lab     # interactive notebook editing
```

Verify the env can see the bundled data:

```bash
uv run python - <<'PY'
import numpy as np
a = np.load('data/processed_data/rds/lane1/2024-07-09.npy')
b = np.load('data/processed_data/motion/lane1/2024-07-09.npy')
print('RDS  ', a.shape, a.dtype, 'nan_frac=', round(float(np.isnan(a).mean()), 4))
print('Motion', b.shape, b.dtype, 'nan_frac=', round(float(np.isnan(b).mean()), 4))
PY
```

Expected output:

```
RDS   (200, 3600) float32 nan_frac= ~0.99
Motion (200, 3600) float32 nan_frac= 0.0
```

GPU is optional; calibration falls back to CPU automatically (`torch.cuda.is_available()` gate in `calibration/train.py`). The default PyTorch wheel on Linux ships with CUDA 12.x; to force CPU-only run `uv sync --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu`.

> **Legacy conda users.** `environment.yml` is still present but no longer authoritative; if you must use conda, run `conda env create -f environment.yml && conda activate ASMx` and replace every `uv run python ...` below with `python ...`.

---

## 3. Quickstart

Run **every** stage end-to-end:

```bash
# from repository root
bash end-to-end/run_pipeline.sh
```

Run a specific stage group:

```bash
bash end-to-end/run_pipeline.sh init         # only uv sync (materialize .venv)
bash end-to-end/run_pipeline.sh preprocess   # stages 1-3
bash end-to-end/run_pipeline.sh calibrate    # stage 4
bash end-to-end/run_pipeline.sh evaluate     # stage 5
bash end-to-end/run_pipeline.sh figures      # stage 6
bash end-to-end/run_pipeline.sh smoke        # uv + logging sanity check
```

Run an individual stage (no auto-logging — see §3.1):

```bash
bash end-to-end/01_rds_raw_to_clean.sh
bash end-to-end/02_rds_clean_to_matrix.sh
bash end-to-end/03_motion_to_matrix.sh
bash end-to-end/04_calibrate.sh
bash end-to-end/05_evaluate.sh
bash end-to-end/06_figures.sh
```

> Every script must be invoked from the repository root. The scripts handle their own `cd` into the relevant subdirectory because the source files use relative paths such as `../dates.csv` and `data/raw_data/...`.

### 3.1 Auto-logging

`run_pipeline.sh` tees every stage's `stdout + stderr` to its own log file under

```
logs/end-to-end/{RUNID}/
├── 00_pipeline.log         # master transcript (everything you saw on the terminal)
├── 01_rds_raw_to_clean.log
├── 02_rds_clean_to_matrix.log
├── 03_motion_to_matrix.log
├── 04a_train_single_date.log
├── 04b_train_day_to_day.log
├── 04c_seed_sensitivity.log
├── 05_evaluate_iou.log
└── 06_figure{N}.log
```

`RUNID` defaults to the local wall-clock at launch (`YYYYMMDD_HHMMSS`); override it with
`PIPELINE_RUNID=my_label bash end-to-end/run_pipeline.sh ...` if you want a memorable directory name.

If a stage exits non-zero the driver prints `✘ FAIL …` to the master log, preserves the failing stage's log, and halts the pipeline with the stage's exit code so downstream stages do not run.

Quick verification (no real work, ~1 second) that the logging plumbing is healthy:

```bash
bash end-to-end/run_pipeline.sh smoke
ls logs/end-to-end/*/00_smoke_*.log
```

---

## 4. Pipeline stages

Every stage below documents: **what runs**, **inputs** read from disk, **outputs** written to disk, the **explicit command**, and the expected **runtime**. Read `data/README.md` first for the unit and axis conventions that all stages share.

### Stage 1 — RDS raw record → cleaned lane-split CSV

| | |
|---|---|
| **Script** | [../preprocessing/rds_fast_processing.py](../preprocessing/rds_fast_processing.py) |
| **Helpers** | [../preprocessing/FAST.py](../preprocessing/FAST.py) |
| **Reads** | `data/raw_record/rds/{YYYY-MM-DD}.csv` (5 files) |
| **Writes** | `data/raw_data/rds/{YYYY-MM-DD}.csv` (5 files) |
| **What it does** | Parses the stringified `lane_dict_text` into per-lane `speed/volume/occupancy` columns, snaps each report's timestamp to a 30 s grid (`round_to_fix`), maps the TDOT mile marker to the canonical grid, drops everything outside `[58.7, 62.7]` mi and `[06:00, 10:00]` CDT, replaces negative readings with NaN, and back-fills missing lanes from the cross-lane mean. |

```bash
# from repository root (uv resolves the project env on the fly)
cd preprocessing && uv run --project .. python rds_fast_processing.py && cd ..
# OR
bash end-to-end/01_rds_raw_to_clean.sh
```

**Runtime:** ≈ 2–5 minutes per date on a laptop CPU (single process; the multiprocessing pool inside the script is commented out by default).

### Stage 2 — RDS cleaned CSV → (space × time) matrix

| | |
|---|---|
| **Script** | [../preprocessing/preprocessing.py](../preprocessing/preprocessing.py) |
| **Reads** | `data/raw_data/rds/{YYYY-MM-DD}.csv` (5 files) |
| **Writes** | `data/processed_data/rds/lane{1..4}/{YYYY-MM-DD}.npy` (20 files, shape `(200, 3600)`, `float32`, mph) |
| **What it does** | Builds a `pandas` matrix indexed by `(time_unix_fix, milemarker)` on the canonical grid (`dx = 0.02 mi`, `dt = 4 s`, `[58.7, 62.7) × [t0, t0 + 4 h)`), assigns every detector report to its nearest cell, transposes to `(space, time)`, and saves as `float32`. Unobserved cells stay `NaN`. |

```bash
# from repository root
cd preprocessing && uv run --project .. python preprocessing.py && cd ..
# OR
bash end-to-end/02_rds_clean_to_matrix.sh
```

**Runtime:** ≈ 1 minute per (lane, date) combination on a laptop CPU. Output shape is `(200, 3600)` — 200 mile-marker bins, 3600 four-second time bins.

### Stage 3 — Motion raw record → ASM-smoothed ground truth

| | |
|---|---|
| **Script** | [../production/ASMxMOTION.py](../production/ASMxMOTION.py) |
| **Reads** | `data/raw_data/motion/{YYYY-MM-DD}/lane_{1..4}_speed_matrix.csv` (20 files, mi/s) |
| **Writes** | `data/processed_data/motion/lane{1..4}/{YYYY-MM-DD}.npy` (20 files, shape `(200, 3600)`, `float32`, mph) |
| **What it does** | Slices the CSV to `[:3600, :200]`, transposes to `(space, time)`, converts units (`× 3600` → mph), and applies an `AdaptiveSmoothing` torch module with the **default (uncalibrated) parameters** to densify the field. The result is the ground-truth tensor `gt` consumed by calibration. |

```bash
# from repository root (paths inside ASMxMOTION.py are relative to repo root)
uv run python production/ASMxMOTION.py
# OR
bash end-to-end/03_motion_to_matrix.sh
```

**Runtime:** ≈ 30 seconds per (lane, date) on CPU; faster on GPU. The smoothing kernel is FFT-based, so memory rises with the matrix area, not the kernel size.

> **Reproducibility caveat.** Stage 3 uses the *default* ASM parameters baked into `ASMxMOTION.py`, not the calibrated ones. The shipped `data/processed_data/motion/…` files were produced with those defaults; rerun this stage only if you intentionally want to change them.

### Stage 4 — Calibration

| | |
|---|---|
| **Scripts** | [../calibration/train.py](../calibration/train.py), [../calibration/train_day_to_day.py](../calibration/train_day_to_day.py), [../calibration/seeds.py](../calibration/seeds.py) |
| **Reads** | `data/processed_data/rds/lane{1..4}/{date}.npy` (model input) <br/> `data/processed_data/motion/lane{1..4}/{date}.npy` (ground truth) <br/> `dates.csv` (the canonical list) |
| **Writes** | `model/{RUNID}/best_model_lane{1..4}.pt` (single-date) <br/> `model/{RUNID}/{date_id}/best_model_lane{1..4}.pt` (day-to-day) <br/> `logs/calibration/{RUNID}/params_history_lane{1..4}.json` <br/> `logs/calibration/{RUNID}/calibration_log.txt` <br/> `calibration/sensitivity_results.csv` (seed sweep) |
| **What it does** | Loads RDS as the input tensor and motion as the masked target. The six ASM parameters are wrapped as `nn.Parameter`s, optimised by Adam at `lr=1e-1` against a `weighted_rmse` loss (10× weight where target speed < 15 mph). Each epoch the parameters are quantised to two decimals; the best validation RMSE is saved. `RUNID` is the local wall-clock timestamp at launch (`YYYYMMDD_HHMMSS`). |

```bash
# from repository root
cd calibration
uv run --project .. python train.py               # single-date calibration
uv run --project .. python train_day_to_day.py    # day-to-day (date_id = 0..4) calibration
uv run --project .. python seeds.py               # seed-sensitivity sweep
cd ..
# OR
bash end-to-end/04_calibrate.sh
```

**Runtime:** ≈ 3–6 minutes per (lane, date) on CPU; under a minute on GPU. `train.py` runs once (1 date × 4 lanes); `train_day_to_day.py` runs 4 lanes × 5 dates = 20 calibrations.

**How to find your run:** the `RUNID` is printed at the start of each script. After `train.py`:

```bash
ls model/         # contains a new <RUNID> directory with best_model_lane{1..4}.pt
ls logs/calibration/   # contains the matching <RUNID> directory with params + log
```

### Stage 5 — Evaluation (IoU)

| | |
|---|---|
| **Notebook** | [../evaluation/IOU.ipynb](../evaluation/IOU.ipynb) |
| **Reads** | `data/processed_data/rds/lane{lane}/{date}.npy`, `data/processed_data/motion/lane{lane}/{date}.npy`, and a saved model checkpoint from `model/<RUNID>/best_model_lane{lane}.pt` |
| **Writes** | IoU tables (printed inline) and intersection visualisations (inline). |
| **What it does** | Reloads the calibrated `AdaptiveSmoothing` module from a checkpoint, applies it to the RDS matrix, and compares the speed-threshold contour against the motion ground truth via IoU for thresholds 5–30 mph. |

```bash
# from repository root
cd evaluation
uv run --project .. jupyter nbconvert --to notebook --execute IOU.ipynb \
  --output IOU.executed.ipynb --ExecutePreprocessor.timeout=600
cd ..
# OR
bash end-to-end/05_evaluate.sh
```

> **Heads-up.** The notebook hard-codes `model/20250607_221107/best_model_lane{lane}.pt` as the calibrated checkpoint (shipped with the repo for reproducibility of the published numbers). To evaluate a freshly-trained run, change `best_model_path` near the top of the notebook to point at `model/<your RUNID>/best_model_lane{lane}.pt`.

### Stage 6 — Figures

| | |
|---|---|
| **Notebooks** | [../visualization/figure1.ipynb](../visualization/figure1.ipynb), [../visualization/figure2-3.ipynb](../visualization/figure2-3.ipynb), [../visualization/figure4.ipynb](../visualization/figure4.ipynb), [../visualization/figure5.ipynb](../visualization/figure5.ipynb), [../visualization/figure6-7-8.ipynb](../visualization/figure6-7-8.ipynb), [../visualization/figure9.ipynb](../visualization/figure9.ipynb), [../visualization/figure10.ipynb](../visualization/figure10.ipynb) |
| **Reads** | Whatever the previous stages produced + the shipped checkpoints in `model/`. |
| **Writes** | PDFs / PNGs in `figures/` and `*.executed.ipynb` mirrors of each notebook. |

```bash
# from repository root
bash end-to-end/06_figures.sh
```

The script executes each notebook in numeric order with a 15-minute per-notebook timeout. Each figure notebook is self-contained and can also be opened interactively in JupyterLab.

---

## 5. Recovering from partial runs

| If you broke … | rerun starting from |
|---|---|
| The RDS CSV cleaning | `01_rds_raw_to_clean.sh` |
| The RDS matrix grid | `02_rds_clean_to_matrix.sh` (Stage 1 outputs are still valid) |
| The motion smoothing parameters | `03_motion_to_matrix.sh` |
| The model checkpoints | `04_calibrate.sh` |
| Anything in `evaluation/` | `05_evaluate.sh` |
| Anything in `figures/` | `06_figures.sh` |

Every stage is idempotent — rerunning it overwrites the same output files. Stage 4 is the only one that generates a *new* `RUNID` directory each time, so old checkpoints are never overwritten in place.

---

## 6. File listing of `end-to-end/`

| File | Purpose |
|---|---|
| [README.md](README.md) | This document — workflow, commands, expected runtimes. |
| [run_pipeline.sh](run_pipeline.sh) | Master driver: `bash run_pipeline.sh [all\|preprocess\|calibrate\|evaluate\|figures]`. |
| [01_rds_raw_to_clean.sh](01_rds_raw_to_clean.sh) | Stage 1: `raw_record/rds → raw_data/rds`. |
| [02_rds_clean_to_matrix.sh](02_rds_clean_to_matrix.sh) | Stage 2: `raw_data/rds → processed_data/rds`. |
| [03_motion_to_matrix.sh](03_motion_to_matrix.sh) | Stage 3: `raw_data/motion → processed_data/motion`. |
| [04_calibrate.sh](04_calibrate.sh) | Stage 4: train + day-to-day + seed sweep. |
| [05_evaluate.sh](05_evaluate.sh) | Stage 5: execute `evaluation/IOU.ipynb`. |
| [06_figures.sh](06_figures.sh) | Stage 6: execute every `visualization/figure*.ipynb`. |

---

## 7. Troubleshooting

- **`uv: command not found`** — install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` and re-open the shell. `run_pipeline.sh` also probes `~/.local/bin` and `~/.cargo/bin` automatically.
- **`FileNotFoundError: ../dates.csv`** — you are running a script from the wrong directory. Use the wrappers in `end-to-end/` or `cd` to the directory documented in the table for that stage.
- **CUDA out of memory during calibration** — set `CUDA_VISIBLE_DEVICES=` to force CPU, or shrink the calibration window in [../calibration/train.py](../calibration/train.py).
- **`jupyter nbconvert: command not found`** — run `uv sync` (the dep is pinned in `pyproject.toml`); the pipeline driver does this for you.
- **PyTorch wants a different CUDA than the host** — re-sync with the CPU index: `uv sync --extra-index-url https://download.pytorch.org/whl/cpu`.
- **IoU notebook reports a missing checkpoint** — point `best_model_path` at the `RUNID` you just produced (see Stage 5 note).
- **A figure notebook fails on `usetex=True`** — install a working LaTeX toolchain (`texlive-latex-base`, `texlive-fonts-recommended`, `dvipng`) or set `plt.rc('text', usetex=False)` at the top of the notebook.
