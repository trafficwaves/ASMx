# ASMx: Implementation of Adaptive Smoothing Method 

This repository contains the data and code for the paper: Calibrating Adaptive Smoothing Methods for Freeway Traffic Reconstruction", currently under review at European Transport Research Review (ETRR).

---

## 📦 Repository Structure

```
ASMx/
├── calibration/    # Calibration scripts and tools
├── data/           # Datasets for experiments (see data/README.md for the data spec)
├── end-to-end/     # End-to-end reproducibility scripts + workflow docs (see end-to-end/README.md)
├── evaluation/     # Evaluation scripts and metrics
├── figures/        # Generated figures and plots
├── logs/           # Log files from runs
├── models/         # Saved model parameters and checkpoints
├── preprocessing/  # Data preprocessing scripts
├── production/     # Production scripts for running the smoothing method
├── results/        # Results from experiments (independet of the raw data for reproducibility)
├── visualization/  # Visualization jupyternotebooks to generate figures (sorted by the figure number)
├── .gitignore      # Git ignore file
├── ASM_utils.py    # Utility functions for the adaptive smoothing method
├── dates.csv       # Dates for the data included in this repository
├── environment.yml # Conda environment file
├── LICENSE         # License file (MIT License)
├── README.md       # This file

```

---

## ⚙️ Environment Setup

We use [`uv`](https://docs.astral.sh/uv/) (Astral's fast Python package manager) as the canonical way to manage the environment. The legacy `environment.yml` (conda) is kept for users who prefer conda but is no longer authoritative.

### Option A — `uv` (recommended)

```bash
# one-time: install uv (skip if `uv --version` works)
curl -LsSf https://astral.sh/uv/install.sh | sh

# from repository root: materialize .venv from pyproject.toml + uv.lock
uv sync
```

Run any Python entry point through `uv run` so the project venv is used regardless of shell state:

```bash
uv run python production/ASMxRDS.py
uv run jupyter lab        # to open the visualization/ notebooks interactively
```

The end-to-end pipeline calls `uv sync` automatically before each stage — see [end-to-end/README.md](end-to-end/README.md).

### Option B — conda (legacy)

```bash
conda env create -f environment.yml
conda activate ASMx
```

---

## 🚦 Reproducibility of the figures in the paper

To reproduce the figures in the paper, you can run the Jupyter notebooks located in the `visualization/` directory. Each notebook corresponds to a specific figure in the paper and is named according to the figure number.

For example, to reproduce Figure 1, you would run `visualization/figure_1.ipynb`.

Make sure to have the environment set up as described above before running the notebooks.

## 🔁 End-to-end reproducibility (preprocessing → evaluation)

For a complete, ordered pipeline — raw record → cleaned data → matrices → calibration → evaluation → figures — see [end-to-end/README.md](end-to-end/README.md). That document spells out every command, the exact directory each script must be invoked from, the inputs and outputs of each stage, and expected runtimes. A one-line master runner is provided:

```bash
bash end-to-end/run_pipeline.sh           # run every stage
bash end-to-end/run_pipeline.sh calibrate # or any single stage group
```

## 🧪 Experimental Settings

For full reproducibility, the parameters, random seeds, and discretization constants used in our experiments are listed below. They match the values hard-coded in [calibration/train.py](calibration/train.py), [calibration/train_day_to_day.py](calibration/train_day_to_day.py), and [calibration/seeds.py](calibration/seeds.py); change them there if you wish to re-run with different settings.

### Random seeds

| Script | Seed(s) | Purpose |
|---|---|---|
| [calibration/train.py:15](calibration/train.py#L15) | `42` | Single-day calibration (main result) |
| [calibration/train_day_to_day.py:17](calibration/train_day_to_day.py#L17) | `42` | Day-to-day generalization runs |
| [calibration/seeds.py:137](calibration/seeds.py#L137) | `1, 2, …, 9` | Seed-sensitivity analysis (`sensitivity_results.csv`) |

All seeds are applied to NumPy, Python `random`, and PyTorch (CPU and CUDA via `torch.cuda.manual_seed_all`). Matmul precision is set to `medium` (`torch.set_float32_matmul_precision('medium')`).

### Spatio-temporal discretization

| Symbol | Value | Unit | Meaning |
|---|---|---|---|
| `dx` | `0.02` | mile | Spatial cell width |
| `dt` | `4.0` | sec | Temporal cell width |
| `kernel_time_window` | `T · dt` | sec | Set from matrix shape at runtime |
| `kernel_space_window` | `X · dx` | mile | Set from matrix shape at runtime |

### Adaptive-smoothing model — initial parameters

Used to construct `AdaptiveSmoothing(...)` at the start of every calibration run.

| Parameter | Init value | Unit | Description |
|---|---|---|---|
| `tau` | `15.0` | sec | Temporal kernel bandwidth |
| `delta` | `0.15` | mile | Spatial kernel bandwidth |
| `c_cong` | `9.3` | mph | Congested wave speed |
| `c_free` | `-43.5` | mph | Free-flow wave speed |
| `v_thr` | `37.3` | mph | Regime-transition speed threshold |
| `v_delta` | `12.4` | mph | Regime-transition smoothing width |

Per-epoch constraints applied in-place: `c_free ≥ -60` (clamped), and all six parameters are quantized to two decimals.

### Optimization

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-1` |
| Epochs | `1000` |
| Loss | Weighted RMSE (see below) |
| Lanes | `1, 2, 3, 4` (calibrated independently) |
| Train/val date | `dates[1]` from [dates.csv](dates.csv) (i.e., `2024-07-09`) |
| Device | CUDA if available, else CPU |

The weighted RMSE up-weights low-speed (congested) samples to balance the loss surface:

- `threshold = 15.0` mph — targets below this are up-weighted
- `high_weight = 10.0` — multiplier applied to those samples

For the optional combined loss (`combined_loss`), the RMSE/Wasserstein mixing coefficient is `alpha = 1.0` (i.e., pure RMSE by default).

### Logging

Each run is tagged with a timestamp `runid` (`YYYYMMDD_HHMMSS`); per-epoch parameter snapshots, train/val RMSE, and the best checkpoint are written under `logs/calibration/<runid>/` and `models/<runid>/`.

## 🧾 Credibility of the Computational Experiments

The log files from the calibration are stored in the `logs/calibration` directory. Each experiment is recorded by the date and time it was run, and the log files are named accordingly. Log files are generated automatically based on your local machine time. When you run the code in `calibration/train.py`, the log files will be created in the `logs/calibration` directory.

## 📊 Availability of the Computational Experiments

All data required for reproducing the experiments—including raw records from the queried database, preprocessed data, and the matrix data used as model input—are provided in this repository. All code and experimental results are also included to ensure full reproducibility. For any questions or further assistance, please contact us.

## 📑 Data Documentation

The contents of every file under `data/`, the meaning of each column, units, and the pipeline that produces them are documented in [data/README.md](data/README.md). It covers the three pipeline stages — `raw_record/`, `raw_data/`, `processed_data/` — for both the RDS detector data and the motion-based ground truth, along with the demonstration assets at the root of `data/`. We welcome feedback from users to help improve and iterate on the data standard moving forward.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributors

This project is maintained by the following contributors:

- Junyi Ji, Vanderbilt University