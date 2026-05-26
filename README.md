# ASMx: Implementation of Adaptive Smoothing Method 

This repository contains the data and code for the paper: Calibrating Adaptive Smoothing Methods for Freeway Traffic Reconstruction", currently under review at European Transport Research Review (ETRR).

---

## 📦 Repository Structure

```
ASMx/
├── calibration/    # Calibration scripts and tools
├── data/           # Datasets for experiments (see data/README.md for the data spec)
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

All dependencies are listed in `environment.yml`.  
To create the environment:

```bash
conda env create -f environment.yml
```

To update the environment file with your current packages:

```bash
conda env export --no-builds > environment.yml
```

---

## 🚦 Reproducibility of the figures in the paper

To reproduce the figures in the paper, you can run the Jupyter notebooks located in the `visualization/` directory. Each notebook corresponds to a specific figure in the paper and is named according to the figure number.

For example, to reproduce Figure 1, you would run `visualization/figure_1.ipynb`.

Make sure to have the environment set up as described above before running the notebooks.

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