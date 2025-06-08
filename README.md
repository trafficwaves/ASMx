# ASMx
Improved Adaptive Smoothing Method Implementation

## Setup Using Conda (`environment.yml`)

If you already have an `environment.yml` file, you can recreate the exact environment with:

```bash
conda env create -f environment.yml
```

Update the environment with:

```bash
conda env export --no-builds > environment.yml
```

## Run the smoothing
```bash
python ASMNN.py
```


File structure:
```
ASMx/
├── calibration/
├── data/
├── figures/
├── logs/
├── models/
```