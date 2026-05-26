# Data Documentation

This directory contains every dataset used to reproduce the experiments in the ASMx paper. The data covers the I-24 westbound corridor near Nashville, TN, mile markers **53.3–70.1**, **four lanes**, **five weekdays from 2024-07-08 through 2024-07-12**, **06:00–10:00 America/Chicago**. The canonical date list is [../dates.csv](../dates.csv).

Two measurement modalities are propagated through three pipeline stages:

- **RDS** — fixed-location loop-detector aggregates queried from the TDOT roadside detector system. Sparse: a detector only reports when something passes.
- **Motion** — vehicle-trajectory–derived speed field. Dense: every space-time cell carries a value. After post-processing it is used as the **ground truth** for ASM calibration.

```
raw_record/   →   raw_data/   →   processed_data/
 (DB dump)     (lane-split,     (regular space-time
                NaN-cleaned)     matrices, .npy)
```

---

## 1. Conventions used across the directory

| Item | Value |
| --- | --- |
| Spatial domain | mile markers 53.3 → 70.1 (westbound) |
| Time zone | America/Chicago (UTC−05:00 in summer) |
| Time-of-day window | 06:00–10:00 local |
| Lane numbering | `lane1` … `lane4` in the canonical pipeline files. The leftmost (passing) lane is `lane1`; the rightmost is `lane4`. The demo files in §4 use `lane0`…`lane3` instead — see the note there. |
| Time encoding | Unix epoch seconds (`time_unix`, `time_unix_fix`); `time_unix_fix` is snapped to a 30 s grid by `round_to_fix` in [../preprocessing/FAST.py](../preprocessing/FAST.py) |
| Missing values | Negative sensor readings (transmission errors) are converted to `NaN` in stage 2; unmeasured space-time cells in stage 3 are stored as `NaN` |
| Processed-array axis order | `(space, time)` — row index = mile-marker bin, column index = time bin |
| Processed-array dtype | `float32` |
| Resolution of processed matrices | `dx = 0.02 mi`, `dt = 4 s` |

---

## 2. Stage 1 — `raw_record/`

Exact dump of the upstream queries. One row per detector report; no cleaning.

### 2.1 `raw_record/rds/{YYYY-MM-DD}.csv` — 5 files (one per date)

8 columns, one row per detector report (≈46 k rows/day, ≈12 MB/day).

| Column | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `link_update_time` | ISO 8601 string with TZ offset | — | Sensor report timestamp, e.g. `2024-07-08 10:59:55.276203-05:00` |
| `speed` | float | mph | Cross-lane mean speed at this mile marker |
| `occupancy` | float | percent | Cross-lane mean occupancy |
| `volume` | float | vehicles per report | Cross-lane aggregated vehicle count for this report |
| `smooth_speed` | float | mph | Vendor-side smoothed speed |
| `smooth_occupancy` | float | percent | Vendor-side smoothed occupancy |
| `milemarker` | float | miles | TDOT detector mile marker |
| `lane_dict_text` | stringified Python dict | mixed | Per-lane raw values: `{sensor_id: [name, speed_mph, volume, occupancy_pct], ...}`. Parsed by `extract_lane_speed`, `extract_lane_volume`, `extract_lane_occupancy` in [../preprocessing/FAST.py](../preprocessing/FAST.py). The lane index lives inside `name`, e.g. `R3G-00I24-59.0W (277)- Lane4`. |

Open item: a small number of rows contain `Lane5` entries — to be confirmed whether they should be dropped or remapped (see §6).

### 2.2 `raw_record/motion/{YYYY-MM-DD}/lane_{1..4}_speed_matrix.csv` — 5 × 4 = 20 files

- Layout: **3600 data rows × 215 columns** plus one header row (`0,1,...,214`) that gives column indices, not data.
- Each row = one time step; each column = one space cell of the source motion grid. Axis coordinates are implicit (no embedded mile-marker or timestamp axis); they are derived in code from the corridor grid.
- **Values: speed in miles per second.** Typical range ≈ 0.001–0.030 (= 3.6–108 mph). To convert to mph, multiply by 3600. This conversion is performed by `production/ASMxMOTION.py` when building stage 3 (see §4.2).
- `NaN` denotes a cell with no trajectory observation.

---

## 3. Stage 2 — `raw_data/`

Lane-split, time-snapped, NaN-cleaned form of `raw_record/`.

### 3.1 `raw_data/rds/{YYYY-MM-DD}.csv` — 5 files

Produced by [../preprocessing/rds_fast_processing.py](../preprocessing/rds_fast_processing.py) with helpers in [../preprocessing/FAST.py](../preprocessing/FAST.py). Each row is one (mile marker, 30 s time bin) cell.

| Column | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `tdot_milemarker` | float | miles | Original TDOT mile marker (pre-snap) |
| `time_unix_fix` | float | Unix seconds | Time snapped to a 30 s grid via `round_to_fix` |
| `lane{1..4}_speed` | float | mph | Per-lane speed |
| `lane{1..4}_volume` | float | vehicles / 30 s bin | Per-lane vehicle count over the 30 s bin |
| `lane{1..4}_occ` | float | percent | Per-lane occupancy |
| `milemarker` | float | miles | Snapped mile marker (after the TDOT-mile-marker → closest-mile-marker remap defined at the top of [../preprocessing/rds_fast_processing.py](../preprocessing/rds_fast_processing.py)) |

Cleaning applied at this stage:

- Negative `speed` / `volume` / `occupancy` values become `NaN` (negatives are TDOT transmission errors).
- When some lanes report but others do not, the missing per-lane values are filled with the across-lane mean for that (mile marker, time) cell (see `raw_lane_level_df_process` in [../preprocessing/FAST.py](../preprocessing/FAST.py)).

### 3.2 `raw_data/motion/{YYYY-MM-DD}/lane_{1..4}_speed_matrix.csv` — 20 files

**Byte-identical pass-through of `raw_record/motion/…/lane_{1..4}_speed_matrix.csv`** — the motion source already arrives clean, so no cleaning is applied. Same layout and units as §2.2. Included so that motion has the same `raw_record → raw_data → processed_data` shape as RDS.

---

## 4. Stage 3 — `processed_data/`

Regular space-time matrices, `dtype=float32`, axis order `(space, time)`. These are what calibration and evaluation actually load.

### 4.1 `processed_data/rds/lane{1..4}/{YYYY-MM-DD}.npy` — 5 × 4 = 20 files

- Producer: [../preprocessing/preprocessing.py](../preprocessing/preprocessing.py) (`fill_space_time_matrix`).
- Source: `raw_data/rds/{date}.csv`.
- Spatial grid: `np.arange(58.7, 62.7, 0.02)` miles → **200 cells**. (Open item: this is narrower than the corridor-wide 53.3–70.1 window defined in [../preprocessing/corridor_preprocessing.py](../preprocessing/corridor_preprocessing.py). See §6.)
- Temporal grid: 4 s step in Unix seconds, spanning each file's `time_unix_fix` range → **3600 cells** for the 4-hour window.
- Shape: `(200, 3600)`. Values: speed in **mph**, ≈ 0–115. Storage: written via `.T.values.astype(np.float32)` so the on-disk array is `(space, time)`. Cells with no detector report remain `NaN` — on a representative file ≈ 99.3 % of cells are `NaN`, reflecting the sparseness of detector coverage.
- Loaded by [../calibration/train.py](../calibration/train.py), [../calibration/train_day_to_day.py](../calibration/train_day_to_day.py), [../calibration/seeds.py](../calibration/seeds.py).

### 4.2 `processed_data/motion/lane{1..4}/{YYYY-MM-DD}.npy` — 20 files

- Producer: [../production/ASMxMOTION.py](../production/ASMxMOTION.py) (`main`).
- Source: `raw_data/motion/{date}/lane_{lane}_speed_matrix.csv`, sliced to `[:3600, :200]` and transposed.
- Two transforms applied before saving:
  1. **Unit conversion** to mph: `speed = 3600 * data` (see [../production/ASMxMOTION.py:179](../production/ASMxMOTION.py#L179)). The upstream CSVs are in mi/s; the on-disk `.npy` is in mph.
  2. **Adaptive Smoothing**: an `AdaptiveSmoothing` torch module (defined in the same file) is applied to densify and de-noise the field before save. The output is therefore **not** a raw down-sampling of the source — it is the ASM-smoothed ground truth used by calibration.
- Shape: `(200, 3600)`, `dtype=float32`. Values: speed in **mph**, ≈ 0.2–101.5. Because of the smoothing step, there are typically 0 `NaN`s in the on-disk array.
- Loaded as the ground-truth tensor `gt_np` by [../calibration/train.py](../calibration/train.py) (e.g. [line 256](../calibration/train.py#L256), [line 271](../calibration/train.py#L271)), [../calibration/train_day_to_day.py](../calibration/train_day_to_day.py), [../calibration/seeds.py](../calibration/seeds.py).

> Because RDS-processed and motion-processed live on the **same** `(200, 3600)` grid with the **same** mph unit, they can be overlaid directly without any rescaling.

---

## 5. Files outside the three-stage pipeline (root of `data/`)

These ship alongside the pipeline files but are demonstration assets, not products of the documented pipeline.

| File | Shape / format | Unit | Notes |
| --- | --- | --- | --- |
| [2023-10-26.csv](2023-10-26.csv) | CSV, columns `tdot_milemarker, time_unix_fix, lane1_speed…lane4_occ, milemarker` | speed in mph | Same schema as `raw_data/rds/…csv` (§3.1). Single-day demo from a date outside the 2024-07 study window. |
| [2025-04-09.csv](2025-04-09.csv) | CSV, columns `time_unix_fix, lane0_speed, lane1_speed, lane2_speed, lane3_speed, milemarker, time_unix` | speed in mph | **Uses `lane0..lane3` numbering, not `lane1..lane4`.** Demo-only file from a different ingestion path. |
| [I95speed.npy](I95speed.npy) | `(750, 4088)`, `float32` | mph (≈ 3–100) | Demo speed matrix used by some visualization notebooks. Not produced by the documented pipeline. |
| [speed.npy](speed.npy) | `(150, 545)`, `float32` | mph (≈ 3–100) | Small demo speed matrix referenced by notebooks. Not produced by the documented pipeline. |

---

## 6. Open items / known caveats

These are intentionally listed rather than resolved silently, so reviewers and external users can see what is still unclear:

1. **Lane numbering direction.** The pipeline assumes Lane 1 = leftmost / passing lane on I-24 westbound. To be verified against the paper's convention.
2. **`Lane5` rows** in some `lane_dict_text` entries of `raw_record/rds/…csv` — current code only extracts `Lane1`–`Lane4`; `Lane5` rows are effectively ignored. Document whether that is intentional.
3. **Spatial window discrepancy.** [../preprocessing/preprocessing.py](../preprocessing/preprocessing.py) uses `min_milemarker = 58.7`, `max_milemarker = 62.7` → 200-cell grid (matches the shipped `processed_data/rds/` files). The corridor-wide window 53.3–70.1 used elsewhere in the repo produces a different (840-cell) grid. The shipped processed files use the 58.7–62.7 window; this is the segment shown in the paper figures.
4. **`volume` unit semantics.** In `raw_record/rds/` the cross-lane `volume` column is per-report; in `raw_data/rds/` the per-lane `lane{i}_volume` is per-30-s bin (after the `round_to_fix` snap). The values are not directly comparable between stages.
5. **Motion source description.** The on-disk units (mi/s in CSV → mph in `.npy`) are confirmed empirically. The upstream provenance of the motion CSVs (vehicle-trajectory pipeline parameters, grid origin, sampling) is not documented here and should be added once the trajectory pipeline is released.

---

## 7. Verifying this documentation against the data

Anyone updating these files can re-run the following sanity checks (conda env `ASMx`):

```bash
# RDS stage-2 schema (should match the table in §3.1)
head -1 data/raw_data/rds/2024-07-08.csv

# Stage-3 matrices: shape, dtype, value range, NaN fraction
python - <<'PY'
import numpy as np
for path in [
    'data/processed_data/rds/lane1/2024-07-08.npy',
    'data/processed_data/motion/lane1/2024-07-08.npy',
]:
    a = np.load(path)
    print(path, a.shape, a.dtype,
          'range=', float(np.nanmin(a)), float(np.nanmax(a)),
          'nan_frac=', round(float(np.isnan(a).mean()), 4))
PY

# Motion CSV at stage 1 vs stage 2 — should be byte-identical
cmp data/raw_record/motion/2024-07-08/lane_1_speed_matrix.csv \
    data/raw_data/motion/2024-07-08/lane_1_speed_matrix.csv && echo OK
```
