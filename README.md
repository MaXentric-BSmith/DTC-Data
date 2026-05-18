# DTC Data Submission — Heart Age Decade Classifier

MaXentric Technologies submission for the DARPA **Digital Twin Consortium (DTC) TA2 Applicant Challenge**: predict each subject’s **age decade** (six classes) from paired **aortic** and **brachial** arterial pressure waveforms sampled at **500 Hz**.

| Class index | Decade label |
|-------------|--------------|
| 0 | 20s |
| 1 | 30s |
| 2 | 40s |
| 3 | 50s |
| 4 | 60s |
| 5 | 70s |

The repository contains training and evaluation code (Python modules and Jupyter notebooks), saved **Random Forest** pipelines, example **submission JSON**, and supporting documentation from the challenge package.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── heart_age_classifier.py      # Main ML library + CLI (RF, k-NN, HGB, XGBoost)
├── cnn_age_classifier.py        # PyTorch ResNet-1D + waveform I/O / preprocessing
├── arterial_waveform_features.py # Beat-level morphology (peaks, notch, PTT, AUC)
├── create_age_classifier.ipynb  # Train tree models and/or CNN; export predictions
├── evaluate_saved_rf_models.ipynb # Holdout evaluation of saved .joblib pipelines
├── data_visualizer.ipynb        # Interactive EDA on *_train* CSVs
├── datasets/                    # Place challenge CSVs here (see Data)
│   ├── train/
│   │   ├── aortaP_train_data.csv
│   │   └── brachP_train_data.csv
│   └── test/
│       ├── aortaP_test_data.csv
│       └── brachP_test_data.csv
├── models/                      # Saved sklearn pipelines (.joblib)
├── MaXentric_Technologies_output.json   # Example test predictions (875 subjects)
├── MaXentric_Technologies_description.pdf
└── TA2 Applicant Challenge instructions.docx
```

> **Note:** Large `datasets/` CSVs may not be checked into git. Copy them from the challenge distribution into `datasets/train` and `datasets/test` before training.

---

## Problem and data format

- **Input:** Two aligned tables per split — aortic pressure (`aortaP_*`) and brachial pressure (`brachP_*`).
- **Waveform columns:** `aorta_t_0` … `aorta_t_335` and `brach_t_0` … `brach_t_335` (**336 samples** ≈ 672 ms at 500 Hz).
- **Training rows:** `subject_index`, 336 time samples, and integer **`target`** in `0`–`5`.
- **Test rows:** `subject_index` and waveforms only (no `target`). The code expects **875** test subjects with indices **`0` … `874`** (`EXPECTED_TEST_INDICES` in `heart_age_classifier.py`).
- **Training file naming:** Training CSV paths must contain `_train` in the filename (enforced by `_require_train_csv`).

---

## Installation

### Python version

Use **Python 3.11 or 3.12**. PyTorch wheels in `requirements.txt` are pinned for `3.11 ≤ python < 3.13`. NumPy is capped below version 2 for compatibility with those wheels.

### Create environment and install

```bash
cd "/path/to/DTC Data"
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Register the kernel for Jupyter / VS Code / Cursor:

```bash
python -m ipykernel install --user --name dtc-age --display-name "DTC age classifier"
```

### Tree models only (no CNN)

If you only need Random Forest / boosting / k-NN and will not run the ResNet cells, install everything except PyTorch:

```bash
pip install numpy pandas scipy scikit-learn joblib xgboost matplotlib seaborn ipywidgets ipykernel
```

### macOS notes

- **Apple Silicon:** `create_age_classifier.ipynb` selects **MPS** when available, otherwise CPU.
- **XGBoost / OpenMP:** If `import xgboost` fails with an OpenMP symbol error, reinstall via conda-forge or install a matching `libomp` (see error text from `xgb_available()` / `_xgb_unavailable_message()` in `heart_age_classifier.py`). Use `--skip-if-xgb-unavailable` on the CLI to train other classifiers without XGBoost.

---

## Quick start — command line

Train with **engineered features** (default: PPA ratio/diff, pulse pressure, rolling ARV/CV/slope, optional Chebyshev-smoothed variants) and write test predictions:

```bash
python heart_age_classifier.py \
  --train-aorta datasets/train/aortaP_train_data.csv \
  --train-brach datasets/train/brachP_train_data.csv \
  --test-aorta datasets/test/aortaP_test_data.csv \
  --test-brach datasets/test/brachP_test_data.csv \
  --classifier rf \
  --features engineered \
  --save-model-rf models/heart_age_rf.joblib \
  --acc-in-model-name \
  --out-json MaXentric_Technologies_output.json
```

### Useful CLI flags

| Flag | Description |
|------|-------------|
| `--classifier` | `rf`, `knn`, `hgb`, `xgb`, or `both` (RF + k-NN) |
| `--features` | `engineered`, `waveform` (672 raw samples), or `waveform_plus` (raw + selected 336-sample traces) |
| `--engineered-columns` | Comma-separated names from `ENGINEERED_FEATURE_NAMES` |
| `--eval-split` | Holdout fraction for validation metrics (default `0.2`) |
| `--seed` | RNG seed; omit for a random seed printed to stdout |
| `--grid-search` | Run `GridSearchCV` before final refit |
| `--acc-in-model-name` | Append `_acc{tenths}` to saved model paths from holdout accuracy |

Full help:

```bash
python heart_age_classifier.py --help
```

---

## Notebooks

### `create_age_classifier.ipynb`

Primary training workflow:

1. **Imports & project root** — adds repo to `sys.path`, detects PyTorch device (MPS/CPU).
2. **Shared settings** — CSV paths, `FEATURE_MODE`, validation split, imputer, output JSON paths.
3. **RF / k-NN data QC** — stacked random waveforms per class (same merge as tree training).
4. **Random Forest / k-NN / HistGradientBoosting / XGBoost** — separate cells calling `heart_age_classifier.main()`.
5. **ResNet-1D CNN** (optional) — two-channel 1-D ResNet with phase-1 preprocessing from `cnn_age_classifier.py`.
6. **Evaluation** — test-set prediction plots and feature importances.
7. **Side-by-side confusion matrices** — compare all trained models in one figure.

Keep the notebook in the same directory as `heart_age_classifier.py`, or adjust `NOTEBOOK_DIR` in the first code cell.

### `evaluate_saved_rf_models.ipynb`

Loads saved `.joblib` pipelines from `models/` and scores them on a **stratified validation holdout** with the same preprocessing and `RANDOM_STATE` as training.

**Important:** `main()` refits on **all** training rows before saving. Predicting on the validation fold with that artifact is in-sample. By default this notebook **clones hyperparameters**, refits on the train fold only, and evaluates on the held-out fold — matching the `_acc###` suffix in filenames. Set `EVAL_ON_SAVED_FULL_FIT=True` only to inspect in-sample behavior.

### `data_visualizer.ipynb`

Interactive exploration of `*_train*` CSVs: class balance, univariate distributions, waveforms by decade, missing-data rates, and correlation heatmaps on sampled waveform columns. Requires `ipywidgets` for the dashboard controls.

---

## Python modules

### `heart_age_classifier.py`

Central library for:

- Loading and merging paired train/test CSVs (`load_train_pair`, `load_test_pair`).
- Feature extraction: **engineered**, **waveform**, **waveform_plus** (`extract_features`, `extract_waveform_plus`).
- sklearn **Pipeline** builders: imputation → scaling → classifier.
- Training entry point `main()` and `python heart_age_classifier.py` CLI.
- Metrics, confusion matrices, permutation importance, submission JSON writers.
- Model I/O via **joblib** (`load_fitted_pipeline`, save paths with optional accuracy suffix).

**Classifiers:** `RandomForestClassifier`, `KNeighborsClassifier`, `HistGradientBoostingClassifier`, `XGBClassifier` (optional).

**Engineered scalars** (default engineered mode) include PPA ratio/diff (raw and Chebyshev-smoothed), per-site pulse pressure, and rolling mean ARV / CV / slope (window default **10**).

### `cnn_age_classifier.py`

PyTorch utilities for the two-channel **ResNet-1D** in `create_age_classifier.ipynb`:

- `load_two_channel_waveforms` → `(n_subjects, 336, 2)`
- Gap imputation, per-subject min-max / z-score, optional phase-1 preprocessing
- `build_two_channel_resnet1d`, checkpoint save/load
- Pulse-pressure tabular side features for the CNN head

### `arterial_waveform_features.py`

Standalone **beat-level** morphology on 500 Hz traces: systolic peaks, dicrotic notch, foot/beat boundaries, pulse pressure, AUC splits, crest time, max dP/dt, form factor, and optional **pulse transit time (PTT)** when brachial pressure is supplied. Useful for feature prototyping and visualization (optional matplotlib axes in `TYPE_CHECKING` paths).

---

## Feature modes (summary)

| Mode | Description |
|------|-------------|
| `engineered` | Scalar summaries per subject (default, best for fast RF/XGB baselines). |
| `waveform` | All 672 waveform samples; gaps linearly imputed along time per channel. |
| `waveform_plus` | 672 samples plus additional **336-sample traces** per name in `engineered_columns` (raw, Chebyshev, CNN-aligned preproc traces, etc.). |

Trace and scalar name registry: `ENGINEERED_FEATURE_NAMES`, `DECADE_TARGET_LABELS`.

---

## Saved models and submission output

### Models (`models/`)

Example filename pattern:

`heart_age_rf_acc674_hp_max_depth-20__min_samples_leaf-1__n_estimators-100.joblib`

- `_acc674` → holdout accuracy **67.4%** (tenths of a percent) before the final full-training refit.
- `_hp_*` → hyperparameters embedded when `embed_grid_search_params_in_save_path` is enabled.

A second file prefixed `BAD_` is retained for comparison / failed experiments.

### Submission JSON

`MaXentric_Technologies_output.json` maps string subject indices to predicted class integers:

```json
{
  "0": 5,
  "1": 1,
  "2": 0,
  ...
}
```

Keys **`"0"` … `"874"`** must all be present for challenge-style test exports.

---

## Reproducing the submission workflow

1. Place train/test CSVs under `datasets/` (see layout above).
2. Create a virtualenv and `pip install -r requirements.txt`.
3. Open `create_age_classifier.ipynb`, set paths in **Shared settings**, run RF (and optionally other) training cells.
4. Run the test evaluation / JSON export cell, or use the CLI with `--out-json`.
5. Validate holdout metrics with `evaluate_saved_rf_models.ipynb` (refit-on-train-fold mode).
6. Submit `MaXentric_Technologies_output.json` plus `MaXentric_Technologies_description.pdf` per challenge instructions.

---

## Dependencies

See [`requirements.txt`](requirements.txt) for pinned versions. Major packages:

| Package | Role |
|---------|------|
| numpy, pandas, scipy | Arrays, tables, Chebyshev filtering, signal helpers |
| scikit-learn, joblib | Pipelines, CV, metrics, model persistence |
| xgboost | Optional gradient boosting classifier |
| matplotlib, seaborn, ipywidgets | Plots and `data_visualizer` dashboard |
| torch, torchvision, torchaudio | ResNet-1D CNN path |
| ipykernel | Jupyter execution |

---

## References

- `TA2 Applicant Challenge instructions.docx` — official task definition and submission rules.
- `MaXentric_Technologies_description.pdf` — team / approach summary for reviewers.

---

## License and attribution

Submitted as part of the MaXentric Technologies DARPA DTC TA2 applicant challenge response. Contact the repository owner for reuse terms outside the challenge context.
