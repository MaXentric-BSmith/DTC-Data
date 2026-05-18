"""
Age-group classifiers for paired aortic / brachial pressure waveforms.

Supports:
  - **RandomForestClassifier**, **KNeighborsClassifier**, **HistGradientBoostingClassifier**
    (histogram gradient boosting), and **XGBoost** (:class:`xgboost.XGBClassifier`) via :func:`main` / CLI.
  - **engineered features** (default): PPA ratio/diff, PP peak per site, rolling-window
    ARV / CV / slope (matches ``data_visualizer.ipynb``).
  - **raw waveforms**: all ``aorta_t_*`` + ``brach_t_*`` columns (672 features), each channel
    linearly imputed along time (``numpy.interp``; all-missing row → 0) before the pipeline.
  - **waveform_plus**: stacks 336-sample traces for each name in ``engineered_columns`` (see
    :data:`ENGINEERED_FEATURE_NAMES`, including ``aorta_raw`` / ``brach_raw`` for the full site rows
    and optional **CNN phase-1–aligned** traces ``aorta_preproc`` / ``brach_preproc`` plus Chebyshev
    variants ``aorta_preproc_cheb`` / ``brach_preproc_cheb``). ``None`` keeps the legacy default set
    (raw + Cheb traces + scalars); ``[]`` is the 672-column waveform matrix only.

Training CSV layout:
  <subject_index>, <{aorta|brach}_t_0..335>, <target>
The first column may be unnamed in the file; targets are encoded 0–5 (20s–70s).

Dependencies: pandas, scikit-learn, numpy. :func:`evaluate_and_visualize_model` also needs matplotlib.
:class:`~xgboost.XGBClassifier` requires the ``xgboost`` package. Optional :func:`main` grid search uses
:class:`~sklearn.model_selection.GridSearchCV`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional (but required for aorta_cheb/brach_cheb waveform_plus features).
try:
    from scipy.signal import cheby1, sosfiltfilt
except ImportError:  # pragma: no cover
    cheby1 = None  # type: ignore[assignment]
    sosfiltfilt = None  # type: ignore[assignment]

# XGBoost can raise xgboost.core.XGBoostError (not ImportError) if libxgboost / OpenMP fails to load.
_XGB_IMPORT_ERROR: Optional[BaseException] = None
try:
    from xgboost import XGBClassifier
except Exception as _xgb_exc:  # pragma: no cover
    XGBClassifier = None  # type: ignore[misc, assignment]
    _XGB_IMPORT_ERROR = _xgb_exc


def _xgb_unavailable_message() -> str:
    return (
        "XGBoost is not usable in this environment. "
        "Typical fixes: reinstall xgboost with a matching OpenMP runtime "
        "(e.g. `conda install -c conda-forge py-xgboost llvm-openmp`). "
        "On macOS, a pip wheel plus Homebrew `libomp` often disagrees on OpenMP symbols "
        "(e.g. ___kmpc_dispatch_deinit). "
        f"Detail: {_XGB_IMPORT_ERROR!r}"
    )


def xgb_available() -> bool:
    """True if :class:`~xgboost.XGBClassifier` is importable and ``libxgboost`` loads."""
    return XGBClassifier is not None

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore[assignment]

# Expected test subjects for submission-style JSON (0 … 874 inclusive).
EXPECTED_TEST_INDICES = range(875)

SUBJECT_COL = "subject_index"
TARGET_COL = "target"

_TRAIN_NAME_MARKER = "_train"


def _require_train_csv(path: Path, role: str) -> None:
    """Training must use only files whose names include ``_train`` (e.g. ``*_train*.csv``)."""
    if _TRAIN_NAME_MARKER not in path.name:
        raise ValueError(
            f"{role} must be a *_train* CSV (filename must contain '{_TRAIN_NAME_MARKER}'); got {path}"
        )


def _normalize_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the leading unnamed subject column is named ``subject_index``."""
    out = df.copy()
    if SUBJECT_COL not in out.columns:
        # Typical export: first header cell is empty -> Unnamed: 0
        unnamed = [c for c in out.columns if str(c).startswith("Unnamed")]
        if unnamed:
            out = out.rename(columns={unnamed[0]: SUBJECT_COL})
        else:
            out.insert(0, SUBJECT_COL, np.arange(len(out), dtype=np.int64))
    return out


def waveform_columns(prefix: str) -> list[str]:
    return [f"{prefix}_t_{i}" for i in range(336)]


ROLL_WINDOW_DEFAULT = 10

# Scalar summaries used by ``feature_mode=\"engineered\"`` (one value per subject).
ENGINEERED_SCALAR_NAMES: List[str] = [
    "ppa_ratio",
    "ppa_diff",
    "ppa_ratio_cheb",
    "ppa_diff_cheb",
    "aorta_pp",
    "brach_pp",
    "aorta_arv_roll_mean",
    "brach_arv_roll_mean",
    "aorta_cv_roll_mean",
    "brach_cv_roll_mean",
    "aorta_slope_roll_mean",
    "brach_slope_roll_mean",
]

# Default ``engineered_columns=None`` / waveform_plus: these traces + all scalars (backward compatible).
ENGINEERED_WAVEFORM_TRACE_NAMES_DEFAULT: List[str] = [
    "aorta_raw",
    "brach_raw",
    "aorta_cheb",
    "brach_cheb",
]

# CNN phase-1–aligned traces (336 samples each); use :func:`extract_waveform_plus` kwargs to match CNN gap/z-score settings.
ENGINEERED_WAVEFORM_TRACE_NAMES_CNN_ALIGNED: List[str] = [
    "aorta_preproc",
    "brach_preproc",
    "aorta_preproc_cheb",
    "brach_preproc_cheb",
]

# Full registry for ``subset_engineered_feature_names`` / CLI validation.
ENGINEERED_FEATURE_NAMES: List[str] = (
    ENGINEERED_WAVEFORM_TRACE_NAMES_DEFAULT
    + ENGINEERED_WAVEFORM_TRACE_NAMES_CNN_ALIGNED
    + ENGINEERED_SCALAR_NAMES
)

_RAW_TRACE_KEYS = frozenset(
    {
        "aorta_raw",
        "brach_raw",
        "aorta_cheb",
        "brach_cheb",
        "aorta_preproc",
        "brach_preproc",
        "aorta_preproc_cheb",
        "brach_preproc_cheb",
    }
)

_CNN_PHASE1_TRACE_NAMES = frozenset(ENGINEERED_WAVEFORM_TRACE_NAMES_CNN_ALIGNED)

# Chebyshev smoothing defaults (matches data_visualizer.ipynb).
CHEBY_SAMPLE_RATE_HZ = 500.0
CHEBY_CUTOFF_HZ = 8.0
CHEBY_RP_DB = 0.8
CHEBY_ORDER = 2


def subset_engineered_feature_names(columns: Sequence[str]) -> List[str]:
    """
    Return ``columns`` in order, validating each name is in :data:`ENGINEERED_FEATURE_NAMES`.

    Use with ``feature_mode="engineered"`` (scalar columns) or ``feature_mode="waveform_plus"``
    (which 336-sample traces to append) while the full waveform matrices are still computed internally.
    """
    cols = list(columns)
    if not cols:
        raise ValueError("engineered_columns must be a non-empty sequence when provided.")
    allowed = set(ENGINEERED_FEATURE_NAMES)
    unknown = [c for c in cols if c not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown engineered feature name(s): {unknown}. "
            f"Allowed names: {ENGINEERED_FEATURE_NAMES}"
        )
    return cols


def _waveform_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Shape (n_rows, n_time); coerce to float."""
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _pp_peak_per_row(W: np.ndarray) -> np.ndarray:
    """Peak pulse pressure per row (max over time samples)."""
    if W.size == 0:
        return np.array([])
    return np.nanmax(W, axis=1)


def _rolling_temporal_means(W: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling-window summaries across time (last axis), then mean over windows per row.
    Returns (arv_roll_mean, cv_roll_mean, slope_roll_mean).
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    n_rows, n_t = W.shape
    if n_t < window:
        nan = np.full(n_rows, np.nan)
        return nan, nan, nan

    win = np.lib.stride_tricks.sliding_window_view(W, window_shape=window, axis=1)
    # All-NaN windows are common when waveforms have gaps; reductions then warn but correctly yield NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arv_roll = np.nanmean(np.abs(np.diff(win, axis=2)), axis=2)
        mu = np.nanmean(win, axis=2)
        sd = np.nanstd(win, axis=2, ddof=1)
        y_mean = np.nanmean(win, axis=2, keepdims=True)
    cv_roll = np.where(mu != 0.0, sd / mu, np.nan)
    tt = np.arange(window, dtype=float)
    tt = tt - tt.mean()
    denom = float(np.sum(tt**2))
    slope_roll = np.nansum(tt * (win - y_mean), axis=2) / denom
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            np.nanmean(arv_roll, axis=1),
            np.nanmean(cv_roll, axis=1),
            np.nanmean(slope_roll, axis=1),
        )


def _per_timestep_window_series(
    W: np.ndarray,
    window: int,
    kind: str,
) -> np.ndarray:
    """
    For each valid trailing window ending at index ``j``, compute ARV, CV, or slope inside the window
    and assign to column ``j``. Earlier indices are NaN (window not full). Shape matches ``W``.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    n_rows, n_t = W.shape
    if n_t < window:
        return np.full((n_rows, n_t), np.nan)

    win = np.lib.stride_tricks.sliding_window_view(W, window_shape=window, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if kind == "arv":
            vals = np.nanmean(np.abs(np.diff(win, axis=2)), axis=2)
        elif kind == "cv":
            mu = np.nanmean(win, axis=2)
            sd = np.nanstd(win, axis=2, ddof=1)
            vals = np.where(mu != 0.0, sd / mu, np.nan)
        elif kind == "slope":
            y_mean = np.nanmean(win, axis=2, keepdims=True)
            tt = np.arange(window, dtype=float)
            tt = tt - tt.mean()
            denom = float(np.sum(tt**2))
            vals = np.nansum(tt * (win - y_mean), axis=2) / denom
        else:
            raise ValueError(f"kind must be arv, cv, or slope; got {kind!r}")

    out = np.full((n_rows, n_t), np.nan)
    nw = vals.shape[1]
    out[:, window - 1 : window - 1 + nw] = vals
    return out


def _require_scipy_for_cheb() -> None:
    if cheby1 is None or sosfiltfilt is None:  # pragma: no cover
        raise ImportError(
            "scipy is required for Chebyshev waveform features (aorta_cheb/brach_cheb). "
            "Install with: pip install scipy"
        )


def _linear_impute_nonfinite_1d(y: np.ndarray, *, fill_all_nan: float = 0.0) -> np.ndarray:
    """Linearly interpolate non-finite samples along one 1-D trace (time).

    Leading/trailing gaps use the nearest finite sample (``numpy.interp`` left/right).
    All-non-finite traces become the constant ``fill_all_nan``. Semantics match
    :func:`cnn_age_classifier._impute_time_series_1d` (no PyTorch import on this module).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    finite = np.isfinite(y)
    if finite.all():
        return y.astype(float)
    if not finite.any():
        return np.full(n, float(fill_all_nan), dtype=float)
    idx = np.arange(n, dtype=np.float64)
    xv = idx[finite]
    yv = y[finite]
    out = np.interp(idx, xv, yv, left=float(yv[0]), right=float(yv[-1]))
    return out.astype(float)


def _fill_nan_waveforms(W: np.ndarray) -> np.ndarray:
    """Fill non-finite samples along time (each row) with linear interpolation for stable filtering."""
    X = np.asarray(W, dtype=float)
    if X.size == 0 or not np.any(~np.isfinite(X)):
        return X
    out = np.empty_like(X, dtype=float)
    for i in range(X.shape[0]):
        out[i, :] = _linear_impute_nonfinite_1d(X[i, :], fill_all_nan=0.0)
    return out


def _linear_impute_raw_waveform_dataframe(
    df: pd.DataFrame,
    *,
    a_cols: list[str],
    b_cols: list[str],
) -> pd.DataFrame:
    """``aorta_t_*`` + ``brach_t_*`` as a DataFrame with per-row linear gap fill on each channel."""
    Wa = _fill_nan_waveforms(_waveform_matrix(df, a_cols))
    Wb = _fill_nan_waveforms(_waveform_matrix(df, b_cols))
    return pd.concat(
        [
            pd.DataFrame(Wa, index=df.index, columns=a_cols),
            pd.DataFrame(Wb, index=df.index, columns=b_cols),
        ],
        axis=1,
    )


_TRACE_COL_RE = re.compile(r"^(.+)_t_(\d+)$")


class TraceBlockLinearImputer(BaseEstimator, TransformerMixin):
    """Residual NaN handling aligned with waveform **linear** gap fill.

    Columns matching ``{name}_t_{k}`` are grouped by ``name``; for each row, NaNs in that
    block are filled with :func:`_linear_impute_nonfinite_1d` along time (same semantics as
    :func:`extract_waveform_plus`). Any other columns use the **median** from ``fit`` (finite
    values only; all-NaN columns → ``0.0``).
    """

    def __init__(self, *, fill_all_nan: float = 0.0):
        self.fill_all_nan = float(fill_all_nan)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_df = X
            cols: list[str] = [str(c) for c in X_df.columns]
            X_arr = X_df.to_numpy(dtype=np.float64, copy=False)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim != 2:
                raise ValueError("Expected a 2-D feature array.")
            cols = [f"x{j}" for j in range(X_arr.shape[1])]

        trace_mask = np.array([_TRACE_COL_RE.match(c) is not None for c in cols], dtype=bool)
        groups: Dict[str, List[Tuple[int, int]]] = {}
        for j, c in enumerate(cols):
            m = _TRACE_COL_RE.match(c)
            if not m:
                continue
            base, tk = m.group(1), int(m.group(2))
            groups.setdefault(base, []).append((j, tk))
        trace_groups_idx: Dict[str, np.ndarray] = {}
        for base, pairs in groups.items():
            pairs_sorted = sorted(pairs, key=lambda p: p[1])
            trace_groups_idx[base] = np.array([p[0] for p in pairs_sorted], dtype=int)

        self._columns = cols
        self._trace_groups_idx = trace_groups_idx
        scalar_idx = np.flatnonzero(~trace_mask)
        self._scalar_indices = scalar_idx
        if scalar_idx.size:
            med = np.nanmedian(X_arr[:, scalar_idx], axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            self.scalar_medians_ = med.astype(float)
        else:
            self.scalar_medians_ = np.empty(0, dtype=float)

        self.n_features_in_ = X_arr.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(cols, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        if isinstance(X, pd.DataFrame):
            cols = [str(c) for c in X.columns]
            if cols != self._columns:
                raise ValueError(
                    f"Feature name/order mismatch: expected {len(self._columns)} columns "
                    f"as at fit time; got {len(cols)} with different names or order."
                )
            X_arr = X.to_numpy(dtype=np.float64, copy=True)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim != 2 or X_arr.shape[1] != int(self.n_features_in_):
                raise ValueError(
                    f"Expected array with {int(self.n_features_in_)} features; "
                    f"got shape {X_arr.shape}."
                )
            X_arr = X_arr.copy()

        for idxs in self._trace_groups_idx.values():
            block = X_arr[:, idxs]
            for i in range(X_arr.shape[0]):
                block[i, :] = _linear_impute_nonfinite_1d(
                    block[i, :], fill_all_nan=self.fill_all_nan
                )

        if self._scalar_indices.size and self.scalar_medians_.size:
            for k, j in enumerate(self._scalar_indices):
                col = X_arr[:, j]
                bad = ~np.isfinite(col)
                if np.any(bad):
                    col[bad] = float(self.scalar_medians_[k])
        return X_arr

    def _more_tags(self):
        return {"allow_nan": True}


def _cheby_lowpass_matrix(
    W: np.ndarray,
    *,
    sample_rate_hz: float = CHEBY_SAMPLE_RATE_HZ,
    cutoff_hz: float = CHEBY_CUTOFF_HZ,
    rp_db: float = CHEBY_RP_DB,
) -> np.ndarray:
    """Zero-phase Chebyshev-I lowpass for each row (axis=1)."""
    _require_scipy_for_cheb()
    X = _fill_nan_waveforms(W)
    if X.size == 0:
        return X
    nyq = 0.5 * float(sample_rate_hz)
    if not (0.0 < float(cutoff_hz) < nyq):
        return X
    # sosfiltfilt needs some minimum length; keep short traces unchanged.
    if X.shape[1] < 9:
        return X
    sos = cheby1(CHEBY_ORDER, float(rp_db), float(cutoff_hz), btype="low", fs=float(sample_rate_hz), output="sos")
    return sosfiltfilt(sos, X, axis=1)


def _waveform_plus_trace_matrix(
    name: str,
    Wa: np.ndarray,
    Wb: np.ndarray,
    *,
    window: int,
    Wa_pre: Optional[np.ndarray] = None,
    Wb_pre: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return shape ``(n_rows, n_time)`` trace for one :data:`ENGINEERED_FEATURE_NAMES` entry."""
    if name == "aorta_raw":
        return np.array(Wa, dtype=float, copy=True)
    if name == "brach_raw":
        return np.array(Wb, dtype=float, copy=True)
    if name == "aorta_cheb":
        return _cheby_lowpass_matrix(Wa)
    if name == "brach_cheb":
        return _cheby_lowpass_matrix(Wb)
    if name == "aorta_preproc":
        if Wa_pre is None:
            raise ValueError("internal: Wa_pre required for aorta_preproc")
        return np.array(Wa_pre, dtype=float, copy=True)
    if name == "brach_preproc":
        if Wb_pre is None:
            raise ValueError("internal: Wb_pre required for brach_preproc")
        return np.array(Wb_pre, dtype=float, copy=True)
    if name == "aorta_preproc_cheb":
        if Wa_pre is None:
            raise ValueError("internal: Wa_pre required for aorta_preproc_cheb")
        return _cheby_lowpass_matrix(Wa_pre)
    if name == "brach_preproc_cheb":
        if Wb_pre is None:
            raise ValueError("internal: Wb_pre required for brach_preproc_cheb")
        return _cheby_lowpass_matrix(Wb_pre)
    if name == "ppa_ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            r = Wb / Wa
        return np.where(np.isfinite(r), r, np.nan)
    if name == "ppa_diff":
        return Wb - Wa
    if name == "ppa_ratio_cheb":
        Wa_c = _cheby_lowpass_matrix(Wa)
        Wb_c = _cheby_lowpass_matrix(Wb)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = Wb_c / Wa_c
        return np.where(np.isfinite(r), r, np.nan)
    if name == "ppa_diff_cheb":
        Wa_c = _cheby_lowpass_matrix(Wa)
        Wb_c = _cheby_lowpass_matrix(Wb)
        return Wb_c - Wa_c
    if name == "aorta_pp":
        peak = np.nanmax(Wa, axis=1, keepdims=True)
        return np.broadcast_to(peak, Wa.shape).copy()
    if name == "brach_pp":
        peak = np.nanmax(Wb, axis=1, keepdims=True)
        return np.broadcast_to(peak, Wb.shape).copy()
    if name == "aorta_arv_roll_mean":
        return _per_timestep_window_series(Wa, window, "arv")
    if name == "brach_arv_roll_mean":
        return _per_timestep_window_series(Wb, window, "arv")
    if name == "aorta_cv_roll_mean":
        return _per_timestep_window_series(Wa, window, "cv")
    if name == "brach_cv_roll_mean":
        return _per_timestep_window_series(Wb, window, "cv")
    if name == "aorta_slope_roll_mean":
        return _per_timestep_window_series(Wa, window, "slope")
    if name == "brach_slope_roll_mean":
        return _per_timestep_window_series(Wb, window, "slope")
    raise ValueError(f"Unknown engineered feature name for waveform_plus: {name!r}")


def extract_engineered_features(
    merged: pd.DataFrame,
    *,
    a_cols: list[str],
    b_cols: list[str],
    window: int = ROLL_WINDOW_DEFAULT,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build the engineered feature matrix aligned with ``merged`` row order.

    PP peaks are row-wise maxima over PP waveform columns. PPA ratio/diff and
    ``ppa_ratio_cheb`` / ``ppa_diff_cheb`` use the same peak definitions on
    Chebyshev-lowpass traces (see :func:`_cheby_lowpass_matrix`). Rolling
    ARV/CV/slope match ``data_visualizer.ipynb``.

    ``columns``: if ``None``, return all scalar summaries in :data:`ENGINEERED_SCALAR_NAMES`;
    otherwise return only those columns (validated). ``aorta_raw`` / ``brach_raw`` are not valid here—use
    ``feature_mode=\"waveform_plus\"``.

    """
    Wa = _waveform_matrix(merged, a_cols)
    Wb = _waveform_matrix(merged, b_cols)

    aorta_pp = _pp_peak_per_row(Wa)
    brach_pp = _pp_peak_per_row(Wb)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppa_ratio = brach_pp / aorta_pp
    ppa_ratio = np.where(np.isfinite(ppa_ratio), ppa_ratio, np.nan)
    ppa_diff = brach_pp - aorta_pp

    Wa_c = _cheby_lowpass_matrix(Wa)
    Wb_c = _cheby_lowpass_matrix(Wb)
    aorta_pp_cheb = _pp_peak_per_row(Wa_c)
    brach_pp_cheb = _pp_peak_per_row(Wb_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppa_ratio_cheb = brach_pp_cheb / aorta_pp_cheb
    ppa_ratio_cheb = np.where(np.isfinite(ppa_ratio_cheb), ppa_ratio_cheb, np.nan)
    ppa_diff_cheb = brach_pp_cheb - aorta_pp_cheb

    a_arv, a_cv, a_slope = _rolling_temporal_means(Wa, window)
    b_arv, b_cv, b_slope = _rolling_temporal_means(Wb, window)

    out = pd.DataFrame(
        {
            "ppa_ratio": ppa_ratio,
            "ppa_diff": ppa_diff,
            "ppa_ratio_cheb": ppa_ratio_cheb,
            "ppa_diff_cheb": ppa_diff_cheb,
            "aorta_pp": aorta_pp,
            "brach_pp": brach_pp,
            "aorta_arv_roll_mean": a_arv,
            "brach_arv_roll_mean": b_arv,
            "aorta_cv_roll_mean": a_cv,
            "brach_cv_roll_mean": b_cv,
            "aorta_slope_roll_mean": a_slope,
            "brach_slope_roll_mean": b_slope,
        },
        index=merged.index,
    )
    if columns is None:
        use_cols = list(ENGINEERED_SCALAR_NAMES)
    else:
        use_cols = subset_engineered_feature_names(columns)
        bad = set(use_cols) & _RAW_TRACE_KEYS
        if bad:
            raise ValueError(
                f"{sorted(bad)} are waveform-only identifiers; use feature_mode='waveform_plus', "
                "not scalar engineered mode."
            )
    return out[use_cols]


def extract_waveform_plus(
    merged: pd.DataFrame,
    *,
    a_cols: list[str],
    b_cols: list[str],
    engineered_columns: Optional[Sequence[str]],
    roll_window: int,
    waveform_plus_cnn_max_gap_samples: Optional[int] = None,
    waveform_plus_cnn_max_gap_ms: Optional[float] = None,
    waveform_plus_cnn_sample_rate_hz: float = CHEBY_SAMPLE_RATE_HZ,
    waveform_plus_cnn_zscore_eps: float = 1e-6,
    waveform_plus_cnn_zscore_mode: Literal["independent", "aorta_reference"] = "aorta_reference",
) -> pd.DataFrame:
    """
    Concatenate **336-sample traces only** (no implicit duplicate prefix): for each name in
    ``engineered_columns``, append ``{name}_t_0..335`` (see :func:`_waveform_plus_trace_matrix`).
    ``aorta_raw`` / ``brach_raw`` are the full aortic / brachial rows (same semantics as
    ``aorta_t_*`` / ``brach_t_*``) after **linear** gap imputation along time on each channel.

    - ``engineered_columns`` ``None`` → default traces (:data:`ENGINEERED_WAVEFORM_TRACE_NAMES_DEFAULT`)
      plus all scalar summaries (same column count as before CNN-aligned traces existed).
    - ``[]`` → only ``aorta_t_*`` + ``brach_t_*`` (**672** columns), linearly imputed like ``waveform`` mode.
    - Otherwise use :func:`subset_engineered_feature_names` order (validated).

    **CNN-aligned traces** (``aorta_preproc``, ``brach_preproc``, ``*_preproc_cheb``) use
    :func:`cnn_age_classifier.preprocess_cnn_phase1_traces_rowwise` with the keyword arguments
    above so you can match :func:`cnn_age_classifier.load_two_channel_waveforms` / the training notebook.
    With defaults ``waveform_plus_cnn_max_gap_samples=None`` and ``max_gap_ms=None``, no subjects
    are blanked for long gaps (only linear imputation + z-score).
    """
    Wa = _fill_nan_waveforms(_waveform_matrix(merged, a_cols))
    Wb = _fill_nan_waveforms(_waveform_matrix(merged, b_cols))
    n_t = Wa.shape[1]
    idx = merged.index

    if engineered_columns is not None and len(list(engineered_columns)) == 0:
        return pd.concat(
            [
                pd.DataFrame(Wa, index=idx, columns=a_cols),
                pd.DataFrame(Wb, index=idx, columns=b_cols),
            ],
            axis=1,
        )

    if engineered_columns is None:
        selected = list(ENGINEERED_WAVEFORM_TRACE_NAMES_DEFAULT) + list(ENGINEERED_SCALAR_NAMES)
    else:
        selected = subset_engineered_feature_names(engineered_columns)

    selected_set = set(selected)
    Wa_pre: Optional[np.ndarray] = None
    Wb_pre: Optional[np.ndarray] = None
    if selected_set & _CNN_PHASE1_TRACE_NAMES:
        from cnn_age_classifier import preprocess_cnn_phase1_traces_rowwise

        Wa_pre, Wb_pre = preprocess_cnn_phase1_traces_rowwise(
            Wa,
            Wb,
            max_gap_samples=waveform_plus_cnn_max_gap_samples,
            max_gap_ms=waveform_plus_cnn_max_gap_ms,
            sample_rate_hz=float(waveform_plus_cnn_sample_rate_hz),
            zscore_eps=float(waveform_plus_cnn_zscore_eps),
            zscore_mode=waveform_plus_cnn_zscore_mode,
        )

    blocks: List[pd.DataFrame] = []
    for name in selected:
        mat = _waveform_plus_trace_matrix(
            name,
            Wa,
            Wb,
            window=roll_window,
            Wa_pre=Wa_pre,
            Wb_pre=Wb_pre,
        )
        blocks.append(
            pd.DataFrame(
                mat,
                index=idx,
                columns=[f"{name}_t_{i}" for i in range(n_t)],
            )
        )
    return pd.concat(blocks, axis=1)


FeatureMode = Literal["engineered", "waveform", "waveform_plus"]
ClassifierChoice = Literal["rf", "knn", "hgb", "xgb", "both"]

# Decade class labels for ``target`` 0..5 (same convention as training CSVs / notebooks).
DECADE_TARGET_LABELS: Dict[int, str] = {
    0: "20s",
    1: "30s",
    2: "40s",
    3: "50s",
    4: "60s",
    5: "70s",
}


def plot_train_pair_waveforms_by_class_stacked(
    aorta_path: Union[Path, str],
    brach_path: Union[Path, str],
    *,
    cheby_cutoff_hz: float = CHEBY_CUTOFF_HZ,
    sample_rate_hz: float = CHEBY_SAMPLE_RATE_HZ,
    rng: Optional[np.random.Generator] = None,
    title_suffix: str = "",
    show: bool = True,
) -> Any:
    """
    Six stacked axes (shared sample index): one **random** subject per ``target`` class (0–5).

    Uses the **same inner merge** on ``subject_index`` as :func:`load_train_pair` (aortic CSV
    supplies ``target``). Plots **raw** ``aorta_t_*`` / ``brach_t_*`` values faintly and
    **Chebyshev-I** lowpass traces (see :func:`_cheby_lowpass_matrix`) at full opacity — same idea
    as ``data_visualizer.ipynb`` ``plot_subject_dual_waveforms_by_class_stacked``.

    This reflects the **CSV waveforms** that feed feature construction for RF / k-NN / HGB / XGB.
    With ``feature_mode=\"waveform_plus\"``, models may additionally use derived traces
    (e.g. Chebyshev columns, CNN-aligned ``*_preproc``); those are built from these rows.
    """
    import matplotlib.pyplot as plt

    aorta_path = Path(aorta_path)
    brach_path = Path(brach_path)
    _require_train_csv(aorta_path, "Aorta training CSV")
    _require_train_csv(brach_path, "Brachial training CSV")

    aorta = _normalize_subject_column(pd.read_csv(aorta_path))
    brach = _normalize_subject_column(pd.read_csv(brach_path))
    a_cols = waveform_columns("aorta")
    b_cols = waveform_columns("brach")
    missing_a = set(a_cols + [TARGET_COL]) - set(aorta.columns)
    missing_b = set(b_cols + [TARGET_COL]) - set(brach.columns)
    if missing_a:
        raise ValueError(f"aorta train CSV missing columns: {sorted(missing_a)}")
    if missing_b:
        raise ValueError(f"brach train CSV missing columns: {sorted(missing_b)}")

    merged = aorta[[SUBJECT_COL] + a_cols + [TARGET_COL]].merge(
        brach[[SUBJECT_COL] + b_cols],
        on=SUBJECT_COL,
        how="inner",
        validate="one_to_one",
    )
    df_a = merged[[SUBJECT_COL, TARGET_COL] + a_cols]
    df_b = merged[[SUBJECT_COL] + b_cols]

    rng = rng or np.random.default_rng()
    class_ids = sorted(DECADE_TARGET_LABELS.keys())
    n_cls = len(class_ids)
    fig, axes = plt.subplots(n_cls, 1, figsize=(11, 2.5 * n_cls), sharex=True)
    if n_cls == 1:
        axes = np.array([axes])

    times_a = np.arange(len(a_cols), dtype=float)
    times_b = np.arange(len(b_cols), dtype=float)

    sub = df_a[[SUBJECT_COL, TARGET_COL]].copy()
    sub[TARGET_COL] = pd.to_numeric(sub[TARGET_COL], errors="coerce")
    sub = sub.dropna(subset=[TARGET_COL])
    sub[TARGET_COL] = sub[TARGET_COL].astype(int)

    for ax, t in zip(axes, class_ids):
        pool = sub.loc[sub[TARGET_COL] == t, SUBJECT_COL].astype(int).unique().tolist()
        if not pool:
            ax.text(
                0.5,
                0.5,
                f"No subjects with target={t} ({DECADE_TARGET_LABELS.get(t, '')})",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        sid = int(rng.choice(pool))
        row_a = df_a.loc[df_a[SUBJECT_COL] == sid].iloc[0]
        row_b = df_b.loc[df_b[SUBJECT_COL] == sid].iloc[0]
        ya = pd.to_numeric(row_a[a_cols], errors="coerce").to_numpy(dtype=float)
        yb = pd.to_numeric(row_b[b_cols], errors="coerce").to_numpy(dtype=float)
        c_a, c_b = "C0", "C1"
        ax.plot(times_a, ya, color=c_a, alpha=0.3, linewidth=1.0, label="Aortic (aorta_t_*), raw")
        ax.plot(times_b, yb, color=c_b, alpha=0.3, linewidth=1.0, label="Brachial (brach_t_*), raw")
        ya_s = _cheby_lowpass_matrix(
            ya[np.newaxis, :],
            sample_rate_hz=sample_rate_hz,
            cutoff_hz=cheby_cutoff_hz,
            rp_db=CHEBY_RP_DB,
        )[0]
        yb_s = _cheby_lowpass_matrix(
            yb[np.newaxis, :],
            sample_rate_hz=sample_rate_hz,
            cutoff_hz=cheby_cutoff_hz,
            rp_db=CHEBY_RP_DB,
        )[0]
        ax.plot(times_a, ya_s, color=c_a, alpha=1.0, linewidth=1.2, label="Aortic, Cheby-1")
        ax.plot(times_b, yb_s, color=c_b, alpha=1.0, linewidth=1.2, label="Brachial, Cheby-1")
        ax.set_ylabel("Value")
        lbl = DECADE_TARGET_LABELS.get(t, str(t))
        ax.set_title(f"{lbl} — subject_index={sid}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Sample index")
    extra = f" {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"RF / k-NN train merge — random subject per class — raw (α=0.3) + Chebyshev-1 "
        f"({cheby_cutoff_hz:g} Hz @ {sample_rate_hz:g} Hz fs){extra}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if show:
        plt.show()
    return fig


def plot_train_pair_model_inputs_by_class_stacked(
    aorta_path: Union[Path, str],
    brach_path: Union[Path, str],
    *,
    feature_mode: FeatureMode,
    roll_window: int = ROLL_WINDOW_DEFAULT,
    engineered_columns: Optional[Sequence[str]] = None,
    rng: Optional[np.random.Generator] = None,
    title_suffix: str = "",
    show: bool = True,
    waveform_plus_cnn_max_gap_samples: Optional[int] = None,
    waveform_plus_cnn_max_gap_ms: Optional[float] = None,
    waveform_plus_cnn_sample_rate_hz: float = CHEBY_SAMPLE_RATE_HZ,
    waveform_plus_cnn_zscore_eps: float = 1e-6,
    waveform_plus_cnn_zscore_mode: Literal["independent", "aorta_reference"] = "aorta_reference",
) -> Any:
    """
    Six stacked axes (shared sample index): one **random** subject per ``target`` class (0–5).

    Plots the **same feature traces** RF / k-NN / HGB / XGB see from :func:`load_train_pair`:

    - ``waveform``: ``aorta_t_*`` and ``brach_t_*`` (672-column model input, linear gap imputed).
    - ``waveform_plus``: each listed ``engineered_columns`` trace (336 samples), built via
      :func:`extract_waveform_plus` including ``waveform_plus_cnn_*`` when ``*_preproc*`` names are used.
    - ``engineered``: only a short message (scalar features, not 336-sample traces).

    When ``feature_mode=\"waveform_plus\"`` and ``engineered_columns`` is ``None``, plots
    :data:`ENGINEERED_WAVEFORM_TRACE_NAMES_DEFAULT`. When ``engineered_columns`` is ``[]`` (bare 672
    matrix, matching :func:`load_train_pair`), behaves like ``waveform``.
    """
    import matplotlib.pyplot as plt

    aorta_path = Path(aorta_path)
    brach_path = Path(brach_path)
    _require_train_csv(aorta_path, "Aorta training CSV")
    _require_train_csv(brach_path, "Brachial training CSV")

    aorta = _normalize_subject_column(pd.read_csv(aorta_path))
    brach = _normalize_subject_column(pd.read_csv(brach_path))
    a_cols = waveform_columns("aorta")
    b_cols = waveform_columns("brach")
    missing_a = set(a_cols + [TARGET_COL]) - set(aorta.columns)
    missing_b = set(b_cols + [TARGET_COL]) - set(brach.columns)
    if missing_a:
        raise ValueError(f"aorta train CSV missing columns: {sorted(missing_a)}")
    if missing_b:
        raise ValueError(f"brach train CSV missing columns: {sorted(missing_b)}")

    merged = aorta[[SUBJECT_COL] + a_cols + [TARGET_COL]].merge(
        brach[[SUBJECT_COL] + b_cols],
        on=SUBJECT_COL,
        how="inner",
        validate="one_to_one",
    )

    rng = rng or np.random.default_rng()
    class_ids = sorted(DECADE_TARGET_LABELS.keys())
    sub = merged[[SUBJECT_COL, TARGET_COL]].copy()
    sub[TARGET_COL] = pd.to_numeric(sub[TARGET_COL], errors="coerce")
    sub = sub.dropna(subset=[TARGET_COL])
    sub[TARGET_COL] = sub[TARGET_COL].astype(int)

    bare_waveform_plus = (
        feature_mode == "waveform_plus"
        and engineered_columns is not None
        and len(list(engineered_columns)) == 0
    )

    if feature_mode == "engineered":
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.text(
            0.5,
            0.5,
            'FEATURE_MODE="engineered" uses scalar summaries only (no 336-sample trace columns).',
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
        )
        ax.set_axis_off()
        fig.suptitle("RF / k-NN model inputs")
        if show:
            plt.show()
        return fig

    if feature_mode == "waveform" or bare_waveform_plus:
        n_t = len(a_cols)
        times = np.arange(n_t, dtype=float)
        fig, axes = plt.subplots(len(class_ids), 1, figsize=(11, 2.5 * len(class_ids)), sharex=True)
        if len(class_ids) == 1:
            axes = np.array([axes])
        for ax, t in zip(axes, class_ids):
            pool = sub.loc[sub[TARGET_COL] == t, SUBJECT_COL].astype(int).unique().tolist()
            if not pool:
                ax.text(
                    0.5,
                    0.5,
                    f"No subjects with target={t}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue
            sid = int(rng.choice(pool))
            row = merged.loc[merged[SUBJECT_COL] == sid].iloc[0]
            ya = _linear_impute_nonfinite_1d(
                pd.to_numeric(row[a_cols], errors="coerce").to_numpy(dtype=float),
                fill_all_nan=0.0,
            )
            yb = _linear_impute_nonfinite_1d(
                pd.to_numeric(row[b_cols], errors="coerce").to_numpy(dtype=float),
                fill_all_nan=0.0,
            )
            ax.plot(times, ya, color="C0", linewidth=1.2, label="aorta_t_*")
            ax.plot(times, yb, color="C1", linewidth=1.2, label="brach_t_*")
            ax.set_ylabel("Value")
            lbl = DECADE_TARGET_LABELS.get(t, str(t))
            ax.set_title(f"{lbl} — subject_index={sid}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Sample index")
        mode_lbl = "waveform" if feature_mode == "waveform" else "waveform_plus (engineered_columns=[])"
        extra = f" {title_suffix}" if title_suffix else ""
        fig.suptitle(f"RF / k-NN model inputs — {mode_lbl} — 672 raw waveform columns{extra}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        if show:
            plt.show()
        return fig

    trace_names: List[str]
    if engineered_columns is not None and len(list(engineered_columns)) > 0:
        trace_names = subset_engineered_feature_names(list(engineered_columns))
    else:
        trace_names = list(ENGINEERED_WAVEFORM_TRACE_NAMES_DEFAULT)

    X = extract_waveform_plus(
        merged,
        a_cols=a_cols,
        b_cols=b_cols,
        engineered_columns=trace_names,
        roll_window=roll_window,
        waveform_plus_cnn_max_gap_samples=waveform_plus_cnn_max_gap_samples,
        waveform_plus_cnn_max_gap_ms=waveform_plus_cnn_max_gap_ms,
        waveform_plus_cnn_sample_rate_hz=waveform_plus_cnn_sample_rate_hz,
        waveform_plus_cnn_zscore_eps=waveform_plus_cnn_zscore_eps,
        waveform_plus_cnn_zscore_mode=waveform_plus_cnn_zscore_mode,
    )

    n_t = 336
    times = np.arange(n_t, dtype=float)
    fig, axes = plt.subplots(len(class_ids), 1, figsize=(11, 2.5 * len(class_ids)), sharex=True)
    if len(class_ids) == 1:
        axes = np.array([axes])

    for ax, t in zip(axes, class_ids):
        pool = sub.loc[sub[TARGET_COL] == t, SUBJECT_COL].astype(int).unique().tolist()
        if not pool:
            ax.text(
                0.5,
                0.5,
                f"No subjects with target={t} ({DECADE_TARGET_LABELS.get(t, '')})",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        sid = int(rng.choice(pool))
        idx = merged.loc[merged[SUBJECT_COL] == sid].index[0]
        row_X = X.loc[idx]
        for j, name in enumerate(trace_names):
            cols = [f"{name}_t_{i}" for i in range(n_t)]
            miss = [c for c in cols if c not in X.columns]
            if miss:
                raise ValueError(
                    f"Missing expected columns for trace {name!r} (first missing: {miss[0]!r})."
                )
            y = pd.to_numeric(row_X[cols], errors="coerce").to_numpy(dtype=float)
            ax.plot(times, y, color=f"C{j % 10}", linewidth=1.2, label=name)
        ax.set_ylabel("Value")
        lbl = DECADE_TARGET_LABELS.get(t, str(t))
        ax.set_title(f"{lbl} — subject_index={sid}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Sample index (per-trace timestep)")
    name_str = ", ".join(trace_names)
    extra = f" {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"RF / k-NN model inputs (waveform_plus) — {name_str} — "
        f"via extract_waveform_plus (same as load_train_pair){extra}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if show:
        plt.show()
    return fig


def load_train_pair(
    aorta_path: Path,
    brach_path: Path,
    *,
    feature_mode: FeatureMode = "engineered",
    roll_window: int = ROLL_WINDOW_DEFAULT,
    engineered_columns: Optional[Sequence[str]] = None,
    waveform_plus_cnn_max_gap_samples: Optional[int] = None,
    waveform_plus_cnn_max_gap_ms: Optional[float] = None,
    waveform_plus_cnn_sample_rate_hz: float = CHEBY_SAMPLE_RATE_HZ,
    waveform_plus_cnn_zscore_eps: float = 1e-6,
    waveform_plus_cnn_zscore_mode: Literal["independent", "aorta_reference"] = "aorta_reference",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load both training CSVs, align rows on ``subject_index``.

    ``feature_mode``:
      - ``engineered``: PPA + PP peaks + Chebyshev PPA summaries + rolling ARV/CV/slope (12 columns by default).
      - ``waveform``: ``aorta_t_*`` + ``brach_t_*`` (672 columns), linearly imputed along time per channel.
      - ``waveform_plus``: stacks **336-sample traces** per selected :data:`ENGINEERED_FEATURE_NAMES`
        entry (see :func:`extract_waveform_plus`). ``aorta_raw`` / ``brach_raw`` are the full site rows.
        ``None`` selects :data:`ENGINEERED_WAVEFORM_TRACE_NAMES_DEFAULT` plus scalars; ``[]`` returns only
        the 672-column waveform matrix.

    ``engineered_columns``: subset of :data:`ENGINEERED_FEATURE_NAMES` when ``feature_mode`` is
    ``engineered`` **or** ``waveform_plus`` (ignored for ``waveform`` with a warning).

    ``waveform_plus_cnn_*``: passed to :func:`extract_waveform_plus` when CNN-aligned trace names are
    selected (``aorta_preproc``, ``brach_preproc``, ``*_preproc_cheb``). Default
    ``waveform_plus_cnn_max_gap_samples=None`` (with ``max_gap_ms=None``) does **not** drop or
    blank rows for long gaps; set a sample count or ``max_gap_ms`` to enable that policy.
    """
    _require_train_csv(aorta_path, "Aorta training CSV")
    _require_train_csv(brach_path, "Brachial training CSV")

    aorta = _normalize_subject_column(pd.read_csv(aorta_path))
    brach = _normalize_subject_column(pd.read_csv(brach_path))

    a_cols = waveform_columns("aorta")
    b_cols = waveform_columns("brach")

    missing_a = set(a_cols + [TARGET_COL]) - set(aorta.columns)
    missing_b = set(b_cols + [TARGET_COL]) - set(brach.columns)
    if missing_a:
        raise ValueError(f"aorta train CSV missing columns: {sorted(missing_a)}")
    if missing_b:
        raise ValueError(f"brach train CSV missing columns: {sorted(missing_b)}")

    merged = aorta[[SUBJECT_COL] + a_cols + [TARGET_COL]].merge(
        brach[[SUBJECT_COL] + b_cols],
        on=SUBJECT_COL,
        how="inner",
        validate="one_to_one",
    )

    if engineered_columns is not None and feature_mode == "waveform":
        warnings.warn(
            "engineered_columns is ignored unless feature_mode is 'engineered' or 'waveform_plus'.",
            UserWarning,
            stacklevel=2,
        )

    if feature_mode == "waveform":
        X = _linear_impute_raw_waveform_dataframe(merged, a_cols=a_cols, b_cols=b_cols)
    elif feature_mode == "waveform_plus":
        X = extract_waveform_plus(
            merged,
            a_cols=a_cols,
            b_cols=b_cols,
            engineered_columns=engineered_columns,
            roll_window=roll_window,
            waveform_plus_cnn_max_gap_samples=waveform_plus_cnn_max_gap_samples,
            waveform_plus_cnn_max_gap_ms=waveform_plus_cnn_max_gap_ms,
            waveform_plus_cnn_sample_rate_hz=waveform_plus_cnn_sample_rate_hz,
            waveform_plus_cnn_zscore_eps=waveform_plus_cnn_zscore_eps,
            waveform_plus_cnn_zscore_mode=waveform_plus_cnn_zscore_mode,
        )
    elif feature_mode == "engineered":
        X = extract_engineered_features(
            merged,
            a_cols=a_cols,
            b_cols=b_cols,
            window=roll_window,
            columns=engineered_columns,
        )
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode!r}")

    y = merged[TARGET_COL]
    return X, y


def load_test_pair(
    aorta_path: Path,
    brach_path: Path,
    *,
    feature_mode: FeatureMode = "engineered",
    roll_window: int = ROLL_WINDOW_DEFAULT,
    engineered_columns: Optional[Sequence[str]] = None,
    waveform_plus_cnn_max_gap_samples: Optional[int] = None,
    waveform_plus_cnn_max_gap_ms: Optional[float] = None,
    waveform_plus_cnn_sample_rate_hz: float = CHEBY_SAMPLE_RATE_HZ,
    waveform_plus_cnn_zscore_eps: float = 1e-6,
    waveform_plus_cnn_zscore_mode: Literal["independent", "aorta_reference"] = "aorta_reference",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load test CSVs (no target). Returns feature matrix and ``subject_index``.

    ``engineered_columns``: same as :func:`load_train_pair`.
    """
    aorta = _normalize_subject_column(pd.read_csv(aorta_path))
    brach = _normalize_subject_column(pd.read_csv(brach_path))

    a_cols = waveform_columns("aorta")
    b_cols = waveform_columns("brach")

    merged = aorta[[SUBJECT_COL] + a_cols].merge(
        brach[[SUBJECT_COL] + b_cols],
        on=SUBJECT_COL,
        how="inner",
        validate="one_to_one",
    )

    if engineered_columns is not None and feature_mode == "waveform":
        warnings.warn(
            "engineered_columns is ignored unless feature_mode is 'engineered' or 'waveform_plus'.",
            UserWarning,
            stacklevel=2,
        )

    if feature_mode == "waveform":
        X = _linear_impute_raw_waveform_dataframe(merged, a_cols=a_cols, b_cols=b_cols)
    elif feature_mode == "waveform_plus":
        X = extract_waveform_plus(
            merged,
            a_cols=a_cols,
            b_cols=b_cols,
            engineered_columns=engineered_columns,
            roll_window=roll_window,
            waveform_plus_cnn_max_gap_samples=waveform_plus_cnn_max_gap_samples,
            waveform_plus_cnn_max_gap_ms=waveform_plus_cnn_max_gap_ms,
            waveform_plus_cnn_sample_rate_hz=waveform_plus_cnn_sample_rate_hz,
            waveform_plus_cnn_zscore_eps=waveform_plus_cnn_zscore_eps,
            waveform_plus_cnn_zscore_mode=waveform_plus_cnn_zscore_mode,
        )
    elif feature_mode == "engineered":
        X = extract_engineered_features(
            merged,
            a_cols=a_cols,
            b_cols=b_cols,
            window=roll_window,
            columns=engineered_columns,
        )
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode!r}")

    subjects = merged[SUBJECT_COL]
    return X, subjects


def build_model(
    *,
    nan_imputer_cls: Type[Any] = SimpleImputer,
    nan_imputer_kwargs: Optional[Dict[str, Any]] = None,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    class_weight: Union[str, dict, None] = "balanced_subsample",
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    """Pipeline: missing-value handling (default ``SimpleImputer``) + ``RandomForestClassifier``."""
    imp_kw: Dict[str, Any] = {}
    if isinstance(nan_imputer_cls, type) and issubclass(nan_imputer_cls, SimpleImputer):
        imp_kw["strategy"] = "median"
    if nan_imputer_kwargs:
        imp_kw.update(nan_imputer_kwargs)
    imputer = nan_imputer_cls(**imp_kw)

    return Pipeline(
        steps=[
            ("imputer", imputer),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight=class_weight,
                    n_jobs=n_jobs,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_knn_pipeline(
    *,
    nan_imputer_cls: Type[Any] = SimpleImputer,
    nan_imputer_kwargs: Optional[Dict[str, Any]] = None,
    n_neighbors: int = 7,
    weights: str = "distance",
    n_jobs: int = -1,
) -> Pipeline:
    """Pipeline: imputer + ``StandardScaler`` + ``KNeighborsClassifier``."""
    imp_kw: Dict[str, Any] = {}
    if isinstance(nan_imputer_cls, type) and issubclass(nan_imputer_cls, SimpleImputer):
        imp_kw["strategy"] = "median"
    if nan_imputer_kwargs:
        imp_kw.update(nan_imputer_kwargs)
    imputer = nan_imputer_cls(**imp_kw)

    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def build_hgb_pipeline(
    *,
    nan_imputer_cls: Type[Any] = SimpleImputer,
    nan_imputer_kwargs: Optional[Dict[str, Any]] = None,
    learning_rate: float = 0.1,
    max_iter: int = 100,
    max_depth: Optional[int] = None,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 0.0,
    class_weight: Union[str, dict, None] = None,
    random_state: Optional[int] = None,
    early_stopping: Union[str, bool] = "auto",
) -> Pipeline:
    """Pipeline: missing-value handling + :class:`~sklearn.ensemble.HistGradientBoostingClassifier`."""
    imp_kw: Dict[str, Any] = {}
    if isinstance(nan_imputer_cls, type) and issubclass(nan_imputer_cls, SimpleImputer):
        imp_kw["strategy"] = "median"
    if nan_imputer_kwargs:
        imp_kw.update(nan_imputer_kwargs)
    imputer = nan_imputer_cls(**imp_kw)

    return Pipeline(
        steps=[
            ("imputer", imputer),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    class_weight=class_weight,
                    random_state=random_state,
                    early_stopping=early_stopping,
                ),
            ),
        ]
    )


def build_xgb_pipeline(
    *,
    nan_imputer_cls: Type[Any] = SimpleImputer,
    nan_imputer_kwargs: Optional[Dict[str, Any]] = None,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_child_weight: float = 1.0,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Pipeline:
    """Pipeline: missing-value handling + :class:`xgboost.XGBClassifier` (multiclass)."""
    if XGBClassifier is None:
        raise ImportError(_xgb_unavailable_message()) from _XGB_IMPORT_ERROR

    imp_kw: Dict[str, Any] = {}
    if isinstance(nan_imputer_cls, type) and issubclass(nan_imputer_cls, SimpleImputer):
        imp_kw["strategy"] = "median"
    if nan_imputer_kwargs:
        imp_kw.update(nan_imputer_kwargs)
    imputer = nan_imputer_cls(**imp_kw)

    return Pipeline(
        steps=[
            ("imputer", imputer),
            (
                "clf",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    reg_alpha=reg_alpha,
                    min_child_weight=min_child_weight,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    tree_method=tree_method,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                ),
            ),
        ]
    )


EvalModelKind = Literal["rf", "knn", "hgb", "xgb", "cnn"]


def infer_eval_model_kind(path: Union[str, Path]) -> EvalModelKind:
    """Infer classifier family from a saved artifact (sklearn pipeline or PyTorch checkpoint).

    - ``.pt`` / ``.pth`` → ``\"cnn\"``
    - ``.joblib`` / ``.pkl`` / ``.pickle``: load with joblib and inspect the last pipeline step
      (or bare estimator). Returns ``\"rf\"``, ``\"knn\"``, ``\"hgb\"``, or ``\"xgb\"``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model path is not an existing file: {p}")
    suf = p.suffix.lower()
    if suf in (".pt", ".pth"):
        return "cnn"
    if joblib is None:
        raise ImportError("joblib is required to load sklearn pipelines for type inference")

    obj: Any = joblib.load(p)
    if isinstance(obj, Pipeline):
        clf = obj.steps[-1][1]
    else:
        clf = obj

    if isinstance(clf, RandomForestClassifier):
        return "rf"
    if isinstance(clf, KNeighborsClassifier):
        return "knn"
    if isinstance(clf, HistGradientBoostingClassifier):
        return "hgb"
    if XGBClassifier is not None and isinstance(clf, XGBClassifier):
        return "xgb"

    cname = type(clf).__name__.lower()
    if "randomforest" in cname:
        return "rf"
    if "kneighbor" in cname:
        return "knn"
    if "histgradient" in cname or "histogramgradient" in cname:
        return "hgb"
    if "xgb" in cname or "xgboost" in cname:
        return "xgb"
    raise ValueError(
        f"Cannot infer model kind from final estimator {type(clf)!r} in {p}. "
        "Expected RF, k-NN, HistGradientBoosting, XGBoost, or a PyTorch .pt checkpoint."
    )


# Default hyperparameter grids for :func:`main` when ``grid_search=True`` (sklearn ``clf__*`` prefixes).
DEFAULT_RF_PARAM_GRID: Dict[str, List[Any]] = {
    "clf__n_estimators": [100, 250],
    "clf__max_depth": [None, 20, 40],
    "clf__min_samples_leaf": [1, 2, 4],
}

DEFAULT_KNN_PARAM_GRID: Dict[str, List[Any]] = {
    "clf__n_neighbors": [3, 5, 7, 11],
    "clf__weights": ["uniform", "distance"],
}

DEFAULT_HGB_PARAM_GRID: Dict[str, List[Any]] = {
    "clf__learning_rate": [0.05, 0.1, 0.15],
    "clf__max_iter": [100, 200],
    "clf__max_depth": [None, 8, 16],
    "clf__max_leaf_nodes": [31, 64],
    "clf__l2_regularization": [0.0, 0.1],
}

DEFAULT_XGB_PARAM_GRID: Dict[str, List[Any]] = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [4, 6, 8],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0],
}


def _print_discovered_hyperparameters(
    best_params: Mapping[str, Any],
    *,
    model_label: str,
    intro: str = "Discovered hyperparameters (best grid point)",
) -> None:
    """Pretty-print sklearn ``best_params_`` after grid search."""
    print(f"=== {intro} — {model_label} ===")
    if not best_params:
        print("  (empty)")
        return
    keys = sorted(best_params.keys())
    width = max(len(k) for k in keys)
    for key in keys:
        val = best_params[key]
        print(f"  {key.ljust(width)}  =  {val!r}")


def _fit_pipeline_with_optional_grid_search(
    pipeline: Pipeline,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    *,
    grid_search: bool,
    param_grid: Optional[Dict[str, Sequence[Any]]],
    default_param_grid: Dict[str, List[Any]],
    cv: int,
    scoring: str,
    n_jobs: int,
    random_state: int,
    verbose: int,
    model_label: str,
) -> Tuple[Pipeline, Optional[Dict[str, Any]]]:
    """
    Fit ``pipeline`` on ``(X_train, y_train)``, optionally via ``GridSearchCV``.

    When ``grid_search`` is False, fits once and prints a validation report on ``(X_val, y_val)``.
    When True, runs stratified *k*-fold CV on the training fold, prints best params / CV score,
    then prints a holdout report on ``(X_val, y_val)`` (same stratified split as :func:`main`,
    not part of CV).

    Returns the fitted best pipeline (training fold only) and optional metadata
    ``{best_params, best_cv_score, scoring, cv}``.
    """
    y_tr = y_train
    y_v = y_val
    if not grid_search:
        pipeline.fit(X_train, y_tr)
        print(f"=== {model_label} — validation ===")
        print(classification_report(y_v, pipeline.predict(X_val), digits=3))
        return pipeline, None

    grid: Dict[str, Sequence[Any]] = dict(param_grid) if param_grid is not None else dict(default_param_grid)
    if not grid:
        raise ValueError(f"{model_label}: empty hyperparameter grid for grid search.")

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipeline,
        grid,
        cv=cv_split,
        scoring=scoring,
        refit=True,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    gs.fit(X_train, y_tr)
    print(f"=== {model_label} — GridSearchCV (scoring={scoring!r}, cv={cv}) ===")
    print(f"Best mean CV score: {gs.best_score_:.5f}")
    _print_discovered_hyperparameters(gs.best_params_, model_label=model_label)
    best = gs.best_estimator_
    print(f"=== {model_label} — holdout (same stratified split as main; not used in CV) ===")
    print(classification_report(y_v, best.predict(X_val), digits=3))
    meta: Dict[str, Any] = {
        "best_params": gs.best_params_,
        "best_cv_score": float(gs.best_score_),
        "scoring": scoring,
        "cv": cv,
    }
    return best, meta


def load_param_grids_from_json(path: Union[Path, str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Load optional per-model grids from a JSON object with keys ``rf``, ``knn``, ``hgb``, ``xgb``.

    Values are dicts mapping parameter names (e.g. ``clf__n_estimators``) to lists of
    candidates; use JSON ``null`` for ``None`` (e.g. unlimited tree depth).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Grid JSON must be a JSON object at the top level.")
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for key in ("rf", "knn", "hgb", "xgb"):
        v = raw.get(key)
        if v is None:
            out[key] = None
        elif isinstance(v, dict):
            out[key] = dict(v)
        else:
            raise ValueError(f"Grid JSON[{key!r}] must be an object or omitted, got {type(v).__name__}")
    return out


def predictions_to_submission_json(
    subject_indices: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> dict[int, int]:
    """
    Map subject_index -> predicted class. Keys and values are Python ints.
    Note: ``json.dumps`` will serialize dict keys as strings (JSON requirement).
    """
    out: dict[int, int] = {}
    for sid, cls in zip(subject_indices, y_pred, strict=True):
        out[int(sid)] = int(cls)
    return out


def write_prediction_json(path: Path, mapping: dict[int, int]) -> None:
    """Write predictions; keys sort numerically for stable diffs."""
    ordered = {str(k): int(mapping[k]) for k in sorted(mapping)}
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


def _as_path(p: Union[Path, str]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _path_with_val_accuracy(path: Union[Path, str], acc: float) -> Path:
    """
    ``heart_age_knn.joblib`` → ``heart_age_knn_acc986.joblib`` when ``acc`` is ``0.9863``
    (98.63%): value is **tenths of a percent** as an integer, ``int(round(acc * 1000))``.
    """
    p = _as_path(path)
    tenths = int(round(float(acc) * 1000))
    return p.parent / f"{p.stem}_acc{tenths}{p.suffix}"


def _format_grid_param_value(val: Any) -> str:
    """Stringify a single grid-search value for use in a filename token."""
    if val is None:
        return "None"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        t = format(val, ".6g")
        return t.replace(".", "p").replace("-", "m")
    if isinstance(val, int):
        return str(val)
    return str(val)


def _sanitize_filename_token(s: str) -> str:
    """Keep alphanumerics, hyphen, underscore; map other characters to underscore."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)


def _grid_params_filename_slug(
    best_params: Mapping[str, Any],
    *,
    max_len: int = 96,
    hash_len: int = 14,
) -> str:
    """
    Build a stable, filesystem-safe token from ``GridSearchCV.best_params_``.

    Sorted by stripped parameter name (``clf__`` prefix removed). If the readable slug
    exceeds ``max_len``, falls back to ``hp_sha`` + a short SHA-256 hex digest of the
    canonical JSON (sorted keys).
    """
    if not best_params:
        return "hp_empty"

    segments: list[str] = []
    for key in sorted(best_params.keys(), key=lambda k: k.split("__", 1)[-1]):
        short_key = key.split("__", 1)[-1]
        val_tok = _format_grid_param_value(best_params[key])
        seg = _sanitize_filename_token(f"{short_key}-{val_tok}")
        while "__" in seg:
            seg = seg.replace("__", "_")
        segments.append(seg)

    slug = "hp_" + "__".join(segments)
    if len(slug) <= max_len:
        return slug

    canonical = json.dumps(dict(best_params), sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:hash_len]
    return f"hp_sha{digest}"


def _path_with_grid_params_slug(path: Union[Path, str], best_params: Mapping[str, Any]) -> Path:
    """Append :func:`_grid_params_filename_slug` to ``path`` stem (before suffix)."""
    p = _as_path(path)
    slug = _grid_params_filename_slug(best_params)
    return p.parent / f"{p.stem}_{slug}{p.suffix}"


def _save_sklearn_pipeline(path: Union[Path, str], pipeline: Pipeline) -> None:
    """Persist a fitted sklearn ``Pipeline`` with joblib."""
    if joblib is None:
        raise ImportError("Saving models requires joblib: pip install joblib")
    p = _as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, p)


def load_fitted_pipeline(path: Union[Path, str]) -> Pipeline:
    """Load a pipeline saved by :func:`_save_sklearn_pipeline`."""
    if joblib is None:
        raise ImportError("Loading models requires joblib: pip install joblib")
    return joblib.load(_as_path(path))


def _resolve_random_state(random_state: Optional[int]) -> int:
    """Use a fixed seed when provided; otherwise draw a random integer and print it."""
    if random_state is not None:
        return random_state
    seed = random.randint(0, 2**31 - 1)
    print(f"Random seed (auto-generated): {seed}")
    return seed


def main(
    *,
    classifier: ClassifierChoice = "both",
    training_csvs: Optional[Sequence[Union[Path, str]]] = None,
    test_csvs: Optional[Sequence[Union[Path, str]]] = None,
    out_json: Optional[Union[Path, str]] = None,
    out_json_knn: Optional[Union[Path, str]] = None,
    out_json_hgb: Optional[Union[Path, str]] = None,
    out_json_xgb: Optional[Union[Path, str]] = None,
    save_model_rf: Optional[Union[Path, str]] = None,
    save_model_knn: Optional[Union[Path, str]] = None,
    save_model_hgb: Optional[Union[Path, str]] = None,
    save_model_xgb: Optional[Union[Path, str]] = None,
    feature_mode: FeatureMode = "engineered",
    roll_window: int = ROLL_WINDOW_DEFAULT,
    engineered_columns: Optional[Sequence[str]] = None,
    val_split: float = 0.2,
    nan_imputer_cls: Type[Any] = SimpleImputer,
    nan_imputer_kwargs: Optional[Dict[str, Any]] = None,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    class_weight: Union[str, dict, None] = "balanced_subsample",
    knn_neighbors: int = 7,
    knn_weights: str = "distance",
    hgb_learning_rate: float = 0.1,
    hgb_max_iter: int = 100,
    hgb_max_depth: Optional[int] = None,
    hgb_max_leaf_nodes: int = 31,
    hgb_l2_regularization: float = 0.0,
    hgb_class_weight: Union[str, dict, None] = None,
    hgb_early_stopping: Union[str, bool] = "auto",
    xgb_n_estimators: int = 400,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.1,
    xgb_subsample: float = 0.9,
    xgb_colsample_bytree: float = 0.9,
    xgb_reg_lambda: float = 1.0,
    xgb_reg_alpha: float = 0.0,
    xgb_min_child_weight: float = 1.0,
    xgb_tree_method: str = "hist",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    embed_validation_accuracy_in_save_path: bool = False,
    embed_grid_search_params_in_save_path: bool = True,
    grid_search: bool = False,
    grid_search_cv: int = 5,
    grid_search_scoring: str = "f1_macro",
    grid_search_verbose: int = 0,
    rf_param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    knn_param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    hgb_param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    xgb_param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    skip_if_xgb_unavailable: bool = False,
) -> Dict[str, Any]:
    """
    Train / evaluate on labeled CSVs and optionally predict on test CSVs.

    ``training_csvs`` — exactly two paths ``[aorta_train.csv, brach_train.csv]``.
    ``test_csvs`` — optional ``[aorta_test.csv, brach_test.csv]``; if omitted, default
    locations under ``datasets/test/`` next to this file are used when present.

    ``classifier`` — ``"rf"``, ``"knn"``, ``"hgb"`` (sklearn histogram boosting), ``"xgb"``
    (XGBoost), or ``"both"`` (RF + k-NN only) on the same feature matrix. Fitted pipelines can
    be written with ``save_model_rf`` / ``save_model_knn`` / ``save_model_hgb`` /
    ``save_model_xgb`` (requires ``joblib``; XGBoost requires the ``xgboost`` package).
    If XGBoost cannot load (missing package or native/OpenMP mismatch) and
    ``skip_if_xgb_unavailable`` is True, the XGBoost branch is skipped with a warning instead
    of raising.

    If ``embed_validation_accuracy_in_save_path`` is True, each save path gets
    ``_acc{tenths}`` before the suffix (tenths of a percent, e.g. 98.63% →
    ``heart_age_knn_acc986.joblib``), using holdout accuracy from the stratified
    ``val_split`` before the final full-data refit.

    If ``embed_grid_search_params_in_save_path`` is True (default) and ``grid_search`` found
    a best parameter set, the save path also gets a token derived from ``best_params_``
    (after any accuracy token), e.g. ``..._acc986_hp_n_estimators-250__max_depth-20.joblib``.
    Very long slugs fall back to a short ``hp_sha{hex}`` digest.

    **Hyperparameter search:** when ``grid_search`` is True, each trained classifier is fit
    with :class:`~sklearn.model_selection.GridSearchCV` on the **training** split only
    (``StratifiedKFold`` with ``grid_search_cv`` folds). Holdout metrics still use the
    stratified ``val_split`` slice. Defaults are :data:`DEFAULT_RF_PARAM_GRID`,
    :data:`DEFAULT_KNN_PARAM_GRID`, :data:`DEFAULT_HGB_PARAM_GRID`, and
    :data:`DEFAULT_XGB_PARAM_GRID`; override with the ``*_param_grid`` arguments or
    :func:`load_param_grids_from_json`. For HGB, ``early_stopping`` is forced to ``False`` during
    grid search so folds are comparable.

    Default features are **engineered** (notebook-aligned); use ``feature_mode="waveform"``
    for raw 672-column waveforms, or ``feature_mode="waveform_plus"`` with
    ``engineered_columns`` selecting which 336-sample traces to stack—including ``aorta_raw``,
    ``brach_raw``, and CNN phase-1–aligned names such as ``aorta_preproc`` (see :func:`extract_waveform_plus`).
    With ``feature_mode="engineered"``, pass
    ``engineered_columns`` using only scalar names from :data:`ENGINEERED_SCALAR_NAMES` (or omit for all).

    Missing values: instantiate ``nan_imputer_cls`` with ``nan_imputer_kwargs``. For
    ``SimpleImputer`` subclasses, defaults include ``strategy="median"`` unless overridden.

    If ``random_state`` is omitted (``None``), a random integer seed is chosen at runtime.

    Returns a dict with ``random_state_used`` (the resolved integer seed for the stratified
    split), optional ``val_accuracy_*``, and ``saved_model_*`` paths (``None`` if not trained or
    not saved). When ``grid_search`` is True, ``grid_search_*`` keys hold best-params metadata
    for each model that was tuned.
    """
    if classifier not in ("rf", "knn", "hgb", "xgb", "both"):
        raise ValueError('classifier must be "rf", "knn", "hgb", "xgb", or "both"')
    rng_seed = _resolve_random_state(random_state)

    default_root = Path(__file__).resolve().parent

    if training_csvs is None:
        train_aorta = default_root / "datasets/train/aortaP_train_data.csv"
        train_brach = default_root / "datasets/train/brachP_train_data.csv"
    else:
        pair = list(training_csvs)
        if len(pair) != 2:
            raise ValueError(
                "training_csvs must have exactly two entries: [aorta_csv, brach_csv]"
            )
        train_aorta, train_brach = _as_path(pair[0]), _as_path(pair[1])

    out_path = _as_path(out_json) if out_json is not None else default_root / "predictions.json"
    knn_out_path = (
        _as_path(out_json_knn)
        if out_json_knn is not None
        else default_root / "predictions_knn.json"
    )
    hgb_out_path = (
        _as_path(out_json_hgb)
        if out_json_hgb is not None
        else default_root / "predictions_hgb.json"
    )
    xgb_out_path = (
        _as_path(out_json_xgb)
        if out_json_xgb is not None
        else default_root / "predictions_xgb.json"
    )

    if test_csvs is None:
        test_aorta = default_root / "datasets/test/aortaP_test_data.csv"
        test_brach = default_root / "datasets/test/brachP_test_data.csv"
    else:
        tpair = list(test_csvs)
        if len(tpair) != 2:
            raise ValueError("test_csvs must have exactly two entries: [aorta_csv, brach_csv]")
        test_aorta, test_brach = _as_path(tpair[0]), _as_path(tpair[1])

    X, y = load_train_pair(
        train_aorta,
        train_brach,
        feature_mode=feature_mode,
        roll_window=roll_window,
        engineered_columns=engineered_columns,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_split,
        random_state=rng_seed,
        stratify=y,
    )

    rf: Optional[Pipeline] = None
    knn: Optional[Pipeline] = None
    hgb: Optional[Pipeline] = None
    xgb: Optional[Pipeline] = None
    grid_meta_rf: Optional[Dict[str, Any]] = None
    grid_meta_knn: Optional[Dict[str, Any]] = None
    grid_meta_hgb: Optional[Dict[str, Any]] = None
    grid_meta_xgb: Optional[Dict[str, Any]] = None

    if grid_search and grid_search_cv < 2:
        raise ValueError("grid_search_cv must be >= 2 when grid_search is True.")

    if classifier in ("rf", "both"):
        rf_pipe = build_model(
            nan_imputer_cls=nan_imputer_cls,
            nan_imputer_kwargs=nan_imputer_kwargs,
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=rng_seed,
        )
        rf, grid_meta_rf = _fit_pipeline_with_optional_grid_search(
            rf_pipe,
            X_train,
            y_train,
            X_val,
            y_val,
            grid_search=grid_search,
            param_grid=rf_param_grid,
            default_param_grid=DEFAULT_RF_PARAM_GRID,
            cv=grid_search_cv,
            scoring=grid_search_scoring,
            n_jobs=n_jobs,
            random_state=rng_seed,
            verbose=grid_search_verbose,
            model_label=f"Random Forest (features={feature_mode})",
        )

    if classifier in ("knn", "both"):
        knn_pipe = build_knn_pipeline(
            nan_imputer_cls=nan_imputer_cls,
            nan_imputer_kwargs=nan_imputer_kwargs,
            n_neighbors=knn_neighbors,
            weights=knn_weights,
            n_jobs=n_jobs,
        )
        knn, grid_meta_knn = _fit_pipeline_with_optional_grid_search(
            knn_pipe,
            X_train,
            y_train,
            X_val,
            y_val,
            grid_search=grid_search,
            param_grid=knn_param_grid,
            default_param_grid=DEFAULT_KNN_PARAM_GRID,
            cv=grid_search_cv,
            scoring=grid_search_scoring,
            n_jobs=n_jobs,
            random_state=rng_seed,
            verbose=grid_search_verbose,
            model_label="k-NN",
        )

    if classifier == "hgb":
        hgb_es: Union[str, bool] = False if grid_search else hgb_early_stopping
        hgb_pipe = build_hgb_pipeline(
            nan_imputer_cls=nan_imputer_cls,
            nan_imputer_kwargs=nan_imputer_kwargs,
            learning_rate=hgb_learning_rate,
            max_iter=hgb_max_iter,
            max_depth=hgb_max_depth,
            max_leaf_nodes=hgb_max_leaf_nodes,
            l2_regularization=hgb_l2_regularization,
            class_weight=hgb_class_weight,
            random_state=rng_seed,
            early_stopping=hgb_es,
        )
        hgb, grid_meta_hgb = _fit_pipeline_with_optional_grid_search(
            hgb_pipe,
            X_train,
            y_train,
            X_val,
            y_val,
            grid_search=grid_search,
            param_grid=hgb_param_grid,
            default_param_grid=DEFAULT_HGB_PARAM_GRID,
            cv=grid_search_cv,
            scoring=grid_search_scoring,
            n_jobs=n_jobs,
            random_state=rng_seed,
            verbose=grid_search_verbose,
            model_label=f"HistGradientBoosting (features={feature_mode})",
        )

    if classifier == "xgb":
        if XGBClassifier is None:
            if skip_if_xgb_unavailable:
                warnings.warn(
                    "Skipping XGBoost: " + _xgb_unavailable_message(),
                    UserWarning,
                    stacklevel=1,
                )
            else:
                raise ImportError(_xgb_unavailable_message()) from _XGB_IMPORT_ERROR
        if XGBClassifier is not None:
            xgb_pipe = build_xgb_pipeline(
                nan_imputer_cls=nan_imputer_cls,
                nan_imputer_kwargs=nan_imputer_kwargs,
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                reg_lambda=xgb_reg_lambda,
                reg_alpha=xgb_reg_alpha,
                min_child_weight=xgb_min_child_weight,
                random_state=rng_seed,
                n_jobs=n_jobs,
                tree_method=xgb_tree_method,
            )
            xgb, grid_meta_xgb = _fit_pipeline_with_optional_grid_search(
                xgb_pipe,
                X_train,
                y_train,
                X_val,
                y_val,
                grid_search=grid_search,
                param_grid=xgb_param_grid,
                default_param_grid=DEFAULT_XGB_PARAM_GRID,
                cv=grid_search_cv,
                scoring=grid_search_scoring,
                n_jobs=n_jobs,
                random_state=rng_seed,
                verbose=grid_search_verbose,
                model_label=f"XGBoost (features={feature_mode})",
            )

    val_accuracy_rf: Optional[float] = None
    val_accuracy_knn: Optional[float] = None
    val_accuracy_hgb: Optional[float] = None
    val_accuracy_xgb: Optional[float] = None
    if rf is not None:
        val_accuracy_rf = float(accuracy_score(y_val, rf.predict(X_val)))
    if knn is not None:
        val_accuracy_knn = float(accuracy_score(y_val, knn.predict(X_val)))
    if hgb is not None:
        val_accuracy_hgb = float(accuracy_score(y_val, hgb.predict(X_val)))
    if xgb is not None:
        val_accuracy_xgb = float(accuracy_score(y_val, xgb.predict(X_val)))

    saved_model_rf: Optional[Path] = None
    saved_model_knn: Optional[Path] = None
    saved_model_hgb: Optional[Path] = None
    saved_model_xgb: Optional[Path] = None

    if rf is not None:
        rf = clone(rf)
        rf.fit(X, y)
        if save_model_rf is not None:
            path_rf = _as_path(save_model_rf)
            if embed_validation_accuracy_in_save_path and val_accuracy_rf is not None:
                path_rf = _path_with_val_accuracy(path_rf, val_accuracy_rf)
            if (
                embed_grid_search_params_in_save_path
                and grid_meta_rf is not None
                and grid_meta_rf.get("best_params")
            ):
                path_rf = _path_with_grid_params_slug(path_rf, grid_meta_rf["best_params"])
            _save_sklearn_pipeline(path_rf, rf)
            saved_model_rf = path_rf
            print(f"Saved Random Forest pipeline: {path_rf}")

    if knn is not None:
        knn = clone(knn)
        knn.fit(X, y)
        if save_model_knn is not None:
            path_knn = _as_path(save_model_knn)
            if embed_validation_accuracy_in_save_path and val_accuracy_knn is not None:
                path_knn = _path_with_val_accuracy(path_knn, val_accuracy_knn)
            if (
                embed_grid_search_params_in_save_path
                and grid_meta_knn is not None
                and grid_meta_knn.get("best_params")
            ):
                path_knn = _path_with_grid_params_slug(path_knn, grid_meta_knn["best_params"])
            _save_sklearn_pipeline(path_knn, knn)
            saved_model_knn = path_knn
            print(f"Saved k-NN pipeline: {path_knn}")

    if hgb is not None:
        hgb = clone(hgb)
        hgb.fit(X, y)
        if save_model_hgb is not None:
            path_hgb = _as_path(save_model_hgb)
            if embed_validation_accuracy_in_save_path and val_accuracy_hgb is not None:
                path_hgb = _path_with_val_accuracy(path_hgb, val_accuracy_hgb)
            if (
                embed_grid_search_params_in_save_path
                and grid_meta_hgb is not None
                and grid_meta_hgb.get("best_params")
            ):
                path_hgb = _path_with_grid_params_slug(path_hgb, grid_meta_hgb["best_params"])
            _save_sklearn_pipeline(path_hgb, hgb)
            saved_model_hgb = path_hgb
            print(f"Saved HistGradientBoosting pipeline: {path_hgb}")

    if xgb is not None:
        xgb = clone(xgb)
        xgb.fit(X, y)
        if save_model_xgb is not None:
            path_xgb = _as_path(save_model_xgb)
            if embed_validation_accuracy_in_save_path and val_accuracy_xgb is not None:
                path_xgb = _path_with_val_accuracy(path_xgb, val_accuracy_xgb)
            if (
                embed_grid_search_params_in_save_path
                and grid_meta_xgb is not None
                and grid_meta_xgb.get("best_params")
            ):
                path_xgb = _path_with_grid_params_slug(path_xgb, grid_meta_xgb["best_params"])
            _save_sklearn_pipeline(path_xgb, xgb)
            saved_model_xgb = path_xgb
            print(f"Saved XGBoost pipeline: {path_xgb}")

    if test_aorta.is_file() and test_brach.is_file():
        X_test, subjects = load_test_pair(
            test_aorta,
            test_brach,
            feature_mode=feature_mode,
            roll_window=roll_window,
            engineered_columns=engineered_columns,
        )
        ref_for_check: Optional[Dict[int, int]] = None
        if rf is not None:
            submission_rf = predictions_to_submission_json(subjects.values, rf.predict(X_test))
            ref_for_check = submission_rf
            write_prediction_json(out_path, submission_rf)
            print(f"Wrote RF predictions: {out_path} ({len(submission_rf)} subjects).")
        if knn is not None:
            submission_knn = predictions_to_submission_json(subjects.values, knn.predict(X_test))
            if ref_for_check is None:
                ref_for_check = submission_knn
            write_prediction_json(knn_out_path, submission_knn)
            print(f"Wrote k-NN predictions: {knn_out_path} ({len(submission_knn)} subjects).")
        if hgb is not None:
            submission_hgb = predictions_to_submission_json(subjects.values, hgb.predict(X_test))
            if ref_for_check is None:
                ref_for_check = submission_hgb
            write_prediction_json(hgb_out_path, submission_hgb)
            print(f"Wrote HGB predictions: {hgb_out_path} ({len(submission_hgb)} subjects).")
        if xgb is not None:
            submission_xgb = predictions_to_submission_json(subjects.values, xgb.predict(X_test))
            if ref_for_check is None:
                ref_for_check = submission_xgb
            write_prediction_json(xgb_out_path, submission_xgb)
            print(f"Wrote XGBoost predictions: {xgb_out_path} ({len(submission_xgb)} subjects).")
        if ref_for_check is not None:
            missing = set(EXPECTED_TEST_INDICES) - set(ref_for_check.keys())
            extra = set(ref_for_check.keys()) - set(EXPECTED_TEST_INDICES)
            if missing or extra:
                print(
                    f"Warning: expected subjects 0–874; "
                    f"missing={len(missing)} extra={len(extra)}"
                )
    else:
        print("Test files not found; skipped prediction JSON. Train metrics only.")

    return {
        "random_state_used": rng_seed,
        "val_accuracy_rf": val_accuracy_rf,
        "val_accuracy_knn": val_accuracy_knn,
        "val_accuracy_hgb": val_accuracy_hgb,
        "val_accuracy_xgb": val_accuracy_xgb,
        "saved_model_rf": saved_model_rf,
        "saved_model_knn": saved_model_knn,
        "saved_model_hgb": saved_model_hgb,
        "saved_model_xgb": saved_model_xgb,
        "grid_search_rf": grid_meta_rf,
        "grid_search_knn": grid_meta_knn,
        "grid_search_hgb": grid_meta_hgb,
        "grid_search_xgb": grid_meta_xgb,
    }


def run_example(
    *,
    classifier: ClassifierChoice = "rf",
    training_csvs: Optional[Sequence[Union[Path, str]]] = None,
    test_csvs: Optional[Sequence[Union[Path, str]]] = None,
    feature_mode: FeatureMode = "engineered",
    roll_window: int = ROLL_WINDOW_DEFAULT,
    engineered_columns: Optional[Sequence[str]] = None,
    val_split: float = 0.2,
    random_state: Optional[int] = 42,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    class_weight: Union[str, dict, None] = "balanced_subsample",
    knn_neighbors: int = 7,
    knn_weights: str = "distance",
    n_jobs: int = -1,
    out_json: Optional[Union[Path, str]] = None,
    out_json_knn: Optional[Union[Path, str]] = None,
    out_json_hgb: Optional[Union[Path, str]] = None,
    out_json_xgb: Optional[Union[Path, str]] = None,
    save_model_rf: Optional[Union[Path, str]] = None,
    save_model_knn: Optional[Union[Path, str]] = None,
    save_model_hgb: Optional[Union[Path, str]] = None,
    save_model_xgb: Optional[Union[Path, str]] = None,
    hgb_learning_rate: float = 0.1,
    hgb_max_iter: int = 100,
    hgb_max_depth: Optional[int] = None,
    hgb_max_leaf_nodes: int = 31,
    hgb_l2_regularization: float = 0.0,
    hgb_class_weight: Union[str, dict, None] = None,
    hgb_early_stopping: Union[str, bool] = "auto",
    xgb_n_estimators: int = 400,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.1,
    xgb_subsample: float = 0.9,
    xgb_colsample_bytree: float = 0.9,
    xgb_reg_lambda: float = 1.0,
    xgb_reg_alpha: float = 0.0,
    xgb_min_child_weight: float = 1.0,
    xgb_tree_method: str = "hist",
    imputer_strategy: str = "median",
    embed_validation_accuracy_in_save_path: bool = False,
    embed_grid_search_params_in_save_path: bool = True,
    grid_search: bool = False,
    grid_search_cv: int = 5,
    grid_search_scoring: str = "f1_macro",
    grid_search_verbose: int = 0,
    rf_param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    knn_param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    hgb_param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    xgb_param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    skip_if_xgb_unavailable: bool = False,
) -> Dict[str, Any]:
    """
    Programmatic example: pick ``classifier`` (``"rf"``, ``"knn"``, ``"hgb"``, ``"xgb"``, or ``"both"``), tune
    hyperparameters, and set ``save_model_*`` paths to persist fitted pipelines with joblib.

    Call from ``if __name__ == "__main__"`` or import and invoke from another script.
    Returns the same dict as :func:`main`.
    """
    return main(
        classifier=classifier,
        training_csvs=training_csvs,
        test_csvs=test_csvs,
        out_json=out_json,
        out_json_knn=out_json_knn,
        out_json_hgb=out_json_hgb,
        out_json_xgb=out_json_xgb,
        save_model_rf=save_model_rf,
        save_model_knn=save_model_knn,
        save_model_hgb=save_model_hgb,
        save_model_xgb=save_model_xgb,
        feature_mode=feature_mode,
        roll_window=roll_window,
        engineered_columns=engineered_columns,
        val_split=val_split,
        nan_imputer_cls=SimpleImputer,
        nan_imputer_kwargs={"strategy": imputer_strategy},
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        knn_neighbors=knn_neighbors,
        knn_weights=knn_weights,
        hgb_learning_rate=hgb_learning_rate,
        hgb_max_iter=hgb_max_iter,
        hgb_max_depth=hgb_max_depth,
        hgb_max_leaf_nodes=hgb_max_leaf_nodes,
        hgb_l2_regularization=hgb_l2_regularization,
        hgb_class_weight=hgb_class_weight,
        hgb_early_stopping=hgb_early_stopping,
        xgb_n_estimators=xgb_n_estimators,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        xgb_subsample=xgb_subsample,
        xgb_colsample_bytree=xgb_colsample_bytree,
        xgb_reg_lambda=xgb_reg_lambda,
        xgb_reg_alpha=xgb_reg_alpha,
        xgb_min_child_weight=xgb_min_child_weight,
        xgb_tree_method=xgb_tree_method,
        n_jobs=n_jobs,
        random_state=random_state,
        embed_validation_accuracy_in_save_path=embed_validation_accuracy_in_save_path,
        embed_grid_search_params_in_save_path=embed_grid_search_params_in_save_path,
        grid_search=grid_search,
        grid_search_cv=grid_search_cv,
        grid_search_scoring=grid_search_scoring,
        grid_search_verbose=grid_search_verbose,
        rf_param_grid=rf_param_grid,
        knn_param_grid=knn_param_grid,
        hgb_param_grid=hgb_param_grid,
        xgb_param_grid=xgb_param_grid,
        skip_if_xgb_unavailable=skip_if_xgb_unavailable,
    )


def _feature_names_for_X(
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[Sequence[str]],
) -> list[str]:
    if feature_names is not None:
        return list(feature_names)
    if isinstance(X, pd.DataFrame):
        return [str(c) for c in X.columns]
    n = int(np.asarray(X).shape[1])
    return [f"x{i}" for i in range(n)]


def evaluate_and_visualize_model(
    pipeline: Pipeline,
    X_eval: Union[pd.DataFrame, np.ndarray],
    y_eval: Union[np.ndarray, pd.Series],
    *,
    feature_names: Optional[Sequence[str]] = None,
    title_prefix: str = "Model",
    top_n_features: int = 20,
    permutation_n_repeats: int = 10,
    permutation_max_samples: Optional[int] = 800,
    random_state: int = 42,
    show: bool = True,
) -> Dict[str, Any]:
    """
    Score a **fitted** classifier ``Pipeline`` on labeled data, print metrics, and plot:

    - Confusion matrix (row-normalized by true label / recall).
    - Feature importance: impurity / gain importances for ``RandomForestClassifier``,
      ``HistGradientBoostingClassifier``, and ``xgboost.XGBClassifier``; ``permutation_importance``
      for ``KNeighborsClassifier``.

    ``X_eval`` / ``y_eval`` are typically a held-out validation set or labeled test CSV
    features built with the same ``feature_mode`` / ``roll_window`` as training.

    For k-NN on large matrices (e.g. waveform mode), ``permutation_max_samples`` subsamples
    rows to keep permutation importance tractable (``None`` uses all rows).

    Returns a dict with ``accuracy``, ``f1_macro``, ``qwk`` (quadratic weighted kappa),
    ``y_pred``, ``feature_names``, ``importance`` (aligned with ``feature_names``),
    ``importance_kind`` (``"forest_gini"`` or ``"permutation_mean"``), and ``figure``
    (matplotlib ``Figure``).
    If ``show`` is False, call ``result["figure"].savefig(...)`` or ``plt.close(result["figure"])`` when done.
    """
    import matplotlib.pyplot as plt

    clf = pipeline.named_steps.get("clf")
    if clf is None:
        raise ValueError("pipeline must have a 'clf' step (expected imputer [+ scaler] + classifier).")

    names = _feature_names_for_X(X_eval, feature_names)
    y_true = np.asarray(y_eval).ravel()
    y_pred = pipeline.predict(X_eval)
    acc = float(accuracy_score(y_true, y_pred))
    mae = float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    qwk = _quadratic_weighted_kappa(y_true, y_pred)

    print(f"=== {title_prefix} — evaluation ===")
    print(f"Accuracy: {acc:.4f}  |  MAE: {mae:.4f}  |  QWK: {qwk:.4f}")
    print(classification_report(y_true, y_pred, digits=3))

    labels = np.unique(np.concatenate([y_true, y_pred]))
    fig = plt.figure(figsize=(11.5, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15])
    ax_cm = fig.add_subplot(gs[0, 0])
    ax_imp = fig.add_subplot(gs[1, 0])

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, normalize="true", ax=ax_cm, colorbar=True
    )
    ax_cm.set_title(
        f"Confusion matrix — normalized by true label (recall)  "
        f"|  QWK={qwk:.3f}  acc={acc:.3f}  MAE={mae:.3f}"
    )

    importance_kind: str
    imp: np.ndarray

    if isinstance(clf, RandomForestClassifier):
        importance_kind = "forest_gini"
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif isinstance(clf, HistGradientBoostingClassifier):
        importance_kind = "histgb_impurity"
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif XGBClassifier is not None and isinstance(clf, XGBClassifier):
        importance_kind = "xgb_gain"
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif isinstance(clf, KNeighborsClassifier):
        importance_kind = "permutation_mean"
        X_perm = X_eval
        y_perm = y_true
        if permutation_max_samples is not None and len(y_perm) > permutation_max_samples:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(y_perm), size=permutation_max_samples, replace=False)
            if isinstance(X_perm, pd.DataFrame):
                X_perm = X_perm.iloc[idx]
            else:
                X_perm = np.asarray(X_perm)[idx]
            y_perm = y_perm[idx]
        perm = permutation_importance(
            pipeline,
            X_perm,
            y_perm,
            n_repeats=permutation_n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        imp = np.asarray(perm.importances_mean, dtype=float)
    else:
        raise TypeError(
            f"Unsupported classifier for importance: {type(clf).__name__} "
            "(expected RandomForestClassifier, HistGradientBoostingClassifier, XGBClassifier, or KNeighborsClassifier)."
        )

    if len(names) != imp.shape[0]:
        raise ValueError(
            f"feature_names length ({len(names)}) does not match importance length ({imp.shape[0]})."
        )

    order = np.argsort(imp)[::-1]
    k = min(top_n_features, len(order))
    top_idx = order[:k]
    top_names = [names[i] for i in top_idx]
    top_imp = imp[top_idx]

    y_pos = np.arange(k)
    ax_imp.barh(y_pos, top_imp[::-1], align="center")
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(top_names[::-1], fontsize=8)
    ax_imp.set_xlabel("Importance")
    ax_imp.set_title(
        f"Top {k} features ({importance_kind})"
        + (
            " — RF mean decrease impurity"
            if importance_kind == "forest_gini"
            else " — HGB impurity-based importance"
            if importance_kind == "histgb_impurity"
            else " — XGBoost gain-based importance"
            if importance_kind == "xgb_gain"
            else " — permutation Δ accuracy"
        )
    )
    ax_imp.invert_yaxis()

    fig.suptitle(
        f"{title_prefix}  |  QWK={qwk:.4f}  acc={acc:.4f}  MAE={mae:.4f}",
        fontsize=12,
    )

    if show:
        plt.show()

    return {
        "accuracy": acc,
        "mae": mae,
        "f1_macro": f1_macro,
        "qwk": qwk,
        "y_pred": y_pred,
        "feature_names": names,
        "importance": imp,
        "importance_kind": importance_kind,
        "figure": fig,
    }


def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted κ for ordinal class indices (e.g. decade labels 0..K-1)."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0:
        return 0.0
    k = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    if k is None or (isinstance(k, float) and np.isnan(k)):
        return 0.0
    return float(k)


def plot_validation_normalized_confusion_matrices(
    y_true: Union[np.ndarray, pd.Series],
    model_predictions: Sequence[Tuple[str, Union[np.ndarray, pd.Series]]],
    *,
    reference_description: str = "validation holdout (train-fold model)",
    show: bool = True,
) -> Any:
    """
    Plot **row-normalized** (recall) confusion matrices for several models on the same label set.

    Prints accuracy, MAE, quadratic weighted kappa (QWK), and a classification report per model.
    Class labels are the union of values in ``y_true`` and all prediction vectors.
    """
    import matplotlib.pyplot as plt

    if not model_predictions:
        raise ValueError("model_predictions must contain at least one (name, y_pred) pair.")

    y_true_arr = np.asarray(y_true).ravel()
    label_pieces = [y_true_arr]
    named_preds: List[Tuple[str, np.ndarray]] = []
    for name, y_pred in model_predictions:
        y_pred_arr = np.asarray(y_pred).ravel()
        if len(y_pred_arr) != len(y_true_arr):
            raise ValueError(
                f"Model {name!r}: len(y_pred)={len(y_pred_arr)} != len(y_true)={len(y_true_arr)}."
            )
        label_pieces.append(y_pred_arr)
        named_preds.append((name, y_pred_arr))

    labels = np.unique(np.concatenate(label_pieces))
    n = len(named_preds)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5.5 * n, 5.6),
        constrained_layout=True,
        squeeze=False,
    )

    qwk_list: list[float] = []
    for j, (name, y_pred_arr) in enumerate(named_preds):
        ax = axes[0, j]
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        mae = float(np.mean(np.abs(np.asarray(y_true_arr, dtype=float) - np.asarray(y_pred_arr, dtype=float))))
        f1m = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        qwk = _quadratic_weighted_kappa(y_true_arr, y_pred_arr)
        qwk_list.append(qwk)
        print(f"=== {name} — {reference_description} ===")
        print(f"Accuracy: {acc:.4f}  |  MAE: {mae:.4f}  |  QWK: {qwk:.4f}")
        print(classification_report(y_true_arr, y_pred_arr, digits=3))
        ConfusionMatrixDisplay.from_predictions(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            normalize="true",
            ax=ax,
            colorbar=True,
        )
        ax.set_title(f"{name}\nQWK={qwk:.3f}  acc={acc:.3f}  MAE={mae:.3f}")

    if len(qwk_list) == 1:
        qwk_hdr = f"QWK={qwk_list[0]:.3f}"
    else:
        qwk_hdr = f"mean QWK={float(np.mean(qwk_list)):.3f}"
    fig.suptitle(
        f"Confusion matrices (normalized by true label / recall) — {reference_description}  "
        f"|  {qwk_hdr}",
        fontsize=12,
    )
    if show:
        plt.show()

    return fig


def plot_confusion_matrices_row_panels(
    panels: Sequence[Tuple[str, Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series]]],
    *,
    reference_description: str = "validation / evaluation",
    print_reports: bool = True,
    fig_width_per_panel: float = 5.5,
    show: bool = True,
) -> Any:
    """
    One row of **row-normalized** (recall) confusion matrices for side-by-side comparison.

    All subplots share the same ``labels`` ordering (union of classes appearing in any panel)
    so axes are aligned. Each panel may use a different ``y_true`` slice (e.g. tree holdout vs
    CNN OOF) — interpret titles accordingly.

    When ``print_reports`` is True, prints accuracy, MAE, QWK, and ``classification_report``
    per panel.
    """
    import matplotlib.pyplot as plt

    if not panels:
        raise ValueError("panels must contain at least one (name, y_true, y_pred) tuple.")

    decade_names = ["20s", "30s", "40s", "50s", "60s", "70s"]
    resolved: List[Tuple[str, np.ndarray, np.ndarray]] = []
    label_pieces: List[np.ndarray] = []
    for name, yt, yp in panels:
        y_true_arr = np.asarray(yt).ravel()
        y_pred_arr = np.asarray(yp).ravel()
        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(
                f"Panel {name!r}: len(y_true)={len(y_true_arr)} != len(y_pred)={len(y_pred_arr)}."
            )
        resolved.append((name, y_true_arr, y_pred_arr))
        label_pieces.append(y_true_arr)
        label_pieces.append(y_pred_arr)

    labels = np.unique(np.concatenate(label_pieces))
    try:
        display_labels = [decade_names[int(c)] for c in labels]
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Class labels must be integer decade indices 0..5; got {labels!r}"
        ) from e

    n = len(resolved)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(fig_width_per_panel * n, 5.75),
        constrained_layout=True,
        squeeze=False,
    )

    qwk_list: list[float] = []
    for j, (name, y_true_arr, y_pred_arr) in enumerate(resolved):
        ax = axes[0, j]
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        mae = float(np.mean(np.abs(np.asarray(y_true_arr, dtype=float) - np.asarray(y_pred_arr, dtype=float))))
        f1m = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        qwk = _quadratic_weighted_kappa(y_true_arr, y_pred_arr)
        qwk_list.append(qwk)
        if print_reports:
            print(f"=== {name} — {reference_description} ===")
            print(f"Accuracy: {acc:.4f}  |  MAE: {mae:.4f}  |  QWK: {qwk:.4f}")
            print(classification_report(y_true_arr, y_pred_arr, digits=3))
        ConfusionMatrixDisplay.from_predictions(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            display_labels=display_labels,
            normalize="true",
            ax=ax,
            colorbar=True,
        )
        ax.set_title(f"{name}\nQWK={qwk:.3f}  acc={acc:.3f}  MAE={mae:.3f}")

    if len(qwk_list) == 1:
        qwk_hdr = f"QWK={qwk_list[0]:.3f}"
    else:
        qwk_hdr = f"mean QWK={float(np.mean(qwk_list)):.3f}"
    fig.suptitle(
        f"Confusion matrices (normalized by true label / recall) — {reference_description}  |  {qwk_hdr}",
        fontsize=11,
    )
    if show:
        plt.show()

    return fig


def visualize_predictions_on_test(
    pipeline: Pipeline,
    X_test: Union[pd.DataFrame, np.ndarray],
    *,
    feature_names: Optional[Sequence[str]] = None,
    title_prefix: str = "Test (unlabeled)",
    top_n_features: int = 20,
    X_labeled_reference: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_labeled_reference: Optional[Union[np.ndarray, pd.Series]] = None,
    y_true_reference: Optional[Union[np.ndarray, pd.Series]] = None,
    y_pred_reference: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_description: str = "validation (from split)",
    show: bool = True,
) -> Dict[str, Any]:
    """
    For **unlabeled** test features (e.g. from :func:`load_test_pair`): predict, plot class counts,
    and show tree-model importances from the fitted ``pipeline`` (RF: Gini; HGB / XGBoost: gain;
    k-NN: message only).

    Optional **confusion matrix** on labeled reference data (test CSVs still have no labels):

    - When shown, only the **row-normalized** matrix (recall by true class) is plotted.
    - Prefer ``y_true_reference`` and ``y_pred_reference`` together (e.g. holdout labels and
      predictions from a model fit on the **train fold only**, matching the stratified split
      used in :func:`main`).
    - Or pass ``X_labeled_reference`` and ``y_labeled_reference``; predictions are then
      ``pipeline.predict(X_labeled_reference)`` (in-sample for a full-data refit).

    Do not pass both precomputed ``y_*_reference`` and ``X_labeled_reference`` / ``y_labeled_reference``.
    For several models on the same holdout labels, use :func:`plot_validation_normalized_confusion_matrices`.

    k-NN permutation importance is not shown here; the importance panel notes that for k-NN.
    """
    import matplotlib.pyplot as plt

    clf = pipeline.named_steps.get("clf")
    if clf is None:
        raise ValueError("pipeline must have a 'clf' step (expected imputer [+ scaler] + classifier).")

    names = _feature_names_for_X(X_test, feature_names)
    y_pred = np.asarray(pipeline.predict(X_test)).ravel()
    classes, counts = np.unique(y_pred, return_counts=True)
    pred_counts = {int(c): int(n) for c, n in zip(classes, counts)}

    ref_acc: Optional[float] = None
    ref_mae: Optional[float] = None
    ref_f1: Optional[float] = None
    ref_qwk: Optional[float] = None
    y_pred_ref: Optional[np.ndarray] = None
    y_true_ref: Optional[np.ndarray] = None

    if (X_labeled_reference is None) ^ (y_labeled_reference is None):
        raise ValueError("Pass both X_labeled_reference and y_labeled_reference, or neither.")
    if (y_true_reference is None) ^ (y_pred_reference is None):
        raise ValueError("Pass both y_true_reference and y_pred_reference, or neither.")
    has_xy_pair = X_labeled_reference is not None and y_labeled_reference is not None
    has_pre_pair = y_true_reference is not None and y_pred_reference is not None
    if has_pre_pair and has_xy_pair:
        raise ValueError(
            "Use either (y_true_reference, y_pred_reference) or "
            "(X_labeled_reference, y_labeled_reference), not both."
        )

    print(f"=== {title_prefix} ===")
    print(
        "Test rows have no ground-truth labels — top row shows predicted distribution on test."
    )
    print(f"Predicted class counts (test): {pred_counts}")
    show_cm = False
    if y_true_reference is not None and y_pred_reference is not None:
        y_true_ref = np.asarray(y_true_reference).ravel()
        y_pred_ref = np.asarray(y_pred_reference).ravel()
        show_cm = True
    elif X_labeled_reference is not None and y_labeled_reference is not None:
        y_true_ref = np.asarray(y_labeled_reference).ravel()
        y_pred_ref = np.asarray(pipeline.predict(X_labeled_reference)).ravel()
        show_cm = True

    if show_cm and y_true_ref is not None and y_pred_ref is not None:
        ref_acc = float(accuracy_score(y_true_ref, y_pred_ref))
        ref_mae = float(np.mean(np.abs(np.asarray(y_true_ref, dtype=float) - np.asarray(y_pred_ref, dtype=float))))
        ref_f1 = float(f1_score(y_true_ref, y_pred_ref, average="macro", zero_division=0))
        ref_qwk = _quadratic_weighted_kappa(y_true_ref, y_pred_ref)
        print(
            f"Labeled reference ({reference_description}): accuracy={ref_acc:.4f}  "
            f"MAE={ref_mae:.4f}  QWK={ref_qwk:.4f}"
        )
        print(classification_report(y_true_ref, y_pred_ref, digits=3))

    fig = plt.figure(figsize=(11, 12 if show_cm else 9), constrained_layout=True)
    if show_cm:
        gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.0, 1.1])
    else:
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax_counts = fig.add_subplot(gs[0, 0])
    ax_frac = fig.add_subplot(gs[0, 1])
    row_imp = 2 if show_cm else 1
    ax_imp = fig.add_subplot(gs[row_imp, :])

    if show_cm and y_true_ref is not None and y_pred_ref is not None:
        cm_labels = np.unique(np.concatenate([y_true_ref, y_pred_ref]))
        ax_cm = fig.add_subplot(gs[1, :])
        ConfusionMatrixDisplay.from_predictions(
            y_true_ref,
            y_pred_ref,
            labels=cm_labels,
            normalize="true",
            ax=ax_cm,
            colorbar=True,
        )
        ax_cm.set_title(
            f"Confusion matrix — {reference_description} (normalized by true label / recall)  "
            f"|  QWK={ref_qwk:.3f}  acc={ref_acc:.3f}  MAE={ref_mae:.3f}"
        )

    ax_counts.bar([str(int(c)) for c in classes], counts, color="steelblue")
    ax_counts.set_xlabel("Predicted class")
    ax_counts.set_ylabel("Count")
    ax_counts.set_title("Predicted class counts (test)")

    frac = counts.astype(float) / max(float(counts.sum()), 1.0)
    ax_frac.bar([str(int(c)) for c in classes], frac, color="coral")
    ax_frac.set_xlabel("Predicted class")
    ax_frac.set_ylabel("Fraction")
    ax_frac.set_title("Predicted class fraction (test)")
    ax_frac.set_ylim(0, 1)

    importance_kind: str
    imp: np.ndarray

    if isinstance(clf, RandomForestClassifier):
        importance_kind = "forest_gini"
        imp = np.asarray(clf.feature_importances_, dtype=float)
        if len(names) != imp.shape[0]:
            raise ValueError(
                f"feature_names length ({len(names)}) does not match importance length ({imp.shape[0]})."
            )
        order = np.argsort(imp)[::-1]
        k = min(top_n_features, len(order))
        top_idx = order[:k]
        top_names = [names[i] for i in top_idx]
        top_imp = imp[top_idx]
        y_pos = np.arange(k)
        ax_imp.barh(y_pos, top_imp[::-1], align="center")
        ax_imp.set_yticks(y_pos)
        ax_imp.set_yticklabels(top_names[::-1], fontsize=8)
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title(f"Top {k} features (RF mean decrease impurity, model-level)")
        ax_imp.invert_yaxis()
    elif isinstance(clf, HistGradientBoostingClassifier):
        importance_kind = "histgb_impurity"
        imp = np.asarray(clf.feature_importances_, dtype=float)
        if len(names) != imp.shape[0]:
            raise ValueError(
                f"feature_names length ({len(names)}) does not match importance length ({imp.shape[0]})."
            )
        order = np.argsort(imp)[::-1]
        k = min(top_n_features, len(order))
        top_idx = order[:k]
        top_names = [names[i] for i in top_idx]
        top_imp = imp[top_idx]
        y_pos = np.arange(k)
        ax_imp.barh(y_pos, top_imp[::-1], align="center")
        ax_imp.set_yticks(y_pos)
        ax_imp.set_yticklabels(top_names[::-1], fontsize=8)
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title(f"Top {k} features (HGB impurity importance, model-level)")
        ax_imp.invert_yaxis()
    elif XGBClassifier is not None and isinstance(clf, XGBClassifier):
        importance_kind = "xgb_gain"
        imp = np.asarray(clf.feature_importances_, dtype=float)
        if len(names) != imp.shape[0]:
            raise ValueError(
                f"feature_names length ({len(names)}) does not match importance length ({imp.shape[0]})."
            )
        order = np.argsort(imp)[::-1]
        k = min(top_n_features, len(order))
        top_idx = order[:k]
        top_names = [names[i] for i in top_idx]
        top_imp = imp[top_idx]
        y_pos = np.arange(k)
        ax_imp.barh(y_pos, top_imp[::-1], align="center")
        ax_imp.set_yticks(y_pos)
        ax_imp.set_yticklabels(top_names[::-1], fontsize=8)
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title(f"Top {k} features (XGBoost gain importance, model-level)")
        ax_imp.invert_yaxis()
    elif isinstance(clf, KNeighborsClassifier):
        importance_kind = "unavailable_knn_no_labels"
        imp = np.array([])
        ax_imp.axis("off")
        ax_imp.text(
            0.5,
            0.5,
            "k-NN permutation importance needs labeled rows.\n"
            "Use labeled data with evaluate_and_visualize_model,\n"
            "or inspect prediction distributions above.",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax_imp.transAxes,
        )
    else:
        raise TypeError(
            f"Unsupported classifier: {type(clf).__name__} "
            "(expected RandomForestClassifier, HistGradientBoostingClassifier, XGBClassifier, or KNeighborsClassifier)."
        )

    n_test = int(len(y_pred))
    sub = f"n_test={n_test}"
    if ref_acc is not None and ref_mae is not None and ref_qwk is not None:
        sub += (
            f"  |  ref QWK={ref_qwk:.4f}  acc={ref_acc:.4f}  MAE={ref_mae:.4f} "
            f"({reference_description})"
        )
    fig.suptitle(f"{title_prefix}  |  {sub}", fontsize=12)

    if show:
        plt.show()

    return {
        "accuracy": None,
        "f1_macro": None,
        "reference_accuracy": ref_acc,
        "reference_mae": ref_mae,
        "reference_f1_macro": ref_f1,
        "reference_qwk": ref_qwk,
        "y_pred": y_pred,
        "y_pred_reference": y_pred_ref,
        "pred_counts": pred_counts,
        "feature_names": names,
        "importance": imp if imp.size else None,
        "importance_kind": importance_kind,
        "figure": fig,
    }


def _parse_engineered_columns_arg(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated engineered feature names; empty/whitespace → ``None`` (use all)."""
    if value is None or not str(value).strip():
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return parts or None


def _cli() -> None:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train Random Forest, k-NN, HistGradientBoosting, and/or XGBoost age-group classifiers on paired waveform CSVs.",
    )
    parser.add_argument(
        "--train-aorta",
        type=Path,
        default=default_root / "datasets/train/aortaP_train_data.csv",
        help="Training aorta CSV (filename must contain '_train')",
    )
    parser.add_argument(
        "--train-brach",
        type=Path,
        default=default_root / "datasets/train/brachP_train_data.csv",
        help="Training brachial CSV (filename must contain '_train')",
    )
    parser.add_argument(
        "--test-aorta",
        type=Path,
        default=default_root / "datasets/test/aortaP_test_data.csv",
        help="Test aorta CSV (optional)",
    )
    parser.add_argument(
        "--test-brach",
        type=Path,
        default=default_root / "datasets/test/brachP_test_data.csv",
        help="Test brachial CSV (optional)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=default_root / "predictions.json",
        help="Where to write Random Forest {subject_index: class} predictions",
    )
    parser.add_argument(
        "--out-json-knn",
        type=Path,
        default=default_root / "predictions_knn.json",
        help="Where to write k-NN {subject_index: class} predictions",
    )
    parser.add_argument(
        "--out-json-hgb",
        type=Path,
        default=default_root / "predictions_hgb.json",
        help="Where to write HistGradientBoosting {subject_index: class} predictions",
    )
    parser.add_argument(
        "--out-json-xgb",
        type=Path,
        default=default_root / "predictions_xgb.json",
        help="Where to write XGBoost {subject_index: class} predictions",
    )
    parser.add_argument(
        "--classifier",
        choices=("rf", "knn", "hgb", "xgb", "both"),
        default="both",
        help="Train RF only, k-NN only, HGB only, XGB only, or both (RF + k-NN)",
    )
    parser.add_argument(
        "--skip-if-xgb-unavailable",
        action="store_true",
        help="With --classifier xgb, warn and exit successfully if XGBoost cannot load (e.g. OpenMP mismatch)",
    )
    parser.add_argument(
        "--save-model-rf",
        type=Path,
        default=None,
        help="Optional path to save fitted RF pipeline (joblib)",
    )
    parser.add_argument(
        "--save-model-knn",
        type=Path,
        default=None,
        help="Optional path to save fitted k-NN pipeline (joblib)",
    )
    parser.add_argument(
        "--save-model-hgb",
        type=Path,
        default=None,
        help="Optional path to save fitted HistGradientBoosting pipeline (joblib)",
    )
    parser.add_argument(
        "--save-model-xgb",
        type=Path,
        default=None,
        help="Optional path to save fitted XGBoost pipeline (joblib)",
    )
    parser.add_argument(
        "--features",
        choices=("engineered", "waveform", "waveform_plus"),
        default="engineered",
        help=(
            "Feature set: engineered (scalar summaries), raw 672 waveforms, or waveform_plus "
            "(672 + one 336-trace per engineered column; see --engineered-columns)"
        ),
    )
    parser.add_argument(
        "--roll-window",
        type=int,
        default=ROLL_WINDOW_DEFAULT,
        help="Rolling window for ARV/CV/slope (engineered and waveform_plus rolling traces)",
    )
    parser.add_argument(
        "--engineered-columns",
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated names from ENGINEERED_FEATURE_NAMES (includes aorta_raw, brach_raw, "
            "aorta_preproc, …). With --features engineered: scalar summaries only, omit = all scalars. "
            "With --features waveform_plus: which 336-sample traces to stack "
            "(omit = default raw+Cheb traces plus scalars, see ENGINEERED_WAVEFORM_TRACE_NAMES_DEFAULT)."
        ),
    )
    parser.add_argument("--knn-neighbors", type=int, default=7, help="k for KNeighborsClassifier")
    parser.add_argument(
        "--knn-weights",
        default="distance",
        choices=("uniform", "distance"),
        help="KNeighborsClassifier weights",
    )
    parser.add_argument("--eval-split", type=float, default=0.2, help="Holdout fraction for metrics.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for split + forest (default: random integer)",
    )
    parser.add_argument(
        "--imputer-strategy",
        default="median",
        choices=("mean", "median", "most_frequent", "constant"),
        help="Passed to SimpleImputer when using default nan_imputer_cls",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument(
        "--class-weight",
        default="balanced_subsample",
        help='RandomForest class_weight (e.g. balanced_subsample, balanced, None)',
    )
    parser.add_argument("--hgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--hgb-max-iter", type=int, default=100)
    parser.add_argument(
        "--hgb-max-depth",
        type=int,
        default=None,
        help="Tree depth for HistGradientBoostingClassifier (default: unlimited)",
    )
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=0.0, help="L2 regularization for HGB")
    parser.add_argument(
        "--hgb-class-weight",
        default=None,
        help="HistGradientBoosting class_weight (e.g. balanced, None)",
    )
    parser.add_argument(
        "--hgb-early-stopping",
        default="auto",
        help='HGB early_stopping: auto, True, False (default: auto)',
    )
    parser.add_argument("--xgb-n-estimators", type=int, default=400, help="XGBoost n_estimators")
    parser.add_argument("--xgb-max-depth", type=int, default=6, help="XGBoost max_depth")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1, help="XGBoost learning_rate")
    parser.add_argument("--xgb-subsample", type=float, default=0.9, help="XGBoost subsample")
    parser.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=0.9,
        help="XGBoost colsample_bytree",
    )
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0, help="XGBoost reg_lambda")
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0, help="XGBoost reg_alpha")
    parser.add_argument(
        "--xgb-min-child-weight",
        type=float,
        default=1.0,
        help="XGBoost min_child_weight",
    )
    parser.add_argument(
        "--xgb-tree-method",
        default="hist",
        help="XGBoost tree_method (e.g. hist, approx, auto)",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--acc-in-model-name",
        action="store_true",
        help="Append _acc{tenths} (tenths of %% accuracy) to save-model paths (see main).",
    )
    parser.add_argument(
        "--omit-grid-params-from-save-path",
        action="store_true",
        help=(
            "Do not append GridSearchCV best_params to joblib filenames "
            "(default: append hp_* token when --grid-search is used)."
        ),
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run GridSearchCV on the training split (StratifiedKFold); see --grid-search-cv.",
    )
    parser.add_argument(
        "--grid-search-cv",
        type=int,
        default=5,
        help="Number of stratified folds for GridSearchCV (>= 2).",
    )
    parser.add_argument(
        "--grid-search-scoring",
        default="f1_macro",
        help="Scikit-learn scoring string for GridSearchCV (default: f1_macro).",
    )
    parser.add_argument(
        "--grid-search-verbose",
        type=int,
        default=0,
        help="Verbosity passed to GridSearchCV (0 = quiet).",
    )
    parser.add_argument(
        "--grid-params-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file: object with keys rf, knn, hgb, xgb mapping to param grids "
            "(clf__* keys → lists). Omitted keys use built-in defaults when --grid-search is set."
        ),
    )
    args = parser.parse_args()

    cw: Union[str, dict, None]
    if args.class_weight.lower() in ("none", "null"):
        cw = None
    else:
        cw = args.class_weight

    hgb_cw: Union[str, dict, None]
    if args.hgb_class_weight is None or str(args.hgb_class_weight).lower() in ("none", "null"):
        hgb_cw = None
    else:
        hgb_cw = args.hgb_class_weight

    es: Union[str, bool]
    es_arg = args.hgb_early_stopping
    if isinstance(es_arg, str):
        el = es_arg.lower()
        if el in ("true", "1", "yes"):
            es = True
        elif el in ("false", "0", "no"):
            es = False
        else:
            es = es_arg
    else:
        es = bool(es_arg)

    rf_grid: Optional[Dict[str, Sequence[Any]]] = None
    knn_grid: Optional[Dict[str, Sequence[Any]]] = None
    hgb_grid: Optional[Dict[str, Sequence[Any]]] = None
    xgb_grid: Optional[Dict[str, Sequence[Any]]] = None
    if args.grid_params_json is not None:
        loaded = load_param_grids_from_json(args.grid_params_json)
        rf_grid = loaded.get("rf")  # type: ignore[assignment]
        knn_grid = loaded.get("knn")  # type: ignore[assignment]
        hgb_grid = loaded.get("hgb")  # type: ignore[assignment]
        xgb_grid = loaded.get("xgb")  # type: ignore[assignment]

    engineered_cols = _parse_engineered_columns_arg(args.engineered_columns)
    if engineered_cols is not None and args.features not in ("engineered", "waveform_plus"):
        parser.error("--engineered-columns is only valid with --features engineered or waveform_plus")

    main(
        classifier=args.classifier,
        training_csvs=(args.train_aorta, args.train_brach),
        test_csvs=(args.test_aorta, args.test_brach),
        out_json=args.out_json,
        out_json_knn=args.out_json_knn,
        out_json_hgb=args.out_json_hgb,
        out_json_xgb=args.out_json_xgb,
        save_model_rf=args.save_model_rf,
        save_model_knn=args.save_model_knn,
        save_model_hgb=args.save_model_hgb,
        save_model_xgb=args.save_model_xgb,
        feature_mode=args.features,
        roll_window=args.roll_window,
        engineered_columns=engineered_cols,
        val_split=args.eval_split,
        nan_imputer_cls=SimpleImputer,
        nan_imputer_kwargs={"strategy": args.imputer_strategy},
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight=cw,
        knn_neighbors=args.knn_neighbors,
        knn_weights=args.knn_weights,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_max_iter=args.hgb_max_iter,
        hgb_max_depth=args.hgb_max_depth,
        hgb_max_leaf_nodes=args.hgb_max_leaf_nodes,
        hgb_l2_regularization=args.hgb_l2,
        hgb_class_weight=hgb_cw,
        hgb_early_stopping=es,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_reg_lambda=args.xgb_reg_lambda,
        xgb_reg_alpha=args.xgb_reg_alpha,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_tree_method=args.xgb_tree_method,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        embed_validation_accuracy_in_save_path=args.acc_in_model_name,
        embed_grid_search_params_in_save_path=not args.omit_grid_params_from_save_path,
        grid_search=args.grid_search,
        grid_search_cv=args.grid_search_cv,
        grid_search_scoring=args.grid_search_scoring,
        grid_search_verbose=args.grid_search_verbose,
        rf_param_grid=rf_grid,
        knn_param_grid=knn_grid,
        hgb_param_grid=hgb_grid,
        xgb_param_grid=xgb_grid,
        skip_if_xgb_unavailable=args.skip_if_xgb_unavailable,
    )


if __name__ == "__main__":
    # Example: set USE_EXAMPLE = False to use argparse CLI instead.
    USE_EXAMPLE = False
    _root = Path(__file__).resolve().parent

    if USE_EXAMPLE:
        run_example(
            classifier="rf",  # "rf" | "knn" | "hgb" | "xgb" | "both"
            feature_mode="engineered",
            roll_window=ROLL_WINDOW_DEFAULT,
            val_split=0.2,
            random_state=42,
            n_estimators=200,
            max_depth=None,
            class_weight="balanced_subsample",
            knn_neighbors=7,
            knn_weights="distance",
            n_jobs=-1,
            out_json=_root / "predictions.json",
            out_json_knn=_root / "predictions_knn.json",
            save_model_rf=_root / "models" / "heart_age_rf.joblib",
            save_model_knn=_root / "models" / "heart_age_knn.joblib",
        )
    else:
        _cli()
