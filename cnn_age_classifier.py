"""CNN age-group classifier utilities (PyTorch).

This module contains reusable pieces for the two-channel ResNet-1D model used in
`create_age_classifier.ipynb`:
- Loading paired aorta/brachial waveform CSVs into ``(n, n_time, 2)``
- **Phase-1 preprocessing** (optional): optional long-gap exclusion (only when ``max_gap_samples`` /
  ``max_gap_ms`` is set), **linear** gap fill
  (interpolate between adjacent finite samples; hold endpoints past edges), per-subject Z-score
- Linear gap fill (``preprocess=\"linear\"`` load path), min-max / z-score helpers
- ResNet-1D model builder

Apple Silicon:
- This model can run on CPU or MPS (Metal) when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required for the CNN model. Install with: pip install torch") from e


def _one_hot_int(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D class vector; got shape {y.shape}")
    if np.any((y < 0) | (y >= n_classes)):
        bad = y[(y < 0) | (y >= n_classes)]
        raise ValueError(f"Class index outside 0..{n_classes-1}: {bad[:10]}")
    return np.eye(n_classes, dtype=np.float32)[y]


def _impute_time_series_1d(y: np.ndarray, *, fill_all_nan: float = 0.0) -> np.ndarray:
    """Linearly interpolate non-finite samples along a single 1-D trace.

    Leading/trailing gaps use the nearest valid sample (constant extrapolation).
    If there are no finite values, returns a constant trace ``fill_all_nan``.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    finite = np.isfinite(y)
    if finite.all():
        return y.astype(np.float32)
    if not finite.any():
        return np.full(n, float(fill_all_nan), dtype=np.float32)

    idx = np.arange(n, dtype=np.float64)
    xv = idx[finite]
    yv = y[finite]
    out = np.interp(idx, xv, yv, left=float(yv[0]), right=float(yv[-1]))
    return out.astype(np.float32)


def impute_waveform_gaps(
    X: np.ndarray,
    *,
    fill_all_nan: float = 0.0,
) -> np.ndarray:
    """Fill gaps (non-finite entries) in ``(n_subjects, n_time, n_channels)`` waveforms.

    **Strategy:** independent **linear interpolation** along time for each subject and
    each channel (aorta and brachial are not coupled). Matches periodic missing samples
    better than a global constant after normalization: impute **before** ``minmax`` /
    ``zscore`` so scaling uses real dynamics.

    Parameters
    ----------
    fill_all_nan
        Value used when an entire channel is missing (rare after CSV merge).
    """
    # NumPy 1.x: ``asarray(..., copy=True)`` is invalid; ``array(..., copy=True)`` works.
    X = np.array(X, dtype=np.float32, copy=True)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (n, time, channels); got {X.shape}")
    n_sub, n_t, n_ch = X.shape
    for i in range(n_sub):
        for c in range(n_ch):
            X[i, :, c] = _impute_time_series_1d(X[i, :, c], fill_all_nan=fill_all_nan)
    return X


def _max_contiguous_nonfinite_run(y: np.ndarray) -> int:
    """Longest run of non-finite values along a 1-D trace."""
    y = np.asarray(y).ravel()
    best = cur = 0
    for v in y:
        if not np.isfinite(v):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _resolve_max_gap_samples(
    max_gap_samples: int | None,
    max_gap_ms: float | None,
    sample_rate_hz: float,
) -> int | None:
    """Resolve contiguous gap threshold in samples.

    - If ``max_gap_ms`` is set, it wins (rounded to nearest sample count, at least 1).
    - Else if ``max_gap_samples`` is ``None``, returns ``None`` (**no** gap-based drops).
    - Else returns ``max(1, int(max_gap_samples))``.
    """
    if max_gap_ms is not None:
        return max(1, int(round(float(max_gap_ms) * float(sample_rate_hz) / 1000.0)))
    if max_gap_samples is None:
        return None
    return max(1, int(max_gap_samples))


def _impute_small_gaps_1d(y: np.ndarray, max_gap_samples: int) -> np.ndarray:
    """Fill non-finite samples by **linear interpolation** between neighboring finite points.

    Delegates to :func:`_impute_time_series_1d` (``numpy.interp``): internal gaps are
    straight segments between the values at the connecting indices; leading/trailing holes
    use the nearest finite sample (constant extrapolation).

    When :func:`preprocess_waveforms_phase1` uses a finite gap threshold, subjects with longer
    contiguous non-finite runs are dropped **before** this runs. ``max_gap_samples`` is unused
    here but kept for API stability with existing call sites.
    """
    _ = max_gap_samples
    return _impute_time_series_1d(y, fill_all_nan=0.0)


def preprocess_waveforms_phase1(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_gap_samples: int | None = None,
    zscore_eps: float = 1e-6,
    zscore_mode: Literal["independent", "aorta_reference"] = "aorta_reference",
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Phase-1 CNN preprocessing: optional gap threshold → linear gap imputation → PP tabular → z-score.

    1. **Drop** (only if ``max_gap_samples`` is not ``None``): remove any subject if **either**
       channel has a contiguous non-finite run strictly longer than ``max_gap_samples``.
       Default ``None`` keeps **all** subjects; only linear imputation is applied.
    2. **Impute** non-finite samples on each channel with **linear interpolation** in time
       between adjacent finite samples (see :func:`_impute_small_gaps_1d`).
    3. **Tabular:** :func:`extract_pulse_pressure_tabular` on the **imputed, raw-scale**
       waveforms (``max - min`` per channel). Passed to the CNN head alongside conv/xcorr.
    4. **Normalize:** ``zscore_mode="aorta_reference"`` (default) uses aortic μ,σ for
       **both** channels; ``"independent"`` uses :func:`zscore_normalize_per_subject`.

    Returns
    -------
    X_out, y_out, n_dropped, pp_tab
        ``pp_tab`` shape ``(n, 2)``: **[aorta_pp, brach_pp]** in raw units (pre z-score).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X (n, time, channels); got {X.shape}")
    n_sub = X.shape[0]
    keep = np.ones(n_sub, dtype=bool)
    if max_gap_samples is not None:
        for i in range(n_sub):
            for c in range(X.shape[2]):
                if _max_contiguous_nonfinite_run(X[i, :, c]) > max_gap_samples:
                    keep[i] = False
                    break
    n_dropped = int(np.sum(~keep))
    if n_sub > 0 and n_dropped == n_sub:
        raise ValueError(
            f"All {n_sub} subjects removed: every row has a gap longer than "
            f"{max_gap_samples} samples on at least one channel."
        )
    X = X[keep].copy()
    y = np.asarray(y)[keep].copy()

    n_sub, _, n_ch = X.shape
    impute_arg = max_gap_samples if max_gap_samples is not None else 0
    for i in range(n_sub):
        for c in range(n_ch):
            if max_gap_samples is not None:
                if _max_contiguous_nonfinite_run(X[i, :, c]) > max_gap_samples:
                    raise RuntimeError("internal: long gap after filter")
            X[i, :, c] = _impute_small_gaps_1d(X[i, :, c], impute_arg)

    pp_tab = extract_pulse_pressure_tabular(X)
    if zscore_mode == "aorta_reference":
        X = zscore_normalize_aorta_referenced(X, eps=zscore_eps)
    elif zscore_mode == "independent":
        X = zscore_normalize_per_subject(X, eps=zscore_eps)
    else:
        raise ValueError(
            f"zscore_mode must be 'aorta_reference' or 'independent'; got {zscore_mode!r}"
        )
    return X, y, n_dropped, pp_tab


def preprocess_cnn_phase1_traces_rowwise(
    Wa: np.ndarray,
    Wb: np.ndarray,
    *,
    max_gap_samples: int | None = None,
    max_gap_ms: float | None = None,
    sample_rate_hz: float = 500.0,
    zscore_eps: float = 1e-6,
    zscore_mode: Literal["independent", "aorta_reference"] = "aorta_reference",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply **phase-1 CNN** gap rules, linear gap imputation, and z-score **per tabular row**.

    For each row (subject), builds a temporary ``(1, n_time, 2)`` stack **[aorta, brach]** and
    applies the same steps as :func:`preprocess_waveforms_phase1` **without** cross-row batching.
    When a finite gap threshold is resolved, **per-row NaNs**: if either channel has a contiguous
    non-finite run strictly longer than that threshold, that row's output traces are all **NaN**
    (matching subjects the CNN batch loader would exclude, while keeping row alignment with merged
    training tables for sklearn imputers). With default ``max_gap_samples=None`` and
    ``max_gap_ms=None``, **no** rows are blanked for gap length.

    Parameters
    ----------
    Wa, Wb
        Shape ``(n_rows, n_time)``, same layout as ``heart_age_classifier`` waveform matrices.
    max_gap_samples, max_gap_ms, sample_rate_hz
        Passed to :func:`_resolve_max_gap_samples` (``max_gap_ms`` overrides sample count when set).
    zscore_eps, zscore_mode
        Match :func:`preprocess_waveforms_phase1` / CNN training defaults.

    Returns
    -------
    Wa_out, Wb_out
        ``float64`` arrays, shape matching inputs; invalid rows are all-NaN.
    """
    Wa = np.asarray(Wa, dtype=np.float64)
    Wb = np.asarray(Wb, dtype=np.float64)
    if Wa.shape != Wb.shape:
        raise ValueError(f"Wa shape {Wa.shape} != Wb shape {Wb.shape}")
    if Wa.ndim != 2:
        raise ValueError(f"Expected Wa/Wb (n_rows, n_time); got {Wa.shape}")

    gap_lim = _resolve_max_gap_samples(
        max_gap_samples,
        max_gap_ms,
        float(sample_rate_hz),
    )
    impute_arg = gap_lim if gap_lim is not None else 0
    n = int(Wa.shape[0])
    Wa_out = np.full_like(Wa, np.nan, dtype=np.float64)
    Wb_out = np.full_like(Wb, np.nan, dtype=np.float64)

    for i in range(n):
        wa = np.asarray(Wa[i], dtype=np.float32)
        wb = np.asarray(Wb[i], dtype=np.float32)
        if gap_lim is not None and (
            _max_contiguous_nonfinite_run(wa) > gap_lim
            or _max_contiguous_nonfinite_run(wb) > gap_lim
        ):
            continue
        wa_i = _impute_small_gaps_1d(wa, impute_arg)
        wb_i = _impute_small_gaps_1d(wb, impute_arg)
        X1 = np.stack([wa_i, wb_i], axis=-1)[np.newaxis, ...].astype(np.float32)
        if zscore_mode == "aorta_reference":
            X1 = zscore_normalize_aorta_referenced(X1, eps=float(zscore_eps))
        elif zscore_mode == "independent":
            X1 = zscore_normalize_per_subject(X1, eps=float(zscore_eps))
        else:
            raise ValueError(
                f"zscore_mode must be 'aorta_reference' or 'independent'; got {zscore_mode!r}"
            )
        Wa_out[i, :] = X1[0, :, 0].astype(np.float64)
        Wb_out[i, :] = X1[0, :, 1].astype(np.float64)

    return Wa_out, Wb_out


def load_two_channel_waveforms(
    *,
    train_aorta_csv: Path | None = None,
    train_brach_csv: Path | None = None,
    base_dir: Path | None = None,
    n_samples: int = 336,
    n_classes: int = 6,
    subject_col: str = "subject_index",
    target_col: str = "target",
    preprocess: Literal["none", "linear", "phase1"] = "phase1",
    max_gap_samples: int | None = None,
    max_gap_ms: float | None = None,
    sample_rate_hz: float = 500.0,
    zscore_eps: float = 1e-6,
    zscore_mode: Literal["independent", "aorta_reference"] = "aorta_reference",
    impute_fill_all_nan: float = 0.0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load training data as ``(X, y, pp_tab)``.

    - **X** shape: (n_subjects, n_samples, 2) where channels are [aorta, brach].
    - **y** shape: (n_subjects, n_classes) one-hot encoded classes.
    - **pp_tab** shape: (n_subjects, 2) raw **[aorta_pp, brach_pp]** if ``preprocess=="phase1"``,
      else ``None`` (caller can pass zeros into the model).

    Non-numeric CSV cells become NaN.

    ``preprocess``:
      - ``"phase1"`` (default): optional gap threshold (set ``max_gap_samples`` or ``max_gap_ms``),
        linear impute, pulse-pressure tabular features, then z-score (see ``zscore_mode``).
      - ``"linear"``: :func:`impute_waveform_gaps` only (no drops, no Z-score).
      - ``"none"``: raw finite/NaN array as loaded.
    """
    if base_dir is None:
        base_dir = Path.cwd()
    if train_aorta_csv is None:
        train_aorta_csv = base_dir / "datasets/train/aortaP_train_data.csv"
    if train_brach_csv is None:
        train_brach_csv = base_dir / "datasets/train/brachP_train_data.csv"

    a = pd.read_csv(train_aorta_csv)
    b = pd.read_csv(train_brach_csv)

    # Normalize subject index column name (same rule as heart_age_classifier).
    if subject_col not in a.columns:
        unnamed = [c for c in a.columns if str(c).startswith("Unnamed")]
        if unnamed:
            a = a.rename(columns={unnamed[0]: subject_col})
    if subject_col not in b.columns:
        unnamed = [c for c in b.columns if str(c).startswith("Unnamed")]
        if unnamed:
            b = b.rename(columns={unnamed[0]: subject_col})

    a_cols = [f"aorta_t_{i}" for i in range(n_samples)]
    b_cols = [f"brach_t_{i}" for i in range(n_samples)]
    need_a = {subject_col, target_col, *a_cols}
    need_b = {subject_col, *b_cols}
    miss_a = need_a - set(a.columns)
    miss_b = need_b - set(b.columns)
    if miss_a:
        raise ValueError(
            f"Aorta CSV missing columns: {sorted(miss_a)[:10]} (and {max(0, len(miss_a) - 10)} more)"
        )
    if miss_b:
        raise ValueError(
            f"Brach CSV missing columns: {sorted(miss_b)[:10]} (and {max(0, len(miss_b) - 10)} more)"
        )

    m = a[[subject_col, target_col] + a_cols].merge(
        b[[subject_col] + b_cols],
        on=subject_col,
        how="inner",
        validate="one_to_one",
    )

    Xa = m[a_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    Xb = m[b_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X = np.stack([Xa, Xb], axis=-1)  # (n, n_samples, 2)

    t = pd.to_numeric(m[target_col], errors="coerce").to_numpy(dtype=np.int64)
    y = _one_hot_int(t, n_classes=n_classes)

    if preprocess == "phase1":
        n_before = int(X.shape[0])
        mg = _resolve_max_gap_samples(max_gap_samples, max_gap_ms, sample_rate_hz)
        X, y, n_drop, pp_tab = preprocess_waveforms_phase1(
            X,
            y,
            max_gap_samples=mg,
            zscore_eps=zscore_eps,
            zscore_mode=zscore_mode,
        )
        n_after = int(X.shape[0])
        if verbose:
            zm = "aortic μ,σ on both channels" if zscore_mode == "aorta_reference" else "independent per channel"
            if mg is None:
                print(
                    "Phase-1 CNN preprocess: no gap-based subject drops "
                    f"({n_after} / {n_before} retained). "
                    f"Linear gap impute; raw PP (max−min) tabular for head; z-score ({zm})."
                )
            else:
                ms_thr = 1000.0 * mg / sample_rate_hz
                print(
                    "Phase-1 CNN preprocess (gap threshold): "
                    f"dropped {n_drop} / {n_before} subjects "
                    f"({n_after} retained); "
                    f"threshold is contiguous non-finite run > {mg} samples "
                    f"(~{ms_thr:.1f} ms @ {sample_rate_hz:g} Hz). "
                    f"Linear gap impute; raw PP (max−min) tabular for head; z-score ({zm})."
                )
        return X, y, pp_tab
    elif preprocess == "linear":
        X = impute_waveform_gaps(X, fill_all_nan=impute_fill_all_nan)
    elif preprocess != "none":
        raise ValueError(f"preprocess must be 'none', 'linear', or 'phase1'; got {preprocess!r}")
    return X, y, None


def extract_pulse_pressure_tabular(X: np.ndarray) -> np.ndarray:
    """Per-subject pulse pressure proxy: ``max(time) - min(time)`` per channel (raw units).

    Expects finite imputed waveforms, shape ``(n, time, 2)``. Returns ``(n, 2)`` with
    columns **[aorta_pp, brach_pp]** before any z-scoring.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[2] != 2:
        raise ValueError(f"Expected X (n, time, 2); got {X.shape}")
    n_sub = X.shape[0]
    out = np.empty((n_sub, 2), dtype=np.float32)
    for i in range(n_sub):
        for c in range(2):
            col = X[i, :, c]
            out[i, c] = float(np.nanmax(col) - np.nanmin(col))
    return out


def zscore_normalize_aorta_referenced(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score **both** channels using the aortic trace's mean and std (per subject).

    For each row: :math:`\\mu,\\sigma` from channel 0 over time; apply
    :math:`(x-\\mu)/\\sigma` to aorta and brachial. Preserves **relative amplitude**
    between sites compared to independent per-channel z-score.
    """
    X = np.array(X, dtype=np.float32, copy=True)
    n_sub, _, n_ch = X.shape
    if n_ch != 2:
        raise ValueError(f"Expected 2 channels; got {n_ch}")
    eps = float(eps)
    for i in range(n_sub):
        a = X[i, :, 0]
        mu = float(np.nanmean(a))
        sd = float(np.nanstd(a)) + eps
        X[i, :, 0] = (X[i, :, 0] - mu) / sd
        X[i, :, 1] = (X[i, :, 1] - mu) / sd
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def zscore_normalize_per_subject(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-subject, per-channel Z-score normalization across time.

    Uses ``nanmean`` / ``nanstd`` so any remaining non-finite values do not poison
    the scale (prefer :func:`impute_waveform_gaps` before this when possible).
    """
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    out = (X - mu) / (sd + eps)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def minmax_normalize_per_subject(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-subject, per-channel min-max to [0, 1] across time.

    Uses ``nanmin`` / ``nanmax`` for bounds; remaining NaNs (if any) are mapped to
    ``0.5``. Prefer :func:`impute_waveform_gaps` **before** normalization so gaps follow
    local waveform shape instead of a flat mid-level.
    """
    X = np.asarray(X, dtype=np.float32)
    mn = np.nanmin(X, axis=1, keepdims=True)
    mx = np.nanmax(X, axis=1, keepdims=True)
    mn = np.nan_to_num(mn, nan=0.0, posinf=0.0, neginf=0.0)
    mx = np.nan_to_num(mx, nan=0.0, posinf=0.0, neginf=0.0)
    denom = mx - mn
    denom = np.where(denom > eps, denom, 1.0)
    out = (X - mn) / denom
    out = np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class _BasicBlock1D(nn.Module):
    """ResNet-1D basic block: Conv-BN-ReLU-Conv-BN + skip (optional projection)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else None

        self.proj: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        if self.drop is not None:
            out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class _InceptionTemporalBlock1D(nn.Module):
    """Inception-style multi-scale temporal filtering.

    Uses parallel Conv1d branches with different kernel sizes (e.g., 3/5/7) to
    capture morphology at multiple time scales without committing to a single
    receptive field early.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_ch % (len(kernel_sizes) + 1) != 0:
            raise ValueError("out_ch must be divisible by (len(kernel_sizes) + 1)")
        branch_ch = out_ch // (len(kernel_sizes) + 1)

        self.branches = nn.ModuleList()
        # Pointwise branch helps cross-channel mixing + cheap features.
        self.branches.append(
            nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(branch_ch),
                nn.ReLU(inplace=True),
            )
        )
        for k in kernel_sizes:
            pad = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, branch_ch, kernel_size=k, padding=pad, bias=False),
                    nn.BatchNorm1d(branch_ch),
                    nn.ReLU(inplace=True),
                )
            )

        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else None
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([b(x) for b in self.branches], dim=1)
        if self.drop is not None:
            y = self.drop(y)
        return self.fuse(y)


def _xcorr_lag_features(
    x: torch.Tensor,
    *,
    max_lag: int = 50,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute normalized cross-correlation features between channel 0 and 1.

    Input:
        x: (N, 2, T)
    Output:
        feats: (N, 2*max_lag+1) where each entry is corr at lag in [-max_lag, +max_lag]
    """
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected (N, 2, T); got {tuple(x.shape)}")
    a = x[:, 0, :]
    b = x[:, 1, :]

    # Zero-mean, unit-norm per row for stable correlation magnitudes.
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    a = a / (a.norm(dim=1, keepdim=True) + eps)
    b = b / (b.norm(dim=1, keepdim=True) + eps)

    T = a.shape[1]
    feats: list[torch.Tensor] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[:, : T + lag]
            bb = b[:, -lag:]
        elif lag > 0:
            aa = a[:, lag:]
            bb = b[:, : T - lag]
        else:
            aa = a
            bb = b
        # Mean of elementwise product is correlation coefficient estimate.
        feats.append((aa * bb).mean(dim=1, keepdim=True))
    return torch.cat(feats, dim=1)


class _SelfAttention1D(nn.Module):
    """Lightweight self-attention over time (batch-first)."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, C)
        y, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + y)
        y = self.ff(x)
        return self.ln2(x + y)


class TwoChannelResNet1D(nn.Module):
    """Two-channel ResNet-1D for waveform classification.

    Input expects shape (batch, 2, T) [channels-first].

    Optional ``pp_tab`` (batch, 2) are raw **pulse pressures** (max−min per channel),
    concatenated at the head with pooled conv features and xcorr embed.

    - Default: returns logits of shape ``(batch, n_classes)``.
    - With ``auxiliary_regression=True``: returns a tuple ``(cls_logits, reg)`` where
      ``cls_logits`` is ``(batch, n_classes)`` and ``reg`` is ``(batch, 1)`` (decade index).
    """

    def __init__(
        self,
        *,
        n_classes: int = 6,
        auxiliary_regression: bool = False,
        tabular_pp_dim: int = 2,
    ) -> None:
        super().__init__()
        self.auxiliary_regression = auxiliary_regression
        self.tabular_pp_dim = int(tabular_pp_dim)
        # Cross-channel lag features. At 500 Hz, lag ±50 samples ≈ ±100 ms (captures long PTT).
        self.xcorr_max_lag = 50
        self.xcorr_embed = nn.Sequential(
            nn.Linear(2 * self.xcorr_max_lag + 1, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        # Multi-scale temporal filtering early (3/5/7) with minimal early downsampling.
        # This is where the network learns digital-filter-like primitives at multiple scales.
        self.inception = _InceptionTemporalBlock1D(2, 64, kernel_sizes=(3, 5, 7), dropout=0.05)

        # Keep stride=1 longer to preserve timing/phase info; downsample later.
        self.stage1 = nn.Sequential(
            _BasicBlock1D(64, 64, kernel_size=5, stride=1, dilation=1, dropout=0.1),
            _BasicBlock1D(64, 64, kernel_size=5, stride=1, dilation=1, dropout=0.1),
        )
        self.stage2 = nn.Sequential(
            _BasicBlock1D(64, 128, kernel_size=5, stride=1, dilation=1, dropout=0.1),
            _BasicBlock1D(128, 128, kernel_size=5, stride=1, dilation=2, dropout=0.1),
        )

        self.stage3 = nn.Sequential(
            _BasicBlock1D(128, 256, kernel_size=3, stride=2, dilation=1, dropout=0.1),
            _BasicBlock1D(256, 256, kernel_size=3, stride=1, dilation=2, dropout=0.1),
        )

        # Self-attention after stage3: shorter sequence (T' ≈ T/2), 256-d channels — forward vs reflected structure.
        self.attn = _SelfAttention1D(d_model=256, n_heads=4, dropout=0.1)

        # 1D Global Average Pooling over the time axis (Keras `GlobalAveragePooling1D` equivalent).
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.head_drop = nn.Dropout(p=0.25)
        d_head = 256 + 64 + self.tabular_pp_dim
        if auxiliary_regression:
            self.fc_cls = nn.Linear(d_head, n_classes)
            self.fc_reg = nn.Linear(d_head, 1)
        else:
            self.fc = nn.Linear(d_head, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        pp_tab: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (N, 2, T)
        n = x.shape[0]
        if pp_tab is None:
            pp_tab = x.new_zeros((n, self.tabular_pp_dim))
        elif pp_tab.shape != (n, self.tabular_pp_dim):
            raise ValueError(
                f"pp_tab must be (N, {self.tabular_pp_dim}); got {tuple(pp_tab.shape)}"
            )
        else:
            pp_tab = pp_tab.to(device=x.device, dtype=x.dtype)

        xcorr = _xcorr_lag_features(x, max_lag=self.xcorr_max_lag)  # (N, 2L+1)
        xcorr = self.xcorr_embed(xcorr)  # (N, 64)

        x = self.inception(x)
        x = self.stage1(x)

        # Preserve timing early; stage2 stays stride=1 and uses dilation to widen view.
        x = self.stage2(x)  # (N, 128, T)

        x = self.stage3(x)  # (N, 256, T')  shorter time axis after stride-2

        # Self-attention over time on coarser grid: (N, C, T') -> (N, T', C) -> attn -> back
        xt = x.transpose(1, 2)
        xt = self.attn(xt)
        x = xt.transpose(1, 2)

        x = self.global_avg_pool(x).squeeze(-1)  # (N, 256)
        x = torch.cat([x, xcorr, pp_tab], dim=1)
        x = self.head_drop(x)
        if self.auxiliary_regression:
            return self.fc_cls(x), self.fc_reg(x)
        return self.fc(x)


def build_two_channel_resnet1d(
    *,
    n_classes: int = 6,
    auxiliary_regression: bool = False,
    tabular_pp_dim: int = 2,
):
    """Build the two-channel ResNet-1D model (PyTorch)."""
    return TwoChannelResNet1D(
        n_classes=n_classes,
        auxiliary_regression=auxiliary_regression,
        tabular_pp_dim=tabular_pp_dim,
    )


def save_two_channel_resnet_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    *,
    objective: str,
    n_classes: int,
    auxiliary_regression: bool,
    tabular_pp_dim: int = 2,
    temperature: float | None = None,
) -> Path:
    """
    Persist a trained :class:`TwoChannelResNet1D` (CPU ``state_dict`` + metadata).

    Use :func:`load_two_channel_resnet_checkpoint` to restore. Filenames can mirror sklearn
    pipelines (e.g. embed holdout accuracy via ``heart_age_classifier._path_with_val_accuracy``).
    """
    p = path if isinstance(path, Path) else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload: dict[str, Any] = {
        "format": "two_channel_resnet1d",
        "version": 1,
        "state_dict": sd,
        "objective": str(objective),
        "n_classes": int(n_classes),
        "auxiliary_regression": bool(auxiliary_regression),
        "tabular_pp_dim": int(tabular_pp_dim),
        "temperature": float(temperature) if temperature is not None else None,
    }
    torch.save(payload, p)
    return p


def load_two_channel_resnet_checkpoint(
    path: Union[str, Path],
    device: torch.device | str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load weights and metadata from :func:`save_two_channel_resnet_checkpoint`."""
    p = path if isinstance(path, Path) else Path(path)
    try:
        payload = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(p, map_location=device)
    if not isinstance(payload, dict) or payload.get("format") != "two_channel_resnet1d":
        raise ValueError(f"Not a two_channel_resnet1d checkpoint: {p}")

    n_classes = int(payload["n_classes"])
    aux = bool(payload["auxiliary_regression"])
    tab = int(payload.get("tabular_pp_dim", 2))
    model = build_two_channel_resnet1d(
        n_classes=n_classes,
        auxiliary_regression=aux,
        tabular_pp_dim=tab,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, meta

