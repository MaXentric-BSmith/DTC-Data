"""
Arterial pressure waveform morphology: peaks, dicrotic notch, AUC splits, and PTT.

Designed for traces sampled at 500 Hz (2 ms per sample) like the DTC train CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

import numpy as np
from scipy.signal import find_peaks

try:
    from numpy import trapezoid as _integrate_trapz
except ImportError:  # NumPy < 2.0
    _integrate_trapz = np.trapz

try:
    from scipy.integrate import simpson as _integrate_simps
except ImportError:  # pragma: no cover
    try:
        from scipy.integrate import simps as _integrate_simps  # type: ignore[attr-defined]
    except ImportError:
        _integrate_simps = None


@dataclass
class BeatFeatures:
    """Features for one cardiac cycle segment."""

    idx_foot: int
    idx_peak: int
    idx_notch: int
    idx_beat_end: int
    p_sys: float
    p_dia: float
    pulse_pressure: float
    p_notch: float
    auc_sys: float
    auc_dia: float
    auc_ratio_sys_over_dia: float
    t_crest_ms: float
    dp_dt_max: float
    form_factor: float


@dataclass
class WaveformAnalysisResult:
    """Full-trace analysis: one row per detected systolic peak."""

    sample_rate_hz: float
    dt_ms: float
    map_mean_pressure: float
    beats: list[BeatFeatures] = field(default_factory=list)
    ptt_ms: float | None = None
    aorta_trace: np.ndarray | None = None
    """Pressure samples actually used for detection (finite-only, NaNs imputed)."""
    brachial_trace: np.ndarray | None = None
    """Same for the brachial channel when ``brachial_pressure`` was supplied."""

    def to_array_dict(self) -> dict[str, np.ndarray]:
        """Stack per-beat scalars into 1-D arrays (empty beats → zero-length arrays)."""
        if not self.beats:
            return {
                "idx_foot": np.array([], dtype=np.int64),
                "idx_peak": np.array([], dtype=np.int64),
                "idx_notch": np.array([], dtype=np.int64),
                "idx_beat_end": np.array([], dtype=np.int64),
                "p_sys": np.array([]),
                "p_dia": np.array([]),
                "pulse_pressure": np.array([]),
                "p_notch": np.array([]),
                "auc_sys": np.array([]),
                "auc_dia": np.array([]),
                "auc_ratio_sys_over_dia": np.array([]),
                "t_crest_ms": np.array([]),
                "dp_dt_max": np.array([]),
                "form_factor": np.array([]),
            }
        return {
            "idx_foot": np.array([b.idx_foot for b in self.beats]),
            "idx_peak": np.array([b.idx_peak for b in self.beats]),
            "idx_notch": np.array([b.idx_notch for b in self.beats]),
            "idx_beat_end": np.array([b.idx_beat_end for b in self.beats]),
            "p_sys": np.array([b.p_sys for b in self.beats]),
            "p_dia": np.array([b.p_dia for b in self.beats]),
            "pulse_pressure": np.array([b.pulse_pressure for b in self.beats]),
            "p_notch": np.array([b.p_notch for b in self.beats]),
            "auc_sys": np.array([b.auc_sys for b in self.beats]),
            "auc_dia": np.array([b.auc_dia for b in self.beats]),
            "auc_ratio_sys_over_dia": np.array([b.auc_ratio_sys_over_dia for b in self.beats]),
            "t_crest_ms": np.array([b.t_crest_ms for b in self.beats]),
            "dp_dt_max": np.array([b.dp_dt_max for b in self.beats]),
            "form_factor": np.array([b.form_factor for b in self.beats]),
        }


def _linear_impute_nonfinite_1d(y: np.ndarray, *, fill_all_nan: float = 0.0) -> np.ndarray:
    """Linearly interpolate non-finite samples along one pressure trace (matches train CSV gaps)."""
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n == 0:
        return y
    finite = np.isfinite(y)
    if not finite.any():
        return np.full(n, float(fill_all_nan), dtype=float)
    idx = np.arange(n, dtype=float)
    xv = idx[finite]
    yv = y[finite]
    return np.interp(idx, xv, yv, left=float(yv[0]), right=float(yv[-1]))


def _auc_segment(
    pressure: np.ndarray,
    i0: int,
    i1: int,
    dt_s: float,
    rule: Literal["trapz", "simps"],
) -> float:
    """Pressure–time integral (mmHg·s) from index i0 through i1 inclusive."""
    i0 = int(max(0, i0))
    i1 = int(min(len(pressure) - 1, i1))
    if i1 <= i0:
        return 0.0
    seg = pressure[i0 : i1 + 1]
    x = (np.arange(seg.size, dtype=float) * dt_s) if dt_s != 1.0 else np.arange(seg.size, dtype=float)
    if rule == "simps" and _integrate_simps is not None and seg.size >= 3 and seg.size % 2 == 1:
        return float(_integrate_simps(seg, x=x))
    return float(_integrate_trapz(seg, x=x))


def pressure_derivatives(
    pressure: np.ndarray, sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """First and second derivatives of pressure w.r.t. time via :func:`numpy.gradient`."""
    dt_s = 1.0 / float(sample_rate_hz)
    p = np.asarray(pressure, dtype=float).ravel()
    dp_dt = np.gradient(p, dt_s)
    d2p_dt2 = np.gradient(dp_dt, dt_s)
    return dp_dt, d2p_dt2


def _find_dicrotic_notch_index(
    pressure: np.ndarray,
    dp_dt: np.ndarray,
    idx_peak: int,
    idx_end: int,
    *,
    min_lag_samples: int = 3,
) -> int:
    """
    First local maximum of :math:`dP/dt` after the systolic peak (valve closure slows the
    pressure drop). Search window starts ``min_lag_samples`` after the peak to avoid
    upstroke/apex artifacts. Falls back to argmax on that window if no peak is found.
    """
    lo = int(idx_peak) + int(max(1, min_lag_samples))
    hi = int(idx_end)
    if hi <= lo + 1:
        lo = int(idx_peak) + 1
        if hi <= lo + 1:
            return int(np.clip(lo, 0, len(pressure) - 1))
    window = dp_dt[lo : hi + 1]
    peaks_rel, _ = find_peaks(window)
    if peaks_rel.size > 0:
        return lo + int(peaks_rel[0])
    return int(lo + int(np.argmax(window)))


def _diastolic_foot_index(
    pressure: np.ndarray,
    lo: int,
    hi: int,
    prominence_mmhg: float,
) -> int:
    """
    Diastolic minimum between samples ``lo`` and ``hi`` (inclusive hi for slicing end).

    Uses :func:`scipy.signal.find_peaks` on the inverted segment (``-pressure``) so local
    minima are detected as peaks, with the same prominence idea as systolic detection.
    Falls back to ``argmin`` if no prominent valley is found.
    """
    lo = int(max(0, lo))
    hi = int(min(len(pressure) - 1, hi))
    if hi <= lo:
        return lo
    seg_neg = -pressure[lo : hi + 1]
    peaks_rel, props = find_peaks(seg_neg, prominence=float(prominence_mmhg))
    if peaks_rel.size > 0:
        prom = props.get("prominences", np.ones(peaks_rel.size))
        return lo + int(peaks_rel[int(np.argmax(prom))])
    return lo + int(np.argmin(pressure[lo : hi + 1]))


def analyze_pressure_waveform(
    pressure: np.ndarray,
    sample_rate_hz: float = 500.0,
    prominence_mmhg: float = 10.0,
    min_peak_distance_samples: int = 200,
    notch_min_lag_samples: int = 3,
    integration_rule: Literal["trapz", "simps"] = "trapz",
    brachial_pressure: np.ndarray | None = None,
    assume_first_sample_is_foot: bool = False,
    assume_one_beat_per_trace: bool = False,
    dicrotic_notch_source: Literal["aorta", "brachial"] = "aorta",
) -> WaveformAnalysisResult:
    """
    Detect systolic peaks and diastolic feet, notch, AUC splits, and optional PTT.

    Parameters
    ----------
    pressure
        Aortic (or single-site) pressure samples.
    sample_rate_hz
        Sampling rate (default 500 Hz → 2 ms per sample).
    prominence_mmhg
        Minimum prominence for systolic peaks (noise rejection).
    min_peak_distance_samples
        Minimum index spacing between systolic peaks (~200 at 500 Hz for physiological HR).
    notch_min_lag_samples
        Earliest index *after* systolic peak where the dicrotic notch search begins.
    integration_rule
        ``trapz`` (default) or ``simps`` (Simpson's rule when SciPy provides it and length is suitable).
    brachial_pressure
        Optional same-length brachial trace for foot-to-foot pulse transit time.
    assume_first_sample_is_foot
        If True, **beat 0** fixes the diastolic foot at sample index ``0`` (first point after
        imputation) instead of searching for a pre-systolic minimum. Later beats are unchanged.
    assume_one_beat_per_trace
        If True, each 1-D trace is treated as **at most one cardiac cycle** (e.g. one beat per
        CSV row/window): only the **first** systolic peak from :func:`~scipy.signal.find_peaks`
        is kept; any additional peaks are ignored and diastolic runoff runs to the end of the trace.
    dicrotic_notch_source
        Which trace sets the **notch time index** (``idx_notch``): ``\"aorta\"`` uses
        :math:`dP/dt` on the aortic signal; ``\"brachial\"`` uses brachial :math:`dP/dt` (often
        clearer peripherally). Aortic **``p_notch``** and **aortic AUC** still use that same index on
        the aortic pressure. Needs aligned ``brachial_pressure``; else falls back to aorta.

    Notes
    -----
    * **NaNs / missing samples** (common in the DTC ``*_train`` CSVs) are replaced by
      linear interpolation along time before peak detection, matching
      ``heart_age_classifier._linear_impute_nonfinite_1d``. MAP and form factor then use
      the mean of this **imputed** working trace.
    * Diastolic foot for beat *k>0*: prominent local minimum (inverted ``find_peaks``) on the
      interval between the previous and current systolic peak; beat 0 uses the same unless
      ``assume_first_sample_is_foot`` is True (foot at index 0).
    * PTT (ms) = (idx_brach_foot − idx_aorta_foot) × (1000 / sample_rate_hz), using the
      first detected beat on each channel independently for foot indices.
    """
    p_raw = np.asarray(pressure, dtype=float).ravel()
    n = p_raw.size
    p = _linear_impute_nonfinite_1d(p_raw)
    pb_clean: np.ndarray | None = None
    if brachial_pressure is not None:
        pb_raw = np.asarray(brachial_pressure, dtype=float).ravel()
        if pb_raw.size == n:
            pb_clean = _linear_impute_nonfinite_1d(pb_raw)

    if n < 3:
        return WaveformAnalysisResult(
            sample_rate_hz=sample_rate_hz,
            dt_ms=1000.0 / float(sample_rate_hz),
            map_mean_pressure=float(np.mean(p)) if n else float("nan"),
            beats=[],
            ptt_ms=None,
            aorta_trace=p.copy() if n else np.array([], dtype=float),
            brachial_trace=pb_clean.copy() if pb_clean is not None and n else None,
        )

    dt_s = 1.0 / float(sample_rate_hz)
    dt_ms = 1000.0 / float(sample_rate_hz)
    map_mean = float(np.mean(p))

    peak_idx, _ = find_peaks(
        p,
        prominence=float(prominence_mmhg),
        distance=int(max(1, min_peak_distance_samples)),
    )
    peak_idx = np.asarray(peak_idx, dtype=np.int64)
    if assume_one_beat_per_trace and peak_idx.size > 1:
        peak_idx = peak_idx[:1]
    if peak_idx.size == 0:
        return WaveformAnalysisResult(
            sample_rate_hz=sample_rate_hz,
            dt_ms=dt_ms,
            map_mean_pressure=map_mean,
            beats=[],
            ptt_ms=None,
            aorta_trace=p.copy(),
            brachial_trace=pb_clean.copy() if pb_clean is not None else None,
        )

    dp_dt, _d2p_dt2 = pressure_derivatives(p, sample_rate_hz)
    use_brach_notch = (
        dicrotic_notch_source == "brachial"
        and pb_clean is not None
        and int(pb_clean.size) == int(p.size)
    )
    if use_brach_notch:
        dp_dt_brach, _ = pressure_derivatives(pb_clean, sample_rate_hz)

    beats: list[BeatFeatures] = []
    for k in range(len(peak_idx)):
        pk = int(peak_idx[k])
        prev_pk = int(peak_idx[k - 1]) if k > 0 else -1
        foot_lo = prev_pk + 1
        if k == 0 and assume_first_sample_is_foot:
            foot = 0
        else:
            foot = _diastolic_foot_index(p, foot_lo, pk, prominence_mmhg)
        foot = int(np.clip(foot, foot_lo, pk))

        if k + 1 < len(peak_idx):
            next_pk = int(peak_idx[k + 1])
            end_lo = pk + 1
            beat_end = _diastolic_foot_index(p, end_lo, next_pk, prominence_mmhg)
        else:
            beat_end = n - 1

        if use_brach_notch:
            seg = pb_clean[foot : beat_end + 1]
            pk_b = foot + int(np.argmax(seg)) if seg.size else pk
            pk_b = int(np.clip(pk_b, foot, beat_end))
            idx_notch = _find_dicrotic_notch_index(
                pb_clean,
                dp_dt_brach,
                pk_b,
                beat_end,
                min_lag_samples=notch_min_lag_samples,
            )
        else:
            idx_notch = _find_dicrotic_notch_index(
                p, dp_dt, pk, beat_end, min_lag_samples=notch_min_lag_samples
            )
        idx_notch = int(np.clip(idx_notch, foot, beat_end))

        p_sys = float(p[pk])
        p_dia = float(p[foot])
        pp = p_sys - p_dia
        p_notch = float(p[idx_notch])

        auc_sys = _auc_segment(p, foot, idx_notch, dt_s, integration_rule)
        auc_dia = _auc_segment(p, idx_notch, beat_end, dt_s, integration_rule)
        if auc_dia > 0:
            auc_ratio = auc_sys / auc_dia
        else:
            auc_ratio = float("inf") if auc_sys > 0 else float("nan")

        t_crest_ms = (pk - foot) * dt_ms
        upstroke = dp_dt[foot : pk + 1]
        dp_max = float(np.max(upstroke)) if upstroke.size else float("nan")

        if pp > 0:
            ff = (map_mean - p_dia) / pp
        else:
            ff = float("nan")

        beats.append(
            BeatFeatures(
                idx_foot=foot,
                idx_peak=pk,
                idx_notch=idx_notch,
                idx_beat_end=beat_end,
                p_sys=p_sys,
                p_dia=p_dia,
                pulse_pressure=pp,
                p_notch=p_notch,
                auc_sys=auc_sys,
                auc_dia=auc_dia,
                auc_ratio_sys_over_dia=auc_ratio,
                t_crest_ms=float(t_crest_ms),
                dp_dt_max=dp_max,
                form_factor=float(ff),
            )
        )

    ptt_ms: float | None = None
    if pb_clean is not None:
        res_a = analyze_pressure_waveform(
            p,
            sample_rate_hz=sample_rate_hz,
            prominence_mmhg=prominence_mmhg,
            min_peak_distance_samples=min_peak_distance_samples,
            notch_min_lag_samples=notch_min_lag_samples,
            integration_rule=integration_rule,
            brachial_pressure=None,
            assume_first_sample_is_foot=assume_first_sample_is_foot,
            assume_one_beat_per_trace=assume_one_beat_per_trace,
        )
        res_b = analyze_pressure_waveform(
            pb_clean,
            sample_rate_hz=sample_rate_hz,
            prominence_mmhg=prominence_mmhg,
            min_peak_distance_samples=min_peak_distance_samples,
            notch_min_lag_samples=notch_min_lag_samples,
            integration_rule=integration_rule,
            brachial_pressure=None,
            assume_first_sample_is_foot=assume_first_sample_is_foot,
            assume_one_beat_per_trace=assume_one_beat_per_trace,
        )
        if res_a.beats and res_b.beats:
            lag_samples = res_b.beats[0].idx_foot - res_a.beats[0].idx_foot
            ptt_ms = float(lag_samples * dt_ms)

    return WaveformAnalysisResult(
        sample_rate_hz=sample_rate_hz,
        dt_ms=dt_ms,
        map_mean_pressure=map_mean,
        beats=beats,
        ptt_ms=ptt_ms,
        aorta_trace=p.copy(),
        brachial_trace=pb_clean.copy() if pb_clean is not None else None,
    )


def display_arterial_waveform_analysis(
    pressure_aorta: np.ndarray,
    pressure_brach: np.ndarray | None = None,
    *,
    sample_rate_hz: float = 500.0,
    prominence_mmhg: float = 10.0,
    min_peak_distance_samples: int = 200,
    notch_min_lag_samples: int = 3,
    assume_first_sample_is_foot: bool = False,
    assume_one_beat_per_trace: bool = False,
    dicrotic_notch_source: Literal["aorta", "brachial"] = "aorta",
    title: str = "Arterial waveform features",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    ax: Axes | None = None,
    result: WaveformAnalysisResult | None = None,
    show_xlabel: bool = True,
    annotation_fontsize: float = 9.0,
    legend_fontsize: float = 8.0,
):
    """
    Run :func:`analyze_pressure_waveform` and plot the pressure trace with detected landmarks.

    If ``pressure_brach`` is provided, it is overlaid and foot-to-foot PTT is annotated.

    Parameters
    ----------
    ax
        Optional matplotlib ``Axes``. When set, draws into this axis (for stacked figures)
        and does not add a new figure-level ``tight_layout``.
    result
        Optional precomputed analysis; when set, skips calling ``analyze_pressure_waveform``.
    assume_first_sample_is_foot
        Forwarded to :func:`analyze_pressure_waveform` when ``result`` is ``None``.
    assume_one_beat_per_trace
        Forwarded to :func:`analyze_pressure_waveform` when ``result`` is ``None``.
    dicrotic_notch_source
        Forwarded to :func:`analyze_pressure_waveform` when ``result`` is ``None``.
    show_xlabel
        If False, omits the time-axis label (useful for shared-x stacked subplots).

    Returns
    -------
    matplotlib.figure.Figure
        The figure handle. When ``show=True`` (default), ``plt.show()`` is also called
        (use ``show=False`` in headless tests or when embedding elsewhere).
    """
    import matplotlib.pyplot as plt

    a = np.asarray(pressure_aorta, dtype=float).ravel()
    b = np.asarray(pressure_brach, dtype=float).ravel() if pressure_brach is not None else None
    if result is None:
        res = analyze_pressure_waveform(
            a,
            sample_rate_hz=sample_rate_hz,
            prominence_mmhg=prominence_mmhg,
            min_peak_distance_samples=min_peak_distance_samples,
            notch_min_lag_samples=notch_min_lag_samples,
            brachial_pressure=b,
            assume_first_sample_is_foot=assume_first_sample_is_foot,
            assume_one_beat_per_trace=assume_one_beat_per_trace,
            dicrotic_notch_source=dicrotic_notch_source,
        )
    else:
        res = result

    pa = res.aorta_trace if res.aorta_trace is not None and res.aorta_trace.size else a
    pb_plot = res.brachial_trace if res.brachial_trace is not None else b

    created = ax is None
    if created:
        if figsize is None:
            figsize = (11.0, 5.5)
        _, ax = plt.subplots(figsize=figsize)
    fig = ax.figure
    t_ms = np.arange(pa.size) * res.dt_ms

    ax.plot(t_ms, pa, color="C0", lw=1.2, label="Aortic (interp gaps)")

    if pb_plot is not None and pb_plot.size == pa.size:
        ax.plot(t_ms, pb_plot, color="C1", lw=1.0, alpha=0.85, label="Brachial (interp gaps)")

    colors = {"foot": "#006400", "peak": "#b22222", "notch": "#4b0082"}
    for j, beat in enumerate(res.beats):
        ax.scatter(
            t_ms[beat.idx_foot],
            beat.p_dia,
            s=70,
            marker="v",
            c=colors["foot"],
            edgecolors="k",
            linewidths=0.6,
            zorder=6,
            label="Diastolic foot" if j == 0 else "_nolegend_",
        )
        ax.scatter(
            t_ms[beat.idx_peak],
            beat.p_sys,
            s=85,
            marker="*",
            c=colors["peak"],
            edgecolors="k",
            linewidths=0.5,
            zorder=6,
            label="Systolic peak" if j == 0 else "_nolegend_",
        )
        notch_on_brach_plot = (
            dicrotic_notch_source == "brachial"
            and pb_plot is not None
            and pb_plot.size == pa.size
        )
        if notch_on_brach_plot:
            notch_y = float(pb_plot[int(beat.idx_notch)])
            notch_lbl = "Dicrotic notch (brachial dP/dt)" if j == 0 else "_nolegend_"
        else:
            notch_y = beat.p_notch
            notch_lbl = "Dicrotic notch" if j == 0 else "_nolegend_"
        ax.scatter(
            t_ms[beat.idx_notch],
            notch_y,
            s=65,
            marker="D",
            c=colors["notch"],
            edgecolors="k",
            linewidths=0.5,
            zorder=6,
            label=notch_lbl,
        )
        ax.axvspan(
            t_ms[beat.idx_foot],
            t_ms[beat.idx_notch],
            alpha=0.12,
            color="green",
        )
        ax.axvspan(
            t_ms[beat.idx_notch],
            t_ms[beat.idx_beat_end],
            alpha=0.12,
            color="orange",
        )

    lines = [f"MAP (working trace) = {res.map_mean_pressure:.2f} mmHg"]
    if not res.beats:
        lines.append(
            "No systolic peaks detected — lower prominence_mmhg or min_peak_distance_samples, "
            "or inspect the raw CSV for gaps/outliers."
        )
    if res.beats:
        b0 = res.beats[0]
        extras = []
        if (
            dicrotic_notch_source == "brachial"
            and pb_plot is not None
            and pb_plot.size == pa.size
        ):
            extras.append(
                f"  P_notch time from brachial dP/dt; P_aorta={b0.p_notch:.1f}, P_brach={float(pb_plot[int(b0.idx_notch)]):.1f} mmHg"
            )
        lines.extend(
            [
                f"Beat 0: P_sys={b0.p_sys:.1f}, P_dia={b0.p_dia:.1f}, PP={b0.pulse_pressure:.1f}",
                f"  AUC_sys={b0.auc_sys:.4f}, AUC_dia={b0.auc_dia:.4f}, ratio={b0.auc_ratio_sys_over_dia:.3f}",
                f"  t_crest={b0.t_crest_ms:.1f} ms, dP/dt_max={b0.dp_dt_max:.1f}, FF={b0.form_factor:.3f}",
            ]
            + extras
        )
    if res.ptt_ms is not None:
        lines.append(f"PTT (foot–foot, beat 0) ≈ {res.ptt_ms:.2f} ms")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        fontsize=annotation_fontsize,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9, edgecolor="0.35"),
    )

    if show_xlabel:
        ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=legend_fontsize)
    ax.grid(True, alpha=0.25)
    if created:
        fig.tight_layout()
        if show:
            plt.show()
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    t = np.arange(0, 1.5, 0.002)
    fs = 500.0
    # Synthetic damped pulses (~75 BPM)
    y_a = np.zeros_like(t)
    for k, t0 in enumerate(np.arange(0.2, 1.5, 0.8)):
        env = np.exp(-(t - t0) ** 2 / 0.012)
        y_a += 80 + 40 * env * np.sin(np.pi * (t - t0) / 0.15 + 0.3 * k).clip(min=0)
    y_a += rng.normal(0, 0.5, size=y_a.shape)
    lag = 12  # samples (~24 ms at 500 Hz)
    y_b = np.roll(y_a, lag)
    y_b[:lag] = y_a[:lag]

    display_arterial_waveform_analysis(
        y_a,
        y_b,
        sample_rate_hz=fs,
        title="Demo: synthetic aortic + delayed brachial",
        show=False,
    )
    import matplotlib.pyplot as plt

    plt.close("all")
    r = analyze_pressure_waveform(y_a, brachial_pressure=y_b, sample_rate_hz=fs)
    print(f"demo: {len(r.beats)} beat(s), PTT_ms={r.ptt_ms}")
