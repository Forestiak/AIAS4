#!/usr/bin/env python3
"""
python3 main.py
python3 main.py --no-candidate
python3 main.py --no-baseline
python3 main.py --show-kappa
python3 main.py --subset 5
python3 main.py --targets Loc-01 Loc-03
python3 main.py --exclude Loc-01 Loc-03
python3 main.py --save
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bocpd_cpt import data as D
from bocpd_cpt.bocpd import BOCPDResult, NIGPredictor, NIGPrior, map_changepoints_paper, run_bocpd
from bocpd_cpt.eval import evaluate_run, global_summary
from bocpd_cpt.extensions import build_neighbor_cp_density, refine_cps_to_gradient
from bocpd_cpt.hazards import (
    HazardFn,
    const_hazard,
    contrast_prior_hazard,
    depth_aware_hazard,
    min_thickness_hazard,
    spatial_prior_hazard,
)


TOLERANCES = [0.5, 1.0, 2.0]


RUN_BASELINE = False
RUN_CANDIDATE = True

CANDIDATE_BASE_MODE = "const"          # "const" | "depth"
CANDIDATE_HAZARD_KAPPA = None          # None -> N for const mode when adaptive is off | 30 -> ~0.6 m mean layer thickness prior for const mode; ignored for depth mode
CANDIDATE_USE_ADAPTIVE_KAPPA = True   # const mode: adaptive per-profile kappa; depth mode: adaptive kappa0_m
CANDIDATE_USE_MIN_THICKNESS = True
CANDIDATE_MIN_THICKNESS_M = 0.75
CANDIDATE_USE_SPATIAL = False
CANDIDATE_SPATIAL_BANDWIDTH_M = 1.5
CANDIDATE_SPATIAL_K_NEIGHBORS = 6
CANDIDATE_SPATIAL_STRENGTH = 0.75
CANDIDATE_USE_REFINEMENT = True
CANDIDATE_REFINE_SEARCH_M = 0.3
CANDIDATE_REFINE_SMOOTH_WINDOW = 0
CANDIDATE_REFINE_STRENGTH_RATIO: float | None = None
CANDIDATE_REFINE_DROP_BELOW_QUANTILE: float | None = None
CANDIDATE_DEPTH_KAPPA0_M = 1.0
CANDIDATE_DEPTH_GROWTH = 0.05

# Left-right window contrast prior: modulate the hazard at each depth by how
# strongly the Ic signal's left and right windows disagree
CANDIDATE_USE_CONTRAST = False
CANDIDATE_CONTRAST_WINDOW_M = 0.5   # one-sided window width in metres
CANDIDATE_CONTRAST_STRENGTH = 0.5   # 0 = no effect, >1 = stronger steering

# Per-profile auto-calibration of the contrast hazard and refinement
# drop-quantile from the profile's own signal statistics (SNR, local noise).
CANDIDATE_USE_AUTO_CONTRAST = True        # overrides CANDIDATE_USE_CONTRAST
CANDIDATE_AUTO_REFINE_DROP_QUANTILE = True # ! overrides CANDIDATE_REFINE_DROP_BELOW_QUANTILE

# Profile-centred NIG prior: replace the paper's mu0=0 with the per-profile
# robust median of the feature signal.  Unsupervised; removes the persistent
# downward bias in the reset-slot predictive density on Ic ∈ [1.5, 3.5]
CANDIDATE_USE_PROFILE_MU0 = False

# Adaptive pre-smoothing of the Ic signal before BOCPD; the raw signal is still
# used for gradient refinement, so boundary localisation is unaffected
CANDIDATE_USE_ADAPTIVE_SMOOTH = True
CANDIDATE_SMOOTH_METHOD = "savgol"               # "tv" or "savgol"
CANDIDATE_SMOOTH_ROUGHNESS_THRESHOLD = 0.080 # no smoothing below this
CANDIDATE_SMOOTH_ROUGHNESS_MAX = 0.150       # lambda is capped above this
CANDIDATE_SMOOTH_LAM_MIN = 0.05             # TV lambda at the threshold
CANDIDATE_SMOOTH_LAM_MAX = 0.30             # TV lambda at roughness_max

# Show the per-profile performance analysis by default
DEFAULT_SHOW_ANALYSIS = True

# Adaptive-kappa estimator settings.  Used only when
# CANDIDATE_USE_ADAPTIVE_KAPPA = True
ADAPTIVE_BASE_THICKNESS_M = 1.0
ADAPTIVE_MIN_THICKNESS_M = 0.75
ADAPTIVE_MAX_THICKNESS_M = 30.0
ADAPTIVE_SMOOTH_WINDOW = 0
ADAPTIVE_STRONG_PEAK_QUANTILE = 0.80
ADAPTIVE_BLEND_BASE = 0.50
ADAPTIVE_BLEND_PEAK_SPACING = 0.35
ADAPTIVE_BLEND_VARIATION_SCALE = 0.15

# Target filters
DEFAULT_INCLUDE_TARGETS: list[str] = [] # empty = include all
DEFAULT_EXCLUDE_TARGETS: list[str] = []

# Optional smoke-test size
DEFAULT_SUBSET: int | None = None

# Output / diagnostics defaults
DEFAULT_SHOW_KAPPA = False
DEFAULT_SAVE_OUTPUTS = False


@dataclass(frozen=True)
class AdaptiveKappaConfig:
    enabled: bool = False
    base_thickness_m: float = 0.6
    min_thickness_m: float = 0.5
    max_thickness_m: float = 2.5
    smooth_window: int = 11
    strong_peak_quantile: float = 0.80
    blend_base: float = 0.50
    blend_peak_spacing: float = 0.35
    blend_variation_scale: float = 0.15


@dataclass(frozen=True)
class AdaptiveKappaStats:
    target: str
    adaptive_thickness_m: float
    adaptive_kappa_samples: float
    peak_spacing_m: float
    variation_scale_m: float
    strong_peak_threshold: float
    n_strong_peaks: int
    signal_iqr: float
    gradient_q75: float


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    label: str
    feature: str = "Ic"
    mu0: float = 0.0
    kappa_prior: float = 1.0
    alpha: float = 1.0
    beta: float = 0.1
    # When True, override mu0 with the per-profile robust median of the feature
    mu0_from_signal: bool = False
    base_mode: str = "const"  # const | depth
    hazard_kappa: float | None = None  # samples, for base_mode="const"
    use_min_thickness: bool = False
    min_thickness_m: float = 0.5
    depth_kappa0_m: float = 1.0
    depth_growth: float = 0.05
    adaptive: AdaptiveKappaConfig = field(default_factory=AdaptiveKappaConfig)
    refine: bool = False
    refine_search_m: float = 0.3
    refine_smooth_window: int = 5
    refine_strength_ratio: float | None = None
    refine_drop_below_quantile: float | None = None
    use_contrast: bool = False
    contrast_window_m: float = 0.5
    contrast_strength: float = 1.0
    # When True, override use_contrast / contrast_window_m / contrast_strength
    # with per-profile estimates derived from the signal's local noise and IQR.
    use_auto_contrast: bool = False
    # When True (and refine=True), override refine_drop_below_quantile with a
    # per-profile estimate based on signal SNR.  Improves precision on noisy
    # profiles without penalising clean ones.
    auto_refine_drop_quantile: bool = False
    use_spatial: bool = False
    spatial_bandwidth_m: float = 1.5
    spatial_k_neighbors: int = 6
    spatial_strength: float = 0.75
    # Adaptive pre-smoothing of the feature signal before BOCPD.
    # Smoothing is applied only for profiles whose roughness exceeds
    use_adaptive_smooth: bool = False
    smooth_method: str = "tv"
    smooth_roughness_threshold: float = 0.080
    smooth_roughness_max: float = 0.150
    smooth_lam_min: float = 0.05
    smooth_lam_max: float = 0.30


def _local_noise_estimate(x: np.ndarray, step_m: float, window_m: float = 0.3) -> float:
    """Estimate within-segment noise as the median of local windowed stds.

    Uses a sliding window of ``window_m`` metres.  Taking the *median* over all
    windows makes the estimate robust to large inter-segment jumps, which would
    inflate a global std.  This gives the typical noise level *inside* a layer,
    not the between-layer variation.
    """
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size < 4:
        return float(np.std(finite)) if finite.size > 1 else 1e-4
    hw = max(2, int(round(window_m / 2.0 / step_m)))
    n = len(x)
    stride = max(1, hw // 2)
    local_stds: list[float] = []
    for i in range(0, n, stride):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 3:
            local_stds.append(float(np.std(seg)))
    return float(np.median(local_stds)) if local_stds else float(np.std(finite))


def denoise_tv_1d(x: np.ndarray, lam: float, n_iter: int = 50) -> np.ndarray:
    """1D total variation denoising via dual projected gradient (Chambolle 2004).

    Solves: min_u (1/2)||u - x||^2 + lam * sum_i |u[i+1] - u[i]|

    Edge-preserving: signal jumps larger than ``lam`` are kept intact while
    within-layer fluctuations smaller than ``lam`` are flattened toward the
    local segment mean.  Ideal for piecewise-constant CPT data.

    Pure numpy, fully vectorised, O(N) per iteration.  No new dependencies.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 or lam <= 0.0:
        return x.copy()
    q = np.zeros(n - 1)   # dual variable; constrained |q_i| ≤ 1
    tau = 0.249            # step size < 1/4 (spectral bound of D D^T in 1D)
    Dx = x[1:] - x[:-1]   # first differences of x, precomputed
    for _ in range(n_iter):
        # D D^T q: vectorised second-difference of dual variable
        DDTq = 2.0 * q.copy()
        DDTq[1:] -= q[:-1]
        DDTq[:-1] -= q[1:]
        # Dual gradient: D(x - lam D^T q) = Dx - lam D D^T q
        q = np.clip(q + tau * (Dx - lam * DDTq), -1.0, 1.0)
    # Primal recovery: u = x - lam D^T q
    Dtq = np.zeros(n)
    Dtq[:-1] -= q
    Dtq[1:] += q
    return x - lam * Dtq


def _adaptive_smooth_signal(
    x: np.ndarray,
    roughness: float,
    step_m: float,
    roughness_threshold: float = 0.080,
    roughness_max: float = 0.150,
    lam_min: float = 0.05,
    lam_max: float = 0.30,
    method: str = "tv",
    n_iter: int = 50,
) -> tuple[np.ndarray, float, bool]:
    """Apply adaptive smoothing to a CPT signal based on its roughness.

    No smoothing is applied when ``roughness < roughness_threshold``.  Above
    the threshold the TV lambda (or Savgol window) ramps linearly from
    ``lam_min`` at the threshold to ``lam_max`` at ``roughness_max``.

    Returns
    -------
    smoothed : ndarray
        Smoothed signal (or the input array when no smoothing is applied).
    lam : float
        The TV lambda actually applied (0.0 when no smoothing).
    applied : bool
        True when smoothing was applied.
    """
    x_arr = np.asarray(x, dtype=float)
    if roughness < roughness_threshold:
        return x_arr, 0.0, False

    span = max(roughness_max - roughness_threshold, 1e-9)
    t = float(np.clip((roughness - roughness_threshold) / span, 0.0, 1.0))
    lam = lam_min + t * (lam_max - lam_min)

    if method == "tv":
        x_smooth = denoise_tv_1d(x_arr, lam, n_iter=n_iter)
    elif method == "savgol":
        # Map t linearly to a window size in metres: 0.30 m at threshold, 1.0 m at max.
        window_m = 0.30 + t * 0.70
        win = int(round(window_m / step_m))
        win = win if win % 2 == 1 else win + 1
        win = max(5, min(win, x_arr.size if x_arr.size % 2 == 1 else x_arr.size - 1))
        poly = min(3, win - 1)
        x_smooth = savgol_filter(x_arr, win, poly)
    else:
        raise ValueError(f"Unknown smooth method: {method!r}")

    return x_smooth, float(lam), True


def estimate_auto_contrast_params(
    x: np.ndarray,
    step_m: float,
    adaptive_stats: "AdaptiveKappaStats | None",
    snr_threshold: float = 2.5,
    base_window_m: float = 0.5,
) -> dict[str, object]:
    """Derive per-profile contrast hazard parameters from the signal itself.

    The signal-to-noise ratio (SNR = IQR / local_noise) drives both the
    decision to enable contrast and the modulation strength.

    Returns a dict with keys: use_contrast, contrast_window_m,
    contrast_strength, local_noise, snr.
    """
    x = np.asarray(x, dtype=float)
    local_noise = _local_noise_estimate(x, step_m)

    # Prefer IQR from the adaptive kappa estimator (already computed) to avoid
    # re-scanning the signal.
    if adaptive_stats is not None:
        iqr = max(float(adaptive_stats.signal_iqr), 1e-4)
    else:
        finite = x[np.isfinite(x)]
        iqr = float(np.subtract(*np.percentile(finite, [75, 25]))) if finite.size > 1 else 1e-4
        iqr = max(iqr, 1e-4)

    snr = iqr / max(local_noise, 1e-4)
    use_contrast = snr >= snr_threshold

    # Strength ramps from 0.30 at SNR=snr_threshold to ~1.20 at SNR=10.
    strength = float(np.clip(0.30 + 0.09 * max(0.0, snr - snr_threshold), 0.10, 1.20))
    contrast_window_m = base_window_m  # 0.5 m matches typical CPT transition width

    return {
        "use_contrast": use_contrast,
        "contrast_window_m": contrast_window_m,
        "contrast_strength": strength if use_contrast else 0.0,
        "local_noise": float(local_noise),
        "snr": float(snr),
    }


def estimate_auto_refine_drop_quantile(snr: float) -> float:
    """Choose a per-profile gradient drop quantile based on signal SNR.

    Higher SNR → true boundaries align with clear gradient peaks → we can
    afford to be more selective (higher threshold), which improves precision.
    Lower SNR (noisy signal) → lower threshold to avoid missing real
    boundaries that have modest gradient support.
    """
    if snr >= 5.0:
        return 0.12
    elif snr >= 3.5:
        return 0.09
    elif snr >= 2.5:
        return 0.06
    else:
        return 0.03


def _odd_window(requested: int, n: int) -> int:
    if n < 3:
        return 0
    win = min(int(requested), n if n % 2 == 1 else n - 1)
    if win < 3:
        return 0
    return win if win % 2 == 1 else win - 1


def compute_lr_contrast(x: np.ndarray, step_m: float, window_m: float) -> np.ndarray:
    """Per-sample left-right mean contrast on a 1-D signal.

    For each index i:
        contrast[i] = |mean(x[i-hw : i]) - mean(x[i+1 : i+1+hw])| / (std(combined) + eps)

    where hw = round(window_m / step_m).  Windows are truncated at the
    profile edges (no zero-padding).  The result is a non-negative array on
    the same grid as x that peaks at sharp layer transitions.

    Edge handling.  When either the left or right window contains fewer than
    ``min_samples = max(2, hw // 2)`` finite samples, the contrast is set to 0.
    This avoids the spurious edge inflation that occurs when an empty window's
    mean defaults to zero against a non-zero-mean signal (Ic ≈ 2-3) — without
    the guard, contrast at i=0 and i=N-1 are ~10× larger than the interior,
    biasing the hazard prior near the top and (more importantly) the bottom of
    every profile.

    Implementation uses cumulative sums for O(N) time.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    hw = max(1, int(round(window_m / step_m)))
    eps = 1e-9
    min_samples = max(2, hw // 2)

    cs = np.empty(n + 1)
    cs[0] = 0.0
    np.cumsum(x, out=cs[1:])
    cs2 = np.empty(n + 1)
    cs2[0] = 0.0
    np.cumsum(x ** 2, out=cs2[1:])

    def _window(lo: int, hi: int):
        cnt = hi - lo
        if cnt <= 0:
            return 0.0, 0.0, 0
        s = cs[hi] - cs[lo]
        s2 = cs2[hi] - cs2[lo]
        mean = s / cnt
        var = max(0.0, s2 / cnt - mean ** 2)
        return mean, var, cnt

    idx = np.arange(n)
    l_lo = np.maximum(0, idx - hw)
    l_hi = idx                          # exclusive
    r_lo = idx + 1
    r_hi = np.minimum(n, idx + 1 + hw)

    contrast = np.zeros(n)
    for i in range(n):
        ml, _, l_cnt = _window(int(l_lo[i]), int(l_hi[i]))
        mr, _, r_cnt = _window(int(r_lo[i]), int(r_hi[i]))
        if l_cnt < min_samples or r_cnt < min_samples:
            # Insufficient evidence on one side: no contrast claim at edges.
            continue
        c_lo, c_hi = int(l_lo[i]), int(r_hi[i])
        _, vc, _ = _window(c_lo, c_hi)
        contrast[i] = abs(ml - mr) / (vc ** 0.5 + eps)

    return contrast


def _safe_float(value: float) -> float:
    if np.isfinite(value):
        return float(value)
    return float("nan")


def _series_stats(values: np.ndarray, prefix: str, step_m: float | None = None) -> dict[str, float | int]:
    """Descriptive statistics for one profile signal.

    The output is intentionally wide: these columns are used for downstream
    error analysis, correlation checks, and identifying profiles where a
    method benefits from smoother, noisier, shorter, or more variable signals.
    """
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    out: dict[str, float | int] = {
        f"{prefix}_n": int(x.size),
        f"{prefix}_n_finite": int(finite.size),
        f"{prefix}_n_nonfinite": int(x.size - finite.size),
        f"{prefix}_finite_frac": float(finite.size / x.size) if x.size else float("nan"),
    }
    if finite.size == 0:
        for name in (
            "mean", "var", "std", "min", "q05", "q25", "median", "q75", "q95",
            "max", "range", "iqr", "mad", "cv", "skew", "kurtosis_excess",
        ):
            out[f"{prefix}_{name}"] = float("nan")
        if step_m is not None:
            for name in ("abs_grad_mean", "abs_grad_std", "abs_grad_q75", "abs_grad_q95", "roughness"):
                out[f"{prefix}_{name}"] = float("nan")
        return out

    mean = float(np.mean(finite))
    var = float(np.var(finite))
    std = float(np.sqrt(var))
    q05, q25, median, q75, q95 = np.percentile(finite, [5, 25, 50, 75, 95])
    centered = finite - mean
    mad = float(np.median(np.abs(finite - median)))
    if std > 0:
        z = centered / std
        skew = float(np.mean(z ** 3))
        kurtosis_excess = float(np.mean(z ** 4) - 3.0)
    else:
        skew = 0.0
        kurtosis_excess = -3.0

    out.update({
        f"{prefix}_mean": mean,
        f"{prefix}_var": var,
        f"{prefix}_std": std,
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_q05": float(q05),
        f"{prefix}_q25": float(q25),
        f"{prefix}_median": float(median),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q95": float(q95),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_range": float(np.max(finite) - np.min(finite)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_mad": mad,
        f"{prefix}_cv": _safe_float(std / abs(mean)) if mean != 0 else float("nan"),
        f"{prefix}_skew": skew,
        f"{prefix}_kurtosis_excess": kurtosis_excess,
    })

    if step_m is not None:
        clean = np.asarray(x, dtype=float).copy()
        bad = ~np.isfinite(clean)
        if bad.any() and finite.size:
            good = ~bad
            idx = np.arange(clean.size)
            clean[bad] = np.interp(idx[bad], idx[good], clean[good])
        if clean.size >= 2 and step_m > 0:
            grad = np.abs(np.gradient(clean, step_m))
            out.update({
                f"{prefix}_abs_grad_mean": float(np.mean(grad)),
                f"{prefix}_abs_grad_std": float(np.std(grad)),
                f"{prefix}_abs_grad_q75": float(np.quantile(grad, 0.75)),
                f"{prefix}_abs_grad_q95": float(np.quantile(grad, 0.95)),
                f"{prefix}_roughness": float(np.std(np.diff(clean))),
            })
        else:
            for name in ("abs_grad_mean", "abs_grad_std", "abs_grad_q75", "abs_grad_q95", "roughness"):
                out[f"{prefix}_{name}"] = float("nan")

    return out


def profile_statistics(profile: D.Profile) -> dict[str, float | int | str]:
    """Return per-profile descriptors useful for ML error analysis."""
    depth = np.asarray(profile.depth, dtype=float)
    stats: dict[str, float | int | str] = {
        "profile_n_samples": int(profile.n),
        "profile_n_point_ids": int(len(profile.point_ids)),
        "profile_depth_min_m": float(np.min(depth)) if depth.size else float("nan"),
        "profile_depth_max_m": float(np.max(depth)) if depth.size else float("nan"),
        "profile_depth_span_m": float(np.max(depth) - np.min(depth)) if depth.size else float("nan"),
        "profile_step_m": float(profile.step),
        "profile_x": float(profile.x) if profile.x is not None else float("nan"),
        "profile_y": float(profile.y) if profile.y is not None else float("nan"),
        "profile_z": float(profile.z) if profile.z is not None else float("nan"),
        "profile_bathymetry": float(profile.bathymetry) if profile.bathymetry is not None else float("nan"),
    }
    if depth.size >= 2:
        dz = np.diff(depth)
        stats.update({
            "profile_step_mean_m": float(np.mean(dz)),
            "profile_step_std_m": float(np.std(dz)),
            "profile_step_min_m": float(np.min(dz)),
            "profile_step_max_m": float(np.max(dz)),
        })
    else:
        stats.update({
            "profile_step_mean_m": float("nan"),
            "profile_step_std_m": float("nan"),
            "profile_step_min_m": float("nan"),
            "profile_step_max_m": float("nan"),
        })

    for feature_name in sorted(profile.features):
        stats.update(_series_stats(profile.features[feature_name], f"profile_{feature_name}", step_m=profile.step))
    return stats


def estimate_adaptive_kappa(profile: D.Profile,
                            feature: str,
                            cfg: AdaptiveKappaConfig,
                            floor_m: float) -> AdaptiveKappaStats:
    """Estimate a profile-specific layer-thickness prior from the signal only.

    The rule blends:
    - a conservative base thickness prior
    - median spacing between strong gradient peaks
    - a variation scale IQR(signal) / Q75(|d signal / dz|)

    All terms are clipped to a geotechnically plausible range before blending.
    """
    step_m = profile.step
    x = np.asarray(profile.features[feature], dtype=float)
    if x.size < 3:
        thickness_m = max(floor_m, cfg.base_thickness_m)
        return AdaptiveKappaStats(
            target=profile.target,
            adaptive_thickness_m=thickness_m,
            adaptive_kappa_samples=thickness_m / step_m,
            peak_spacing_m=thickness_m,
            variation_scale_m=thickness_m,
            strong_peak_threshold=0.0,
            n_strong_peaks=0,
            signal_iqr=0.0,
            gradient_q75=0.0,
        )

    sw = _odd_window(cfg.smooth_window, x.size)
    if sw >= 3:
        xs = savgol_filter(x, sw, min(3, sw - 1))
    else:
        xs = x

    grad = np.abs(np.gradient(xs, step_m))
    peaks = np.flatnonzero((grad[1:-1] > grad[:-2]) & (grad[1:-1] >= grad[2:])) + 1
    peak_threshold = float(np.quantile(grad, cfg.strong_peak_quantile))
    strong_peaks = peaks[grad[peaks] >= peak_threshold]

    min_allowed = max(floor_m, cfg.min_thickness_m, step_m)
    max_allowed = max(min_allowed, cfg.max_thickness_m)

    if strong_peaks.size >= 2:
        peak_spacing_m = float(np.median(np.diff(profile.depth[strong_peaks])))
    else:
        peak_spacing_m = cfg.base_thickness_m
    peak_spacing_m = float(np.clip(peak_spacing_m, min_allowed, max_allowed))

    signal_iqr = float(np.subtract(*np.percentile(xs, [75, 25])))
    gradient_q75 = float(np.quantile(grad, 0.75))
    if gradient_q75 > 0:
        variation_scale_m = signal_iqr / gradient_q75
    else:
        variation_scale_m = cfg.base_thickness_m
    variation_scale_m = float(np.clip(variation_scale_m, min_allowed, max_allowed))

    thickness_m = (
        cfg.blend_base * cfg.base_thickness_m
        + cfg.blend_peak_spacing * peak_spacing_m
        + cfg.blend_variation_scale * variation_scale_m
    )
    thickness_m = float(np.clip(thickness_m, min_allowed, max_allowed))

    return AdaptiveKappaStats(
        target=profile.target,
        adaptive_thickness_m=thickness_m,
        adaptive_kappa_samples=thickness_m / step_m,
        peak_spacing_m=peak_spacing_m,
        variation_scale_m=variation_scale_m,
        strong_peak_threshold=peak_threshold,
        n_strong_peaks=int(strong_peaks.size),
        signal_iqr=signal_iqr,
        gradient_q75=gradient_q75,
    )


def build_composed_hazard(profile: D.Profile,
                          cfg: PipelineConfig,
                          spatial_density: np.ndarray | None = None
                          ) -> tuple[HazardFn, dict[str, float | int | str]]:
    """Build one composed hazard function for a single profile."""
    diag: dict[str, float | int | str] = {
        "base_mode": cfg.base_mode,
        "adaptive_enabled": int(cfg.adaptive.enabled),
        "min_thickness_enabled": int(cfg.use_min_thickness),
        "contrast_enabled": int(cfg.use_contrast or cfg.use_auto_contrast),
        "spatial_enabled": int(cfg.use_spatial),
    }

    floor_m = cfg.min_thickness_m if cfg.use_min_thickness else profile.step
    adaptive_stats: AdaptiveKappaStats | None = None
    if cfg.adaptive.enabled:
        adaptive_stats = estimate_adaptive_kappa(profile, cfg.feature, cfg.adaptive, floor_m=floor_m)
        diag.update({
            "adaptive_thickness_m": adaptive_stats.adaptive_thickness_m,
            "adaptive_kappa_samples": adaptive_stats.adaptive_kappa_samples,
            "peak_spacing_m": adaptive_stats.peak_spacing_m,
            "variation_scale_m": adaptive_stats.variation_scale_m,
            "strong_peak_threshold": adaptive_stats.strong_peak_threshold,
            "n_strong_peaks": adaptive_stats.n_strong_peaks,
            "signal_iqr": adaptive_stats.signal_iqr,
            "gradient_q75": adaptive_stats.gradient_q75,
        })

    if cfg.base_mode == "const":
        if adaptive_stats is not None:
            kappa_samples = adaptive_stats.adaptive_kappa_samples
        else:
            kappa_samples = cfg.hazard_kappa if cfg.hazard_kappa is not None else float(profile.n)
        base = const_hazard(kappa_samples)
        diag["hazard_kappa_samples"] = float(kappa_samples)
        diag["hazard_thickness_m"] = float(kappa_samples * profile.step)
    elif cfg.base_mode == "depth":
        if adaptive_stats is not None:
            kappa0_m = adaptive_stats.adaptive_thickness_m
        else:
            kappa0_m = cfg.depth_kappa0_m
        base = depth_aware_hazard(
            profile.depth,
            kappa0_m=kappa0_m,
            growth_m_per_m=cfg.depth_growth,
            step_m=profile.step,
        )
        diag["depth_kappa0_m"] = float(kappa0_m)
        diag["depth_growth_m_per_m"] = float(cfg.depth_growth)
    else:
        raise ValueError(f"Unknown base_mode: {cfg.base_mode}")

    if cfg.use_min_thickness:
        r_min = int(round(cfg.min_thickness_m / profile.step))
        base = min_thickness_hazard(base, r_min)
        diag["min_run_samples"] = int(r_min)

    if cfg.use_auto_contrast:
        x_feat = np.asarray(profile.features[cfg.feature], dtype=float)
        auto_c = estimate_auto_contrast_params(
            x_feat, profile.step, adaptive_stats, base_window_m=cfg.contrast_window_m
        )
        diag["auto_contrast"] = int(auto_c["use_contrast"])
        diag["auto_snr"] = float(auto_c["snr"])
        diag["auto_local_noise"] = float(auto_c["local_noise"])
        if auto_c["use_contrast"]:
            contrast = compute_lr_contrast(x_feat, profile.step, float(auto_c["contrast_window_m"]))
            base = contrast_prior_hazard(
                base, contrast_signal=contrast, strength=float(auto_c["contrast_strength"])
            )
            diag["contrast_window_m"] = float(auto_c["contrast_window_m"])
            diag["contrast_strength"] = float(auto_c["contrast_strength"])
            diag["contrast_mean"] = float(np.mean(contrast))
            diag["contrast_max"] = float(np.max(contrast))
    elif cfg.use_contrast:
        x = np.asarray(profile.features[cfg.feature], dtype=float)
        contrast = compute_lr_contrast(x, profile.step, cfg.contrast_window_m)
        base = contrast_prior_hazard(base, contrast_signal=contrast, strength=cfg.contrast_strength)
        diag["contrast_window_m"] = float(cfg.contrast_window_m)
        diag["contrast_strength"] = float(cfg.contrast_strength)
        diag["contrast_mean"] = float(np.mean(contrast))
        diag["contrast_max"] = float(np.max(contrast))

    if cfg.use_spatial:
        if spatial_density is None:
            raise ValueError(f"Spatial pipeline {cfg.name!r} requires a prior density.")
        base = spatial_prior_hazard(base, prior_density=spatial_density, strength=cfg.spatial_strength)
        diag["spatial_density_mean"] = float(np.mean(spatial_density))
        diag["spatial_density_max"] = float(np.max(spatial_density))

    return base, diag


def run_profile(profile: D.Profile,
                cfg: PipelineConfig,
                spatial_density: np.ndarray | None = None
                ) -> tuple[BOCPDResult, np.ndarray, dict[str, float | int | str]]:
    x = profile.features[cfg.feature]
    if cfg.mu0_from_signal:
        x_finite = np.asarray(x, dtype=float)
        x_finite = x_finite[np.isfinite(x_finite)]
        mu0 = float(np.median(x_finite)) if x_finite.size else cfg.mu0
    else:
        mu0 = cfg.mu0
    prior = NIGPrior(mu0=mu0, kappa=cfg.kappa_prior, alpha=cfg.alpha, beta=cfg.beta)
    hazard, diag = build_composed_hazard(profile, cfg, spatial_density=spatial_density)
    diag["nig_mu0"] = float(mu0)

    # Adaptive pre-smoothing: denoise x for BOCPD detection while keeping the
    # raw x for gradient refinement (preserves precise boundary localisation).
    x_raw = np.asarray(x, dtype=float)
    if cfg.use_adaptive_smooth:
        roughness = float(np.std(np.diff(x_raw))) if x_raw.size >= 2 else 0.0
        x_bocpd, smooth_lam, smooth_applied = _adaptive_smooth_signal(
            x_raw, roughness, profile.step,
            roughness_threshold=cfg.smooth_roughness_threshold,
            roughness_max=cfg.smooth_roughness_max,
            lam_min=cfg.smooth_lam_min,
            lam_max=cfg.smooth_lam_max,
            method=cfg.smooth_method,
        )
        diag["smooth_applied"] = int(smooth_applied)
        diag["smooth_lambda"] = float(smooth_lam)
        diag["smooth_roughness"] = float(roughness)
    else:
        x_bocpd = x_raw
        diag["smooth_applied"] = 0
        diag["smooth_lambda"] = 0.0
        diag["smooth_roughness"] = float(np.std(np.diff(x_raw))) if x_raw.size >= 2 else 0.0

    factory = lambda capacity=profile.n + 1: NIGPredictor(prior, capacity=capacity)
    result = run_bocpd(x_bocpd, factory, hazard=hazard, full_posterior=True)

    min_run = int(round(cfg.min_thickness_m / profile.step)) if cfg.use_min_thickness else 1
    cps_idx = map_changepoints_paper(result.run_length_posterior, min_run_length=max(1, min_run))

    if cfg.refine and cps_idx.size:
        # Auto-calibrate drop-quantile from per-profile SNR when requested.
        if cfg.auto_refine_drop_quantile:
            snr = float(diag.get("auto_snr", 0.0))
            if snr == 0.0:
                # SNR not yet computed (auto_contrast disabled): derive it now and
                # store so the analysis output can display it.
                x_arr = np.asarray(profile.features[cfg.feature], dtype=float)
                local_noise = _local_noise_estimate(x_arr, profile.step)
                finite = x_arr[np.isfinite(x_arr)]
                iqr = float(np.subtract(*np.percentile(finite, [75, 25]))) if finite.size > 1 else 1e-4
                snr = max(iqr, 1e-4) / max(local_noise, 1e-4)
                diag["auto_snr"] = float(snr)
                diag["auto_local_noise"] = float(local_noise)
            drop_q: float | None = estimate_auto_refine_drop_quantile(snr)
            diag["auto_drop_quantile"] = float(drop_q)
        else:
            drop_q = cfg.refine_drop_below_quantile
        cps_idx = refine_cps_to_gradient(
            x,
            cps_idx,
            profile.step,
            search_radius_m=cfg.refine_search_m,
            smooth_window=cfg.refine_smooth_window,
            min_sep_m=(cfg.min_thickness_m if cfg.use_min_thickness else 0.0),
            strength_ratio=cfg.refine_strength_ratio,
            drop_below_quantile=drop_q,
        )

    diag["n_pred"] = int(cps_idx.size)
    return result, cps_idx, diag


def filter_to_located(profiles: list[D.Profile]) -> tuple[list[D.Profile], list[D.Profile]]:
    def has_location(profile: D.Profile) -> bool:
        return (
            profile.x is not None
            and profile.y is not None
            and np.isfinite(profile.x)
            and np.isfinite(profile.y)
        )

    located = [p for p in profiles if has_location(p)]
    dropped = [p for p in profiles if not has_location(p)]
    return located, dropped


def run_pipeline(profiles: list[D.Profile],
                 cfg: PipelineConfig,
                 pred_source: dict[str, np.ndarray] | None = None
                 ) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    xy = {
        p.target: (p.x, p.y)
        for p in profiles
        if p.x is not None and p.y is not None and np.isfinite(p.x) and np.isfinite(p.y)
    }
    preds: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for profile in profiles:
        profile_stats = profile_statistics(profile)
        density = None
        if cfg.use_spatial:
            if pred_source is None:
                raise ValueError(f"Pipeline {cfg.name!r} requires source predictions for the spatial prior.")
            density = build_neighbor_cp_density(
                profile.target,
                pred_source,
                xy,
                profile.depth,
                bandwidth_m=cfg.spatial_bandwidth_m,
                k_neighbors=cfg.spatial_k_neighbors,
            )
        _, cps_idx, diag = run_profile(profile, cfg, spatial_density=density)
        preds[profile.target] = profile.depth[cps_idx]
        rows.append({"Target": profile.target, **profile_stats, **diag})
    diag_df = pd.DataFrame(rows).sort_values("Target").reset_index(drop=True)
    return preds, diag_df


def print_summary(label: str, gs: dict[str, float], elapsed: float) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {label}  [{elapsed:.1f}s]")
    print(f"{'─' * 70}")
    for tol in TOLERANCES:
        print(
            f"  @{tol:.1f}m   "
            f"F1={gs[f'pooled_f1_{tol}']:.3f}  "
            f"P={gs[f'pooled_precision_{tol}']:.3f}  "
            f"R={gs[f'pooled_recall_{tol}']:.3f}  "
            f"mean-F1={gs[f'mean_f1_{tol}']:.3f}"
        )


def _diagnose_profile(row: pd.Series, f1_tol: float = 1.0) -> list[str]:
    """Return a list of human-readable diagnostic strings for one profile.

    Diagnoses are derived *only* from the profile's own CPT statistics and the
    model's prediction counts — no ground truth is used for diagnosis.
    """
    diagnoses: list[str] = []

    n_pred = int(row.get("n_pred", 0))
    n_true = int(row.get("n_true", 0))
    precision = float(row.get(f"precision_{f1_tol}", 0.0))
    recall = float(row.get(f"recall_{f1_tol}", 0.0))

    # Over / under segmentation
    if n_true > 0:
        ratio = n_pred / n_true
        if ratio > 1.5:
            diagnoses.append(
                f"Over-segmented ({n_pred} pred vs {n_true} true, {ratio:.1f}×): "
                "too many false boundaries → precision loss"
            )
        elif ratio < 0.55:
            diagnoses.append(
                f"Under-segmented ({n_pred} pred vs {n_true} true, {ratio:.1f}×): "
                "layers missed → recall loss"
            )
    elif n_pred > 2:
        diagnoses.append(f"No ground-truth boundaries but {n_pred} predicted")

    # Precision vs recall asymmetry
    if precision < 0.45 and recall > 0.65 and not any("Over-seg" in d for d in diagnoses):
        diagnoses.append("Precision dominated: many spurious boundaries without matching truth")
    if recall < 0.45 and precision > 0.65 and not any("Under-seg" in d for d in diagnoses):
        diagnoses.append("Recall dominated: model misses many true layer boundaries")

    # Signal contrast
    ic_iqr = float(row.get("profile_Ic_iqr", float("nan")))
    if np.isfinite(ic_iqr) and ic_iqr < 0.35:
        diagnoses.append(
            f"Low Ic contrast (IQR={ic_iqr:.3f}): layers have similar Ic values, "
            "making boundaries hard to detect"
        )

    # Signal roughness — lower threshold because even moderate roughness (0.09+)
    # visibly increases false positives.
    ic_roughness = float(row.get("profile_Ic_roughness", float("nan")))
    if np.isfinite(ic_roughness) and ic_roughness > 0.09:
        diagnoses.append(
            f"Rough Ic signal (roughness={ic_roughness:.3f}): within-layer noise "
            "can create spurious detected boundaries"
        )

    # SNR (auto-contrast diagnostic)
    snr = float(row.get("auto_snr", float("nan")))
    if np.isfinite(snr) and snr < 2.5:
        diagnoses.append(
            f"Low signal-to-noise ratio (SNR={snr:.1f}): contrast hazard disabled "
            "for this profile because the signal is too noisy to guide it"
        )

    # Adaptive kappa — if kappa is at the minimum floor, flag that the model
    # detects fine-scale geology that may exceed ground-truth annotation grain.
    kappa_m = float(row.get("adaptive_thickness_m", float("nan")))
    if np.isfinite(kappa_m):
        if kappa_m > 3.0 and recall < 0.55:
            diagnoses.append(
                f"Thick kappa prior ({kappa_m:.2f} m): the model expects thick layers "
                "and may suppress detection of thinner real ones"
            )
        elif kappa_m <= 0.60 and precision < 0.60:
            diagnoses.append(
                f"Kappa at minimum floor ({kappa_m:.2f} m): the signal has many "
                "thin-layer transitions (~0.5 m scale) that the model faithfully "
                "detects but that may be finer than the ground-truth annotation "
                "grain (major strata boundaries)"
            )

    # Very short profile
    depth_span = float(row.get("profile_depth_span_m", float("nan")))
    if np.isfinite(depth_span) and depth_span < 6.0:
        diagnoses.append(
            f"Short profile ({depth_span:.1f} m depth): fewer data points constrain "
            "the model less and can inflate noise effects"
        )

    if not diagnoses:
        f1 = float(row.get(f"f1_{f1_tol}", 0.0))
        if f1 >= 0.75:
            diagnoses.append("Well-performing profile — signal and model are well matched")
        else:
            diagnoses.append(
                "Moderate performance — no dominant single cause from available statistics"
            )
    return diagnoses


def print_performance_analysis(
    eval_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    f1_tol: float = 1.0,
    n_worst: int = 8,
) -> None:
    """Print a per-profile diagnostic analysis explaining F1 variation.

    Two sections:
    1. Spearman correlations between signal statistics and F1 (exploratory).
    2. Per-profile table sorted by F1, with human-readable root-cause
       diagnoses for the worst-performing and best-performing profiles.

    All analysis is based entirely on the CPT signal statistics and model
    diagnostics — ground truth is used only for the F1 values themselves.
    """
    analysis = eval_df.merge(diag_df, on="Target", how="left", suffixes=("", "_diag"))
    f1_col = f"f1_{f1_tol}"
    if f1_col not in analysis.columns:
        print(f"\n  [analysis] Column {f1_col!r} not found in evaluation data.")
        return

    n_profiles = len(analysis)
    print(f"\n{'═' * 70}")
    print(f"  PERFORMANCE ANALYSIS  (N={n_profiles} profiles, F1@{f1_tol:.1f}m)")
    print(f"{'═' * 70}")

    # --- Spearman correlation table (exploratory) ---
    candidate_signal_cols = [
        "profile_Ic_iqr", "profile_Ic_roughness", "profile_Ic_abs_grad_mean",
        "profile_Ic_abs_grad_q75", "profile_Ic_std", "profile_Ic_range",
        "profile_Ic_cv", "profile_depth_span_m", "profile_n_samples",
        "adaptive_thickness_m", "adaptive_kappa_samples", "n_strong_peaks",
        "auto_snr", "auto_local_noise", "auto_drop_quantile",
        "contrast_strength",
        "smooth_applied", "smooth_lambda",
    ]
    available = [c for c in candidate_signal_cols if c in analysis.columns]
    if available:
        f1_vals = analysis[f1_col].astype(float)
        corrs: list[tuple[float, str]] = []
        for col in available:
            col_vals = analysis[col].astype(float)
            if col_vals.dropna().nunique() < 3:
                continue
            rho = float(f1_vals.corr(col_vals, method="spearman"))
            if np.isfinite(rho):
                corrs.append((rho, col))
        corrs.sort(key=lambda t: abs(t[0]), reverse=True)

        print(
            f"\n  Spearman ρ with F1@{f1_tol:.1f}m  "
            "(exploratory only — n is small so treat with caution)"
        )
        print(f"  {'Statistic':<40s}  {'ρ':>6s}  {'Direction'}")
        print(f"  {'-' * 40}  {'-' * 6}  {'-' * 25}")
        for rho, col in corrs[:10]:
            direction = "↑ better with more" if rho > 0 else "↑ better with less"
            print(f"  {col:<40s}  {rho:+.3f}  {direction}")

    # --- Per-profile diagnostic table ---
    sorted_df = analysis.sort_values(f1_col, ascending=True).reset_index(drop=True)

    def _fmt_row(row: pd.Series) -> str:
        tgt = str(row["Target"])
        f1 = float(row.get(f1_col, float("nan")))
        prec = float(row.get(f"precision_{f1_tol}", float("nan")))
        rec = float(row.get(f"recall_{f1_tol}", float("nan")))
        n_p = int(row.get("n_pred", -1))
        n_t = int(row.get("n_true", -1))
        iqr = float(row.get("profile_Ic_iqr", float("nan")))
        rough = float(row.get("profile_Ic_roughness", float("nan")))
        kap = float(row.get("adaptive_thickness_m", float("nan")))
        snr_v = float(row.get("auto_snr", float("nan")))
        return (
            f"  {tgt:<12s}  F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  "
            f"pred={n_p:>3d}  true={n_t:>3d}  "
            f"IcIQR={iqr:.2f}  rough={rough:.3f}  κ={kap:.2f}m  SNR={snr_v:.1f}"
        )

    print(f"\n  {'─' * 68}")
    print(f"  Worst {n_worst} profiles (most room for improvement)")
    print(f"  {'─' * 68}")
    for _, row in sorted_df.head(n_worst).iterrows():
        print(_fmt_row(row))
        for d in _diagnose_profile(row, f1_tol):
            print(f"    → {d}")

    best_n = min(5, max(0, n_profiles - n_worst))
    if best_n:
        print(f"\n  {'─' * 68}")
        print(f"  Best {best_n} profiles (reference for what works well)")
        print(f"  {'─' * 68}")
        for _, row in sorted_df.tail(best_n).iloc[::-1].iterrows():
            print(_fmt_row(row))
            for d in _diagnose_profile(row, f1_tol):
                print(f"    → {d}")

    # --- Summary statistics ---
    f1_series = sorted_df[f1_col].astype(float)
    n_low = int((f1_series < 0.60).sum())
    n_med = int(((f1_series >= 0.60) & (f1_series < 0.80)).sum())
    n_high = int((f1_series >= 0.80).sum())
    print(f"\n  F1@{f1_tol:.1f}m distribution:")
    print(f"    <0.60  (hard):   {n_low:>3d} profiles")
    print(f"    0.60–0.80 (OK):  {n_med:>3d} profiles")
    print(f"    ≥0.80  (good):   {n_high:>3d} profiles")
    print(f"    median={f1_series.median():.3f}  mean={f1_series.mean():.3f}  "
          f"std={f1_series.std():.3f}")
    print(f"{'═' * 70}")




def print_profile_metrics(eval_df: pd.DataFrame, sort_tol: float = 1.0) -> None:
    if eval_df.empty:
        print("\n  Per-profile metrics: no evaluated profiles.")
        return

    sort_key = f"f1_{sort_tol}"
    if sort_key not in eval_df.columns:
        sort_tol = TOLERANCES[0]
        sort_key = f"f1_{sort_tol}"

    cols = ["Target", "n_pred", "n_true"]
    rename = {"n_pred": "pred", "n_true": "true"}
    metric_cols = []
    for tol in TOLERANCES:
        for metric, label in (("f1", "F1"), ("precision", "P"), ("recall", "R")):
            col = f"{metric}_{tol}"
            cols.append(col)
            metric_cols.append(col)
            rename[col] = f"{label}@{tol:.1f}"

    table = (
        eval_df[cols]
        .sort_values([sort_key, "Target"], ascending=[True, True])
        .rename(columns=rename)
        .reset_index(drop=True)
    )
    fmt = {rename[col]: "{:.3f}".format for col in metric_cols}

    print(f"\n  Per-profile metrics (worst first by F1@{sort_tol:.1f}m)")
    print(table.to_string(index=False, formatters=fmt))


def print_kappa_summary(diag_df: pd.DataFrame) -> None:
    if "adaptive_kappa_samples" not in diag_df.columns:
        return
    kappa = diag_df["adaptive_kappa_samples"].astype(float)
    thick = diag_df["adaptive_thickness_m"].astype(float)
    print(
        "  adaptive κ: "
        f"median={kappa.median():.1f} samples, "
        f"range=[{kappa.min():.1f}, {kappa.max():.1f}], "
        f"median thickness={thick.median():.2f} m"
    )


def print_pipeline_options(cfg: PipelineConfig) -> None:
    print(f"\n{cfg.label}")
    print(f"  Base hazard: {'constant' if cfg.base_mode == 'const' else 'depth-aware'}")
    if cfg.base_mode == "const":
        if cfg.adaptive.enabled:
            print("      - kappa: adaptive per profile")
        elif cfg.hazard_kappa is None:
            print("      - kappa: N for that profile, because hazard_kappa=None")
        else:
            print(f"      - kappa: {cfg.hazard_kappa}")
    else:
        if cfg.adaptive.enabled:
            print("      - kappa0_m: adaptive per profile")
        else:
            print(f"      - kappa0_m: {cfg.depth_kappa0_m}")
        print(f"      - depth growth: {cfg.depth_growth} m per metre depth")
    print(f"      - Adaptive kappa: {'yes' if cfg.adaptive.enabled else 'no'}")
    print(f"      - Min thickness: {'yes' if cfg.use_min_thickness else 'no'}")
    if cfg.use_min_thickness:
        print(f"      - min_thickness_m: {cfg.min_thickness_m}")
    if cfg.use_auto_contrast:
        print("      - L-R contrast prior: auto (per-profile SNR calibration)")
    else:
        print(f"      - L-R contrast prior: {'yes' if cfg.use_contrast else 'no'}")
        if cfg.use_contrast:
            print(f"      - contrast_window_m: {cfg.contrast_window_m}")
            print(f"      - contrast_strength: {cfg.contrast_strength}")
    print(f"      - Spatial prior: {'yes' if cfg.use_spatial else 'no'}")
    if cfg.use_spatial:
        print(f"      - spatial bandwidth: {cfg.spatial_bandwidth_m} m")
        print(f"      - spatial neighbors: {cfg.spatial_k_neighbors}")
        print(f"      - spatial strength: {cfg.spatial_strength}")
    print(f"      - Refinement: {'yes' if cfg.refine else 'no'}")
    if cfg.refine:
        print(f"      - refine_search_m: {cfg.refine_search_m}")
        if cfg.auto_refine_drop_quantile:
            print("      - drop-quantile filter: auto (per-profile SNR)")
    if cfg.use_adaptive_smooth:
        print(
            f"      - Adaptive pre-smooth: yes ({cfg.smooth_method.upper()},"
            f" threshold={cfg.smooth_roughness_threshold:.3f},"
            f" λ=[{cfg.smooth_lam_min:.2f},{cfg.smooth_lam_max:.2f}])"
        )
    else:
        print("      - Adaptive pre-smooth: no")


def print_delta_vs_baseline(baseline_label: str,
                            baseline_summary: dict[str, float],
                            candidate_label: str,
                            candidate_summary: dict[str, float]) -> None:
    print(f"\n{'─' * 70}")
    print(f"  Delta vs {baseline_label}")
    print(f"{'─' * 70}")
    vals = []
    for tol in TOLERANCES:
        delta = candidate_summary[f"pooled_f1_{tol}"] - baseline_summary[f"pooled_f1_{tol}"]
        vals.append(f"@{tol:.1f}m {delta:+.3f}")
    print(f"  {candidate_label:16s}  " + "  ".join(vals))


def save_outputs(outdir: Path,
                 selected: list[str],
                 preds_by_name: dict[str, dict[str, np.ndarray]],
                 eval_by_name: dict[str, pd.DataFrame],
                 diag_by_name: dict[str, pd.DataFrame],
                 summary_by_name: dict[str, dict[str, float]]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    bounds_dir = outdir / "predicted_boundary_depths"
    bounds_dir.mkdir(exist_ok=True)

    summary = pd.DataFrame(summary_by_name).T
    summary.index.name = "method"
    summary.to_csv(outdir / "main__summary.csv")

    for name in selected:
        eval_by_name[name].to_csv(outdir / f"main__{name}.csv", index=False)
        diag_by_name[name].to_csv(outdir / f"main__{name}__diagnostics.csv", index=False)
        analysis = eval_by_name[name].merge(diag_by_name[name], on="Target", how="left", suffixes=("", "_diag"))
        analysis.to_csv(outdir / f"main__{name}__analysis.csv", index=False)
        rows = [
            {"Target": target, "Boundary_depth": depth}
            for target, depths in sorted(preds_by_name[name].items())
            for depth in depths
        ]
        pd.DataFrame(rows).to_csv(bounds_dir / f"{name}_predicted_boundaries.csv", index=False)


def make_baseline_config() -> PipelineConfig:
    return PipelineConfig(
        name="baseline",
        label="BASELINE",
        base_mode="const",
        hazard_kappa=None,
        use_min_thickness=False,
        refine=False,
        use_spatial=False,
    )


def make_candidate_config() -> PipelineConfig:
    adaptive = AdaptiveKappaConfig(
        enabled=CANDIDATE_USE_ADAPTIVE_KAPPA,
        base_thickness_m=ADAPTIVE_BASE_THICKNESS_M if CANDIDATE_BASE_MODE == "const" else max(1.0, ADAPTIVE_BASE_THICKNESS_M),
        min_thickness_m=ADAPTIVE_MIN_THICKNESS_M,
        max_thickness_m=ADAPTIVE_MAX_THICKNESS_M if CANDIDATE_BASE_MODE == "const" else max(ADAPTIVE_MAX_THICKNESS_M, 3.0),
        smooth_window=ADAPTIVE_SMOOTH_WINDOW,
        strong_peak_quantile=ADAPTIVE_STRONG_PEAK_QUANTILE,
        blend_base=ADAPTIVE_BLEND_BASE,
        blend_peak_spacing=ADAPTIVE_BLEND_PEAK_SPACING,
        blend_variation_scale=ADAPTIVE_BLEND_VARIATION_SCALE,
    )
    return PipelineConfig(
        name="candidate",
        label="CANDIDATE",
        mu0_from_signal=CANDIDATE_USE_PROFILE_MU0,
        base_mode=CANDIDATE_BASE_MODE,
        hazard_kappa=CANDIDATE_HAZARD_KAPPA,
        use_min_thickness=CANDIDATE_USE_MIN_THICKNESS,
        min_thickness_m=CANDIDATE_MIN_THICKNESS_M,
        depth_kappa0_m=CANDIDATE_DEPTH_KAPPA0_M,
        depth_growth=CANDIDATE_DEPTH_GROWTH,
        adaptive=adaptive,
        refine=CANDIDATE_USE_REFINEMENT,
        refine_search_m=CANDIDATE_REFINE_SEARCH_M,
        refine_smooth_window=CANDIDATE_REFINE_SMOOTH_WINDOW,
        refine_strength_ratio=CANDIDATE_REFINE_STRENGTH_RATIO,
        refine_drop_below_quantile=CANDIDATE_REFINE_DROP_BELOW_QUANTILE,
        use_contrast=CANDIDATE_USE_CONTRAST,
        contrast_window_m=CANDIDATE_CONTRAST_WINDOW_M,
        contrast_strength=CANDIDATE_CONTRAST_STRENGTH,
        use_auto_contrast=CANDIDATE_USE_AUTO_CONTRAST,
        auto_refine_drop_quantile=CANDIDATE_AUTO_REFINE_DROP_QUANTILE,
        use_spatial=CANDIDATE_USE_SPATIAL,
        spatial_bandwidth_m=CANDIDATE_SPATIAL_BANDWIDTH_M,
        spatial_k_neighbors=CANDIDATE_SPATIAL_K_NEIGHBORS,
        spatial_strength=CANDIDATE_SPATIAL_STRENGTH,
        use_adaptive_smooth=CANDIDATE_USE_ADAPTIVE_SMOOTH,
        smooth_method=CANDIDATE_SMOOTH_METHOD,
        smooth_roughness_threshold=CANDIDATE_SMOOTH_ROUGHNESS_THRESHOLD,
        smooth_roughness_max=CANDIDATE_SMOOTH_ROUGHNESS_MAX,
        smooth_lam_min=CANDIDATE_SMOOTH_LAM_MIN,
        smooth_lam_max=CANDIDATE_SMOOTH_LAM_MAX,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Composable BOCPD runner with adaptive per-profile kappa.")
    ap.add_argument(
        "--baseline",
        action=argparse.BooleanOptionalAction,
        default=RUN_BASELINE,
        help="Run the fixed paper baseline.",
    )
    ap.add_argument(
        "--candidate",
        action=argparse.BooleanOptionalAction,
        default=RUN_CANDIDATE,
        help="Run the editable candidate defined at the top of the file.",
    )
    ap.add_argument(
        "--targets",
        nargs="+",
        metavar="T",
        default=(DEFAULT_INCLUDE_TARGETS or None),
        help="Run only these Target IDs.",
    )
    ap.add_argument(
        "--exclude",
        nargs="+",
        metavar="T",
        default=(DEFAULT_EXCLUDE_TARGETS or None),
        help="Skip these Target IDs.",
    )
    ap.add_argument(
        "--subset",
        type=int,
        default=DEFAULT_SUBSET,
        help="Run only the first N profiles.",
    )
    ap.add_argument(
        "--show-kappa",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_KAPPA,
        help="Print adaptive-kappa summaries.",
    )
    ap.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_OUTPUTS,
        help="Write CSV outputs to --outdir.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "main",
        help="Directory used when --save is enabled.",
    )
    ap.add_argument(
        "--dataset-prefix",
        default="",
        help="Dataset file prefix, e.g. '' for original data or '2_' for the second dataset.",
    )
    ap.add_argument(
        "--contrast",
        action=argparse.BooleanOptionalAction,
        default=CANDIDATE_USE_CONTRAST,
        help="Enable left-right window contrast hazard prior.",
    )
    ap.add_argument(
        "--contrast-window-m",
        type=float,
        default=CANDIDATE_CONTRAST_WINDOW_M,
        dest="contrast_window_m",
        help="One-sided window width for L-R contrast (metres).",
    )
    ap.add_argument(
        "--contrast-strength",
        type=float,
        default=CANDIDATE_CONTRAST_STRENGTH,
        dest="contrast_strength",
        help="Hazard modulation strength for L-R contrast (0=off, 1=default).",
    )
    ap.add_argument(
        "--analyze",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_ANALYSIS,
        help=(
            "Print per-profile performance analysis after the candidate run, "
            "including Spearman correlations and root-cause diagnostics."
        ),
    )
    args = ap.parse_args()
    if not args.baseline and not args.candidate:
        sys.exit("Both baseline and candidate are disabled. Enable at least one.")

    baseline_cfg = make_baseline_config()
    candidate_cfg = make_candidate_config()
    if args.candidate:
        candidate_cfg = replace(
            candidate_cfg,
            use_contrast=args.contrast,
            contrast_window_m=args.contrast_window_m,
            contrast_strength=args.contrast_strength,
        )

    if args.baseline:
        print_pipeline_options(baseline_cfg)
    if args.candidate:
        print_pipeline_options(candidate_cfg)

    D.configure_dataset(prefix=args.dataset_prefix)
    profiles = D.load_profiles()
    if args.targets:
        wanted = set(args.targets)
        profiles = [p for p in profiles if p.target in wanted]
        if not profiles:
            sys.exit(f"No profiles found for targets: {args.targets}")
    elif args.subset is not None:
        profiles = profiles[: args.subset]

    if args.exclude:
        skip = set(args.exclude)
        profiles = [p for p in profiles if p.target not in skip]
        if not profiles:
            sys.exit("All profiles were excluded.")

    if args.candidate and candidate_cfg.use_spatial:
        profiles, dropped = filter_to_located(profiles)
        if not profiles:
            sys.exit("No geo-located profiles remain after filtering.")
        if dropped:
            print(
                f"Dropped {len(dropped)} profile(s) with no location data: "
                + ", ".join(p.target for p in dropped)
            )

    print(f"Running on {len(profiles)} profile(s).")

    strata = D.load_strata()
    truth = {p.target: D.true_boundaries_for(p.target, strata) for p in profiles}

    preds_by_name: dict[str, dict[str, np.ndarray]] = {}
    diag_by_name: dict[str, pd.DataFrame] = {}
    eval_by_name: dict[str, pd.DataFrame] = {}
    summary_by_name: dict[str, dict[str, float]] = {}
    if args.baseline:
        t0 = time.time()
        preds, diag_df = run_pipeline(profiles, baseline_cfg)
        elapsed = time.time() - t0

        eval_df = evaluate_run(preds, truth, TOLERANCES)
        summary = global_summary(eval_df, TOLERANCES)

        preds_by_name["baseline"] = preds
        diag_by_name["baseline"] = diag_df
        eval_by_name["baseline"] = eval_df
        summary_by_name["baseline"] = summary
        print_summary(baseline_cfg.label, summary, elapsed)
        print_profile_metrics(eval_df)

    if args.candidate:
        pred_source = None
        if candidate_cfg.use_spatial:
            source_cfg = replace(
                candidate_cfg,
                name="_candidate_source",
                label="CANDIDATE SOURCE",
                use_spatial=False,
            )
            pred_source, _ = run_pipeline(profiles, source_cfg)

        t0 = time.time()
        preds, diag_df = run_pipeline(profiles, candidate_cfg, pred_source=pred_source)
        elapsed = time.time() - t0

        eval_df = evaluate_run(preds, truth, TOLERANCES)
        summary = global_summary(eval_df, TOLERANCES)

        preds_by_name["candidate"] = preds
        diag_by_name["candidate"] = diag_df
        eval_by_name["candidate"] = eval_df
        summary_by_name["candidate"] = summary
        print_summary(candidate_cfg.label, summary, elapsed)
        print_profile_metrics(eval_df)
        if args.show_kappa and candidate_cfg.adaptive.enabled:
            print_kappa_summary(diag_df)
        if args.analyze:
            print_performance_analysis(eval_df, diag_df)

    if "baseline" in summary_by_name and "candidate" in summary_by_name:
        print_delta_vs_baseline(
            baseline_cfg.label,
            summary_by_name["baseline"],
            candidate_cfg.label,
            summary_by_name["candidate"],
        )

    if args.save:
        save_outputs(
            args.outdir,
            [name for name in ("baseline", "candidate") if name in summary_by_name],
            preds_by_name,
            eval_by_name,
            diag_by_name,
            summary_by_name,
        )
        print(f"\nSaved CSVs to {args.outdir}/")


if __name__ == "__main__":
    main()
