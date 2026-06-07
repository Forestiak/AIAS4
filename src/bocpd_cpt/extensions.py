"""Unsupervised BOCPD extensions for CPT-based soil layering.

Overview
--------
* ``refine_cps_to_gradient`` — snap predicted change-points to the nearest
  gradient peak of the raw signal.
* ``run_univariate`` — NIG BOCPD on a single feature (e.g. Ic), with a
  configurable hazard (const / depth-aware / minimum-thickness) and optional
  gradient refinement.
* ``run_multivariate`` — Normal–Inverse–Wishart BOCPD on stacked features
  such as ``[logQtn, logFr]``.
* ``build_neighbor_cp_density`` — depth-density of change-points pooled from
  the model's *own* predictions on neighbour profiles (no ground truth).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.signal import savgol_filter

from .bocpd import (
    NIGPredictor, NIGPrior, NIWPredictor, NIWPrior,
    run_bocpd, map_changepoints_paper, BOCPDResult,
)
from .data import Profile
from .hazards import (
    const_hazard, depth_aware_hazard,
    min_thickness_hazard, spatial_prior_hazard, HazardFn,
)


def refine_cps_to_gradient(signal: np.ndarray,
                            cps_idx: np.ndarray,
                            step_m: float,
                            search_radius_m: float = 0.7,
                            smooth_window: int = 5,
                            min_sep_m: float = 0.0,
                            strength_ratio: float | None = None,
                            drop_below_quantile: float | None = None,
                            snap_mode: str = "argmax"
                            ) -> np.ndarray:
    """Snap each predicted change-point index to the nearest peak of
    ``|d signal / dz|`` inside a ±search_radius window.

    Rationale.  The BOCPD forward pass is informative about *whether* a
    transition is happening but can be off by ~1 m on *where* it happens
    because the Gaussian conjugate predictor averages over a window.  The
    true layer boundary sits at the steepest rate of change in the raw
    signal, which is what gradient peak-picking finds.  This step is
    fully unsupervised (uses only the signal itself).

    Parameters
    ----------
    signal : (N,) array
        The raw feature used for BOCPD (e.g. Ic).
    cps_idx : (K,) int array
        BOCPD-proposed change-point indices (0-based, in sample units).
    step_m : float
        Sampling step in metres (so radius / step = half-window samples).
    search_radius_m : float, default 0.7
        How far (in metres) we look around each proposal for a steeper
        gradient.  Geotechnical motivation: typical transition width in
        CPT is < 1 m.
    smooth_window : int, default 5
        Savitzky-Golay window used to compute the derivative robustly.
        Odd integer; 0 or 1 disables smoothing.
    min_sep_m : float
        Minimum inter-CP separation in metres applied *after* refinement.
        Lets us collapse nearby duplicates created when two proposals
        snap to the same peak.
    strength_ratio : float | None
        Conditional snap.  Only move a CP when the in-window peak gradient
        exceeds the gradient at the original CP by at least this factor.
        ``None`` (default) = always snap (legacy behaviour).  ``1.5`` means
        only move if the new location is >=50% steeper than the old.  This
        preserves already-accurate localisations (which BOCPD often gets to
        within 0.5 m) and only corrects clear mislocalisations.
    drop_below_quantile : float | None
        If set (e.g. 0.15), drop any refined CP whose local |grad| is
        below the given quantile of the full gradient profile.  Removes
        BOCPD false positives that lack a matching gradient feature.
    snap_mode : {"argmax", "nearest_peak"}
        ``"argmax"`` snaps to the global max |grad| inside the window.
        ``"nearest_peak"`` snaps to the *nearest* local maximum of
        |grad| inside the window (falls back to argmax if no strict
        local max exists in the window).  Nearest-peak preserves tight
        localisations better than argmax when multiple peaks lie nearby.
    """
    if cps_idx.size == 0:
        return cps_idx.astype(int)
    x = np.asarray(signal, dtype=float)
    if smooth_window is not None and smooth_window >= 3:
        sw = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        poly = min(3, sw - 1)
        xs = savgol_filter(x, sw, poly)
    else:
        xs = x
    grad = np.abs(np.gradient(xs, step_m))

    r = max(1, int(round(search_radius_m / step_m)))

    # Precompute strict local maxima of |grad| for nearest-peak mode.
    if snap_mode == "nearest_peak":
        peaks = np.flatnonzero(
            (grad[1:-1] > grad[:-2]) & (grad[1:-1] >= grad[2:])
        ) + 1
    else:
        peaks = None

    refined = []
    refined_grads = []
    for c in cps_idx:
        a = max(0, int(c) - r)
        b = min(len(grad), int(c) + r + 1)
        local = grad[a:b]
        k_argmax = int(np.argmax(local))
        argmax_idx = a + k_argmax
        if snap_mode == "nearest_peak" and peaks is not None:
            mask = (peaks >= a) & (peaks < b)
            window_peaks = peaks[mask]
            if window_peaks.size:
                new_idx = int(window_peaks[np.argmin(np.abs(window_peaks - int(c)))])
            else:
                new_idx = argmax_idx
        else:
            new_idx = argmax_idx
        if strength_ratio is not None:
            orig_g = grad[int(c)]
            new_g = grad[new_idx]
            if orig_g > 0 and new_g < strength_ratio * orig_g:
                new_idx = int(c)
        refined.append(new_idx)
        refined_grads.append(grad[new_idx])

    if drop_below_quantile is not None and refined:
        thr = float(np.quantile(grad, drop_below_quantile))
        refined = [i for i, g in zip(refined, refined_grads) if g >= thr]

    refined = sorted(set(refined))

    if min_sep_m > 0 and refined:
        min_sep = max(1, int(round(min_sep_m / step_m)))
        kept = [refined[0]]
        for c in refined[1:]:
            if c - kept[-1] >= min_sep:
                kept.append(c)
        refined = kept

    return np.asarray(refined, dtype=int)


# -----------------------------------------------------------------------------
# Univariate (paper baseline + variants)
# -----------------------------------------------------------------------------

@dataclass
class UnivariateConfig:
    feature: str = "Ic"
    mu0: float = 0.0
    kappa_prior: float = 1.0
    alpha: float = 1.0
    beta: float = 0.1
    hazard_kappa: float | None = None  # None -> N (paper)
    hazard: str = "const"              # const | depth | minthick | depth+minthick
    min_thickness_m: float = 0.5
    depth_kappa0_m: float = 1.0
    depth_growth: float = 0.05         # expected-thickness growth per metre
    # Snap each predicted CP to the nearest gradient peak (see
    # refine_cps_to_gradient).
    refine: bool = False
    refine_search_m: float = 0.7
    refine_smooth_window: int = 5
    refine_strength_ratio: float | None = None
    refine_drop_below_quantile: float | None = None


def _make_hazard(cfg: UnivariateConfig, profile: Profile, N: int) -> HazardFn:
    step_m = profile.step
    kappa = cfg.hazard_kappa if cfg.hazard_kappa is not None else float(N)
    base_const = const_hazard(kappa)

    if cfg.hazard == "const":
        base = base_const
    elif cfg.hazard == "depth":
        base = depth_aware_hazard(profile.depth,
                                   kappa0_m=cfg.depth_kappa0_m,
                                   growth_m_per_m=cfg.depth_growth,
                                   step_m=step_m)
    elif cfg.hazard == "minthick":
        base = base_const
    elif cfg.hazard == "depth+minthick":
        base = depth_aware_hazard(profile.depth,
                                   kappa0_m=cfg.depth_kappa0_m,
                                   growth_m_per_m=cfg.depth_growth,
                                   step_m=step_m)
    else:
        raise ValueError(f"Unknown hazard mode: {cfg.hazard}")

    if cfg.hazard in {"minthick", "depth+minthick"}:
        r_min = int(round(cfg.min_thickness_m / step_m))
        base = min_thickness_hazard(base, r_min)

    return base


def run_univariate(profile: Profile, cfg: UnivariateConfig,
                   extra_hazard_wrap: Callable[[HazardFn], HazardFn] | None = None
                   ) -> tuple[BOCPDResult, np.ndarray]:
    """Run BOCPD-ON + paper-Eq.11 MAP extraction on a single profile."""
    x = profile.features[cfg.feature]
    N = profile.n
    prior = NIGPrior(mu0=cfg.mu0, kappa=cfg.kappa_prior,
                     alpha=cfg.alpha, beta=cfg.beta)
    hazard = _make_hazard(cfg, profile, N)
    if extra_hazard_wrap is not None:
        hazard = extra_hazard_wrap(hazard)

    factory = lambda capacity=N + 1: NIGPredictor(prior, capacity=capacity)
    res = run_bocpd(x, factory, hazard=hazard, full_posterior=True)

    # If min-thickness was applied via hazard, the extractor can still stop
    # closer than r_min because the MAP recursion is over posterior values
    # (not over raw hazard gating).  Apply a final r_min post-filter just to
    # make sure no pair is closer than the geotechnical floor.
    step_m = profile.step
    min_run = max(1, int(round(cfg.min_thickness_m / step_m))) \
        if cfg.hazard in {"minthick", "depth+minthick"} else 1
    cps_idx = map_changepoints_paper(res.run_length_posterior,
                                      min_run_length=min_run)

    if cfg.refine and cps_idx.size:
        cps_idx = refine_cps_to_gradient(
            x, cps_idx, step_m,
            search_radius_m=cfg.refine_search_m,
            smooth_window=cfg.refine_smooth_window,
            min_sep_m=(cfg.min_thickness_m
                       if cfg.hazard in {"minthick", "depth+minthick"}
                       else 0.0),
            strength_ratio=cfg.refine_strength_ratio,
            drop_below_quantile=cfg.refine_drop_below_quantile,
        )

    return res, cps_idx


# -----------------------------------------------------------------------------
# Multivariate (log-transformed) BOCPD
# -----------------------------------------------------------------------------

@dataclass
class MultivariateConfig:
    features: tuple[str, ...] = ("logQtn", "logFr")
    standardise: bool = True
    # NIW prior defaults follow Xuan & Murphy (2007) "weakly informative"
    lambda0: float = 1.0
    nu0_extra: float = 2.0    # nu0 = d + nu0_extra
    V0_scale: float = 1.0     # V0 = V0_scale * empirical cov
    hazard_kappa: float | None = None
    hazard: str = "const"
    min_thickness_m: float = 0.5
    depth_kappa0_m: float = 1.0
    depth_growth: float = 0.05


def _stack_features(profile: Profile, names: Iterable[str],
                    standardise: bool) -> tuple[np.ndarray, dict]:
    cols = []
    stats = {}
    for n in names:
        v = profile.features[n].astype(float)
        if standardise:
            mu, sd = float(np.mean(v)), float(np.std(v) + 1e-9)
            stats[n] = (mu, sd)
            v = (v - mu) / sd
        cols.append(v)
    X = np.stack(cols, axis=1)
    return X, stats


def run_multivariate(profile: Profile, cfg: MultivariateConfig,
                     extra_hazard_wrap: Callable[[HazardFn], HazardFn] | None = None
                     ) -> tuple[BOCPDResult, np.ndarray]:
    X, _ = _stack_features(profile, cfg.features, cfg.standardise)
    d = X.shape[1]
    N = profile.n
    # Empirical covariance for the V0 scale
    emp_cov = np.cov(X, rowvar=False)
    if d == 1:
        emp_cov = np.atleast_2d(emp_cov)
    V0 = cfg.V0_scale * emp_cov + 1e-6 * np.eye(d)
    prior = NIWPrior(
        mu0=np.zeros(d),
        lambda0=cfg.lambda0,
        V0=V0,
        nu0=float(d) + cfg.nu0_extra,
    )

    kappa = cfg.hazard_kappa if cfg.hazard_kappa is not None else float(N)
    if cfg.hazard == "const":
        hazard = const_hazard(kappa)
    elif cfg.hazard == "depth":
        hazard = depth_aware_hazard(profile.depth,
                                     kappa0_m=cfg.depth_kappa0_m,
                                     growth_m_per_m=cfg.depth_growth,
                                     step_m=profile.step)
    elif cfg.hazard == "depth+minthick":
        hazard = depth_aware_hazard(profile.depth,
                                     kappa0_m=cfg.depth_kappa0_m,
                                     growth_m_per_m=cfg.depth_growth,
                                     step_m=profile.step)
        r_min = int(round(cfg.min_thickness_m / profile.step))
        hazard = min_thickness_hazard(hazard, r_min)
    elif cfg.hazard == "minthick":
        r_min = int(round(cfg.min_thickness_m / profile.step))
        hazard = min_thickness_hazard(const_hazard(kappa), r_min)
    else:
        raise ValueError(f"Unknown hazard: {cfg.hazard}")
    if extra_hazard_wrap is not None:
        hazard = extra_hazard_wrap(hazard)

    factory = lambda capacity=N + 1: NIWPredictor(prior, capacity=capacity)
    res = run_bocpd(X, factory, hazard=hazard, full_posterior=True)

    step_m = profile.step
    min_run = max(1, int(round(cfg.min_thickness_m / step_m))) \
        if cfg.hazard in {"minthick", "depth+minthick"} else 1
    cps_idx = map_changepoints_paper(res.run_length_posterior,
                                      min_run_length=min_run)
    return res, cps_idx


# -----------------------------------------------------------------------------
# Leakage-safe spatial boundary density
# -----------------------------------------------------------------------------

def build_neighbor_cp_density(target: str,
                               pred_by_target: dict[str, np.ndarray],
                               profile_xy: dict[str, tuple[float, float]],
                               grid_depth: np.ndarray,
                               bandwidth_m: float = 1.5,
                               k_neighbors: int = 6) -> np.ndarray:
    """Density of *predicted* change-points from the nearest ``k`` neighbours.

    Inputs must be pre-computed model predictions (NOT ground truth) from
    neighbour profiles.  The target profile's own predictions are excluded.

    Returns a density in units of [predicted CPs / metre] on ``grid_depth``.
    """
    xy_t = profile_xy.get(target)
    if xy_t is None:
        return np.zeros_like(grid_depth)
    others = [(tgt, xy) for tgt, xy in profile_xy.items() if tgt != target]
    if not others:
        return np.zeros_like(grid_depth)
    dists = np.array([np.hypot(xy[0] - xy_t[0], xy[1] - xy_t[1])
                      for _, xy in others])
    order = np.argsort(dists)
    neigh = [others[i][0] for i in order[:k_neighbors]]
    neigh_dists = dists[order[:k_neighbors]]
    weights = 1.0 / (neigh_dists + 1.0)
    weights = weights / weights.sum()

    density = np.zeros_like(grid_depth, dtype=float)
    for tgt, w in zip(neigh, weights):
        cp_m = pred_by_target.get(tgt)
        if cp_m is None or len(cp_m) == 0:
            continue
        # Gaussian kernel at each predicted depth
        for cp in cp_m:
            density += w * np.exp(-0.5 * ((grid_depth - cp) / bandwidth_m) ** 2) \
                        / (np.sqrt(2 * np.pi) * bandwidth_m)
    return density
