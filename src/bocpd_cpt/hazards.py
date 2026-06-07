"""Hazard functions for BOCPD.

The vectorised convention is ``hazard(t: int, R: int) -> np.ndarray`` where
``t`` is the current (1-indexed) observation count, ``R = t + 1`` is the
length of the run-length vector (values 0..t), and the returned array gives
``p(r_{t+1}=0 | r_t = r)`` for each r.

All implementations return values clipped to ``[0, 1-eps]``.

Extensions beyond the paper's constant hazard:

* :func:`const_hazard` — paper baseline.
* :func:`depth_aware_hazard` — expected layer thickness grows with depth,
  motivated by marine sedimentary consolidation (shallow layers are thinner).
* :func:`min_thickness_hazard` — per-r gating that forces P(change) = 0
  until at least ``r_min`` samples have elapsed.  Embeds the minimum layer
  thickness *inside* the model, not as post-processing.
* :func:`spatial_prior_hazard` — rescales a base hazard by an unsupervised,
  leakage-safe depth-density of predicted change-points from neighbour
  profiles.
* :func:`contrast_prior_hazard` — rescales a base hazard by a left-right
  window contrast signal computed from the profile's own Ic signal.  High
  contrast at depth d → raised hazard → BOCPD is more likely to place a
  boundary there.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


HazardFn = Callable[[int, int], np.ndarray]


_MAX_H = 1.0 - 1e-12


def const_hazard(kappa: float) -> HazardFn:
    inv = min(1.0 / max(float(kappa), 1.0), _MAX_H)
    def _h(t: int, R: int) -> np.ndarray:
        return np.full(R, inv)
    return _h


def depth_aware_hazard(depth: np.ndarray,
                        kappa0_m: float = 1.0,
                        growth_m_per_m: float = 0.05,
                        min_kappa_m: float = 0.5,
                        step_m: float = 0.02) -> HazardFn:
    """Depth-dependent geometric hazard.

    E[layer thickness](d) = kappa0_m + growth_m_per_m * d

    Per-sample hazard at depth d: h(d) = step_m / E[layer thickness](d).

    Defaults (kappa0_m=1.0 m, growth=0.05) encode a marine sedimentary prior:
    ~1 m expected thickness at the seabed, growing to ~6 m at 100 m depth.
    Hyperparameters are not tuned against ground truth.
    """
    depth = np.asarray(depth, dtype=float)
    expected_m = np.maximum(min_kappa_m, kappa0_m + growth_m_per_m * depth)
    h_per_t = np.clip(step_m / expected_m, 1e-12, _MAX_H)

    def _h(t: int, R: int) -> np.ndarray:
        return np.full(R, h_per_t[t])

    return _h


def min_thickness_hazard(base: HazardFn, min_run_length: int) -> HazardFn:
    """Gate a base hazard to 0 for r < min_run_length.

    Geotechnical rationale: spurious very thin layers are not physical; this
    pushes any such boundary out of the posterior entirely (rather than
    filtering them post-hoc).
    """
    def _h(t: int, R: int) -> np.ndarray:
        h = base(t, R).copy()
        cap = min(min_run_length, R)
        h[:cap] = 0.0
        return h
    return _h


def _rate_preserving_factor(raw_factor: np.ndarray) -> np.ndarray:
    """Renormalise a multiplicative hazard factor so its depth-mean equals 1.

    Without this step, a shape like ``factor = 1 + s * (c/c_bar - 1)`` clipped
    to ``[1e-3, 100]`` has a depth-mean that drifts away from 1, which silently
    inflates or deflates the global expected number of change-points set by
    the base hazard's κ.  Dividing by ``mean(factor)`` preserves the κ-implied
    rate while keeping the relative depth modulation intact.
    """
    m = float(np.mean(raw_factor))
    if not np.isfinite(m) or m <= 0.0:
        return np.ones_like(raw_factor)
    return raw_factor / m


def contrast_prior_hazard(base: HazardFn,
                           contrast_signal: np.ndarray,
                           strength: float = 1.0,
                           rate_preserving: bool = True) -> HazardFn:
    """Modulate a base hazard by a per-depth left-right window contrast signal.

    ``contrast_signal`` is the normalised |mean_left - mean_right| / std_combined
    at every sample of the profile, computed from the profile's own Ic signal.
    Positions with above-average contrast get a raised hazard; below-average
    positions get a lower one.  The formula is identical to
    :func:`spatial_prior_hazard`:

        h_eff(r, d) = base(r, d) * factor(d)
        factor(d)   = clip(1 + strength * (c(d)/c_bar - 1), 1e-3, 100)

    When ``rate_preserving`` (default) the factor is divided by its depth-mean
    so the hazard's expected number of change-points stays equal to the base
    hazard's.  ``strength=0`` recovers the base hazard exactly.
    """
    c = np.asarray(contrast_signal, dtype=float)
    c_bar = max(float(np.mean(c)), 1e-12)
    factor = np.clip(1.0 + strength * (c / c_bar - 1.0), 1e-3, 100.0)
    if rate_preserving:
        factor = _rate_preserving_factor(factor)

    def _h(t: int, R: int) -> np.ndarray:
        h = base(t, R) * factor[t]
        return np.clip(h, 0.0, _MAX_H)

    return _h


def spatial_prior_hazard(base: HazardFn,
                          prior_density: np.ndarray,
                          strength: float = 1.0,
                          rate_preserving: bool = True) -> HazardFn:
    """Modulate a base hazard by a depth-dependent prior boundary density.

    ``prior_density`` must be on the same depth grid as the profile being
    analysed.  It must come from unsupervised neighbour evidence only —
    never from ground-truth boundaries of any profile.  The effective
    hazard at depth d is

        h_eff(r, d) = base(r, d) * factor(d)
        factor(d)   = clip(1 + strength * (rho(d)/rho_bar - 1), 1e-3, 100)

    When ``rate_preserving`` (default) the factor is divided by its depth-mean
    so the global expected number of CPs is preserved.  ``strength=0`` recovers
    the base hazard exactly.
    """
    rho = np.asarray(prior_density, dtype=float)
    rho_bar = max(float(np.mean(rho)), 1e-12)
    factor = np.clip(1.0 + strength * (rho / rho_bar - 1.0), 1e-3, 100.0)
    if rate_preserving:
        factor = _rate_preserving_factor(factor)

    def _h(t: int, R: int) -> np.ndarray:
        h = base(t, R) * factor[t]
        return np.clip(h, 0.0, _MAX_H)

    return _h
