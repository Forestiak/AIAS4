"""Core Bayesian Online Change-Point Detection (Adams & MacKay 2007).

Univariate Gaussian likelihood with Normal–Inverse–Gamma conjugate prior
(Student-t predictive), and multivariate Gaussian likelihood with
Normal–Inverse–Wishart conjugate prior (matrix-t predictive).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import special as sp_special


# =============================================================================
# Univariate Student-t predictive under Normal-Inverse-Gamma prior
# =============================================================================

@dataclass
class NIGPrior:
    """Normal–Inverse–Gamma prior hyperparameters.

    The paper's defaults are mu0=0, kappa=1, alpha=1, beta=0.1.
    """
    mu0: float = 0.0
    kappa: float = 1.0
    alpha: float = 1.0
    beta: float = 0.1


def _student_t_logpdf(x: np.ndarray, mu: np.ndarray,
                      scale2: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Log pdf of Student-t.  scale2 is the squared scale parameter."""
    diff = x - mu
    half_df = 0.5 * df
    log_coeff = (sp_special.gammaln(half_df + 0.5)
                 - sp_special.gammaln(half_df)
                 - 0.5 * np.log(df * np.pi * scale2))
    log_kernel = -(half_df + 0.5) * np.log1p(diff * diff / (df * scale2))
    return log_coeff + log_kernel


class NIGPredictor:
    """Maintains NIG sufficient statistics for every run length, via numpy
    arrays that grow by one each step.  Preallocating to capacity ``N+1``
    avoids O(N) reallocations.
    """

    def __init__(self, prior: NIGPrior, capacity: int = 0):
        self.prior = prior
        # Length of the active prefix inside preallocated arrays
        self._n = 1
        cap = max(1, capacity)
        self.mu = np.empty(cap, dtype=float); self.mu[0] = prior.mu0
        self.kappa = np.empty(cap, dtype=float); self.kappa[0] = prior.kappa
        self.alpha = np.empty(cap, dtype=float); self.alpha[0] = prior.alpha
        self.beta = np.empty(cap, dtype=float); self.beta[0] = prior.beta

    def _ensure(self, n: int) -> None:
        if n <= self.mu.size:
            return
        new_cap = max(n, int(self.mu.size * 2))
        for name in ("mu", "kappa", "alpha", "beta"):
            arr = getattr(self, name)
            new = np.empty(new_cap, dtype=float)
            new[: arr.size] = arr
            setattr(self, name, new)

    def log_pred_prob(self, x: float) -> np.ndarray:
        n = self._n
        mu = self.mu[:n]; kappa = self.kappa[:n]
        alpha = self.alpha[:n]; beta = self.beta[:n]
        df = 2.0 * alpha
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)
        diff = x - mu
        half_df = 0.5 * df
        log_coeff = (sp_special.gammaln(half_df + 0.5)
                     - sp_special.gammaln(half_df)
                     - 0.5 * np.log(df * np.pi * scale2))
        log_kernel = -(half_df + 0.5) * np.log1p(diff * diff / (df * scale2))
        return log_coeff + log_kernel

    def update(self, x: float) -> None:
        """Insert the fresh r=0 prior at slot 0, shifting the rest right."""
        n = self._n
        self._ensure(n + 1)
        # r+1 posteriors go to slots 1..n (overwrite shifted)
        mu_prev = self.mu[:n].copy()
        kappa_prev = self.kappa[:n].copy()
        alpha_prev = self.alpha[:n].copy()
        beta_prev = self.beta[:n].copy()
        self.mu[1 : n + 1] = (kappa_prev * mu_prev + x) / (kappa_prev + 1.0)
        self.kappa[1 : n + 1] = kappa_prev + 1.0
        self.alpha[1 : n + 1] = alpha_prev + 0.5
        self.beta[1 : n + 1] = (beta_prev + 0.5 * kappa_prev * (x - mu_prev) ** 2
                                / (kappa_prev + 1.0))
        self.mu[0] = self.prior.mu0
        self.kappa[0] = self.prior.kappa
        self.alpha[0] = self.prior.alpha
        self.beta[0] = self.prior.beta
        self._n = n + 1


# =============================================================================
# Multivariate predictive under Normal-Inverse-Wishart (Xuan & Murphy 2007)
# =============================================================================

@dataclass
class NIWPrior:
    """Normal–Inverse–Wishart prior.

    Parameters follow Xuan & Murphy (2007): the prior is
      mu | Sigma ~ N(mu0, Sigma / lambda0)
      Sigma     ~ IW(V0, nu0)   with degrees of freedom nu0 > d - 1
    """
    mu0: np.ndarray
    lambda0: float
    V0: np.ndarray
    nu0: float


def _multivariate_t_logpdf_batch(x: np.ndarray,
                                  mu: np.ndarray,
                                  Sigma: np.ndarray,
                                  df: np.ndarray) -> np.ndarray:
    """Vectorised log-pdf of the multivariate-t over a batch of (mu, Sigma, df).

    ``mu`` shape (R, d); ``Sigma`` shape (R, d, d); ``df`` shape (R,).
    Returns array of length R giving the log density at the single point ``x``
    """
    R, d = mu.shape
    diff = x[None, :] - mu                     # (R, d)
    # batched Cholesky with jitter fallback
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(d))
    # Solve L z = diff  =>  z shape (R, d)
    z = np.linalg.solve(L, diff[..., None])[..., 0]
    mahal = np.einsum("ij,ij->i", z, z)
    logdet = 2.0 * np.sum(np.log(np.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    return (sp_special.gammaln(0.5 * (df + d))
            - sp_special.gammaln(0.5 * df)
            - 0.5 * d * np.log(df * np.pi)
            - 0.5 * logdet
            - 0.5 * (df + d) * np.log1p(mahal / df))


class NIWPredictor:
    """Run-length-indexed NIW posterior store for multivariate BOCPD.

    Uses preallocated numpy arrays for mu (R, d), V (R, d, d), lam, nu — and
    vectorised Cholesky for the multivariate-t log pdf.  Runs in O(N^2 d^3)
    overall, dominated by the batched Cholesky
    """

    def __init__(self, prior: NIWPrior, capacity: int = 0):
        self.prior = prior
        d = prior.mu0.size
        cap = max(1, capacity)
        self._n = 1
        self.mu = np.empty((cap, d), dtype=float); self.mu[0] = prior.mu0
        self.lam = np.empty(cap, dtype=float); self.lam[0] = prior.lambda0
        self.V = np.empty((cap, d, d), dtype=float); self.V[0] = prior.V0
        self.nu = np.empty(cap, dtype=float); self.nu[0] = prior.nu0
        self.d = d

    def _ensure(self, n: int) -> None:
        if n <= self.mu.shape[0]:
            return
        cap = max(n, int(self.mu.shape[0] * 2))
        d = self.d
        for name, shape in (("mu", (cap, d)), ("lam", (cap,)),
                            ("V", (cap, d, d)), ("nu", (cap,))):
            arr = getattr(self, name)
            new = np.empty(shape, dtype=float)
            new[: arr.shape[0]] = arr
            setattr(self, name, new)

    def log_pred_prob(self, x: np.ndarray) -> np.ndarray:
        n = self._n
        d = self.d
        mu = self.mu[:n]; lam = self.lam[:n]; V = self.V[:n]; nu = self.nu[:n]
        df = nu - d + 1.0                         # (n,)
        scale = V * ((lam + 1.0) / (lam * df))[:, None, None]
        return _multivariate_t_logpdf_batch(np.asarray(x, dtype=float), mu, scale, df)

    def update(self, x: np.ndarray) -> None:
        n = self._n
        self._ensure(n + 1)
        x = np.asarray(x, dtype=float)
        mu_prev = self.mu[:n].copy()
        lam_prev = self.lam[:n].copy()
        V_prev = self.V[:n].copy()
        nu_prev = self.nu[:n].copy()
        diff = x[None, :] - mu_prev
        outer = diff[..., :, None] * diff[..., None, :]
        self.mu[1 : n + 1] = (lam_prev[:, None] * mu_prev + x[None, :]) / (lam_prev[:, None] + 1.0)
        self.V[1 : n + 1] = V_prev + (lam_prev / (lam_prev + 1.0))[:, None, None] * outer
        self.lam[1 : n + 1] = lam_prev + 1.0
        self.nu[1 : n + 1] = nu_prev + 1.0
        self.mu[0] = self.prior.mu0
        self.lam[0] = self.prior.lambda0
        self.V[0] = self.prior.V0
        self.nu[0] = self.prior.nu0
        self._n = n + 1


# =============================================================================
# BOCPD recursion
# =============================================================================

HazardFn = Callable[[int, int], np.ndarray]
"""hazard(t, R) -> array of length R.

Entry r of the returned array is P(r_{t+1}=0 | r_t=r), i.e. the probability
that the next observation starts a new layer given a current run length r
at position t (0-indexed).

The paper uses a constant hazard = 1/kappa with kappa = N (profile length).
"""


def const_hazard(kappa: float) -> HazardFn:
    inv = 1.0 / max(float(kappa), 1.0)
    def _h(t: int, R: int) -> np.ndarray:
        return np.full(R, inv)
    return _h


@dataclass
class BOCPDResult:
    """Output of one BOCPD pass."""
    run_length_posterior: np.ndarray   # shape (N, N+1); rlp[t, r] = P(r_t=r | x_{1:t})
    cp_posterior: np.ndarray            # shape (N,); P(r_t = 0 | x_{1:t})
    log_evidence: np.ndarray            # shape (N,)

    @property
    def N(self) -> int:
        return self.run_length_posterior.shape[0]


def run_bocpd(x: np.ndarray,
              predictor_factory: Callable[[], "NIGPredictor | NIWPredictor"],
              hazard: HazardFn | None = None,
              full_posterior: bool = True) -> BOCPDResult:
    """Run the Adams & MacKay 2007 recursion on a 1D or multi-D signal.

    * ``x`` — shape (N,) for univariate or (N, d) for multivariate.
    * ``predictor_factory`` — zero-arg factory returning a fresh predictor.
    * ``hazard`` — see :data:`HazardFn`.  Defaults to ``const_hazard(N)``.
    * ``full_posterior`` — keep the whole run-length posterior matrix.

    The growth/reset update is done in log space for numerical stability.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        N = x.size
    else:
        N = x.shape[0]
    if hazard is None:
        hazard = const_hazard(N)

    # Pass the preallocation capacity when the factory accepts it.
    try:
        predictor = predictor_factory(N + 1)
    except TypeError:
        predictor = predictor_factory()

    log_R = np.full(1, 0.0, dtype=float)
    log_evidence = np.empty(N, dtype=float)
    cp_post = np.empty(N, dtype=float)

    if full_posterior:
        rlp = np.zeros((N, N + 1), dtype=float)

    for t in range(N):
        xt = x[t] if x.ndim == 1 else x[t]
        log_pi = predictor.log_pred_prob(xt)             # length t+1

        h = hazard(t, t + 1).astype(float)
        h = np.clip(h, 0.0, 1.0 - 1e-12)

        # For h == 0 we must not take log(0); instead we suppress the
        # "reset" contribution at that r using a mask.
        safe_h = np.where(h <= 0.0, 1e-300, h)
        log_h = np.log(safe_h)
        log_1mh = np.log1p(-h)

        log_growth = log_R + log_pi + log_1mh
        # Reset contribution: only r's with h>0 contribute
        reset_terms = log_R + log_pi + log_h
        reset_terms = np.where(h <= 0.0, -np.inf, reset_terms)
        log_reset = sp_special.logsumexp(reset_terms) if np.isfinite(reset_terms).any() else -np.inf

        new_log_R = np.empty(t + 2, dtype=float)
        new_log_R[0] = log_reset
        new_log_R[1:] = log_growth

        log_Z = sp_special.logsumexp(new_log_R)
        log_evidence[t] = log_Z
        post = np.exp(new_log_R - log_Z)
        cp_post[t] = post[0]
        if full_posterior:
            rlp[t, : t + 2] = post

        log_R = new_log_R

        predictor.update(xt)

    return BOCPDResult(
        run_length_posterior=rlp if full_posterior else np.zeros((0, 0)),
        cp_posterior=cp_post,
        log_evidence=log_evidence,
    )


# =============================================================================
# Change-point extraction
# =============================================================================

def map_changepoints_paper(rlp: np.ndarray,
                            min_run_length: int = 1) -> np.ndarray:
    """Paper Eq. 11 MAP changepoint extraction.

        C(z) = max_{j in 1..z-1} [ p(r_z = z-j | x_{1:z})
                                  * p(r_j = 0 | x_{1:j})
                                  * C(j-1) ]

    Returns indices of change-points (the t-indices where a new layer starts,
    0-indexed).  The t=0 boundary is NOT returned (it is the profile top).

    Numerics: we work in log space with proper ``-inf`` for posterior zeros
    (rather than clipping at ``log(1e-300) = -690``).  On long profiles the
    clip caused ``log_C`` to plateau at ``-690`` and the Viterbi backtrace to
    collapse to the "no change-point" branch.  Using ``-inf`` lets the
    recursion keep descending along the valid MAP path.
    """
    N = rlp.shape[0]
    log_C = np.full(N + 1, -np.inf)
    log_C[0] = 0.0
    back = np.zeros(N + 1, dtype=int)

    with np.errstate(divide="ignore"):
        log_rlp = np.log(rlp)                  # -inf where rlp == 0 (correct)
    log_p_r0 = log_rlp[:, 0]

    for z in range(1, N + 1):
        # "no change-point anywhere up to z": whole 1..z block is one run,
        # i.e. r_z = z.  In 0-indexed rlp that's rlp[z-1, z].
        no_cp = log_rlp[z - 1, z]

        j_idx = np.arange(1, z)
        if j_idx.size == 0:
            log_C[z] = no_cp
            back[z] = 0
            continue
        log_term_rl = log_rlp[z - 1, z - j_idx]
        log_term_cp = log_p_r0[j_idx - 1]
        log_prev = log_C[j_idx - 1]
        cand = log_term_rl + log_term_cp + log_prev

        cand_full = np.concatenate([cand, [no_cp]])
        j_full = np.concatenate([j_idx, [0]])
        # If every candidate is -inf, keep log_C[z] = -inf, back = 0 — this
        # just means the MAP cannot distinguish any CP placement at this z
        # and inherits the previous decision by convention.
        if not np.isfinite(cand_full).any():
            log_C[z] = -np.inf
            back[z] = 0
            continue
        k = int(np.nanargmax(np.where(np.isnan(cand_full), -np.inf, cand_full)))
        log_C[z] = cand_full[k]
        back[z] = int(j_full[k])

    # Backtrack from N
    cps: list[int] = []
    z = N
    while z > 0:
        j = back[z]
        if j == 0:
            break
        cps.append(j)
        z = j - 1
    cps = sorted(cps)
    cps = [c - 1 for c in cps]          # 0-indexed depth index
    if min_run_length > 1 and cps:
        kept = [cps[0]]
        for c in cps[1:]:
            if c - kept[-1] >= min_run_length:
                kept.append(c)
        cps = kept
    return np.asarray(cps, dtype=int)


def map_changepoints_argmax(rlp: np.ndarray,
                             min_drop: int = 5,
                             min_run_length: int = 1) -> np.ndarray:
    """Heuristic MAP — declare a CP where the posterior argmax run-length
    drops (i.e. the model thinks a new layer has started).

    A CP is placed at index t if ``argmax_r rlp[t, r] < argmax_r rlp[t-1, r] -
    min_drop``.  Consecutive CPs closer than ``min_run_length`` samples are
    merged (the first is kept).  Less sensitive to the product-of-small-
    numbers collapse that can degrade the paper's Eq.11 on very long or very
    narrow-posterior MV profiles.
    """
    N = rlp.shape[0]
    ml = np.argmax(rlp, axis=1)
    cps: list[int] = []
    for t in range(1, N):
        if ml[t - 1] - ml[t] >= min_drop:
            cps.append(t)
    if min_run_length > 1 and cps:
        kept = [cps[0]]
        for c in cps[1:]:
            if c - kept[-1] >= min_run_length:
                kept.append(c)
        cps = kept
    return np.asarray(cps, dtype=int)
