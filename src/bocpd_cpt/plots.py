from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_profile_segmentation(depth: np.ndarray, signal: np.ndarray,
                              pred_cps: np.ndarray, true_cps: np.ndarray | None,
                              title: str = "", signal_name: str = "Ic",
                              ax: plt.Axes | None = None) -> plt.Axes:
    """Depth on y-axis (increasing downward)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 8))
    ax.plot(signal, depth, "k-", lw=0.7, alpha=0.8)
    ax.invert_yaxis()
    ax.set_ylabel("Depth [m]")
    ax.set_xlabel(signal_name)
    ax.set_title(title)
    if true_cps is not None:
        for d in true_cps:
            ax.axhline(d, color="tab:green", lw=1.0, alpha=0.6, linestyle="--")
    for d in pred_cps:
        ax.axhline(d, color="tab:red", lw=1.0, alpha=0.8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_runlength_posterior(rlp: np.ndarray, depth: np.ndarray,
                              ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 8))
    rlp = rlp.copy()
    rlp[rlp < 1e-6] = np.nan
    ax.imshow(-np.log10(np.clip(rlp.T, 1e-6, None)),
              aspect="auto", cmap="viridis",
              extent=(0, rlp.shape[0], rlp.shape[1], 0))
    ax.set_xlabel("t (depth index)")
    ax.set_ylabel("run length r")
    ax.set_title("-log10 P(r_t = r | x_{1:t})")
    return ax
