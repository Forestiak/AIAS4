"""
one-to-one per profile: each predicted boundary can
match at most one true boundary (the nearest still-unmatched one within the
tolerance window)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class Match:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def match_boundaries(pred: np.ndarray, truth: np.ndarray,
                     tol_m: float) -> Match:
    """Greedy nearest-neighbour one-to-one match within tolerance."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if pred.size == 0 and truth.size == 0:
        return Match(0, 0, 0)
    if truth.size == 0:
        return Match(0, pred.size, 0)
    if pred.size == 0:
        return Match(0, 0, truth.size)

    # pair (i, j) with cost |pred_i - truth_j|
    cost = np.abs(pred[:, None] - truth[None, :])
    # Repeatedly pick global minimum, strike row & column
    tp = 0
    remaining = cost.copy()
    pred_mask = np.ones(pred.size, dtype=bool)
    truth_mask = np.ones(truth.size, dtype=bool)
    while pred_mask.any() and truth_mask.any():
        sub = remaining[np.ix_(pred_mask, truth_mask)]
        if sub.size == 0:
            break
        k = int(np.argmin(sub))
        i_rel, j_rel = np.unravel_index(k, sub.shape)
        pred_idx = np.flatnonzero(pred_mask)[i_rel]
        truth_idx = np.flatnonzero(truth_mask)[j_rel]
        if sub[i_rel, j_rel] > tol_m:
            break
        tp += 1
        pred_mask[pred_idx] = False
        truth_mask[truth_idx] = False

    fp = int(pred_mask.sum())
    fn = int(truth_mask.sum())
    return Match(tp, fp, fn)


def evaluate_profile(pred_depths: np.ndarray, truth_depths: np.ndarray,
                     tolerances: Iterable[float] = (0.5, 1.0, 2.0)) -> dict:
    out = {}
    for tol in tolerances:
        m = match_boundaries(pred_depths, truth_depths, tol)
        out[f"tp_{tol}"] = m.tp
        out[f"fp_{tol}"] = m.fp
        out[f"fn_{tol}"] = m.fn
        out[f"precision_{tol}"] = m.precision
        out[f"recall_{tol}"] = m.recall
        out[f"f1_{tol}"] = m.f1
    return out


def evaluate_run(pred_by_target: dict[str, np.ndarray],
                 truth_by_target: dict[str, np.ndarray],
                 tolerances: Iterable[float] = (0.5, 1.0, 2.0)) -> pd.DataFrame:
    rows = []
    for tgt, pred in pred_by_target.items():
        if tgt not in truth_by_target:
            continue
        row = {"Target": tgt, "n_pred": len(pred), "n_true": len(truth_by_target[tgt])}
        row.update(evaluate_profile(pred, truth_by_target[tgt], tolerances))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Target").reset_index(drop=True)
    return df


def global_summary(df: pd.DataFrame,
                   tolerances: Iterable[float] = (0.5, 1.0, 2.0)) -> dict:
    out = {}
    for tol in tolerances:
        tp = int(df[f"tp_{tol}"].sum())
        fp = int(df[f"fp_{tol}"].sum())
        fn = int(df[f"fn_{tol}"].sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[f"pooled_precision_{tol}"] = p
        out[f"pooled_recall_{tol}"] = r
        out[f"pooled_f1_{tol}"] = f1
        out[f"mean_f1_{tol}"] = float(df[f"f1_{tol}"].mean())
        out[f"median_f1_{tol}"] = float(df[f"f1_{tol}"].median())
    return out
