"""
python experiments/refine_sweep.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bocpd_cpt import data as D
from bocpd_cpt.extensions import (
    UnivariateConfig, run_univariate, refine_cps_to_gradient,
)
from bocpd_cpt.eval import evaluate_run, global_summary

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

# Base (un-refined) config — best non-refined variant from the ablation.
BASE_CFG = UnivariateConfig(
    feature="Ic", hazard="minthick",
    min_thickness_m=0.5, hazard_kappa=30.0,
    refine=False,
)

# Refinement variants to sweep.  None of these re-run BOCPD.
REFINE_VARIANTS = [
    # search radius sweep, unconditional (snap to argmax)
    dict(search_radius_m=0.3, snap_mode="argmax"),
    dict(search_radius_m=0.5, snap_mode="argmax"),
    dict(search_radius_m=0.7, snap_mode="argmax"),
    dict(search_radius_m=1.0, snap_mode="argmax"),
    # nearest_peak: snap to closest local max
    dict(search_radius_m=0.3, snap_mode="nearest_peak"),
    dict(search_radius_m=0.5, snap_mode="nearest_peak"),
    dict(search_radius_m=0.7, snap_mode="nearest_peak"),
    dict(search_radius_m=1.0, snap_mode="nearest_peak"),
    dict(search_radius_m=1.5, snap_mode="nearest_peak"),
    # nearest_peak + conditional snap
    dict(search_radius_m=0.7, snap_mode="nearest_peak", strength_ratio=1.5),
    dict(search_radius_m=1.0, snap_mode="nearest_peak", strength_ratio=1.5),
    # argmax + conservative conditional snap (sanity)
    dict(search_radius_m=0.7, snap_mode="argmax", strength_ratio=2.0),
]

RAW_CACHE = Path("/tmp/refine_sweep_raw.pkl")


def variant_name(v: dict) -> str:
    bits = [f"r{v['search_radius_m']:.1f}"]
    bits.append(v.get("snap_mode", "argmax"))
    if v.get("strength_ratio") is not None:
        bits.append(f"sr{v['strength_ratio']}")
    if v.get("drop_below_quantile") is not None:
        bits.append(f"dq{v['drop_below_quantile']:.2f}")
    return "_".join(bits)


def main() -> None:
    t0 = time.time()
    profiles = D.load_profiles()
    print(f"Loaded {len(profiles)} profiles in {time.time()-t0:.1f}s")
    strata = D.load_strata()
    truth = {p.target: D.true_boundaries_for(p.target, strata) for p in profiles}

    # One BOCPD pass per profile, store raw CPs + signal + posterior at r=0.
    # Cache to disk so we can iterate on refinement cheaply across reruns.
    if RAW_CACHE.exists():
        with open(RAW_CACHE, "rb") as fh:
            raw = pickle.load(fh)
        print(f"Loaded cached raw CPs from {RAW_CACHE}")
    else:
        t0 = time.time()
        raw = {}
        for p in profiles:
            res, cps_idx = run_univariate(p, BASE_CFG)
            rlp_r0 = res.run_length_posterior[:, 0]  # p(r_t=0 | x_{1:t})
            raw[p.target] = (cps_idx,
                             p.features[BASE_CFG.feature],
                             p.step, p.depth, rlp_r0)
        with open(RAW_CACHE, "wb") as fh:
            pickle.dump(raw, fh)
        print(f"BOCPD pass on {len(profiles)} profiles in {time.time()-t0:.1f}s "
              f"(cached to {RAW_CACHE})")

    # No-refine baseline
    preds0 = {t: v[3][v[0]] for t, v in raw.items()}
    df0 = evaluate_run(preds0, truth)
    gs0 = global_summary(df0)
    print(f"[{ 'base_no_refine':38s}] F1@0.5={gs0['pooled_f1_0.5']:.3f} "
          f"F1@1.0={gs0['pooled_f1_1.0']:.3f} F1@2.0={gs0['pooled_f1_2.0']:.3f}")

    rows = [dict(variant="base_no_refine",
                 f1_05=gs0['pooled_f1_0.5'],
                 f1_10=gs0['pooled_f1_1.0'],
                 f1_20=gs0['pooled_f1_2.0'],
                 mean_f1_10=gs0['mean_f1_1.0'])]

    def _score(name: str, preds: dict) -> None:
        df = evaluate_run(preds, truth)
        gs = global_summary(df)
        print(f"[{name:38s}] F1@0.5={gs['pooled_f1_0.5']:.3f} "
              f"F1@1.0={gs['pooled_f1_1.0']:.3f} "
              f"F1@2.0={gs['pooled_f1_2.0']:.3f} "
              f"mean(F1@1.0)={gs['mean_f1_1.0']:.3f}")
        rows.append(dict(variant=name,
                         f1_05=gs['pooled_f1_0.5'],
                         f1_10=gs['pooled_f1_1.0'],
                         f1_20=gs['pooled_f1_2.0'],
                         mean_f1_10=gs['mean_f1_1.0']))

    for v in REFINE_VARIANTS:
        preds = {}
        for t, (cps_idx, sig, step_m, depth, _rlp) in raw.items():
            refined_idx = refine_cps_to_gradient(
                sig, cps_idx, step_m,
                search_radius_m=v["search_radius_m"],
                smooth_window=5,
                min_sep_m=0.5,
                strength_ratio=v.get("strength_ratio"),
                drop_below_quantile=v.get("drop_below_quantile"),
                snap_mode=v.get("snap_mode", "argmax"),
            )
            preds[t] = depth[refined_idx]
        _score(variant_name(v), preds)

    # --- BOCPD posterior-mass filter ---
    # Drop raw CPs whose BOCPD p(r=0 | x_{1:t}) is below the q-quantile of
    # the confidence across the same profile's CP set.  Apply BEFORE
    # refinement so we aren't confused by dedup in the refiner.
    print("\n--- posterior-confidence filter (pre-refine) on r0.3 argmax ---")
    for q in (0.0, 0.10, 0.20, 0.30, 0.50):
        preds = {}
        for t, (cps_idx, sig, step_m, depth, rlp_r0) in raw.items():
            if cps_idx.size and q > 0:
                conf = np.array([rlp_r0[int(c)] for c in cps_idx])
                thr = float(np.quantile(conf, q))
                keep_mask = conf >= thr
                cps_kept = cps_idx[keep_mask]
            else:
                cps_kept = cps_idx
            refined_idx = refine_cps_to_gradient(
                sig, cps_kept, step_m,
                search_radius_m=0.3, smooth_window=5, min_sep_m=0.5,
                snap_mode="argmax",
            )
            preds[t] = depth[refined_idx]
        _score(f"r0.3_argmax_conf_q{q:.2f}", preds)

    # --- Smoothing-window sweep at r=0.3 argmax.
    print("\n--- savgol smoothing-window sweep at r=0.3 argmax ---")
    for sw in (0, 5, 11, 21, 41, 81):
        preds = {}
        for t, (cps_idx, sig, step_m, depth, _rlp) in raw.items():
            refined_idx = refine_cps_to_gradient(
                sig, cps_idx, step_m,
                search_radius_m=0.3, smooth_window=sw, min_sep_m=0.5,
                snap_mode="argmax",
            )
            preds[t] = depth[refined_idx]
        _score(f"r0.3_argmax_sw{sw}", preds)

    # --- Smoothing-window sweep at r=0.7 argmax.
    print("\n--- savgol smoothing-window sweep at r=0.7 argmax ---")
    for sw in (5, 11, 21, 41, 81):
        preds = {}
        for t, (cps_idx, sig, step_m, depth, _rlp) in raw.items():
            refined_idx = refine_cps_to_gradient(
                sig, cps_idx, step_m,
                search_radius_m=0.7, smooth_window=sw, min_sep_m=0.5,
                snap_mode="argmax",
            )
            preds[t] = depth[refined_idx]
        _score(f"r0.7_argmax_sw{sw}", preds)

    # --- Hybrid: start from r=0.3 but use r=0.7 argmax if much stronger
    from scipy.signal import savgol_filter as _sgf
    print("\n--- hybrid r=0.3 primary, r=0.7 fallback if stronger ---")
    for ratio in (1.3, 1.5, 2.0, 3.0):
        preds = {}
        for t, (cps_idx, sig, step_m, depth, _rlp) in raw.items():
            sw = 5
            xs = _sgf(sig, sw, 3) if sw >= 3 else sig
            grad = np.abs(np.gradient(xs, step_m))
            r03 = max(1, int(round(0.3 / step_m)))
            r07 = max(1, int(round(0.7 / step_m)))
            refined = []
            for c in cps_idx:
                a3, b3 = max(0, int(c)-r03), min(len(grad), int(c)+r03+1)
                a7, b7 = max(0, int(c)-r07), min(len(grad), int(c)+r07+1)
                k3 = a3 + int(np.argmax(grad[a3:b3]))
                k7 = a7 + int(np.argmax(grad[a7:b7]))
                if grad[k7] >= ratio * grad[k3]:
                    refined.append(k7)
                else:
                    refined.append(k3)
            refined = sorted(set(refined))
            min_sep = max(1, int(round(0.5 / step_m)))
            if refined:
                kept = [refined[0]]
                for cc in refined[1:]:
                    if cc - kept[-1] >= min_sep:
                        kept.append(cc)
                refined = kept
            preds[t] = depth[np.asarray(refined, dtype=int)]
        _score(f"hybrid_r03_r07_ratio{ratio}", preds)

    # --- Augment with strong gradient peaks BOCPD missed
    # Union of BOCPD CPs and top-X% gradient peaks that are >=0.5m away from
    # any BOCPD CP.  Fully unsupervised.
    from scipy.signal import savgol_filter
    print("\n--- augment BOCPD with strong extra gradient peaks ---")
    for top_pct in (0.80, 0.85, 0.90, 0.95):
        preds = {}
        for t, (cps_idx, sig, step_m, depth, _rlp) in raw.items():
            # Compute gradient and its peaks
            sw = 5
            xs = savgol_filter(sig, sw, 3)
            grad = np.abs(np.gradient(xs, step_m))
            # strict local maxima
            peaks = np.flatnonzero(
                (grad[1:-1] > grad[:-2]) & (grad[1:-1] >= grad[2:])
            ) + 1
            if peaks.size == 0:
                preds[t] = depth[cps_idx]
                continue
            peak_g = grad[peaks]
            g_thr = float(np.quantile(peak_g, top_pct))
            strong_peaks = peaks[peak_g >= g_thr]
            # Keep only those >= 0.5m from any BOCPD CP (else already covered)
            min_sep_samples = max(1, int(round(0.5 / step_m)))
            if cps_idx.size:
                keep = []
                for p_idx in strong_peaks:
                    if np.min(np.abs(cps_idx - p_idx)) >= min_sep_samples:
                        keep.append(int(p_idx))
                extra = np.array(keep, dtype=int)
            else:
                extra = strong_peaks.astype(int)
            combined = np.sort(np.unique(np.concatenate([cps_idx, extra])))
            refined_idx = refine_cps_to_gradient(
                sig, combined, step_m,
                search_radius_m=0.3, smooth_window=5, min_sep_m=0.5,
                snap_mode="argmax",
            )
            preds[t] = depth[refined_idx]
        _score(f"aug_topg{top_pct:.2f}_r0.3_argmax", preds)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "refine_sweep.csv", index=False)
    print("\n=== Sweep summary (sorted by F1@1.0) ===")
    print(out.sort_values("f1_10", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
