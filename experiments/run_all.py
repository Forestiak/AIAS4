"""
python experiments/run_all.py [--subset N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bocpd_cpt import data as D
from bocpd_cpt.extensions import (
    UnivariateConfig, MultivariateConfig,
    run_univariate, run_multivariate,
    build_neighbor_cp_density,
)
from bocpd_cpt.hazards import spatial_prior_hazard
from bocpd_cpt.eval import evaluate_run, global_summary

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Experiment definitions
# -----------------------------------------------------------------------------

UNI_EXPERIMENTS = [
    # Paper-faithful baseline: Ic + NIG(0, 1, 1, 0.1) + const hazard (kappa = N).
    ("baseline_paper",
     UnivariateConfig(feature="Ic", hazard="const")),
    # Lower (data-agnostic) hazard — kappa=30 corresponds to a geotechnical
    # prior of ~0.6 m mean layer thickness on this 0.02 m grid, much closer
    # to observed marine CPT layering than the paper's one-CP-per-profile
    # prior.
    ("uni_Ic_k30_const",
     UnivariateConfig(feature="Ic", hazard="const", hazard_kappa=30.0)),
    # Depth-dependent hazard (expected layer thickness grows with depth).
    ("uni_Ic_depth",
     UnivariateConfig(feature="Ic", hazard="depth")),
    # Embed 0.5 m minimum thickness directly into hazard.
    ("uni_Ic_k30_minthick05",
     UnivariateConfig(feature="Ic", hazard="minthick",
                      min_thickness_m=0.5, hazard_kappa=30.0)),
    ("uni_Ic_depth_minthick05",
     UnivariateConfig(feature="Ic", hazard="depth+minthick",
                      min_thickness_m=0.5)),
    # ++ gradient-peak refinement (our main extension).
    # Aggressive (r=0.7): best F1@1.0 but loses F1@0.5 precision because the
    # global-argmax snap can move already-accurate CPs up to 0.7 m.
    ("uni_Ic_k30_const_refine",
     UnivariateConfig(feature="Ic", hazard="const", hazard_kappa=30.0,
                      refine=True)),
    ("uni_Ic_k30_minthick05_refine",
     UnivariateConfig(feature="Ic", hazard="minthick",
                      min_thickness_m=0.5, hazard_kappa=30.0,
                      refine=True)),
    ("uni_Ic_depth_minthick05_refine",
     UnivariateConfig(feature="Ic", hazard="depth+minthick",
                      min_thickness_m=0.5, refine=True)),
    # Conservative (r=0.3): Pareto-optimal from experiments/refine_sweep.py.
    # Improves F1@1.0/F1@2.0 vs no-refine without hurting tight-tolerance
    # precision.  Preferred default.
    ("uni_Ic_k30_minthick05_refine_r03",
     UnivariateConfig(feature="Ic", hazard="minthick",
                      min_thickness_m=0.5, hazard_kappa=30.0,
                      refine=True, refine_search_m=0.3)),
]

# Multivariate variants are kept available (import path intact) but excluded
# from the default ablation: the NIW forward pass is expensive and the
# extensions targeted for this thesis are univariate.  Enable with --mv.
MV_EXPERIMENTS = [
    ("mv_logQtn_logFr_const",
     MultivariateConfig(features=("logQtn", "logFr"), hazard="const")),
    ("mv_logQtn_logFr_depth_minthick",
     MultivariateConfig(features=("logQtn", "logFr"),
                        hazard="depth+minthick", min_thickness_m=0.5)),
    ("mv_logQtn_logFr_U2_depth_minthick",
     MultivariateConfig(features=("logQtn", "logFr", "U2"),
                        hazard="depth+minthick", min_thickness_m=0.5)),
]


def run_all(subset: int | None = None, include_mv: bool = False) -> None:
    t0 = time.time()
    profiles = D.load_profiles()
    if subset is not None:
        profiles = profiles[:subset]
    print(f"Loaded {len(profiles)} profiles in {time.time()-t0:.1f}s")

    strata = D.load_strata()
    truth_by_target = {p.target: D.true_boundaries_for(p.target, strata)
                       for p in profiles}
    for p in profiles:
        if not len(truth_by_target[p.target]):
            print(f"  WARNING: no strata for {p.target}")

    xy = {p.target: (p.x, p.y) for p in profiles
          if p.x is not None and p.y is not None}

    all_tables: dict[str, pd.DataFrame] = {}
    global_summaries: dict[str, dict] = {}

    # --- Univariate experiments ---
    for name, cfg in UNI_EXPERIMENTS:
        preds, timings = _run_single(profiles, cfg, runner=run_univariate)
        df = evaluate_run(preds, truth_by_target)
        gs = global_summary(df)
        all_tables[name] = df
        global_summaries[name] = gs
        _print_summary(name, gs, np.mean(timings))
        df.to_csv(RESULTS / f"{name}.csv", index=False)
        _save_predictions(preds, name)

    # --- Multivariate experiments (optional; expensive) ---
    if include_mv:
        for name, cfg in MV_EXPERIMENTS:
            preds, timings = _run_single(profiles, cfg, runner=run_multivariate)
            df = evaluate_run(preds, truth_by_target)
            gs = global_summary(df)
            all_tables[name] = df
            global_summaries[name] = gs
            _print_summary(name, gs, np.mean(timings))
            df.to_csv(RESULTS / f"{name}.csv", index=False)
            _save_predictions(preds, name)

    # --- Spatial extension built on top of the strongest non-spatial method ---
    # Pick best by pooled F1 @ 1.0 m — purely on pooled metrics, no ground-truth
    # inspection of individual profiles.
    best_name = max(global_summaries, key=lambda k: global_summaries[k]["pooled_f1_1.0"])
    print(f"\n>>> Spatial prior built on predictions of: {best_name}")
    base_pred = _load_predictions(best_name, {p.target for p in profiles})

    # Reconstruct the chosen config
    base_cfg = dict(UNI_EXPERIMENTS + MV_EXPERIMENTS)[best_name]
    is_mv = isinstance(base_cfg, MultivariateConfig)
    runner = run_multivariate if is_mv else run_univariate

    preds_sp: dict[str, np.ndarray] = {}
    timings_sp: list[float] = []
    for p in profiles:
        # For each target, build density from OTHER profiles only — never use
        # ground-truth boundaries.  We also use the *model's* predictions, not
        # the true layering, so nothing leaks.
        dens = build_neighbor_cp_density(
            p.target, base_pred, xy, p.depth,
            bandwidth_m=1.5, k_neighbors=6,
        )
        wrap = lambda h, d=dens: spatial_prior_hazard(
            h, prior_density=d, strength=0.75)
        t0 = time.time()
        _, cps_idx = runner(p, base_cfg, extra_hazard_wrap=wrap)
        timings_sp.append(time.time() - t0)
        preds_sp[p.target] = p.depth[cps_idx]

    df_sp = evaluate_run(preds_sp, truth_by_target)
    gs_sp = global_summary(df_sp)
    name_sp = f"{best_name}__spatial"
    all_tables[name_sp] = df_sp
    global_summaries[name_sp] = gs_sp
    _print_summary(name_sp, gs_sp, np.mean(timings_sp))
    df_sp.to_csv(RESULTS / f"{name_sp}.csv", index=False)

    # --- Combined global summary table ---
    combo = pd.DataFrame(global_summaries).T
    combo.index.name = "method"
    cols = [c for c in combo.columns
            if c.startswith(("pooled_f1_", "pooled_precision_", "pooled_recall_",
                             "mean_f1_"))]
    combo = combo[cols]
    combo.to_csv(RESULTS / "global_summary.csv")
    print("\n=== Global summary (pooled) ===")
    print(combo.round(3).to_string())

    with open(RESULTS / "global_summary.json", "w") as fh:
        json.dump(global_summaries, fh, indent=2, default=float)

    print(f"\nTotal wallclock: {time.time()-t0:.1f}s")


def _run_single(profiles, cfg, runner):
    preds: dict[str, np.ndarray] = {}
    timings: list[float] = []
    for p in profiles:
        t0 = time.time()
        _, cps_idx = runner(p, cfg)
        timings.append(time.time() - t0)
        preds[p.target] = p.depth[cps_idx]
    return preds, timings


def _save_predictions(preds, name: str) -> None:
    rows = []
    for t, cps in preds.items():
        for c in cps:
            rows.append({"Target": t, "depth": c})
    pd.DataFrame(rows).to_csv(RESULTS / f"pred__{name}.csv", index=False)


def _load_predictions(name: str, targets: set) -> dict[str, np.ndarray]:
    path = RESULTS / f"pred__{name}.csv"
    if not path.exists():
        return {t: np.array([]) for t in targets}
    df = pd.read_csv(path)
    out = {t: np.array([]) for t in targets}
    for t, g in df.groupby("Target"):
        out[t] = g["depth"].to_numpy(dtype=float)
    return out


def _print_summary(name: str, gs: dict, mean_time: float) -> None:
    print(
        f"[{name:45s}] "
        f"F1@0.5={gs['pooled_f1_0.5']:.3f} "
        f"F1@1.0={gs['pooled_f1_1.0']:.3f} "
        f"F1@2.0={gs['pooled_f1_2.0']:.3f} "
        f"mean(F1@1.0)={gs['mean_f1_1.0']:.3f} "
        f"[{mean_time:.2f}s/profile]"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None,
                    help="Run only the first N profiles (smoke test).")
    ap.add_argument("--mv", action="store_true",
                    help="Include the (expensive) multivariate variants.")
    args = ap.parse_args()
    run_all(subset=args.subset, include_mv=args.mv)
