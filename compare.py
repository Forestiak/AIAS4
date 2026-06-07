"""
baseline — NIG, const hazard, κ=N (paper-faithful)
local    — minthick κ=30 + gradient-peak refinement r=0.3 m (Pareto-best)
spatial  — local + neighbour-density prior (overall best, needs full set)

All three methods are evaluated on the same set of profiles.  Because the
spatial prior requires x,y coordinates, only profiles present in all three
source files (CPT, Strata, Location) are used

python compare.py                    # geo-located profiles
python compare.py --targets T1 T2    # specific targets
python compare.py --subset 10        # first N profiles
python compare.py --save             # write CSVs to results/

"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bocpd_cpt import data as D
from bocpd_cpt.extensions import (
    UnivariateConfig, run_univariate,
    build_neighbor_cp_density,
)
from bocpd_cpt.hazards import spatial_prior_hazard
from bocpd_cpt.eval import evaluate_run, global_summary

# =============================================================================
# Settings
# =============================================================================

BASELINE_CFG = UnivariateConfig(
    feature="Ic",
    hazard="const",          # kappa = N (one CP per profile prior, per paper)
)

LOCAL_CFG = UnivariateConfig(
    feature="Ic",
    hazard="minthick",       # const hazard + 0.5 m minimum layer thickness
    hazard_kappa=30.0,       # ~0.6 m mean layer thickness prior
    min_thickness_m=0.5,     # geotechnical floor (no layers < 0.5 m)
    refine=True,             # snap CPs to nearest |dIc/dz| peak
    refine_search_m=0.3,     # search ±0.3 m (Pareto-optimal from sweep)
)

# Spatial extension hyperparameters (applied on top of LOCAL_CFG)
SPATIAL_BANDWIDTH_M = 1.5
SPATIAL_K_NEIGHBORS = 6
SPATIAL_STRENGTH    = 0.75

# Tolerances (metres) at which F1 / precision / recall are reported
TOLERANCES = [0.5, 1.0, 2.0]

# =============================================================================


def filter_to_located(profiles: list) -> tuple[list, list]:
    """Split profiles into (located, dropped).

    Located = has x,y coordinates from the location file.
    All three methods are run only on located profiles so that the spatial
    prior has valid neighbour coordinates for every profile in the set.
    """
    located = [p for p in profiles if p.x is not None and p.y is not None]
    dropped = [p for p in profiles if p.x is None or p.y is None]
    return located, dropped


def run_pipeline(profiles: list, cfg: UnivariateConfig) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    for p in profiles:
        _, cps_idx = run_univariate(p, cfg)
        preds[p.target] = p.depth[cps_idx]
    return preds


def run_spatial(profiles: list,
                pred_local: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Re-run LOCAL_CFG with a neighbour-density hazard built from pred_local."""
    xy = {p.target: (p.x, p.y) for p in profiles}
    preds: dict[str, np.ndarray] = {}
    for p in profiles:
        dens = build_neighbor_cp_density(
            p.target, pred_local, xy, p.depth,
            bandwidth_m=SPATIAL_BANDWIDTH_M,
            k_neighbors=SPATIAL_K_NEIGHBORS,
        )
        wrap = lambda h, d=dens: spatial_prior_hazard(
            h, prior_density=d, strength=SPATIAL_STRENGTH)
        _, cps_idx = run_univariate(p, LOCAL_CFG, extra_hazard_wrap=wrap)
        preds[p.target] = p.depth[cps_idx]
    return preds


def print_summary(label: str, gs: dict, elapsed: float) -> None:
    print(f"\n{'─'*70}")
    print(f"  {label}  [{elapsed:.1f}s]")
    print(f"{'─'*70}")
    for t in TOLERANCES:
        print(
            f"  @{t:.1f}m   "
            f"F1={gs[f'pooled_f1_{t}']:.3f}  "
            f"P={gs[f'pooled_precision_{t}']:.3f}  "
            f"R={gs[f'pooled_recall_{t}']:.3f}  "
            f"mean-F1={gs[f'mean_f1_{t}']:.3f}"
        )


def print_deltas(gs_base: dict, gs_local: dict, gs_spatial: dict) -> None:
    print(f"\n{'─'*70}")
    print("  Delta vs baseline")
    print(f"  {'':6s}  {'local':>10s}  {'spatial':>10s}")
    print(f"{'─'*70}")
    fmt = lambda d: f"{'+'if d>=0 else ''}{d:.3f}"
    for t in TOLERANCES:
        key = f"pooled_f1_{t}"
        print(
            f"  @{t:.1f}m   "
            f"{fmt(gs_local[key] - gs_base[key]):>10s}  "
            f"{fmt(gs_spatial[key] - gs_base[key]):>10s}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Baseline vs. local vs. spatial BOCPD on CPT profiles."
    )
    ap.add_argument(
        "--targets", nargs="+", metavar="T",
        help="Run only these Target IDs (e.g. Loc-01 Loc-03)."
    )
    ap.add_argument(
        "--exclude", nargs="+", metavar="T",
        help="Skip these Target IDs (e.g. Loc-01 Loc-03)."
    )
    ap.add_argument(
        "--subset", type=int, default=None,
        help="Run only the first N profiles (smoke-test mode)."
    )
    ap.add_argument(
        "--save", action="store_true",
        help="Write per-profile CSVs and a summary CSV to results/."
    )
    args = ap.parse_args()

    profiles = D.load_profiles()

    if args.targets:
        want = set(args.targets)
        profiles = [p for p in profiles if p.target in want]
        if not profiles:
            sys.exit(f"No profiles found for targets: {args.targets}")
    elif args.subset is not None:
        profiles = profiles[: args.subset]

    if args.exclude:
        skip = set(args.exclude)
        profiles = [p for p in profiles if p.target not in skip]
        if not profiles:
            sys.exit("All profiles were excluded.")

    profiles, dropped = filter_to_located(profiles)
    if not profiles:
        sys.exit("No geo-located profiles remain after filtering.")
    if dropped:
        print(
            f"Dropped {len(dropped)} profile(s) with no location data "
            f"(not in Input_Location file): "
            + ", ".join(p.target for p in dropped)
        )
    print(f"Running on {len(profiles)} profile(s).")

    strata = D.load_strata()
    truth = {p.target: D.true_boundaries_for(p.target, strata) for p in profiles}

    # --- Baseline ---
    t0 = time.time()
    pred_base = run_pipeline(profiles, BASELINE_CFG)
    t_base = time.time() - t0
    gs_base = global_summary(evaluate_run(pred_base, truth))
    print_summary("BASELINE  (NIG, const hazard, κ=N)", gs_base, t_base)

    # --- Local best ---
    t0 = time.time()
    pred_local = run_pipeline(profiles, LOCAL_CFG)
    t_local = time.time() - t0
    gs_local = global_summary(evaluate_run(pred_local, truth))
    print_summary("LOCAL  (minthick κ=30, min 0.5 m, refine r=0.3 m)", gs_local, t_local)

    # --- Spatial (built on local predictions as neighbour prior) ---
    t0 = time.time()
    pred_spatial = run_spatial(profiles, pred_local)
    t_spatial = time.time() - t0
    gs_spatial = global_summary(evaluate_run(pred_spatial, truth))
    print_summary(
        f"SPATIAL  (local + neighbour density bw={SPATIAL_BANDWIDTH_M}m "
        f"k={SPATIAL_K_NEIGHBORS} s={SPATIAL_STRENGTH})",
        gs_spatial, t_spatial,
    )

    print_deltas(gs_base, gs_local, gs_spatial)

    if args.save:
        out = Path(__file__).parent / "results"
        out.mkdir(exist_ok=True)
        evaluate_run(pred_base, truth).to_csv(out / "compare__baseline.csv", index=False)
        evaluate_run(pred_local, truth).to_csv(out / "compare__local.csv", index=False)
        evaluate_run(pred_spatial, truth).to_csv(out / "compare__spatial.csv", index=False)
        summary = pd.DataFrame({
            "baseline": gs_base,
            "local": gs_local,
            "spatial": gs_spatial,
        }).T
        summary.index.name = "method"
        summary.to_csv(out / "compare__summary.csv")

        bounds_dir = out / "predicted_boundary_depths"
        bounds_dir.mkdir(exist_ok=True)
        for name, preds in (
            ("baseline", pred_base),
            ("local",    pred_local),
            ("spatial",  pred_spatial),
        ):
            rows = [
                {"Target": target, "depth_m": depth}
                for target, depths in sorted(preds.items())
                for depth in depths
            ]
            pd.DataFrame(rows).to_csv(
                bounds_dir / f"{name}_predicted_boundaries.csv", index=False
            )
        print(f"\nSaved CSVs to {out}/")
        print(f"Saved boundary depths to {bounds_dir}/")


if __name__ == "__main__":
    main()
