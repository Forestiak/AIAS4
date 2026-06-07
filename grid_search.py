#!/usr/bin/env python3
"""Grid search over key BOCPD pipeline parameters

python3 grid_search.py
python3 grid_search.py --standard
python3 grid_search.py --extended
python3 grid_search.py --full
python3 grid_search.py --targets Loc-08 Loc-35 Loc-41
python3 grid_search.py --out results/my_search.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import main as M  # noqa: E402
from bocpd_cpt import data as D  # noqa: E402
from bocpd_cpt.eval import evaluate_run, global_summary  # noqa: E402


# ---------------------------------------------------------------------------
# Representative subset: stratified by F1 performance (from a prior full run)
# ---------------------------------------------------------------------------
REPRESENTATIVE_TARGETS = [
    "Loc-41",
    "Loc-52",
    "Loc-28",
    "Loc-22",
    "Loc-14",
    "Loc-50",
    "Loc-25",
    "Loc-08",
    "Loc-35",
    "Loc-34",
]


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

def build_grid(mode: str = "quick") -> list[dict]:
    """Return parameter combinations.

    mode="quick"    — ~30 combos (focuses on the two most impactful axes)
    mode="standard" — ~160 combos (adds refine_search variation)
    mode="extended" — ~640 combos (adds base_mode + auto_contrast)
    """
    # --- Axis 1: layer-thickness floor (the biggest single lever) ---
    # adaptive_min_thickness_m: floor for the kappa estimator (soft prior)
    # min_thickness_m: hard gate in the hazard function
    # Test both "tied" (both same value) and the user-discovered "decoupled"
    # setting (large soft prior, small hard gate).
    thickness_pairs_quick = [
        # (adaptive_min, hard_min)
        (0.50, 0.50),   # original default — both small
        (1.00, 0.50),   # decoupled: large soft prior, small hard gate
        (1.00, 1.00),   # tied at 1.0 m
        (1.50, 0.50),   # decoupled: very large soft prior
        (1.50, 1.50),   # tied at 1.5 m
    ]
    thickness_pairs_standard = thickness_pairs_quick + [
        (0.75, 0.75),
        (1.25, 1.25),
        (2.00, 2.00),
    ]

    # --- Axis 2: adaptive pre-smoothing ---
    # (use_adaptive_smooth, smooth_method, smooth_lam_max)
    smooth_combos_quick = [
        (False, "tv",     0.30),   # off
        (True,  "tv",     0.30),   # TV medium
        (True,  "savgol", 0.30),   # Savgol medium
    ]
    smooth_combos_standard = [
        (False, "tv",     0.30),   # off
        (True,  "tv",     0.15),   # TV gentle
        (True,  "tv",     0.30),   # TV medium
        (True,  "tv",     0.50),   # TV strong
        (True,  "savgol", 0.30),   # Savgol medium
    ]

    # --- Axis 3: gradient refinement settings ---
    drop_q_options = [True, False]

    refine_search_quick    = [0.5]
    refine_search_standard = [0.3, 0.5, 0.7]

    # --- Axis 4 (extended only): base_mode and auto-contrast ---
    base_modes       = ["const", "depth"] if mode == "extended" else ["const"]
    contrast_options = [False, True]      if mode == "extended" else [False]

    # --- Build grid ---
    thickness_pairs = (
        thickness_pairs_standard if mode in ("standard", "extended")
        else thickness_pairs_quick
    )
    smooth_combos = (
        smooth_combos_standard if mode in ("standard", "extended")
        else smooth_combos_quick
    )
    refine_search = (
        refine_search_standard if mode in ("standard", "extended")
        else refine_search_quick
    )

    grid = []
    for (amt, hmt), (use_sm, sm_meth, sm_lam), dq, rs, bm, ct in itertools.product(
        thickness_pairs,
        smooth_combos,
        drop_q_options,
        refine_search,
        base_modes,
        contrast_options,
    ):
        grid.append({
            "adaptive_min_thickness_m": amt,
            "min_thickness_m":          hmt,
            "use_adaptive_smooth":      use_sm,
            "smooth_method":            sm_meth,
            "smooth_lam_max":           sm_lam,
            "auto_refine_drop_quantile": dq,
            "refine_search_m":          rs,
            "base_mode":                bm,
            "use_auto_contrast":        ct,
        })
    return grid


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(combo: dict) -> M.PipelineConfig:
    amt = float(combo["adaptive_min_thickness_m"])
    adaptive = M.AdaptiveKappaConfig(
        enabled=True,
        base_thickness_m=M.ADAPTIVE_BASE_THICKNESS_M,
        min_thickness_m=amt,
        max_thickness_m=max(float(M.ADAPTIVE_MAX_THICKNESS_M), amt * 4.0),
        smooth_window=M.ADAPTIVE_SMOOTH_WINDOW,
        strong_peak_quantile=M.ADAPTIVE_STRONG_PEAK_QUANTILE,
        blend_base=M.ADAPTIVE_BLEND_BASE,
        blend_peak_spacing=M.ADAPTIVE_BLEND_PEAK_SPACING,
        blend_variation_scale=M.ADAPTIVE_BLEND_VARIATION_SCALE,
    )
    bm = combo.get("base_mode", "const")
    return M.PipelineConfig(
        name="search",
        label="SEARCH",
        base_mode=bm,
        hazard_kappa=None,
        use_min_thickness=True,
        min_thickness_m=float(combo["min_thickness_m"]),
        depth_kappa0_m=M.CANDIDATE_DEPTH_KAPPA0_M,
        depth_growth=M.CANDIDATE_DEPTH_GROWTH,
        adaptive=adaptive,
        refine=True,
        refine_search_m=float(combo["refine_search_m"]),
        refine_smooth_window=M.CANDIDATE_REFINE_SMOOTH_WINDOW,
        refine_strength_ratio=None,
        refine_drop_below_quantile=None,
        use_contrast=False,
        contrast_window_m=M.CANDIDATE_CONTRAST_WINDOW_M,
        contrast_strength=M.CANDIDATE_CONTRAST_STRENGTH,
        use_auto_contrast=bool(combo.get("use_auto_contrast", False)),
        auto_refine_drop_quantile=bool(combo["auto_refine_drop_quantile"]),
        use_spatial=False,
        use_adaptive_smooth=bool(combo["use_adaptive_smooth"]),
        smooth_method=str(combo.get("smooth_method", "tv")),
        smooth_roughness_threshold=M.CANDIDATE_SMOOTH_ROUGHNESS_THRESHOLD,
        smooth_roughness_max=M.CANDIDATE_SMOOTH_ROUGHNESS_MAX,
        smooth_lam_min=M.CANDIDATE_SMOOTH_LAM_MIN,
        smooth_lam_max=float(combo["smooth_lam_max"]),
    )


# ---------------------------------------------------------------------------
# Single-combination runner
# ---------------------------------------------------------------------------

def run_combo(
    profiles: list[D.Profile],
    truth: dict[str, np.ndarray],
    combo: dict,
) -> dict:
    cfg = build_config(combo)
    t0 = time.time()
    preds, _ = M.run_pipeline(profiles, cfg)
    elapsed = time.time() - t0

    eval_df = evaluate_run(preds, truth, M.TOLERANCES)
    gs = global_summary(eval_df, M.TOLERANCES)

    row: dict = {}
    for k, v in combo.items():
        row[k] = v
    row["elapsed_s"] = round(elapsed, 1)
    for tol in M.TOLERANCES:
        row[f"f1_{tol}m"]        = round(gs[f"pooled_f1_{tol}"],        4)
        row[f"precision_{tol}m"] = round(gs[f"pooled_precision_{tol}"], 4)
        row[f"recall_{tol}m"]    = round(gs[f"pooled_recall_{tol}"],    4)
        row[f"mean_f1_{tol}m"]   = round(gs[f"mean_f1_{tol}"],          4)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="BOCPD pipeline hyperparameter grid search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode_grp = ap.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--standard", action="store_true",
        help="Standard scan (~160 combos, includes refine_search variation and more smoothing options).",
    )
    mode_grp.add_argument(
        "--extended", action="store_true",
        help="Extended scan (~640 combos, also sweeps base_mode and auto-contrast).",
    )

    target_grp = ap.add_mutually_exclusive_group()
    target_grp.add_argument(
        "--full", action="store_true",
        help="Run on all available profiles (slowest, most reliable).",
    )
    target_grp.add_argument(
        "--subset", type=int, default=None,
        help="Use the first N profiles (alphabetical order).",
    )
    target_grp.add_argument(
        "--targets", nargs="+", default=None,
        help="Explicit Target IDs to evaluate.",
    )

    ap.add_argument(
        "--out", type=Path, default=Path("grid_search_results.csv"),
        help="Output CSV path (JSON with top-20 is saved alongside).",
    )
    ap.add_argument("--dataset-prefix", default="")
    args = ap.parse_args()

    # --- Determine search mode ---
    if args.extended:
        grid_mode = "extended"
    elif args.standard:
        grid_mode = "standard"
    else:
        grid_mode = "quick"

    # --- Load profiles ---
    D.configure_dataset(prefix=args.dataset_prefix)
    all_profiles = D.load_profiles()

    if args.full:
        profiles = all_profiles
        profile_desc = f"all {len(profiles)}"
    elif args.targets:
        wanted = set(args.targets)
        profiles = [p for p in all_profiles if p.target in wanted]
        profile_desc = f"{len(profiles)} specified"
    elif args.subset is not None:
        profiles = all_profiles[:args.subset]
        profile_desc = f"first {len(profiles)}"
    else:
        wanted = set(REPRESENTATIVE_TARGETS)
        profiles = [p for p in all_profiles if p.target in wanted]
        if not profiles:
            profiles = all_profiles[:10]
        profile_desc = f"{len(profiles)} representative"

    if not profiles:
        sys.exit("No profiles loaded.")

    strata = D.load_strata()
    truth = {p.target: D.true_boundaries_for(p.target, strata) for p in profiles}

    # --- Build grid ---
    grid = build_grid(mode=grid_mode)
    n = len(grid)
    est_s = n * len(profiles) * 2.7   # ~2.7 s per profile per combo
    print(
        f"Grid search ({grid_mode} mode): {n} combinations | "
        f"{profile_desc} profiles | "
        f"~{est_s/60:.0f} min estimated"
    )
    print("Profiles:", sorted(p.target for p in profiles))
    print()

    # --- Run ---
    results: list[dict] = []
    t_wall = time.time()

    for i, combo in enumerate(grid, 1):
        row = run_combo(profiles, truth, combo)
        results.append(row)

        f1  = row["f1_1.0m"]
        p1  = row["precision_1.0m"]
        r1  = row["recall_1.0m"]
        elapsed_i = row["elapsed_s"]
        done_so_far = time.time() - t_wall
        eta_s = done_so_far / i * (n - i)

        sm_tag = (
            "off" if not combo["use_adaptive_smooth"]
            else f"{combo['smooth_method']}:{combo['smooth_lam_max']:.2f}"
        )
        print(
            f"[{i:3d}/{n}] "
            f"amt={combo['adaptive_min_thickness_m']:.2f} "
            f"hmt={combo['min_thickness_m']:.2f} "
            f"sm={sm_tag:<12s} "
            f"dq={'Y' if combo['auto_refine_drop_quantile'] else 'N'} "
            f"rs={combo['refine_search_m']:.1f}  "
            f"F1@1m={f1:.4f}  P={p1:.3f}  R={r1:.3f}  "
            f"[{elapsed_i:.1f}s | eta {eta_s/60:.1f}min]"
        )

    total_elapsed = time.time() - t_wall

    # --- Build output DataFrame ---
    df = pd.DataFrame(results)
    df = df.sort_values(
        ["f1_1.0m", "f1_0.5m", "f1_2.0m"],
        ascending=False,
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    # --- Save CSV ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # --- Save JSON (top-20) ---
    json_out = args.out.with_suffix(".json")
    top20 = df.head(20).to_dict(orient="records")
    with open(json_out, "w") as fh:
        json.dump(top20, fh, indent=2)

    # --- Console summary ---
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Results saved to: {args.out}")
    print(f"Top-20 JSON saved to: {json_out}")

    # Top-10 table
    disp_cols = [
        "rank",
        "adaptive_min_thickness_m", "min_thickness_m",
        "use_adaptive_smooth", "smooth_method", "smooth_lam_max",
        "auto_refine_drop_quantile", "refine_search_m",
        "f1_0.5m", "f1_1.0m", "f1_2.0m",
        "precision_1.0m", "recall_1.0m",
    ]
    if args.extended:
        disp_cols.insert(8, "base_mode")
        disp_cols.insert(9, "use_auto_contrast")
    print("\n── Top 10 by F1@1.0m ──────────────────────────────────────────────────")
    print(df[[c for c in disp_cols if c in df.columns]].head(10).to_string(index=False))

    # Best config detail
    best = df.iloc[0]
    param_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in ("f1_", "precision_", "recall_", "mean_f1_", "rank", "elapsed"))
    ]
    print("\n── Best configuration ──────────────────────────────────────────────────")
    for col in param_cols:
        print(f"  {col:<35s} = {best[col]}")
    for tol in M.TOLERANCES:
        print(f"  F1@{tol:.1f}m  = {best[f'f1_{tol}m']:.4f}   "
              f"P={best[f'precision_{tol}m']:.4f}  R={best[f'recall_{tol}m']:.4f}")

    # Show how to replicate best config in main.py
    print("\n── Paste these lines into main.py to replicate the best result ─────────")
    print(f"  ADAPTIVE_MIN_THICKNESS_M           = {best['adaptive_min_thickness_m']}")
    print(f"  CANDIDATE_MIN_THICKNESS_M          = {best['min_thickness_m']}")
    print(f"  CANDIDATE_USE_ADAPTIVE_SMOOTH      = {best['use_adaptive_smooth']}")
    if best['use_adaptive_smooth']:
        print(f"  CANDIDATE_SMOOTH_METHOD            = \"{best['smooth_method']}\"")
        print(f"  CANDIDATE_SMOOTH_LAM_MAX           = {best['smooth_lam_max']}")
    print(f"  CANDIDATE_AUTO_REFINE_DROP_QUANTILE= {best['auto_refine_drop_quantile']}")
    print(f"  CANDIDATE_REFINE_SEARCH_M          = {best['refine_search_m']}")
    if args.extended:
        print(f"  CANDIDATE_BASE_MODE                = \"{best.get('base_mode', 'const')}\"")
        print(f"  CANDIDATE_USE_AUTO_CONTRAST        = {best.get('use_auto_contrast', False)}")


if __name__ == "__main__":
    main()
