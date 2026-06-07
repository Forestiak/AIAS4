from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import main as M  # noqa: E402
from bocpd_cpt import data as D  # noqa: E402
from bocpd_cpt.eval import evaluate_run, global_summary  # noqa: E402


METHOD_LABELS = {
    "baseline": "Baseline",
    "local_adapt": "Local adaptive",
    "final_spatial_adapt": "Final spatial/adaptive",
}

METHOD_COLORS = {
    "baseline": "#2563eb",
    "local_adapt": "#f97316",
    "final_spatial_adapt": "#dc2626",
}


def _boundary_rows(preds: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    return [
        {"Target": target, "depth_m": float(depth)}
        for target, depths in sorted(preds.items())
        for depth in depths
    ]


def save_outputs(
    outdir: Path,
    preds_by_name: dict[str, dict[str, np.ndarray]],
    eval_by_name: dict[str, pd.DataFrame],
    diag_by_name: dict[str, pd.DataFrame],
    summary_by_name: dict[str, dict[str, float]],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    bounds_dir = outdir / "predicted_boundary_depths"
    bounds_dir.mkdir(exist_ok=True)

    summary = pd.DataFrame(summary_by_name).T
    summary.index.name = "method"
    summary.to_csv(outdir / "summary.csv")

    for name in preds_by_name:
        eval_by_name[name].to_csv(outdir / f"{name}.csv", index=False)
        diag_by_name[name].to_csv(outdir / f"{name}__diagnostics.csv", index=False)
        analysis = eval_by_name[name].merge(diag_by_name[name], on="Target", how="left", suffixes=("", "_diag"))
        analysis.to_csv(outdir / f"{name}__analysis.csv", index=False)
        pd.DataFrame(_boundary_rows(preds_by_name[name])).to_csv(
            bounds_dir / f"{name}_predicted_boundaries.csv",
            index=False,
        )


def pooled_counts(eval_df: pd.DataFrame, tol: float) -> tuple[int, int, int, float, float, float]:
    tp = int(eval_df[f"tp_{tol}"].sum())
    fp = int(eval_df[f"fp_{tol}"].sum())
    fn = int(eval_df[f"fn_{tol}"].sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f1


def corr(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return 0.0
    x = x[ok]
    y = y[ok]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_ci(values: pd.Series, n_boot: int = 5000, seed: int = 12) -> tuple[float, float, float]:
    x = values.astype(float).to_numpy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def build_profile_statistics(outdir: Path) -> pd.DataFrame:
    baseline = pd.read_csv(outdir / "baseline.csv").set_index("Target")
    local = pd.read_csv(outdir / "local_adapt.csv").set_index("Target")
    final = pd.read_csv(outdir / "final_spatial_adapt.csv").set_index("Target")
    diag = pd.read_csv(outdir / "final_spatial_adapt__diagnostics.csv").set_index("Target")

    rows = []
    for target in baseline.index:
        span = float(diag.loc[target, "profile_depth_span_m"])
        true_n = int(baseline.loc[target, "n_true"])
        base_pred = int(baseline.loc[target, "n_pred"])
        local_pred = int(local.loc[target, "n_pred"])
        final_pred = int(final.loc[target, "n_pred"])
        delta = float(final.loc[target, "f1_1.0"] - baseline.loc[target, "f1_1.0"])
        rows.append(
            {
                "Target": target,
                "n_samples": int(diag.loc[target, "profile_n_samples"]),
                "depth_span_m": span,
                "step_mean_m": float(diag.loc[target, "profile_step_mean_m"]),
                "true_boundaries": true_n,
                "baseline_pred_boundaries": base_pred,
                "local_pred_boundaries": local_pred,
                "final_pred_boundaries": final_pred,
                "true_avg_layer_m": span / (true_n + 1) if true_n >= 0 else float("nan"),
                "baseline_avg_run_m": span / (base_pred + 1),
                "local_avg_run_m": span / (local_pred + 1),
                "final_avg_run_m": span / (final_pred + 1),
                "Ic_mean": float(diag.loc[target, "profile_Ic_mean"]),
                "Ic_variance": float(diag.loc[target, "profile_Ic_var"]),
                "Ic_std": float(diag.loc[target, "profile_Ic_std"]),
                "Ic_iqr": float(diag.loc[target, "profile_Ic_iqr"]),
                "Ic_mad": float(diag.loc[target, "profile_Ic_mad"]),
                "Ic_roughness": float(diag.loc[target, "profile_Ic_roughness"]),
                "Ic_abs_grad_mean": float(diag.loc[target, "profile_Ic_abs_grad_mean"]),
                "Ic_abs_grad_q75": float(diag.loc[target, "profile_Ic_abs_grad_q75"]),
                "Ic_abs_grad_q95": float(diag.loc[target, "profile_Ic_abs_grad_q95"]),
                "adaptive_thickness_m": float(diag.loc[target, "adaptive_thickness_m"]),
                "adaptive_kappa_samples": float(diag.loc[target, "adaptive_kappa_samples"]),
                "strong_peak_count": int(diag.loc[target, "n_strong_peaks"]),
                "spatial_density_mean": float(diag.loc[target, "spatial_density_mean"]),
                "spatial_density_max": float(diag.loc[target, "spatial_density_max"]),
                "baseline_f1_05m": float(baseline.loc[target, "f1_0.5"]),
                "baseline_f1_1m": float(baseline.loc[target, "f1_1.0"]),
                "baseline_f1_2m": float(baseline.loc[target, "f1_2.0"]),
                "local_f1_05m": float(local.loc[target, "f1_0.5"]),
                "local_f1_1m": float(local.loc[target, "f1_1.0"]),
                "local_f1_2m": float(local.loc[target, "f1_2.0"]),
                "final_f1_05m": float(final.loc[target, "f1_0.5"]),
                "final_f1_1m": float(final.loc[target, "f1_1.0"]),
                "final_f1_2m": float(final.loc[target, "f1_2.0"]),
                "delta_local_vs_baseline_f1_1m": float(local.loc[target, "f1_1.0"] - baseline.loc[target, "f1_1.0"]),
                "delta_final_vs_local_f1_1m": float(final.loc[target, "f1_1.0"] - local.loc[target, "f1_1.0"]),
                "delta_final_vs_baseline_f1_1m": delta,
            }
        )
    stats_df = pd.DataFrame(rows).sort_values("Target").reset_index(drop=True)
    stats_df.to_csv(outdir / "profile_statistics.csv", index=False)
    return stats_df


def _setup_depth_axis(ax: plt.Axes, profile: D.Profile, signal_name: str = "Ic") -> None:
    ax.invert_yaxis()
    ax.set_ylabel("Depth [m]")
    ax.set_xlabel(signal_name)
    ax.grid(True, alpha=0.22, linewidth=0.7)


def plot_final_boundaries(
    outdir: Path,
    profiles: list[D.Profile],
    truth: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    eval_df: pd.DataFrame,
) -> None:
    plot_dir = outdir / "boundary_plots"
    plot_dir.mkdir(exist_ok=True)
    metrics = eval_df.set_index("Target")

    for profile in profiles:
        fig, ax = plt.subplots(figsize=(5.2, 8.6))
        ax.plot(profile.features["Ic"], profile.depth, color="#111827", lw=0.75, alpha=0.78, label="Raw Ic signal")
        for depth in truth.get(profile.target, []):
            ax.axhline(depth, color="#15803d", lw=1.2, ls="--", alpha=0.82, label="True boundary")
        for depth in preds.get(profile.target, []):
            ax.axhline(depth, color="#dc2626", lw=1.0, alpha=0.9, label="Final spatial/adaptive prediction")
        _setup_depth_axis(ax, profile)
        row = metrics.loc[profile.target]
        ax.set_title(
            f"{profile.target} | true={int(row['n_true'])}, pred={int(row['n_pred'])}, F1@1m={row['f1_1.0']:.3f}",
            fontsize=10,
        )
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(plot_dir / f"{profile.target}.svg")
        plt.close(fig)

    write_plot_gallery(
        outdir / "boundary_plots_gallery.md",
        "Final Spatial Boundary Plots",
        "Black traces are raw Ic. Green dashed lines are true boundaries. Red lines are final spatial/adaptive predictions.",
        "boundary_plots",
        [p.target for p in profiles],
    )


def plot_method_comparison(
    outdir: Path,
    profiles: list[D.Profile],
    truth: dict[str, np.ndarray],
    preds_by_name: dict[str, dict[str, np.ndarray]],
    eval_by_name: dict[str, pd.DataFrame],
) -> None:
    plot_dir = outdir / "method_comparison_plots"
    plot_dir.mkdir(exist_ok=True)
    final_metrics = eval_by_name["final_spatial_adapt"].set_index("Target")

    for profile in profiles:
        fig, ax = plt.subplots(figsize=(5.8, 8.8))
        ax.plot(profile.features["Ic"], profile.depth, color="#111827", lw=0.72, alpha=0.58, label="Raw Ic signal")
        for depth in truth.get(profile.target, []):
            ax.axhline(depth, color="#15803d", lw=1.25, ls="--", alpha=0.8, label="True boundary")
        for name, linestyle in [("baseline", ":"), ("local_adapt", "-."), ("final_spatial_adapt", "-")]:
            for depth in preds_by_name[name].get(profile.target, []):
                ax.axhline(
                    depth,
                    color=METHOD_COLORS[name],
                    lw=1.05 if name != "final_spatial_adapt" else 1.25,
                    ls=linestyle,
                    alpha=0.86,
                    label=METHOD_LABELS[name],
                )
        _setup_depth_axis(ax, profile)
        row = final_metrics.loc[profile.target]
        ax.set_title(
            f"{profile.target} | final pred={int(row['n_pred'])}, final F1@1m={row['f1_1.0']:.3f}",
            fontsize=10,
        )
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(plot_dir / f"{profile.target}.svg")
        plt.close(fig)

    write_plot_gallery(
        outdir / "method_comparison_plots_gallery.md",
        "Method Comparison Boundary Plots",
        "Black traces are raw Ic. Green dashed lines are true boundaries. Blue is baseline, orange is local adaptive, red is final spatial/adaptive.",
        "method_comparison_plots",
        [p.target for p in profiles],
    )


def write_plot_gallery(path: Path, title: str, intro: str, plot_dir: str, targets: list[str]) -> None:
    lines = [f"# {title}", "", intro, ""]
    for target in targets:
        lines += [f"## {target}", "", f"![{target}]({plot_dir}/{target}.svg)", ""]
    path.write_text("\n".join(lines))


def plot_aggregate_figures(outdir: Path, stats_df: pd.DataFrame, summary_by_name: dict[str, dict[str, float]]) -> None:
    plot_dir = outdir / "summary_plots"
    plot_dir.mkdir(exist_ok=True)

    methods = ["baseline", "local_adapt", "final_spatial_adapt"]
    x = np.arange(len(methods))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, tol in enumerate(M.TOLERANCES):
        vals = [summary_by_name[m][f"pooled_f1_{tol}"] for m in methods]
        ax.bar(x + (i - 1) * width, vals, width=width, label=f"F1 @ {tol:g} m")
    ax.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=12, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pooled F1")
    ax.set_title("Boundary Detection Performance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(plot_dir / "method_f1_summary.svg")
    plt.close(fig)

    ordered = stats_df.sort_values("delta_final_vs_baseline_f1_1m")
    fig, ax = plt.subplots(figsize=(9.2, max(5.0, len(ordered) * 0.18)))
    colors = np.where(ordered["delta_final_vs_baseline_f1_1m"] >= 0, "#15803d", "#dc2626")
    ax.barh(ordered["Target"], ordered["delta_final_vs_baseline_f1_1m"], color=colors)
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_xlabel("F1 @ 1.0 m delta, final spatial/adaptive minus baseline")
    ax.set_title("Per-profile Improvement")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(plot_dir / "per_profile_delta_f1_1m.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(stats_df["true_boundaries"], stats_df["baseline_pred_boundaries"], label="Baseline", color=METHOD_COLORS["baseline"], alpha=0.75)
    ax.scatter(stats_df["true_boundaries"], stats_df["local_pred_boundaries"], label="Local adaptive", color=METHOD_COLORS["local_adapt"], alpha=0.75)
    ax.scatter(stats_df["true_boundaries"], stats_df["final_pred_boundaries"], label="Final spatial/adaptive", color=METHOD_COLORS["final_spatial_adapt"], alpha=0.75)
    max_n = int(max(stats_df[["true_boundaries", "baseline_pred_boundaries", "local_pred_boundaries", "final_pred_boundaries"]].max()))
    ax.plot([0, max_n], [0, max_n], color="#111827", lw=0.8, alpha=0.5)
    ax.set_xlabel("True boundary count")
    ax.set_ylabel("Predicted boundary count")
    ax.set_title("Boundary Count Calibration")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(plot_dir / "predicted_vs_true_counts.svg")
    plt.close(fig)

    for col, label in [
        ("Ic_std", "Ic standard deviation"),
        ("Ic_roughness", "Ic roughness"),
        ("true_avg_layer_m", "Average true layer thickness [m]"),
        ("spatial_density_max", "Maximum spatial-prior density"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.scatter(stats_df[col], stats_df["delta_final_vs_baseline_f1_1m"], color="#334155", alpha=0.78)
        ax.axhline(0, color="#111827", lw=0.8, alpha=0.55)
        ax.set_xlabel(label)
        ax.set_ylabel("Final minus baseline F1 @ 1.0 m")
        ax.set_title(f"Improvement vs {label}")
        ax.grid(alpha=0.22)
        fig.tight_layout()
        fig.savefig(plot_dir / f"delta_vs_{col}.svg")
        plt.close(fig)


def write_excluded_profiles(outdir: Path, dropped: list[D.Profile]) -> None:
    pd.DataFrame(
        [
            {
                "Target": p.target,
                "x": p.x,
                "y": p.y,
                "reason": "missing or non-finite X/Y coordinate",
            }
            for p in dropped
        ]
    ).to_csv(outdir / "excluded_profiles.csv", index=False)


def write_config(
    outdir: Path,
    dataset_prefix: str,
    paths: tuple[Path, Path, Path],
    configs: dict[str, M.PipelineConfig],
    elapsed_by_name: dict[str, float],
) -> None:
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_prefix": dataset_prefix,
        "dataset_paths": {
            "cpt": str(paths[0]),
            "locations": str(paths[1]),
            "strata": str(paths[2]),
        },
        "tolerances_m": M.TOLERANCES,
        "elapsed_seconds": elapsed_by_name,
        "methods": {name: asdict(cfg) for name, cfg in configs.items()},
    }
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2))


def write_report(
    outdir: Path,
    profiles: list[D.Profile],
    dropped: list[D.Profile],
    eval_by_name: dict[str, pd.DataFrame],
    summary_by_name: dict[str, dict[str, float]],
    stats_df: pd.DataFrame,
    elapsed_by_name: dict[str, float],
) -> None:
    lines = [
        "# BOCPD Current Dataset Results",
        "",
        "## Executive Summary",
        "",
        f"The comparison used {len(profiles)} profiles with finite spatial coordinates. "
        f"{len(dropped)} profile(s) were excluded because spatial information was missing or non-finite; the same filtered set was used for baseline, local adaptive, and final spatial/adaptive methods.",
        "",
        "| Method | Predicted boundaries | F1 @ 0.5 m | F1 @ 1.0 m | F1 @ 2.0 m | Mean profile F1 @ 1.0 m | Runtime s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["baseline", "local_adapt", "final_spatial_adapt"]:
        df = eval_by_name[name]
        s = summary_by_name[name]
        lines.append(
            f"| {METHOD_LABELS[name]} | {int(df['n_pred'].sum())} | "
            f"{s['pooled_f1_0.5']:.3f} | {s['pooled_f1_1.0']:.3f} | {s['pooled_f1_2.0']:.3f} | "
            f"{s['mean_f1_1.0']:.3f} | {elapsed_by_name[name]:.1f} |"
        )

    lines += [
        "",
        "At 1.0 m tolerance, the pooled counts are:",
        "",
        "| Method | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["baseline", "local_adapt", "final_spatial_adapt"]:
        tp, fp, fn, p, r, f1 = pooled_counts(eval_by_name[name], 1.0)
        lines.append(f"| {METHOD_LABELS[name]} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | {f1:.3f} |")

    base_f1 = summary_by_name["baseline"]["pooled_f1_1.0"]
    local_f1 = summary_by_name["local_adapt"]["pooled_f1_1.0"]
    final_f1 = summary_by_name["final_spatial_adapt"]["pooled_f1_1.0"]
    local_delta = local_f1 - base_f1
    spatial_delta = final_f1 - local_f1
    final_delta = final_f1 - base_f1

    mean_delta, lo_delta, hi_delta = bootstrap_ci(stats_df["delta_final_vs_baseline_f1_1m"])
    mean_local_delta, lo_local_delta, hi_local_delta = bootstrap_ci(stats_df["delta_local_vs_baseline_f1_1m"])
    mean_spatial_delta, lo_spatial_delta, hi_spatial_delta = bootstrap_ci(stats_df["delta_final_vs_local_f1_1m"])

    delta = stats_df["delta_final_vs_baseline_f1_1m"]
    improved = int((delta > 1e-9).sum())
    same = int((delta.abs() <= 1e-9).sum())
    worse = int((delta < -1e-9).sum())
    top_gain = stats_df.sort_values("delta_final_vs_baseline_f1_1m", ascending=False).head(5)
    top_loss = stats_df.sort_values("delta_final_vs_baseline_f1_1m").head(5)

    lines += [
        "",
        "## Main Interpretation",
        "",
        f"At the headline 1.0 m tolerance, local adaptive changes improve pooled F1 by {local_delta:+.3f} over baseline. "
        f"The final spatial prior adds {spatial_delta:+.3f} beyond local adaptive, giving {final_delta:+.3f} total improvement over baseline.",
        "",
        "Using profile-level deltas and bootstrap resampling across profiles:",
        "",
        f"- Final minus baseline mean profile F1 delta: {mean_delta:+.3f} with approximate 95% CI [{lo_delta:+.3f}, {hi_delta:+.3f}].",
        f"- Local adaptive minus baseline mean profile F1 delta: {mean_local_delta:+.3f} with approximate 95% CI [{lo_local_delta:+.3f}, {hi_local_delta:+.3f}].",
        f"- Final spatial/adaptive minus local adaptive mean profile F1 delta: {mean_spatial_delta:+.3f} with approximate 95% CI [{lo_spatial_delta:+.3f}, {hi_spatial_delta:+.3f}].",
        "",
        "This means the largest contribution comes from the local changes: adaptive hazard scale, minimum thickness, and gradient refinement. The spatial prior is leakage-safe because it uses neighbouring predicted boundaries rather than true labels, but on this dataset its incremental gain is small.",
        "",
        f"Profile-level final-vs-baseline result at F1 @ 1.0 m: {improved} improved, {same} unchanged, {worse} worsened.",
        "",
        "Largest gains: "
        + ", ".join(f"`{r.Target}` ({r.delta_final_vs_baseline_f1_1m:+.3f})" for r in top_gain.itertuples(index=False))
        + ".",
        "",
        "Largest losses: "
        + ", ".join(f"`{r.Target}` ({r.delta_final_vs_baseline_f1_1m:+.3f})" for r in top_loss.itertuples(index=False))
        + ".",
        "",
        "## What Drives Profile Differences",
        "",
        "The full table is saved as `profile_statistics.csv`. It includes sample count, mean/variance/std of Ic, gradient strength, roughness, true and predicted boundary counts, average true layer thickness, average predicted run length, adaptive thickness, and spatial-prior statistics.",
        "",
    ]

    corr_cols = [
        ("true_boundaries", "true boundary count"),
        ("true_avg_layer_m", "average true layer thickness"),
        ("baseline_pred_boundaries", "baseline predicted boundary count"),
        ("Ic_std", "Ic standard deviation"),
        ("Ic_roughness", "Ic roughness"),
        ("Ic_abs_grad_q95", "upper-tail Ic gradient"),
        ("adaptive_thickness_m", "adaptive thickness"),
        ("spatial_density_max", "maximum spatial density"),
    ]
    y = stats_df["delta_final_vs_baseline_f1_1m"]
    corr_text = [f"`{label}` r={corr(stats_df[col], y):+.2f}" for col, label in corr_cols]
    lines += [
        "Simple Pearson correlations with final-vs-baseline F1 @ 1.0 m delta:",
        "",
        ", ".join(corr_text) + ".",
        "",
        "These correlations are descriptive, not causal. They are useful for explaining why some profiles are easier: cleaner signal jumps, realistic boundary density, and layer spacing compatible with the minimum-thickness/adaptive prior usually help; noisy or very dense stratigraphy can hurt.",
        "",
        "## Report-ready Plots",
        "",
        "Summary plots:",
        "",
        "![Method F1 summary](summary_plots/method_f1_summary.svg)",
        "",
        "![Per-profile F1 delta](summary_plots/per_profile_delta_f1_1m.svg)",
        "",
        "![Predicted vs true counts](summary_plots/predicted_vs_true_counts.svg)",
        "",
        "Additional diagnostic scatter plots are in `summary_plots/`.",
        "",
        "Per-profile boundary plots:",
        "",
        "- `boundary_plots_gallery.md`: raw Ic, true boundaries, and final spatial/adaptive predictions.",
        "- `method_comparison_plots_gallery.md`: raw Ic, true boundaries, and predictions from all three methods.",
        "",
        "## Per-profile Digest",
        "",
        "| Profile | Samples | True | Pred B | Pred L | Pred F | Avg true layer m | Avg final run m | F1 B | F1 L | F1 F | Delta F-B | Ic mean | Ic var | Ic std | Ic rough. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in stats_df.itertuples(index=False):
        lines.append(
            f"| {r.Target} | {r.n_samples} | {r.true_boundaries} | {r.baseline_pred_boundaries} | "
            f"{r.local_pred_boundaries} | {r.final_pred_boundaries} | {r.true_avg_layer_m:.2f} | "
            f"{r.final_avg_run_m:.2f} | {r.baseline_f1_1m:.3f} | {r.local_f1_1m:.3f} | "
            f"{r.final_f1_1m:.3f} | {r.delta_final_vs_baseline_f1_1m:+.3f} | "
            f"{r.Ic_mean:.3f} | {r.Ic_variance:.3f} | {r.Ic_std:.3f} | {r.Ic_roughness:.4f} |"
        )

    lines += [
        "",
        "## Method Notes",
        "",
        "Baseline is the standard univariate Ic BOCPD with a constant hazard. Local adaptive keeps the same signal but adds an unsupervised profile-specific hazard scale, minimum-thickness gating, and gradient-based refinement. Final spatial/adaptive adds a spatial prior from neighbouring local-adaptive predictions while excluding the target profile itself from its neighbour prior.",
        "",
        "The result should be presented as BOCPD with domain-informed priors and unsupervised post-processing, not as a supervised learned classifier.",
    ]
    (outdir / "comprehensive_summary_report.md").write_text("\n".join(lines))


def run(outdir: Path, dataset_prefix: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = D.configure_dataset(prefix=dataset_prefix)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Dataset files are missing:\n" + "\n".join(missing))

    profiles_all = D.load_profiles()
    profiles, dropped = M.filter_to_located(profiles_all)
    if not profiles:
        raise SystemExit("No profiles with finite X/Y coordinates are available.")

    strata = D.load_strata()
    truth = {p.target: D.true_boundaries_for(p.target, strata) for p in profiles}

    baseline_cfg = replace(M.make_baseline_config(), name="baseline", label="BASELINE")
    local_cfg = replace(M.make_candidate_config(), name="local_adapt", label="LOCAL ADAPTIVE", use_spatial=False)
    final_cfg = replace(M.make_candidate_config(), name="final_spatial_adapt", label="FINAL SPATIAL/ADAPT", use_spatial=True)
    configs = {
        "baseline": baseline_cfg,
        "local_adapt": local_cfg,
        "final_spatial_adapt": final_cfg,
    }

    preds_by_name: dict[str, dict[str, np.ndarray]] = {}
    diag_by_name: dict[str, pd.DataFrame] = {}
    eval_by_name: dict[str, pd.DataFrame] = {}
    summary_by_name: dict[str, dict[str, float]] = {}
    elapsed_by_name: dict[str, float] = {}

    for name in ["baseline", "local_adapt"]:
        t0 = time.time()
        preds, diag = M.run_pipeline(profiles, configs[name])
        elapsed_by_name[name] = time.time() - t0
        eval_df = evaluate_run(preds, truth, M.TOLERANCES)
        preds_by_name[name] = preds
        diag_by_name[name] = diag
        eval_by_name[name] = eval_df
        summary_by_name[name] = global_summary(eval_df, M.TOLERANCES)
        print(f"{METHOD_LABELS[name]} F1@1.0={summary_by_name[name]['pooled_f1_1.0']:.3f} [{elapsed_by_name[name]:.1f}s]")

    t0 = time.time()
    preds, diag = M.run_pipeline(profiles, final_cfg, pred_source=preds_by_name["local_adapt"])
    elapsed_by_name["final_spatial_adapt"] = time.time() - t0
    eval_df = evaluate_run(preds, truth, M.TOLERANCES)
    preds_by_name["final_spatial_adapt"] = preds
    diag_by_name["final_spatial_adapt"] = diag
    eval_by_name["final_spatial_adapt"] = eval_df
    summary_by_name["final_spatial_adapt"] = global_summary(eval_df, M.TOLERANCES)
    print(
        f"{METHOD_LABELS['final_spatial_adapt']} F1@1.0="
        f"{summary_by_name['final_spatial_adapt']['pooled_f1_1.0']:.3f} "
        f"[{elapsed_by_name['final_spatial_adapt']:.1f}s]"
    )

    save_outputs(outdir, preds_by_name, eval_by_name, diag_by_name, summary_by_name)
    write_excluded_profiles(outdir, dropped)
    stats_df = build_profile_statistics(outdir)
    plot_final_boundaries(outdir, profiles, truth, preds_by_name["final_spatial_adapt"], eval_by_name["final_spatial_adapt"])
    plot_method_comparison(outdir, profiles, truth, preds_by_name, eval_by_name)
    plot_aggregate_figures(outdir, stats_df, summary_by_name)
    write_config(outdir, dataset_prefix, paths, configs, elapsed_by_name)
    write_report(outdir, profiles, dropped, eval_by_name, summary_by_name, stats_df, elapsed_by_name)

    print(f"Saved comprehensive outputs to {outdir}")


def default_outdir(dataset_prefix: str) -> Path:
    dataset_id = "dataset2" if dataset_prefix == "2_" else (dataset_prefix.strip("_") or "default")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO / "results" / "runs" / f"{dataset_id}_{stamp}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-prefix", default="2_", help="Dataset file prefix, e.g. '' for original data or '2_' for the second dataset.")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    run(args.outdir if args.outdir is not None else default_outdir(args.dataset_prefix), args.dataset_prefix)


if __name__ == "__main__":
    main()
