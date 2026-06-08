"""
Depth profile plots and Robertson SBT charts.
Side-by-side ground-truth strata vs predicted units for each target.
Saves PNGs to the run's plots/ subfolder.
"""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .config import Config


# ── Global soil unit colours (consistent across all plots) ────────────
UNIT_COLORS: dict[str, str] = {
    "Unit 1A siSa":      "#e6194b",   # red
    "Unit 1B siSa/siCl": "#f58231",   # orange
    "Unit 1C siCl":      "#ffe119",   # yellow
    "Unit 2A siSa":      "#3cb44b",   # green
    "Unit 2B siCl":      "#42d4f4",   # cyan
    "Unit 3A siCl":      "#4363d8",   # blue
    "Unit 3B siSa":      "#911eb4",   # purple
    "Unit 3C Sa":        "#f032e6",   # magenta
    "Unit 3D Cl":        "#a9a9a9",   # grey
    "Unit 3E Sa":        "#800000",   # maroon
}
_FALLBACK_PALETTE = [
    "#fabebe", "#008080", "#e6beff", "#9a6324",
    "#fffac8", "#aaffc3", "#808000", "#ffd8b1",
]


def _get_color(label: str) -> str:
    """Return consistent colour for a soil unit label."""
    if label in UNIT_COLORS:
        return UNIT_COLORS[label]
    # Assign a stable fallback for any unknown units
    if label not in _dynamic_colors:
        idx = len(_dynamic_colors) % len(_FALLBACK_PALETTE)
        _dynamic_colors[label] = _FALLBACK_PALETTE[idx]
    return _dynamic_colors[label]

_dynamic_colors: dict[str, str] = {}


def generate_profile_plots(mapped: pd.DataFrame, strata: pd.DataFrame, config: Config, run_dir: Path) -> list[Path]:
    """Create one PNG per target. Returns list of saved paths."""
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    plot_frames = build_profile_plot_frames(mapped, strata, config)
    for target, frame in plot_frames.items():
        path = plots_dir / f"{target}.png"
        _plot_target(frame["reference"], frame["predicted"], str(target), path)
        saved.append(path)
    return saved


def build_profile_plot_frames(
    mapped: pd.DataFrame,
    strata: pd.DataFrame,
    config: Config,
) -> dict[str, dict[str, pd.DataFrame]]:
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    targets = sorted({str(t) for t in mapped["target"].dropna().astype(str)} | {str(t) for t in strata[config.target_col].dropna().astype(str)})

    for target in targets:
        reference = _reference_profile_rows(strata, target, config)
        predicted = _predicted_profile_rows(mapped, target)
        if reference.empty and predicted.empty:
            continue
        frames[target] = {
            "reference": reference,
            "predicted": predicted,
        }

    return frames


def _reference_profile_rows(strata: pd.DataFrame, target: str, config: Config) -> pd.DataFrame:
    target_strata = strata[strata[config.target_col].astype(str) == str(target)].copy()
    if target_strata.empty:
        return pd.DataFrame(columns=["top", "bottom", "unit"])

    reference = pd.DataFrame(
        {
            "top": pd.to_numeric(target_strata[config.strata_top_col], errors="coerce"),
            "bottom": pd.to_numeric(target_strata[config.strata_bottom_col], errors="coerce"),
            "unit": target_strata[config.strata_unit_col].astype(str),
        }
    )
    reference = reference.dropna(subset=["top", "bottom"]).sort_values("top").reset_index(drop=True)
    return reference


def _predicted_profile_rows(mapped: pd.DataFrame, target: str) -> pd.DataFrame:
    target_segments = mapped[mapped["target"].astype(str) == str(target)].copy()
    if target_segments.empty:
        return pd.DataFrame(columns=["top", "bottom", "unit"])

    predicted = pd.DataFrame(
        {
            "top": pd.to_numeric(target_segments["top"], errors="coerce"),
            "bottom": pd.to_numeric(target_segments["bottom"], errors="coerce"),
            "unit": target_segments["predicted_unit"],
        }
    )
    predicted = predicted.dropna(subset=["top", "bottom"]).sort_values("top").reset_index(drop=True)
    return predicted


def _plot_target(reference: pd.DataFrame, predicted: pd.DataFrame, target: str, path: Path) -> None:
    height_units = max(len(reference), len(predicted), 1)
    fig, (ax_ref, ax_pred) = plt.subplots(1, 2, figsize=(5, max(6, height_units * 0.4)), sharey=True)

    all_units = sorted(
        set(reference["unit"].dropna().unique()) | set(predicted["unit"].dropna().unique())
    )
    all_depths = pd.concat(
        [reference[["top", "bottom"]], predicted[["top", "bottom"]]],
        ignore_index=True,
    ).dropna(how="all")

    if not all_depths.empty:
        depth_min = float(all_depths.min().min())
        depth_max = float(all_depths.max().max())
        ax_ref.set_ylim(depth_max, depth_min)

    for _, row in reference.iterrows():
        unit = row.get("unit")
        color = _get_color(unit) if pd.notna(unit) else "#333333"
        ax_ref.barh(y=(row["top"] + row["bottom"]) / 2, width=1, height=row["bottom"] - row["top"],
                     color=color, edgecolor="black", linewidth=0.5)
    ax_ref.set_ylabel("Depth (m)")
    ax_ref.set_title("Ground Truth")
    ax_ref.set_xlim(0, 1)
    ax_ref.set_xticks([])

    for _, row in predicted.iterrows():
        unit = row.get("unit")
        color = _get_color(unit) if pd.notna(unit) else "#333333"
        ax_pred.barh(y=(row["top"] + row["bottom"]) / 2, width=1, height=row["bottom"] - row["top"],
                      color=color, edgecolor="black", linewidth=0.5)
    ax_pred.set_title("Predicted")
    ax_pred.set_xlim(0, 1)
    ax_pred.set_xticks([])

    if all_units:
        patches = [mpatches.Patch(color=_get_color(u), label=u) for u in all_units]
        fig.legend(handles=patches, fontsize=6, loc="lower center", ncol=min(3, len(patches)),
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Target: {target}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Robertson SBT chart ───────────────────────────────────────────────

# Robertson (1990) SBT zone definitions
# Zones 1-6 follow Ic contours; zones 7-9 are the high-Qtn upper region.
_ROBERTSON_ZONES: list[dict] = [
    {"id": 1, "label": "1\nSensitive\nfine-grained", "ic_lo": 3.60, "ic_hi": None,  "color": "#d4e6f1"},
    {"id": 2, "label": "2\nOrganic /\nPeat",          "ic_lo": 2.95, "ic_hi": 3.60, "color": "#a9dfbf"},
    {"id": 3, "label": "3\nClays",                    "ic_lo": 2.60, "ic_hi": 2.95, "color": "#7dcea0"},
    {"id": 4, "label": "4\nSilt\nmixtures",           "ic_lo": 2.05, "ic_hi": 2.60, "color": "#f9e79f"},
    {"id": 5, "label": "5\nSand-silt\nmixtures",      "ic_lo": 1.31, "ic_hi": 2.05, "color": "#f0b27a"},
    {"id": 6, "label": "6\nClean\nsands",             "ic_lo": None, "ic_hi": 1.31, "color": "#e59866"},
    # Zones 7-9: same Ic boundaries but restricted to Qtn > QTN_UPPER_THRESHOLD
    {"id": 7, "label": "7\nGravelly\nsand",           "ic_lo": None, "ic_hi": 1.31, "color": "#cb4335", "upper": True},
    {"id": 8, "label": "8\nVery stiff\nsand",         "ic_lo": 1.31, "ic_hi": 2.05, "color": "#a93226", "upper": True},
    {"id": 9, "label": "9\nVery stiff\nfine-grained", "ic_lo": 2.05, "ic_hi": None,  "color": "#7b241c", "upper": True},
]
_QTN_UPPER = 160.0   # Qtn threshold separating zones 6→7, 5→8, 1-4→9
_FR_LO     = 0.1     # chart x-axis lower bound (%)
_FR_HI     = 10.0    # chart x-axis upper bound (%)
_QTN_LO    = 1.0     # chart y-axis lower bound
_QTN_HI    = 1000.0  # chart y-axis upper bound
_GRID_N    = 400     # resolution of the background mesh


def _robertson_ic(qtn: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """Robertson (1990) soil behaviour type index Ic."""
    return np.sqrt((3.47 - np.log10(np.maximum(qtn, 1e-6))) ** 2
                   + (np.log10(np.maximum(fr, 1e-6)) + 1.22) ** 2)


def _draw_robertson_background(ax: plt.Axes) -> None:
    """Fill the Robertson SBT zone background on a log-log axes."""
    fr_grid  = np.logspace(np.log10(_FR_LO),  np.log10(_FR_HI),  _GRID_N)
    qtn_grid = np.logspace(np.log10(_QTN_LO), np.log10(_QTN_HI), _GRID_N)
    FR, QTN = np.meshgrid(fr_grid, qtn_grid)
    IC = _robertson_ic(QTN, FR)

    # Build a zone-ID grid (0 = unassigned; zones checked in priority order
    # so that upper zones 7-9 overwrite their lower-chart equivalents)
    zone_ids = np.zeros_like(IC, dtype=int)
    for zone in _ROBERTSON_ZONES:
        upper_only = zone.get("upper", False)
        mask = np.ones_like(IC, dtype=bool)
        if zone["ic_lo"] is not None:
            mask &= IC >= zone["ic_lo"]
        if zone["ic_hi"] is not None:
            mask &= IC < zone["ic_hi"]
        mask &= (QTN >= _QTN_UPPER) if upper_only else (QTN < _QTN_UPPER)
        zone_ids[mask] = zone["id"]

    # ListedColormap: index 0 = white (unassigned), indices 1-9 = zone colours
    zone_colors = ["white"] + [z["color"] for z in _ROBERTSON_ZONES]
    cmap = matplotlib.colors.ListedColormap(zone_colors)
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, len(zone_colors)), len(zone_colors))
    ax.pcolormesh(FR, QTN, zone_ids, cmap=cmap, norm=norm,
                  alpha=0.50, shading="auto", rasterized=True)

    # Ic iso-contour boundary lines
    for ic_val in [1.31, 2.05, 2.60, 2.95, 3.60]:
        ax.contour(FR, QTN, IC, levels=[ic_val], colors=["#555555"],
                   linewidths=0.8, linestyles="--", alpha=0.7)

    # Horizontal line separating zones 1-6 from 7-9
    ax.axhline(_QTN_UPPER, color="#555555", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(_FR_LO, _FR_HI)
    ax.set_ylim(_QTN_LO, _QTN_HI)
    ax.set_xlabel("Normalised friction ratio  Fr (%)", fontsize=9)
    ax.set_ylabel("Normalised cone resistance  Qtn (−)", fontsize=9)

    # Zone ID labels
    _zone_label_pos = [
        (1, 0.13, 500), (2, 0.13, 180), (3, 0.30, 55),
        (4, 0.55, 18),  (5, 1.40,  8),  (6, 4.00,  3),
        (7, 0.13, 500), (8, 0.55, 350), (9, 2.50, 350),
    ]
    for zone_id, fr_pos, qtn_pos in _zone_label_pos:
        zone = next(z for z in _ROBERTSON_ZONES if z["id"] == zone_id)
        ax.text(fr_pos, qtn_pos, zone["label"],
                ha="center", va="center", fontsize=5.5, color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.5))


def _valid_robertson_rows(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["SCPT_NFR", "SCPT_NQT"])
    return valid[(valid["SCPT_NFR"] > 0) & (valid["SCPT_NQT"] > 0)]


def _scatter_robertson(ax: plt.Axes, df: pd.DataFrame, unit_col: str, title: str) -> list:
    """Scatter plot of (Fr, Qtn) colored by unit_col. Returns legend patches."""
    valid = _valid_robertson_rows(df)

    units_present = sorted(valid[unit_col].dropna().unique())
    for unit in units_present:
        sub = valid[valid[unit_col] == unit]
        ax.scatter(sub["SCPT_NFR"], sub["SCPT_NQT"],
                   c=_get_color(unit), s=4, alpha=0.45,
                   linewidths=0, label=unit, rasterized=True)

    ax.set_title(title, fontsize=10, fontweight="bold")
    patches = [mpatches.Patch(color=_get_color(u), label=u) for u in units_present]
    return patches


def _save_robertson_figure(df: pd.DataFrame, title: str, path: Path) -> None:
    """Two-panel Robertson chart: left=reference, right=predicted."""
    fig, (ax_ref, ax_pred) = plt.subplots(1, 2, figsize=(13, 7))

    for ax in (ax_ref, ax_pred):
        _draw_robertson_background(ax)

    patches_ref  = _scatter_robertson(ax_ref,  df, "reference_unit", "Reference units")
    patches_pred = _scatter_robertson(ax_pred, df, "predicted_unit", "Predicted units")

    all_units = sorted(
        set(df["reference_unit"].dropna().unique()) |
        set(df["predicted_unit"].dropna().unique())
    )
    shared_patches = [mpatches.Patch(color=_get_color(u), label=u) for u in all_units]
    fig.legend(handles=shared_patches, fontsize=7, loc="lower center",
               ncol=min(5, len(shared_patches)), bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_robertson_class_figure(df: pd.DataFrame, unit: str, path: Path) -> None:
    """Two-panel Robertson chart for one unit/class aggregated across all locations."""
    fig, (ax_ref, ax_pred) = plt.subplots(1, 2, figsize=(13, 7))

    valid = _valid_robertson_rows(df)
    reference = valid[valid["reference_unit"] == unit]
    predicted = valid[valid["predicted_unit"] == unit]
    unit_color = _get_color(unit)

    for ax in (ax_ref, ax_pred):
        _draw_robertson_background(ax)

    if not reference.empty:
        ax_ref.scatter(
            reference["SCPT_NFR"],
            reference["SCPT_NQT"],
            c=unit_color,
            s=8,
            alpha=0.75,
            linewidths=0,
            rasterized=True,
        )
    if not predicted.empty:
        ax_pred.scatter(
            predicted["SCPT_NFR"],
            predicted["SCPT_NQT"],
            c=unit_color,
            s=8,
            alpha=0.75,
            linewidths=0,
            rasterized=True,
        )

    ax_ref.set_title(f"Reference: {unit} (n={len(reference)})", fontsize=10, fontweight="bold")
    ax_pred.set_title(f"Predicted: {unit} (n={len(predicted)})", fontsize=10, fontweight="bold")

    legend_patch = [mpatches.Patch(color=unit_color, label=unit)]
    fig.legend(handles=legend_patch, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Robertson SBT — class {unit}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _slugify_unit_label(unit: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(unit).strip()).strip("_").lower()
    return slug or "unknown"


def generate_robertson_charts(
    seg_rows: pd.DataFrame,
    mapped: pd.DataFrame,
    run_dir: Path,
) -> list[Path]:
    """
    Create one Robertson SBT chart per unit/class, aggregated across all locations.

    seg_rows : raw CPT rows with SCPT_NFR (Fr) and SCPT_NQT (Qtn) columns,
               tagged with (target, segment_id) by build_segments.
    mapped   : segment-level table with (target, segment_id, predicted_unit,
               reference_unit) from evaluate.
    """
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    label_cols = ["target", "segment_id", "predicted_unit", "reference_unit"]
    labels = mapped[label_cols].drop_duplicates(subset=["target", "segment_id"])

    annotated = seg_rows.merge(labels, on=["target", "segment_id"], how="left")

    saved: list[Path] = []

    units = sorted(
        set(annotated["reference_unit"].dropna().unique())
        | set(annotated["predicted_unit"].dropna().unique())
    )
    for unit in units:
        path = plots_dir / f"robertson_class_{_slugify_unit_label(unit)}.png"
        _save_robertson_class_figure(annotated, str(unit), path)
        saved.append(path)

    return saved
