"""
Interactive 3D spatial map of clustered segments.
Shows vertical columns at each CPT location (X, Y) coloured by predicted
soil unit.  A depth slider allows cutting from the surface downward.
Rotation is handled natively by matplotlib's 3D mouse interaction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
)

from . import constants as C


# ── Colours (same as segmentation_v2/src/plots.py) ───────────────────

_UNIT_COLORS: dict[str, str] = {
    "Unit 1A siSa":      "#e6194b",
    "Unit 1B siSa/siCl": "#f58231",
    "Unit 1C siCl":      "#ffe119",
    "Unit 2A siSa":      "#3cb44b",
    "Unit 2B siCl":      "#42d4f4",
    "Unit 3A siCl":      "#4363d8",
    "Unit 3B siSa":      "#911eb4",
    "Unit 3C Sa":        "#f032e6",
    "Unit 3D Cl":        "#a9a9a9",
    "Unit 3E Sa":        "#800000",
}
_FALLBACK = [
    "#fabebe", "#008080", "#e6beff", "#9a6324",
    "#fffac8", "#aaffc3", "#808000", "#ffd8b1",
]
_dyn_colors: dict[str, str] = {}


def _unit_color(label) -> str:
    if pd.isna(label):
        return "#555555"
    label = str(label)
    if label in _UNIT_COLORS:
        return _UNIT_COLORS[label]
    if label not in _dyn_colors:
        _dyn_colors[label] = _FALLBACK[len(_dyn_colors) % len(_FALLBACK)]
    return _dyn_colors[label]


def _filter_outlier_locations(
    locations: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    """
    Remove locations whose X or Y is a clear outlier (> 3x IQR from
    the quartiles).  Catches data-entry errors like missing leading digits.
    """
    if len(locations) < 4:
        return locations
    xs = np.array([v[0] for v in locations.values()])
    ys = np.array([v[1] for v in locations.values()])

    def _mask(arr: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        return (arr >= lo) & (arr <= hi)

    keep_x = _mask(xs)
    keep_y = _mask(ys)
    keep = keep_x & keep_y

    filtered: dict[str, tuple[float, float, float]] = {}
    for i, key in enumerate(locations):
        if keep[i]:
            filtered[key] = locations[key]
    return filtered


# ── Dialog ────────────────────────────────────────────────────────────

class Map3DDialog(QDialog):
    """Popup window with an interactive 3D map and a depth-cut slider."""

    def __init__(self, run_dir: str, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.resize(1050, 770)
        self._run_dir = Path(run_dir)

        self._segments: pd.DataFrame | None = None
        self._locations: dict[str, tuple[float, float, float]] = {}
        self._max_depth: float = 100.0
        self._cut_depth: float = 0.0

        self._build_ui()
        self._load_data()
        self._draw()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Matplotlib 3D canvas
        self.fig = Figure(facecolor="#1e1e2e")
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, 1)

        # Depth-cut slider
        row = QHBoxLayout()
        row.addWidget(QLabel("Cut from surface:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_moved)
        row.addWidget(self.slider, 1)
        self.depth_label = QLabel("0.0 m")
        self.depth_label.setMinimumWidth(70)
        row.addWidget(self.depth_label)
        layout.addLayout(row)

        # Debounce timer so rapid slider drags don't stall
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(80)
        self._redraw_timer.timeout.connect(self._draw)

    # ── Data ──────────────────────────────────────────────────────────

    def _load_data(self):
        # Clustered segments from the run
        csv = self._run_dir / "clustered_segments.csv"
        if csv.exists():
            self._segments = pd.read_csv(csv)

        # Location XYZ
        loc_csv = C.CORE_DIR.parent / "data" / "Input_Location_clean.csv"
        if loc_csv.exists():
            df = pd.read_csv(loc_csv)
            df.columns = df.columns.str.strip()
            for _, r in df.iterrows():
                t = str(r["Target"]).strip()
                if t not in self._locations:
                    self._locations[t] = (float(r["X"]), float(r["Y"]), float(r["Z"]))

            # Filter coordinate outliers (e.g. missing leading digits)
            self._locations = _filter_outlier_locations(self._locations)

        # Slider range
        if self._segments is not None and not self._segments.empty:
            self._max_depth = float(self._segments["bottom"].max())
            self.slider.setRange(0, int(self._max_depth * 10))

        # Window title from run summary
        title = "3D Spatial Map"
        summary_path = self._run_dir / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            title += f"  \u2014  {s.get('model_id', '')}  ({self._run_dir.name})"
        self.setWindowTitle(title)

    # ── Slider ────────────────────────────────────────────────────────

    def _on_slider_moved(self, val: int):
        self._cut_depth = val / 10.0
        self.depth_label.setText(f"{self._cut_depth:.1f} m")
        self._redraw_timer.start()

    # ── Drawing ───────────────────────────────────────────────────────

    def _draw(self):
        # Preserve the current camera angle across redraws
        elev, azim = 25, -50
        if self.fig.axes:
            old = self.fig.axes[0]
            elev, azim = old.elev, old.azim

        self.fig.clear()
        seg = self._segments
        if seg is None or seg.empty or not self._locations:
            self.canvas.draw_idle()
            return

        ax = self.fig.add_subplot(111, projection="3d", facecolor="#1e1e2e")

        # Targets with both segments and locations
        targets = [t for t in seg["target"].unique() if t in self._locations]
        if not targets:
            self.canvas.draw_idle()
            return

        coords = {t: self._locations[t] for t in targets}
        xs = np.array([coords[t][0] for t in targets])
        ys = np.array([coords[t][1] for t in targets])
        zs = np.array([coords[t][2] for t in targets])
        x0, y0 = float(xs.min()), float(ys.min())

        # Bar width = 12% of the minimum inter-location distance
        min_dist = float("inf")
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                d = float(np.hypot(xs[i] - xs[j], ys[i] - ys[j]))
                if d > 0:
                    min_dist = min(min_dist, d)
        bar_w = min_dist * 0.12 if min_dist < float("inf") else 50.0

        cut = self._cut_depth
        color_col = "predicted_unit" if "predicted_unit" in seg.columns else None

        # Collect all bars first (single bar3d call is faster)
        bx, by, bz, bdx, bdy, bdz, bcolors = [], [], [], [], [], [], []
        legend_units: set[str] = set()

        for target in targets:
            gx = coords[target][0] - x0
            gy = coords[target][1] - y0
            gz = coords[target][2]  # ground-level elevation

            t_seg = seg[seg["target"] == target].sort_values("top")
            for _, row in t_seg.iterrows():
                top = float(row["top"])
                bottom = float(row["bottom"])
                vis_top = max(top, cut)
                if vis_top >= bottom:
                    continue

                label = row[color_col] if color_col else None
                color = _unit_color(label)
                if pd.notna(label):
                    legend_units.add(str(label))

                elev_bottom = gz - bottom
                dz = (gz - vis_top) - elev_bottom   # = bottom - vis_top

                bx.append(gx - bar_w / 2)
                by.append(gy - bar_w / 2)
                bz.append(elev_bottom)
                bdx.append(bar_w)
                bdy.append(bar_w)
                bdz.append(dz)
                bcolors.append(color)

        if bx:
            ax.bar3d(
                bx, by, bz, bdx, bdy, bdz,
                color=bcolors, edgecolor="black", linewidth=0.3, alpha=0.92,
            )

        # Location labels above each column
        for target in targets:
            gx = coords[target][0] - x0
            gy = coords[target][1] - y0
            gz = coords[target][2]
            ax.text(gx, gy, gz + 3, target,
                    fontsize=7, ha="center", va="bottom", color="#cdd6f4")

        # ── Cut plane ─────────────────────────────────────────────────
        if cut > 0:
            gxs = np.array([coords[t][0] - x0 for t in targets])
            gys = np.array([coords[t][1] - y0 for t in targets])
            pad = bar_w * 2
            avg_gz = float(zs.mean())
            z_cut = avg_gz - cut
            verts = [
                [float(gxs.min() - pad), float(gys.min() - pad), z_cut],
                [float(gxs.max() + pad), float(gys.min() - pad), z_cut],
                [float(gxs.max() + pad), float(gys.max() + pad), z_cut],
                [float(gxs.min() - pad), float(gys.max() + pad), z_cut],
            ]
            plane = Poly3DCollection(
                [verts], alpha=0.12, facecolor="#89b4fa",
                edgecolor="#89b4fa", linewidth=0.5,
            )
            ax.add_collection3d(plane)

        # ── Axes styling (dark theme) ─────────────────────────────────
        ax.set_xlabel("X offset (m)", fontsize=8, color="#a6adc8", labelpad=8)
        ax.set_ylabel("Y offset (m)", fontsize=8, color="#a6adc8", labelpad=8)
        ax.set_zlabel("Elevation (m)", fontsize=8, color="#a6adc8", labelpad=8)
        ax.tick_params(colors="#a6adc8", labelsize=7)

        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("#181825")
            pane.set_edgecolor("#45475a")
        grid_color = (0.27, 0.28, 0.35, 0.4)
        ax.xaxis._axinfo["grid"]["color"] = grid_color
        ax.yaxis._axinfo["grid"]["color"] = grid_color
        ax.zaxis._axinfo["grid"]["color"] = grid_color

        # Restore camera angle
        ax.view_init(elev=elev, azim=azim)

        # Legend
        if legend_units:
            patches = [mpatches.Patch(color=_unit_color(u), label=u)
                       for u in sorted(legend_units)]
            ax.legend(
                handles=patches, fontsize=7, loc="upper left",
                framealpha=0.6, facecolor="#313244", edgecolor="#45475a",
                labelcolor="#cdd6f4",
            )

        self.fig.tight_layout()
        self.canvas.draw_idle()
