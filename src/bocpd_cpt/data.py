"""Loading, cleaning, and per-target alignment of the CPT dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"

def dataset_paths(prefix: str = "", data_dir: Path | None = None) -> tuple[Path, Path, Path]:
    """Return CPT, location, and strata paths for a named dataset prefix.

    The repository currently contains the original dataset and a second dataset
    whose files are prefixed with ``2_``.  Keeping this explicit avoids source
    edits just to switch experiments.
    """
    root = DATA_DIR if data_dir is None else Path(data_dir)
    return (
        root / f"{prefix}CPT_clean.csv",
        root / f"{prefix}Input_Location_unique_targets.csv",
        root / f"{prefix}Input_Strata_only_loc_targets.csv",
    )


CPT_PATH, LOC_PATH, STRATA_PATH = dataset_paths()


def configure_dataset(prefix: str = "", data_dir: Path | None = None) -> tuple[Path, Path, Path]:
    """Set module-level default dataset paths and return them."""
    global CPT_PATH, LOC_PATH, STRATA_PATH
    CPT_PATH, LOC_PATH, STRATA_PATH = dataset_paths(prefix=prefix, data_dir=data_dir)
    return CPT_PATH, LOC_PATH, STRATA_PATH


CPT_COLS = [
    "PointID", "Target", "Target_number", "SCPG_TESN", "Depth",
    "SCPT_RES",    # qc
    "SCPT_FRES",   # fs
    "SCPT_PWP2",   # u2
    "SCPT_QT",     # qt
    "SCPT_QNET",   # qn
    "SCPT_NQT",    # Qtn
    "SCPT_NFR",    # Fr [%]
    "SCPT_NU2",    # U2
    "SCPT_ICBE",   # Ic
    "n_exp",
    "UNIT",        # only for diagnostics, never used for inference
    "EXCLUDE",
]


@dataclass
class Profile:
    """A single CPT profile at a single target (spatial x,y,z).

    `depth` is strictly monotonic
    Feature arrays share the same length as `depth`.
    """

    target: str
    point_ids: tuple[str, ...]
    depth: np.ndarray
    features: dict[str, np.ndarray]
    x: float | None = None
    y: float | None = None
    z: float | None = None
    bathymetry: float | None = None

    @property
    def n(self) -> int:
        return int(self.depth.size)

    @property
    def step(self) -> float:
        if self.n < 2:
            return 0.02
        return float(np.median(np.diff(self.depth)))

    def get(self, name: str) -> np.ndarray:
        return self.features[name]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_locations(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(LOC_PATH if path is None else path)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def load_strata(path: Path | None = None) -> pd.DataFrame:
    """Ground truth — EVALUATION ONLY.

    The strata file uses a column named ``PointID`` to hold what is actually
    the Target name (``Loc-XX``) in the CPT file.  We rename it to ``Target``
    for consistency.
    """
    df = pd.read_csv(STRATA_PATH if path is None else path)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    # The strata file has both PointID (=Loc-XX, what CPT calls Target) and a
    # Target column that is always identical.  Drop the redundant Target,
    # then rename PointID -> Target for consistency with the CPT file.
    if "Target" in df.columns and "PointID" in df.columns:
        df = df.drop(columns=["Target"])
    df = df.rename(columns={"PointID": "Target"})
    return df


def _collapse_pushes(df: pd.DataFrame, depth_step: float = 0.02) -> pd.DataFrame:
    """Merge overlapping pushes within a single target into one depth grid.

    For overlapping depth rows we average the numeric features.  This mirrors
    the paper's implicit assumption of a single monotone depth profile.
    """
    df = df.sort_values(["Depth"]).copy()
    # Bin depth to the step grid to prevent floating-point duplicates
    df["_d_bin"] = np.round(df["Depth"] / depth_step) * depth_step
    agg = {
        c: "mean"
        for c in df.columns
        if c not in {"PointID", "Target", "Target_number", "SCPG_TESN",
                     "Depth", "_d_bin", "UNIT", "EXCLUDE"}
    }
    g = df.groupby("_d_bin", as_index=False).agg(agg)
    g = g.rename(columns={"_d_bin": "Depth"})
    return g.sort_values("Depth").reset_index(drop=True)


def _safe_log10(x: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    return np.log10(np.clip(x, floor, None))


def _finite_fill(x: np.ndarray) -> np.ndarray:
    """Replace non-finite values with linearly interpolated neighbours.

    Only applied to derived features that can fail (e.g., log of near-zero
    Qtn); raw CPT_clean values are already clean.
    """
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    good = ~bad
    if good.sum() == 0:
        return np.zeros_like(x)
    idx = np.arange(x.size)
    x[bad] = np.interp(idx[bad], idx[good], x[good])
    return x


def _optional_finite_float(row: pd.Series | None, column: str) -> float | None:
    if row is None or column not in row:
        return None
    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def load_profiles(targets: Iterable[str] | None = None,
                  depth_step: float = 0.02,
                  add_log_features: bool = True,
                  cpt_path: Path | None = None,
                  loc_path: Path | None = None) -> list[Profile]:
    """Load all (or selected) CPT profiles, aggregated to one depth column.

    Returns a list of :class:`Profile`.  Ground-truth strata are **not**
    touched — evaluation code loads them separately.
    """
    loc = load_locations(loc_path).set_index("Target")
    cpt = pd.read_csv(CPT_PATH if cpt_path is None else cpt_path, usecols=CPT_COLS, low_memory=False)

    out: list[Profile] = []
    keep_targets = set(targets) if targets is not None else None
    for tgt, grp in cpt.groupby("Target", sort=False):
        if keep_targets is not None and tgt not in keep_targets:
            continue
        agg = _collapse_pushes(grp, depth_step=depth_step)
        feats: dict[str, np.ndarray] = {}
        feats["Ic"] = agg["SCPT_ICBE"].to_numpy(dtype=float)
        feats["Qtn"] = agg["SCPT_NQT"].to_numpy(dtype=float)
        feats["Fr"] = agg["SCPT_NFR"].to_numpy(dtype=float)
        feats["Qt"] = agg["SCPT_QT"].to_numpy(dtype=float)
        feats["qc"] = agg["SCPT_RES"].to_numpy(dtype=float)
        feats["fs"] = agg["SCPT_FRES"].to_numpy(dtype=float)
        feats["u2"] = agg["SCPT_PWP2"].to_numpy(dtype=float)
        feats["U2"] = agg["SCPT_NU2"].to_numpy(dtype=float)
        if add_log_features:
            feats["logQtn"] = _finite_fill(_safe_log10(feats["Qtn"]))
            # Fr can be <=0; floor at 0.1 %
            feats["logFr"] = _finite_fill(_safe_log10(feats["Fr"]))
            feats["logQt"] = _finite_fill(_safe_log10(feats["Qt"]))

        row = loc.loc[tgt] if tgt in loc.index else None
        p = Profile(
            target=tgt,
            point_ids=tuple(sorted(grp["PointID"].dropna().unique())),
            depth=agg["Depth"].to_numpy(dtype=float),
            features=feats,
            x=_optional_finite_float(row, "X"),
            y=_optional_finite_float(row, "Y"),
            z=_optional_finite_float(row, "Z"),
            bathymetry=_optional_finite_float(row, "Bathymetry"),
        )
        out.append(p)
    out.sort(key=lambda p: p.target)
    return out


def true_boundaries_for(target: str, strata: pd.DataFrame | None = None,
                        include_top_bottom: bool = False) -> np.ndarray:
    """Return the interior ground-truth boundary depths (in metres).

    Evaluation only.  By default the top (depth=0) and the profile bottom
    are excluded — they are not interior layer boundaries but profile ends.
    """
    if strata is None:
        strata = load_strata()
    s = strata[strata["Target"] == target].sort_values("Top")
    tops = s["Top"].to_numpy(dtype=float)
    if include_top_bottom:
        bottom = s["Bottom"].to_numpy(dtype=float).max()
        return np.unique(np.concatenate([tops, [bottom]]))
    # drop the very top (top of first layer == surface)
    interior = tops[tops > 0]
    return np.unique(interior)
