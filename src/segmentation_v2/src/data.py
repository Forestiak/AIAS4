"""
Data loading and splitting.
Reads CPT measurements and strata (boundaries + evaluation reference).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config


def load_cpt(config: Config) -> pd.DataFrame:
    df = _read_and_strip(config.cpt_path)
    df[config.depth_col] = pd.to_numeric(df[config.depth_col], errors="coerce")
    for col in config.feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[config.target_col, config.depth_col]).sort_values(
        [config.target_col, config.depth_col]
    )


def load_strata(config: Config) -> pd.DataFrame:
    df = _read_and_strip(config.strata_path)
    df = _supplement_missing_strata_targets(df, config)
    for col in (config.strata_top_col, config.strata_bottom_col):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[config.target_col, config.strata_top_col, config.strata_bottom_col, config.strata_unit_col])
    # Keep only the first PointID per Target to avoid duplicate layers
    first_pid = df.groupby(config.target_col)[config.point_id_col].first()
    df = df[df.apply(lambda r: r[config.point_id_col] == first_pid[r[config.target_col]], axis=1)]
    return df


def _supplement_missing_strata_targets(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Fill targets missing from the primary strata file using the loc-only backup file when available."""
    fallback_path = config.project_dir / "data" / "Input_Strata_merged_boundaries_loc_only.csv"
    if not fallback_path.exists():
        return df

    fallback = _read_and_strip(fallback_path)
    if config.target_col not in df.columns or config.target_col not in fallback.columns:
        return df

    present_targets = set(df[config.target_col].dropna().astype(str))
    fallback_targets = fallback[config.target_col].dropna().astype(str)
    missing_targets = set(fallback_targets) - present_targets
    if not missing_targets:
        return df

    supplement = fallback[fallback[config.target_col].astype(str).isin(missing_targets)].copy()
    if supplement.empty:
        return df

    return pd.concat([df, supplement], ignore_index=True)


def split_by_target(df: pd.DataFrame, target_col: str) -> dict[str, pd.DataFrame]:
    return {str(t): g for t, g in df.groupby(target_col, sort=False)}


def boundaries_from_strata(strata: pd.DataFrame, config: Config) -> dict[str, list[float]]:
    """Extract internal boundary depths from strata Top/Bottom columns."""
    out: dict[str, list[float]] = {}
    for target, group in strata.groupby(config.target_col, sort=False):
        tops = sorted(group[config.strata_top_col].dropna().unique())
        # Skip the surface (first Top); internal transitions only
        out[str(target)] = [float(d) for d in tops[1:]]
    return out


def boundaries_from_exported(config: Config) -> dict[str, list[float]]:
    """Read predicted boundaries exported by the BOCPD workflow."""
    return _boundaries_from_csv(config.exported_boundaries_path)


def boundaries_from_perfect_recall(config: Config) -> dict[str, list[float]]:
    """Read boundaries from the perfect_recall reference file."""
    return _boundaries_from_csv(config.perfect_recall_boundaries_path)


def _boundaries_from_csv(path: Path) -> dict[str, list[float]]:
    df = _read_and_strip(path)

    # Normalise column names to lowercase for flexible matching
    col_map = {c.lower(): c for c in df.columns}
    target_col = col_map.get("target")
    depth_col = col_map.get("boundary_depth") or col_map.get("depth")
    if target_col is None or depth_col is None:
        raise ValueError(
            f"{path.name} must contain Target and boundary_depth/Depth columns "
            f"(found: {list(df.columns)})"
        )

    df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
    df = df.dropna(subset=[target_col, depth_col])

    out: dict[str, list[float]] = {}
    for target, group in df.groupby(target_col, sort=False):
        out[str(target)] = sorted(float(v) for v in group[depth_col].unique())
    return out


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _read_and_strip(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df
