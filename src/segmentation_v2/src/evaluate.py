"""
Evaluation against strata reference (used only after clustering).
Attaches the dominant strata unit to each segment by overlap, then
maps cluster IDs to soil units and computes standard metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)

from .config import Config


def attach_reference_units(features: pd.DataFrame, strata: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Assign one reference unit per segment by dominant overlap."""
    rows: list[dict] = []
    for feat in features.itertuples(index=False):
        target_strata = strata[strata[config.target_col].astype(str) == str(feat.target)]
        best_unit = _best_overlap_unit(feat.top, feat.bottom, target_strata, config)
        row = feat._asdict()
        row["reference_unit"] = best_unit
        rows.append(row)
    return pd.DataFrame(rows)


def attach_reference_units_to_measurements(rows: pd.DataFrame, strata: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Assign the reference unit to each CPT measurement row from its depth."""
    output_parts: list[pd.DataFrame] = []

    for target, group in rows.groupby("target", sort=False):
        target_strata = strata[strata[config.target_col].astype(str) == str(target)]
        enriched = group.copy()
        enriched["reference_unit"] = _reference_units_for_depths(
            enriched[config.depth_col],
            target_strata,
            config,
        )
        output_parts.append(enriched)

    if not output_parts:
        output = rows.copy()
        output["reference_unit"] = None
        return output

    return pd.concat(output_parts, ignore_index=True)


def build_cluster_mapping(df: pd.DataFrame) -> dict[int, str]:
    mapping: dict[int, str] = {}
    valid = df.dropna(subset=["reference_unit"])
    for cid, group in valid.groupby("cluster_id"):
        modes = group["reference_unit"].mode()
        if not modes.empty:
            mapping[int(cid)] = str(modes.iloc[0])
    return mapping


def apply_cluster_mapping(df: pd.DataFrame, mapping: dict[int, str]) -> pd.DataFrame:
    out = df.copy()
    out["predicted_unit"] = out["cluster_id"].map(mapping)
    return out


def map_clusters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    mapping = build_cluster_mapping(df)
    return apply_cluster_mapping(df, mapping), mapping


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    work = df.dropna(subset=["reference_unit", "predicted_unit"])
    if work.empty:
        return {}
    y_true = work["reference_unit"]
    y_pred = work["predicted_unit"]
    cluster_ids = work["cluster_id"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ari": float(adjusted_rand_score(y_true, cluster_ids)),
        "nmi": float(normalized_mutual_info_score(y_true, cluster_ids)),
    }


def per_class_report(df: pd.DataFrame) -> str:
    """Sklearn classification report as a formatted string."""
    work = df.dropna(subset=["reference_unit", "predicted_unit"])
    if work.empty:
        return ""
    return classification_report(work["reference_unit"], work["predicted_unit"], zero_division=0)


def _best_overlap_unit(seg_top: float, seg_bottom: float, strata: pd.DataFrame, config: Config) -> str | None:
    if strata.empty:
        return None
    tops = strata[config.strata_top_col].astype(float)
    bottoms = strata[config.strata_bottom_col].astype(float)
    overlaps = (strata[config.strata_unit_col].values,
                (bottoms.clip(upper=seg_bottom) - tops.clip(lower=seg_top)).clip(lower=0).values)
    units, amounts = overlaps
    if amounts.max() <= 0:
        return None
    return str(units[amounts.argmax()])


def _reference_units_for_depths(depths: pd.Series, strata: pd.DataFrame, config: Config) -> pd.Series:
    assigned = pd.Series(index=depths.index, data=None, dtype=object)
    if strata.empty:
        return assigned

    ordered = strata.sort_values(config.strata_top_col, kind="mergesort").reset_index(drop=True)
    tops = pd.to_numeric(ordered[config.strata_top_col], errors="coerce").to_numpy(dtype=float, copy=False)
    bottoms = pd.to_numeric(ordered[config.strata_bottom_col], errors="coerce").to_numpy(dtype=float, copy=False)
    units = ordered[config.strata_unit_col].astype(str).to_numpy(dtype=object, copy=False)

    depth_values = pd.to_numeric(depths, errors="coerce").to_numpy(dtype=float, copy=False)
    valid_mask = np.isfinite(depth_values)
    if not valid_mask.any():
        return assigned

    valid_positions = np.flatnonzero(valid_mask)
    valid_depths = depth_values[valid_mask]
    candidate_idx = np.searchsorted(tops, valid_depths, side="right") - 1
    in_bounds = candidate_idx >= 0
    if not in_bounds.any():
        return assigned

    bounded_positions = valid_positions[in_bounds]
    bounded_depths = valid_depths[in_bounds]
    bounded_candidates = candidate_idx[in_bounds]
    inside_interval = bounded_depths <= bottoms[bounded_candidates]
    if not inside_interval.any():
        return assigned

    matched_positions = bounded_positions[inside_interval]
    matched_units = units[bounded_candidates[inside_interval]]
    assigned.iloc[matched_positions] = matched_units
    return assigned
