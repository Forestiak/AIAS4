"""
Segment builder.
Converts boundary depths into depth intervals and assigns each CPT row
to a segment. Merges segments thinner than the configured minimum.
"""
from __future__ import annotations

import pandas as pd

from .config import Config


def build_segments(
    profiles: dict[str, pd.DataFrame],
    boundaries: dict[str, list[float]],
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (segment_table, rows_with_segment_id)."""
    all_segments: list[pd.DataFrame] = []
    all_rows: list[pd.DataFrame] = []

    # Only process profiles that appear in the boundary map
    targets = boundaries.keys() if boundaries else profiles.keys()

    for target in targets:
        profile = profiles.get(target)
        if profile is None:
            continue
        depths = boundaries.get(target, [])
        seg = _segments_for_profile(target, profile, depths, config)
        if seg.empty:
            continue
        rows = _assign_rows(profile, seg, config)
        all_segments.append(seg)
        all_rows.append(rows)

    return pd.concat(all_segments, ignore_index=True), pd.concat(all_rows, ignore_index=True)


def _segments_for_profile(
    target: str,
    profile: pd.DataFrame,
    boundary_depths: list[float],
    config: Config,
) -> pd.DataFrame:
    depth = profile[config.depth_col]
    top, bottom = float(depth.min()), float(depth.max())

    # Keep only boundaries inside the profile range
    edges = sorted({top, bottom} | {float(b) for b in boundary_depths if top < float(b) < bottom})

    rows = [
        {"target": target, "segment_id": i, "top": edges[i], "bottom": edges[i + 1]}
        for i in range(len(edges) - 1)
    ]
    seg = pd.DataFrame(rows)
    return _merge_thin(seg, config.min_segment_thickness_m)


def _merge_thin(seg: pd.DataFrame, min_thick: float) -> pd.DataFrame:
    seg = seg.reset_index(drop=True)
    while len(seg) > 1:
        seg["thickness"] = seg["bottom"] - seg["top"]
        thin = seg.index[seg["thickness"] < min_thick].tolist()
        if not thin:
            break
        i = thin[0]
        # Merge into the neighbor above; if first segment, merge into below
        if i == 0:
            seg.loc[i + 1, "top"] = seg.loc[i, "top"]
        else:
            seg.loc[i - 1, "bottom"] = seg.loc[i, "bottom"]
        seg = seg.drop(i).reset_index(drop=True)

    seg["segment_id"] = range(len(seg))
    seg["thickness"] = seg["bottom"] - seg["top"]
    return seg


def _assign_rows(
    profile: pd.DataFrame,
    seg: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    last_id = int(seg["segment_id"].max())

    for row in seg.itertuples(index=False):
        if row.segment_id == last_id:
            mask = (profile[config.depth_col] >= row.top) & (profile[config.depth_col] <= row.bottom)
        else:
            mask = (profile[config.depth_col] >= row.top) & (profile[config.depth_col] < row.bottom)

        chunk = profile.loc[mask].copy()
        chunk["target"] = row.target
        chunk["segment_id"] = row.segment_id
        chunk["seg_top"] = row.top
        chunk["seg_bottom"] = row.bottom
        parts.append(chunk)

    return pd.concat(parts, ignore_index=True)
