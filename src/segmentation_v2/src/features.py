"""
Segment-level feature computation.

Supports:
- summary: segment-level statistics
- resample: fixed-length segment shape representation
- paa: compressed fixed-length segment shape representation
- hybrid: thesis-oriented segment representation combining:
    - distribution/statistical features
    - resampled shape features
    - derivative/roughness features
    - engineered CPT features
    - real depth/geometry context
- fft: segment-level spectral representation using fixed frequency-band features

Important:
- Boundaries define segments.
- Relative depth [0, 1] is used only for shape interpolation.
- Sensor values are not normalized per segment.
- Global imputation/scaling/PCA happens later in model.py.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .config import Config
from .fft_features import build_fft_features, fft_feature_matrix_columns


_STAT_FUNCS: dict[str, Callable[[pd.Series], float]] = {
    "mean": lambda v: float(v.mean()),
    "std": lambda v: float(v.std(ddof=0)),
    "median": lambda v: float(v.median()),
    "min": lambda v: float(v.min()),
    "max": lambda v: float(v.max()),
}

_GROUP_COLS = ["target", "segment_id", "seg_top", "seg_bottom"]


def build_features(segmented_rows: pd.DataFrame, config: Config) -> pd.DataFrame:
    builders = {
        "summary": build_summary_features,
        "resample": build_resampled_features,
        "paa": build_paa_features,
        "hybrid": build_hybrid_features,
        "fft": build_fft_features,
    }

    builder = builders.get(config.representation_type)
    if builder is None:
        raise ValueError(f"Unknown representation type: {config.representation_type}")

    return builder(segmented_rows, config)


def feature_matrix_columns(config: Config) -> list[str]:
    """Return ordered feature columns used by the clustering model."""
    if config.representation_type == "summary":
        return _summary_feature_matrix_columns(config)

    if config.representation_type == "resample":
        return _shape_feature_matrix_columns(
            config,
            base_prefix="r",
            derivative_prefix="d",
            size=config.representation_length or 32,
        )

    if config.representation_type == "paa":
        return _shape_feature_matrix_columns(
            config,
            base_prefix="paa",
            derivative_prefix="paa_d",
            size=config.representation_bins or 8,
        )

    if config.representation_type == "hybrid":
        return _hybrid_feature_matrix_columns(config)

    if config.representation_type == "fft":
        return fft_feature_matrix_columns(config)

    raise ValueError(f"Unknown representation type: {config.representation_type}")


def build_summary_features(segmented_rows: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows: list[dict] = []

    for keys, group in _segment_groups(segmented_rows, config):
        row = _metadata_row(keys, group)

        for col in config.feature_columns:
            vals = _finite_array(group[col] if col in group.columns else pd.Series(dtype=float))
            for stat_name in config.segment_stats:
                func = _STAT_FUNCS.get(stat_name)
                if func is not None:
                    row[f"{col}_{stat_name}"] = func(pd.Series(vals)) if len(vals) else np.nan

        _append_extra_derived_features(row, group, config)
        rows.append(row)

    return pd.DataFrame(rows)


def build_resampled_features(segmented_rows: pd.DataFrame, config: Config) -> pd.DataFrame:
    length = config.representation_length or 32
    grid = np.linspace(0.0, 1.0, length)
    rows: list[dict] = []

    for keys, group in _segment_groups(segmented_rows, config):
        row = _metadata_row(keys, group)
        _, _, top, bottom = keys

        rel_depth = _relative_depth(group[config.depth_col], float(top), float(bottom))

        for col in config.feature_columns:
            values = _numeric_array(group[col] if col in group.columns else pd.Series(dtype=float))
            sampled = _interpolate_to_grid(rel_depth, values, grid)
            _assign_indexed_features(row, col, sampled, "r")

            if config.include_derivatives:
                _assign_indexed_features(row, col, _gradient_features(sampled, grid), "d")

        _append_extra_derived_features(row, group, config)
        rows.append(row)

    return pd.DataFrame(rows)


def build_paa_features(segmented_rows: pd.DataFrame, config: Config) -> pd.DataFrame:
    bins = config.representation_bins or 8
    internal_grid = np.linspace(0.0, 1.0, max(bins * 8, bins))
    centers = np.linspace(0.0, 1.0, bins)
    rows: list[dict] = []

    for keys, group in _segment_groups(segmented_rows, config):
        row = _metadata_row(keys, group)
        _, _, top, bottom = keys

        rel_depth = _relative_depth(group[config.depth_col], float(top), float(bottom))

        for col in config.feature_columns:
            values = _numeric_array(group[col] if col in group.columns else pd.Series(dtype=float))
            sampled = _interpolate_to_grid(rel_depth, values, internal_grid)
            paa = _paa_from_resampled(sampled, bins)

            _assign_indexed_features(row, col, paa, "paa")

            if config.include_derivatives:
                _assign_indexed_features(row, col, _gradient_features(paa, centers), "paa_d")

        _append_extra_derived_features(row, group, config)
        rows.append(row)

    return pd.DataFrame(rows)


def build_hybrid_features(segmented_rows: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    One row = one boundary-defined segment.

    Hybrid combines:
    - real material statistics
    - fixed-length resampled shape
    - derivative/roughness features in real depth units
    - engineered CPT features
    - depth/geometry context
    """
    rows: list[dict] = []

    shape_length = config.representation_length or getattr(config, "hybrid_shape_length", 24) or 24
    grid = np.linspace(0.0, 1.0, int(shape_length))
    profile_ranges = _profile_depth_ranges(segmented_rows, config)

    for keys, group in _segment_groups(segmented_rows, config):
        target, _, top, bottom = keys
        top = float(top)
        bottom = float(bottom)

        row = _metadata_row(keys, group)
        rel_depth = _relative_depth(group[config.depth_col], top, bottom)

        _append_geometry_features(row, str(target), top, bottom, group, profile_ranges, config)

        for col in config.feature_columns:
            values = _numeric_array(group[col] if col in group.columns else pd.Series(dtype=float))
            depths = _numeric_array(group[config.depth_col])

            _append_distribution_features(row, col, values)

            sampled = _interpolate_to_grid(rel_depth, values, grid)
            _assign_indexed_features(row, col, sampled, "r")

            derivative_shape = _gradient_features(sampled, grid)
            _assign_indexed_features(row, col, derivative_shape, "dr")

            _append_real_depth_derivative_features(row, col, depths, values)

        _append_engineered_cpt_features(row, group)

        rows.append(row)

    return pd.DataFrame(rows)


def _summary_feature_matrix_columns(config: Config) -> list[str]:
    cols: list[str] = []

    for sensor in config.feature_columns:
        for stat_name in config.segment_stats:
            if stat_name in _STAT_FUNCS:
                cols.append(f"{sensor}_{stat_name}")

    return _append_optional_feature_columns(cols, config)


def _shape_feature_matrix_columns(
    config: Config,
    base_prefix: str,
    derivative_prefix: str,
    size: int,
) -> list[str]:
    cols: list[str] = []

    for sensor in config.feature_columns:
        cols.extend(_indexed_feature_names(sensor, base_prefix, size))
        if config.include_derivatives:
            cols.extend(_indexed_feature_names(sensor, derivative_prefix, size))

    return _append_optional_feature_columns(cols, config)


def _hybrid_feature_matrix_columns(config: Config) -> list[str]:
    shape_length = int(config.representation_length or getattr(config, "hybrid_shape_length", 24) or 24)

    cols: list[str] = [
        "seg_top_depth",
        "seg_bottom_depth",
        "seg_mid_depth",
        "seg_thickness",
        "seg_log_thickness",
        "seg_n_rows",
        "seg_row_density",
        "profile_min_depth",
        "profile_max_depth",
        "profile_depth_range",
        "seg_rel_top_profile",
        "seg_rel_mid_profile",
        "seg_rel_bottom_profile",
    ]

    for sensor in config.feature_columns:
        cols.extend(_distribution_feature_names(sensor))
        cols.extend(_indexed_feature_names(sensor, "r", shape_length))
        cols.extend(_indexed_feature_names(sensor, "dr", shape_length))
        cols.extend(
            [
                f"d_{sensor}_dz_mean",
                f"d_{sensor}_dz_std",
                f"d_{sensor}_dz_median",
                f"d_{sensor}_dz_min",
                f"d_{sensor}_dz_max",
                f"d_{sensor}_dz_abs_mean",
                f"{sensor}_roughness",
            ]
        )

    cols.extend(
        [
            "Rf_mean",
            "Rf_std",
            "Rf_median",
            "Rf_q25",
            "Rf_q75",
            "Rf_iqr",
            "log_qc_mean",
            "log_qc_std",
            "log_qc_median",
            "log_qc_q25",
            "log_qc_q75",
            "log_SCPT_NQT_mean",
            "log_SCPT_NQT_std",
            "log_SCPT_NFR_mean",
            "log_SCPT_NFR_std",
        ]
    )

    return _dedupe_preserve_order(cols)


def _append_optional_feature_columns(cols: list[str], config: Config) -> list[str]:
    if config.include_thickness:
        cols.append("thickness")

    if "Rf" in config.extra_derived:
        cols.extend(["Rf_mean", "Rf_std", "Rf_median"])

    if "log_qc" in config.extra_derived:
        cols.extend(["log_qc_mean", "log_qc_std"])

    return _dedupe_preserve_order(cols)


def _append_distribution_features(row: dict, sensor: str, values: np.ndarray) -> None:
    names = _distribution_feature_names(sensor)
    for name in names:
        row[name] = np.nan

    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return

    q10, q25, q75, q90 = np.quantile(finite, [0.10, 0.25, 0.75, 0.90])

    row[f"{sensor}_mean"] = float(np.mean(finite))
    row[f"{sensor}_std"] = float(np.std(finite, ddof=0))
    row[f"{sensor}_median"] = float(np.median(finite))
    row[f"{sensor}_min"] = float(np.min(finite))
    row[f"{sensor}_max"] = float(np.max(finite))
    row[f"{sensor}_q10"] = float(q10)
    row[f"{sensor}_q25"] = float(q25)
    row[f"{sensor}_q75"] = float(q75)
    row[f"{sensor}_q90"] = float(q90)
    row[f"{sensor}_iqr"] = float(q75 - q25)
    row[f"{sensor}_range"] = float(np.max(finite) - np.min(finite))


def _distribution_feature_names(sensor: str) -> list[str]:
    return [
        f"{sensor}_mean",
        f"{sensor}_std",
        f"{sensor}_median",
        f"{sensor}_min",
        f"{sensor}_max",
        f"{sensor}_q10",
        f"{sensor}_q25",
        f"{sensor}_q75",
        f"{sensor}_q90",
        f"{sensor}_iqr",
        f"{sensor}_range",
    ]


def _append_real_depth_derivative_features(
    row: dict,
    sensor: str,
    depths: np.ndarray,
    values: np.ndarray,
) -> None:
    names = [
        f"d_{sensor}_dz_mean",
        f"d_{sensor}_dz_std",
        f"d_{sensor}_dz_median",
        f"d_{sensor}_dz_min",
        f"d_{sensor}_dz_max",
        f"d_{sensor}_dz_abs_mean",
        f"{sensor}_roughness",
    ]

    for name in names:
        row[name] = np.nan

    valid = np.isfinite(depths) & np.isfinite(values)
    if valid.sum() < 2:
        return

    d = depths[valid]
    v = values[valid]

    order = np.argsort(d, kind="mergesort")
    d = d[order]
    v = v[order]

    collapsed = (
        pd.DataFrame({"depth": d, "value": v})
        .groupby("depth", sort=True, as_index=False)["value"]
        .mean()
    )

    d = collapsed["depth"].to_numpy(dtype=float, copy=False)
    v = collapsed["value"].to_numpy(dtype=float, copy=False)

    if len(d) < 2 or float(d[-1] - d[0]) <= 0:
        return

    grad = np.gradient(v, d)
    finite_grad = grad[np.isfinite(grad)]

    if len(finite_grad):
        row[f"d_{sensor}_dz_mean"] = float(np.mean(finite_grad))
        row[f"d_{sensor}_dz_std"] = float(np.std(finite_grad, ddof=0))
        row[f"d_{sensor}_dz_median"] = float(np.median(finite_grad))
        row[f"d_{sensor}_dz_min"] = float(np.min(finite_grad))
        row[f"d_{sensor}_dz_max"] = float(np.max(finite_grad))
        row[f"d_{sensor}_dz_abs_mean"] = float(np.mean(np.abs(finite_grad)))

    diff = np.diff(v)
    dz = np.diff(d)
    valid_steps = np.isfinite(diff) & np.isfinite(dz) & (dz > 0)

    if valid_steps.any():
        row[f"{sensor}_roughness"] = float(np.mean(np.abs(diff[valid_steps] / dz[valid_steps])))


def _append_engineered_cpt_features(row: dict, group: pd.DataFrame) -> None:
    engineered_cols = [
        "Rf_mean",
        "Rf_std",
        "Rf_median",
        "Rf_q25",
        "Rf_q75",
        "Rf_iqr",
        "log_qc_mean",
        "log_qc_std",
        "log_qc_median",
        "log_qc_q25",
        "log_qc_q75",
        "log_SCPT_NQT_mean",
        "log_SCPT_NQT_std",
        "log_SCPT_NFR_mean",
        "log_SCPT_NFR_std",
    ]

    for col in engineered_cols:
        row[col] = np.nan

    if "SCPT_RES" in group.columns and "SCPT_FRES" in group.columns:
        qc = _numeric_array(group["SCPT_RES"])
        fs = _numeric_array(group["SCPT_FRES"])
        rf = np.where(qc > 0, fs / qc, np.nan)
        _append_quantile_summary(row, "Rf", rf)

    if "SCPT_RES" in group.columns:
        qc = _numeric_array(group["SCPT_RES"])
        log_qc = np.where(np.isfinite(qc), np.log1p(np.clip(qc, 0, None)), np.nan)
        _append_log_summary(row, "log_qc", log_qc)

    if "SCPT_NQT" in group.columns:
        nqt = _numeric_array(group["SCPT_NQT"])
        log_nqt = np.where((nqt > 0) & np.isfinite(nqt), np.log1p(nqt), np.nan)
        _append_mean_std(row, "log_SCPT_NQT", log_nqt)

    if "SCPT_NFR" in group.columns:
        nfr = _numeric_array(group["SCPT_NFR"])
        log_nfr = np.where((nfr > 0) & np.isfinite(nfr), np.log1p(nfr), np.nan)
        _append_mean_std(row, "log_SCPT_NFR", log_nfr)


def _append_quantile_summary(row: dict, prefix: str, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return

    q25, q75 = np.quantile(finite, [0.25, 0.75])

    row[f"{prefix}_mean"] = float(np.mean(finite))
    row[f"{prefix}_std"] = float(np.std(finite, ddof=0))
    row[f"{prefix}_median"] = float(np.median(finite))
    row[f"{prefix}_q25"] = float(q25)
    row[f"{prefix}_q75"] = float(q75)
    row[f"{prefix}_iqr"] = float(q75 - q25)


def _append_log_summary(row: dict, prefix: str, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return

    q25, q75 = np.quantile(finite, [0.25, 0.75])

    row[f"{prefix}_mean"] = float(np.mean(finite))
    row[f"{prefix}_std"] = float(np.std(finite, ddof=0))
    row[f"{prefix}_median"] = float(np.median(finite))
    row[f"{prefix}_q25"] = float(q25)
    row[f"{prefix}_q75"] = float(q75)


def _append_mean_std(row: dict, prefix: str, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return

    row[f"{prefix}_mean"] = float(np.mean(finite))
    row[f"{prefix}_std"] = float(np.std(finite, ddof=0))


def _append_geometry_features(
    row: dict,
    target: str,
    top: float,
    bottom: float,
    group: pd.DataFrame,
    profile_ranges: dict[str, tuple[float, float]],
    config: Config,
) -> None:
    thickness = bottom - top
    mid = (top + bottom) / 2.0
    n_rows = len(group)

    profile_min, profile_max = profile_ranges.get(target, (top, bottom))
    profile_range = profile_max - profile_min

    row["seg_top_depth"] = top
    row["seg_bottom_depth"] = bottom
    row["seg_mid_depth"] = mid
    row["seg_thickness"] = thickness
    row["seg_log_thickness"] = float(np.log1p(max(thickness, 0.0)))
    row["seg_n_rows"] = float(n_rows)
    row["seg_row_density"] = float(n_rows / thickness) if thickness > 0 else np.nan

    row["profile_min_depth"] = profile_min
    row["profile_max_depth"] = profile_max
    row["profile_depth_range"] = profile_range

    if profile_range > 0:
        row["seg_rel_top_profile"] = (top - profile_min) / profile_range
        row["seg_rel_mid_profile"] = (mid - profile_min) / profile_range
        row["seg_rel_bottom_profile"] = (bottom - profile_min) / profile_range
    else:
        row["seg_rel_top_profile"] = 0.0
        row["seg_rel_mid_profile"] = 0.0
        row["seg_rel_bottom_profile"] = 0.0


def _profile_depth_ranges(
    segmented_rows: pd.DataFrame,
    config: Config,
) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}

    for target, group in segmented_rows.groupby("target", sort=False):
        depths = _finite_array(group[config.depth_col])
        if len(depths):
            ranges[str(target)] = (float(depths.min()), float(depths.max()))

    return ranges


def _segment_groups(segmented_rows: pd.DataFrame, config: Config):
    for keys, group in segmented_rows.groupby(_GROUP_COLS, sort=True):
        yield keys, group.sort_values(config.depth_col, kind="mergesort")


def _metadata_row(keys: tuple, group: pd.DataFrame) -> dict:
    target, seg_id, top, bottom = keys

    return {
        "target": target,
        "segment_id": int(seg_id),
        "top": float(top),
        "bottom": float(bottom),
        "thickness": float(bottom) - float(top),
        "n_rows": len(group),
    }


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _numeric_array(series: pd.Series) -> np.ndarray:
    return _safe_numeric_series(series).to_numpy(dtype=float, copy=False)


def _finite_array(series: pd.Series) -> np.ndarray:
    arr = _numeric_array(series)
    return arr[np.isfinite(arr)]


def _relative_depth(depths: pd.Series, top: float, bottom: float) -> np.ndarray:
    depth_values = _numeric_array(depths)
    thickness = float(bottom) - float(top)

    if thickness <= 0:
        return np.zeros(len(depth_values), dtype=float)

    rel = (depth_values - float(top)) / thickness
    return np.clip(rel, 0.0, 1.0)


def _interpolate_to_grid(rel_depth: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(rel_depth) & np.isfinite(values)

    if not valid.any():
        return np.full_like(grid, np.nan, dtype=float)

    rel = rel_depth[valid]
    vals = values[valid]

    order = np.argsort(rel, kind="mergesort")
    rel = rel[order]
    vals = vals[order]

    collapsed = (
        pd.DataFrame({"rel_depth": rel, "value": vals})
        .groupby("rel_depth", sort=True, as_index=False)["value"]
        .mean()
    )

    unique_depth = collapsed["rel_depth"].to_numpy(dtype=float, copy=False)
    unique_values = collapsed["value"].to_numpy(dtype=float, copy=False)

    if len(unique_depth) == 1:
        return np.full_like(grid, unique_values[0], dtype=float)

    return np.interp(
        grid,
        unique_depth,
        unique_values,
        left=unique_values[0],
        right=unique_values[-1],
    )


def _gradient_features(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).any():
        return np.full_like(values, np.nan, dtype=float)

    if len(values) == 1:
        return np.array([0.0 if np.isfinite(values[0]) else np.nan], dtype=float)

    return np.gradient(values, positions)


def _assign_indexed_features(row: dict, column: str, values: np.ndarray, prefix: str) -> None:
    for name, value in zip(_indexed_feature_names(column, prefix, len(values)), values, strict=True):
        row[name] = float(value) if np.isfinite(value) else np.nan


def _indexed_feature_names(column: str, prefix: str, size: int) -> list[str]:
    width = max(2, len(str(max(size - 1, 0))))
    return [f"{column}_{prefix}{idx:0{width}d}" for idx in range(size)]


def _paa_from_resampled(values: np.ndarray, bins: int) -> np.ndarray:
    if not np.isfinite(values).any():
        return np.full(bins, np.nan, dtype=float)

    chunks = np.array_split(values, bins)
    out = np.empty(bins, dtype=float)

    for idx, chunk in enumerate(chunks):
        finite = chunk[np.isfinite(chunk)]
        out[idx] = float(finite.mean()) if len(finite) else np.nan

    return out


def _dedupe_preserve_order(cols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)

    return out


def _append_extra_derived_features(row: dict, group: pd.DataFrame, config: Config) -> None:
    if "Rf" in config.extra_derived:
        if "SCPT_RES" not in group.columns or "SCPT_FRES" not in group.columns:
            row["Rf_mean"] = np.nan
            row["Rf_std"] = np.nan
            row["Rf_median"] = np.nan
        else:
            qc = _safe_numeric_series(group["SCPT_RES"])
            fs = _safe_numeric_series(group["SCPT_FRES"])
            rf = (fs / qc.replace(0, np.nan)).dropna()
            row["Rf_mean"] = float(rf.mean()) if len(rf) else np.nan
            row["Rf_std"] = float(rf.std(ddof=0)) if len(rf) else np.nan
            row["Rf_median"] = float(rf.median()) if len(rf) else np.nan

    if "log_qc" in config.extra_derived:
        if "SCPT_RES" not in group.columns:
            row["log_qc_mean"] = np.nan
            row["log_qc_std"] = np.nan
        else:
            qc = _safe_numeric_series(group["SCPT_RES"])
            log_qc = np.log1p(qc.clip(lower=0)).dropna()
            row["log_qc_mean"] = float(log_qc.mean()) if len(log_qc) else np.nan
            row["log_qc_std"] = float(log_qc.std(ddof=0)) if len(log_qc) else np.nan