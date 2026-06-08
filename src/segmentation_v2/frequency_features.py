"""
STFT-based frequency-domain segment features for CPT profiles.

The method:
1. Uses the full CPT profile per Target and sensor.
2. Resamples the signal to a uniform depth grid.
3. Computes a depth-localized frequency representation using STFT/spectrogram.
4. Summarizes fixed frequency bins inside each boundary-defined segment.

Output:
- One row per segment.
- Fixed number of frequency features regardless of segment thickness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import spectrogram

try:
    from .config import Config
except ImportError:
    from config import Config


@dataclass(frozen=True)
class FrequencyFeatureConfig:
    depth_step_m: float = 0.05
    window_length_m: float = 1.0
    window_overlap_m: float = 0.5
    frequency_min_cpm: float = 0.25
    frequency_max_cpm: float = 5.0
    frequency_step_cpm: float = 0.25
    min_segment_windows: int = 1
    normalize_power: bool = True


def build_stft_segment_features(
    segmented_rows: pd.DataFrame,
    config: Config,
    freq_config: FrequencyFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Build fixed-size frequency-domain features for boundary-defined segments.

    Args:
        segmented_rows:
            CPT rows after segment assignment. Expected columns:
            target, segment_id, seg_top, seg_bottom, depth, and sensor columns.

        config:
            Main pipeline config.

        freq_config:
            Frequency extraction configuration.

    Returns:
        DataFrame with one row per segment and fixed frequency features.
    """
    freq_config = freq_config or FrequencyFeatureConfig()

    segment_index = _segment_index(segmented_rows)
    feature_rows = segment_index.copy()

    for sensor in config.feature_columns:
        if sensor not in segmented_rows.columns:
            continue

        sensor_features = _build_sensor_stft_features(
            segmented_rows=segmented_rows,
            segment_index=segment_index,
            sensor=sensor,
            config=config,
            freq_config=freq_config,
        )

        feature_rows = feature_rows.merge(
            sensor_features,
            on=["target", "segment_id", "top", "bottom"],
            how="left",
        )

    return feature_rows


def frequency_feature_columns(
    config: Config,
    freq_config: FrequencyFeatureConfig | None = None,
) -> list[str]:
    """
    Return the ordered STFT feature columns expected by the model.
    """
    freq_config = freq_config or FrequencyFeatureConfig()
    grid = _frequency_grid(freq_config)

    cols: list[str] = []
    for sensor in config.feature_columns:
        for freq in grid:
            cols.append(_feature_name(sensor, freq))
    return cols


def _build_sensor_stft_features(
    segmented_rows: pd.DataFrame,
    segment_index: pd.DataFrame,
    sensor: str,
    config: Config,
    freq_config: FrequencyFeatureConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for target, profile in segmented_rows.groupby("target", sort=False):
        target = str(target)
        profile = profile.sort_values(config.depth_col, kind="mergesort")

        depth = pd.to_numeric(profile[config.depth_col], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(profile[sensor], errors="coerce").to_numpy(dtype=float)

        stft_result = _profile_spectrogram(
            depth=depth,
            values=values,
            freq_config=freq_config,
        )

        target_segments = segment_index[segment_index["target"].astype(str) == target]

        if stft_result is None:
            for seg in target_segments.itertuples(index=False):
                rows.append(_empty_feature_row(seg, sensor, freq_config))
            continue

        window_depths, freq_grid, power_grid = stft_result

        for seg in target_segments.itertuples(index=False):
            row = {
                "target": seg.target,
                "segment_id": int(seg.segment_id),
                "top": float(seg.top),
                "bottom": float(seg.bottom),
            }

            mask = (window_depths >= float(seg.top)) & (window_depths <= float(seg.bottom))

            if mask.sum() < freq_config.min_segment_windows:
                for freq in freq_grid:
                    row[_feature_name(sensor, freq)] = np.nan
            else:
                segment_power = power_grid[:, mask]

                if freq_config.normalize_power:
                    total_power = np.nansum(segment_power)
                    if total_power > 0:
                        segment_power = segment_power / total_power

                mean_power_by_freq = np.nanmean(segment_power, axis=1)

                for freq, value in zip(freq_grid, mean_power_by_freq, strict=True):
                    row[_feature_name(sensor, freq)] = float(value) if np.isfinite(value) else np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def _profile_spectrogram(
    depth: np.ndarray,
    values: np.ndarray,
    freq_config: FrequencyFeatureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    valid = np.isfinite(depth) & np.isfinite(values)

    if valid.sum() < 8:
        return None

    depth = depth[valid]
    values = values[valid]

    order = np.argsort(depth, kind="mergesort")
    depth = depth[order]
    values = values[order]

    collapsed = (
        pd.DataFrame({"depth": depth, "value": values})
        .groupby("depth", as_index=False, sort=True)["value"]
        .mean()
    )

    depth = collapsed["depth"].to_numpy(dtype=float)
    values = collapsed["value"].to_numpy(dtype=float)

    depth_min = float(depth.min())
    depth_max = float(depth.max())

    if depth_max <= depth_min:
        return None

    uniform_depth = np.arange(depth_min, depth_max + freq_config.depth_step_m, freq_config.depth_step_m)

    if len(uniform_depth) < 8:
        return None

    uniform_values = np.interp(
        uniform_depth,
        depth,
        values,
        left=values[0],
        right=values[-1],
    )

    uniform_values = _detrend_linear(uniform_depth, uniform_values)

    fs = 1.0 / freq_config.depth_step_m

    nperseg = max(8, int(round(freq_config.window_length_m / freq_config.depth_step_m)))
    noverlap = int(round(freq_config.window_overlap_m / freq_config.depth_step_m))
    noverlap = min(noverlap, nperseg - 1)

    if len(uniform_values) < nperseg:
        return None

    frequencies, window_positions, power = spectrogram(
        uniform_values,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="psd",
    )

    # Convert STFT window position from sample-domain distance to absolute depth.
    window_depths = depth_min + window_positions

    target_freq_grid = _frequency_grid(freq_config)

    if len(frequencies) < 2:
        return None

    # Interpolate each depth-window spectrum onto a fixed physical frequency grid.
    interpolated_power = np.empty((len(target_freq_grid), power.shape[1]), dtype=float)

    for window_idx in range(power.shape[1]):
        interpolated_power[:, window_idx] = np.interp(
            target_freq_grid,
            frequencies,
            power[:, window_idx],
            left=np.nan,
            right=np.nan,
        )

    return window_depths, target_freq_grid, interpolated_power


def _segment_index(segmented_rows: pd.DataFrame) -> pd.DataFrame:
    index = (
        segmented_rows[["target", "segment_id", "seg_top", "seg_bottom"]]
        .drop_duplicates()
        .rename(columns={"seg_top": "top", "seg_bottom": "bottom"})
        .sort_values(["target", "segment_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    index["segment_id"] = index["segment_id"].astype(int)
    index["top"] = index["top"].astype(float)
    index["bottom"] = index["bottom"].astype(float)

    return index


def _frequency_grid(freq_config: FrequencyFeatureConfig) -> np.ndarray:
    return np.round(
        np.arange(
            freq_config.frequency_min_cpm,
            freq_config.frequency_max_cpm + freq_config.frequency_step_cpm / 2.0,
            freq_config.frequency_step_cpm,
        ),
        8,
    )


def _feature_name(sensor: str, frequency_cpm: float) -> str:
    freq_label = f"{frequency_cpm:.2f}".replace(".", "_")
    return f"{sensor}_stft_pwr_{freq_label}"


def _empty_feature_row(
    segment: object,
    sensor: str,
    freq_config: FrequencyFeatureConfig,
) -> dict[str, float | int | str]:
    row = {
        "target": segment.target,
        "segment_id": int(segment.segment_id),
        "top": float(segment.top),
        "bottom": float(segment.bottom),
    }

    for freq in _frequency_grid(freq_config):
        row[_feature_name(sensor, freq)] = np.nan

    return row


def _detrend_linear(depth: np.ndarray, values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & np.isfinite(values)

    if valid.sum() < 2:
        return values - np.nanmean(values)

    coefficients = np.polyfit(depth[valid], values[valid], deg=1)
    trend = np.polyval(coefficients, depth)

    return values - trend