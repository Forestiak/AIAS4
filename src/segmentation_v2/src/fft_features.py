from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


_GROUP_COLS = ["target", "segment_id", "seg_top", "seg_bottom"]

_DEFAULT_FFT_BANDS: tuple[tuple[float, float], ...] = (
    (0.25, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
)


def build_fft_features(segmented_rows: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Build segment-level FFT features.

    One row = one boundary-defined segment.

    Features per signal:
    - normalized band energy in fixed frequency bins
    - total/log spectral energy in the thesis band
    - dominant frequency
    - dominant amplitude
    - spectral centroid
    - low/high energy ratio
    """
    rows: list[dict] = []

    for keys, group in _segment_groups(segmented_rows, config):
        row = _metadata_row(keys, group)

        for sensor in config.feature_columns:
            _append_sensor_fft_features(row, group, sensor, config)

        rows.append(row)

    return pd.DataFrame(rows)


def fft_feature_matrix_columns(config: Config) -> list[str]:
    cols: list[str] = []

    if config.include_thickness:
        cols.append("thickness")

    for sensor in config.feature_columns:
        cols.extend(_sensor_fft_feature_names(sensor, config))

    return _dedupe_preserve_order(cols)


def _append_sensor_fft_features(
    row: dict,
    group: pd.DataFrame,
    sensor: str,
    config: Config,
) -> None:
    for name in _sensor_fft_feature_names(sensor, config):
        row[name] = np.nan

    if sensor not in group.columns:
        return

    result = _compute_segment_fft(group, sensor, config)
    if result is None:
        return

    freq, amp = result
    power = amp ** 2

    min_freq = _fft_min_frequency(config)
    max_freq = _fft_max_frequency(config)

    full_mask = (freq >= min_freq) & (freq <= max_freq)
    if not full_mask.any():
        return

    full_freq = freq[full_mask]
    full_amp = amp[full_mask]
    full_power = power[full_mask]

    total_energy = float(np.sum(full_power))
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        return

    full_suffix = _band_suffix(min_freq, max_freq)

    row[f"fft_{sensor}_total_energy_{full_suffix}"] = total_energy
    row[f"fft_{sensor}_log_total_energy_{full_suffix}"] = float(np.log1p(total_energy))

    peak_idx = int(np.argmax(full_amp))
    dominant_freq = float(full_freq[peak_idx])
    dominant_amp = float(full_amp[peak_idx])

    row[f"fft_{sensor}_dominant_freq_{full_suffix}"] = dominant_freq
    row[f"fft_{sensor}_dominant_amp_{full_suffix}"] = dominant_amp

    centroid = float(np.sum(full_freq * full_power) / total_energy)
    row[f"fft_{sensor}_spectral_centroid_{full_suffix}"] = centroid

    for band_min, band_max in _fft_bands(config):
        band_mask = (freq >= band_min) & (freq < band_max)
        band_energy = float(np.sum(power[band_mask])) if band_mask.any() else 0.0
        band_suffix = _band_suffix(band_min, band_max)

        row[f"fft_{sensor}_energy_{band_suffix}_norm"] = band_energy / total_energy

    low_mask = (freq >= min_freq) & (freq < 2.0)
    high_mask = (freq >= 2.0) & (freq <= max_freq)

    low_energy = float(np.sum(power[low_mask])) if low_mask.any() else 0.0
    high_energy = float(np.sum(power[high_mask])) if high_mask.any() else 0.0

    row[f"fft_{sensor}_low_high_energy_ratio_{full_suffix}"] = (
        low_energy / high_energy if high_energy > 0 else np.nan
    )


def _compute_segment_fft(
    segment: pd.DataFrame,
    signal_col: str,
    config: Config,
) -> tuple[np.ndarray, np.ndarray] | None:
    depth_col = config.depth_col
    min_points = _fft_min_points(config)
    dz = _fft_depth_step(config)

    work = segment[[depth_col, signal_col]].dropna().copy()
    if len(work) < min_points:
        return None

    work[depth_col] = pd.to_numeric(work[depth_col], errors="coerce")
    work[signal_col] = pd.to_numeric(work[signal_col], errors="coerce")
    work = work.dropna()

    if len(work) < min_points:
        return None

    work = work.groupby(depth_col, as_index=False)[signal_col].mean()
    work = work.sort_values(depth_col)

    depth = work[depth_col].to_numpy(dtype=float)
    values = work[signal_col].to_numpy(dtype=float)

    if len(depth) < min_points:
        return None

    start = float(depth.min())
    stop = float(depth.max())
    if stop <= start:
        return None

    uniform_depth = np.arange(start, stop + dz, dz)
    if len(uniform_depth) < min_points:
        return None

    y = np.interp(uniform_depth, depth, values)

    if _fft_detrend(config):
        x = uniform_depth - uniform_depth.mean()
        slope, intercept = np.polyfit(x, y, deg=1)
        y = y - (slope * x + intercept)

    y = y - np.mean(y)

    window = _fft_window(config)
    if window == "hann":
        y = y * np.hanning(len(y))
    elif window != "none":
        raise ValueError(f"Unsupported FFT window: {window}")

    fft = np.fft.rfft(y)
    freq = np.fft.rfftfreq(len(y), d=dz)

    # Amplitude spectrum. Same scaling as the diagnostic script.
    amp = (2.0 / len(y)) * np.abs(fft)

    return freq, amp


def _sensor_fft_feature_names(sensor: str, config: Config) -> list[str]:
    min_freq = _fft_min_frequency(config)
    max_freq = _fft_max_frequency(config)
    full_suffix = _band_suffix(min_freq, max_freq)

    names: list[str] = []

    for band_min, band_max in _fft_bands(config):
        band_suffix = _band_suffix(band_min, band_max)
        names.append(f"fft_{sensor}_energy_{band_suffix}_norm")

    names.extend(
        [
            f"fft_{sensor}_total_energy_{full_suffix}",
            f"fft_{sensor}_log_total_energy_{full_suffix}",
            f"fft_{sensor}_dominant_freq_{full_suffix}",
            f"fft_{sensor}_dominant_amp_{full_suffix}",
            f"fft_{sensor}_spectral_centroid_{full_suffix}",
            f"fft_{sensor}_low_high_energy_ratio_{full_suffix}",
        ]
    )

    return names


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


def _fft_depth_step(config: Config) -> float:
    return float(getattr(config, "fft_depth_step_m", 0.02))


def _fft_min_points(config: Config) -> int:
    return int(getattr(config, "fft_min_points_per_segment", 8))


def _fft_min_frequency(config: Config) -> float:
    return float(getattr(config, "fft_min_frequency_cycles_per_m", 0.25))


def _fft_max_frequency(config: Config) -> float:
    return float(getattr(config, "fft_max_frequency_cycles_per_m", 8.0))


def _fft_bands(config: Config) -> tuple[tuple[float, float], ...]:
    bands = getattr(config, "fft_bands_cycles_per_m", _DEFAULT_FFT_BANDS)
    return tuple((float(lo), float(hi)) for lo, hi in bands)


def _fft_detrend(config: Config) -> bool:
    return bool(getattr(config, "fft_detrend", True))


def _fft_window(config: Config) -> str:
    return str(getattr(config, "fft_window", "hann"))


def _band_suffix(low: float, high: float) -> str:
    return f"{int(round(low * 100)):03d}_{int(round(high * 100)):03d}"


def _dedupe_preserve_order(cols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)

    return out