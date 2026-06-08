"""
STFT frequency-domain analysis for CPT segment representation.

Purpose:
- Load shared CPT data from ../data, not segmentation_v2/data.
- Build boundary-defined segments from strata or exported boundaries.
- Apply STFT/windowed FFT to each full CPT profile along depth.
- Aggregate fixed frequency-bin powers inside each segment.
- Export CSV feature matrix, quality report, JSON summary, and plots.

Default:
    pixi run python .\\frequency_analysis.py

Explicit:
    pixi run python .\\frequency_analysis.py --data-dir C:\\Studies\\master_thesis\\data
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram


SCRIPT_DIR = Path(__file__).resolve().parent

TARGET_COL = "Target"
DEPTH_COL = "Depth"
POINT_ID_COL = "PointID"

STRATA_TOP_COL = "Top"
STRATA_BOTTOM_COL = "Bottom"
STRATA_UNIT_COL = "UNIT"

DEFAULT_SENSORS = ("SCPT_RES", "SCPT_FRES", "SCPT_PWP2")


@dataclass(frozen=True)
class AnalysisConfig:
    data_dir: Path
    output_dir: Path
    boundary_source: str
    boundary_file: str
    min_segment_thickness_m: float


@dataclass(frozen=True)
class StftConfig:
    depth_step_m: float
    window_length_m: float
    window_overlap_m: float
    frequency_min_cpm: float
    frequency_max_cpm: float
    frequency_step_cpm: float
    min_profile_points: int
    min_segment_windows: int
    normalize_segment_power: bool
    detrend_signal: bool
    max_plotted_targets: int


@dataclass(frozen=True)
class ProfileSpectrogram:
    target: str
    sensor: str
    window_depths: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = args.output_dir or SCRIPT_DIR / "outputs" / "frequency_analysis_stft"

    analysis_config = AnalysisConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        boundary_source=args.boundary_source,
        boundary_file=args.boundary_file,
        min_segment_thickness_m=args.min_segment_thickness_m,
    )

    stft_config = StftConfig(
        depth_step_m=args.depth_step_m,
        window_length_m=args.window_length_m,
        window_overlap_m=args.window_overlap_m,
        frequency_min_cpm=args.frequency_min_cpm,
        frequency_max_cpm=args.frequency_max_cpm,
        frequency_step_cpm=args.frequency_step_cpm,
        min_profile_points=args.min_profile_points,
        min_segment_windows=args.min_segment_windows,
        normalize_segment_power=not args.no_normalize,
        detrend_signal=not args.no_detrend,
        max_plotted_targets=args.max_plotted_targets,
    )

    sensors = tuple(args.sensors) if args.sensors else DEFAULT_SENSORS

    run_analysis(analysis_config, stft_config, sensors)


def run_analysis(
    analysis_config: AnalysisConfig,
    stft_config: StftConfig,
    sensors: tuple[str, ...],
) -> None:
    plots_dir = analysis_config.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print_header("STFT frequency analysis")
    print(f"Data folder:        {analysis_config.data_dir}")
    print(f"CPT file:           {analysis_config.data_dir / 'CPT_clean.csv'}")
    print(f"Strata file:        {analysis_config.data_dir / 'Input_Strata_merged_boundaries.csv'}")
    print(f"Output folder:      {analysis_config.output_dir}")
    print(f"Boundary source:    {analysis_config.boundary_source}")
    print(f"Sensors:            {', '.join(sensors)}")
    print(f"Frequency grid:     {stft_config.frequency_min_cpm} to {stft_config.frequency_max_cpm} cycles/m")
    print(f"Frequency step:     {stft_config.frequency_step_cpm} cycles/m")
    print(f"Window length:      {stft_config.window_length_m} m")
    print(f"Window overlap:     {stft_config.window_overlap_m} m")
    print()

    print_step("1/6 Loading CPT and strata data")
    cpt = load_cpt(analysis_config.data_dir, sensors)
    strata = load_strata(analysis_config.data_dir)
    print(f"Loaded CPT rows:    {len(cpt):,}")
    print(f"Loaded strata rows: {len(strata):,}")
    print()

    print_step("2/6 Building boundary-defined segments")
    profiles = split_by_target(cpt)
    boundaries = load_boundaries(strata, analysis_config)
    segment_table, segmented_rows = build_segments(
        profiles=profiles,
        boundaries=boundaries,
        min_segment_thickness_m=analysis_config.min_segment_thickness_m,
    )
    segment_reference = attach_reference_units(segment_table, strata)
    print(f"Profiles:           {len(profiles):,}")
    print(f"Segments:           {len(segment_table):,}")
    print()

    print_step("3/6 Computing STFT features")
    fixed_frequencies = frequency_grid(stft_config)
    all_features: list[pd.DataFrame] = []
    plotted_targets = 0
    spectra_computed = 0
    spectra_skipped = 0

    for target, profile in segmented_rows.groupby("target", sort=False):
        target = str(target)
        target_segments = segment_reference[
            segment_reference["target"].astype(str) == target
        ].copy()

        if target_segments.empty:
            continue

        should_plot = plotted_targets < stft_config.max_plotted_targets

        for sensor in sensors:
            if sensor not in profile.columns:
                print(f"Skipping missing sensor column: {sensor}")
                continue

            spec = compute_profile_spectrogram(
                profile=profile,
                target=target,
                sensor=sensor,
                stft_config=stft_config,
            )

            if spec is None:
                spectra_skipped += 1
                segment_features = empty_segment_frequency_features(
                    target_segments=target_segments,
                    sensor=sensor,
                    frequencies=fixed_frequencies,
                )
            else:
                spectra_computed += 1
                segment_features = aggregate_spectrogram_by_segment(
                    target_segments=target_segments,
                    spec=spec,
                    stft_config=stft_config,
                )

                if should_plot:
                    plot_raw_signal_with_segments(
                        profile=profile,
                        target_segments=target_segments,
                        target=target,
                        sensor=sensor,
                        output_path=plots_dir / f"raw_{safe_name(target)}_{safe_name(sensor)}.png",
                    )
                    plot_spectrogram(
                        spec=spec,
                        output_path=plots_dir / f"spectrogram_{safe_name(target)}_{safe_name(sensor)}.png",
                    )
                    plot_segment_frequency_matrix(
                        segment_features=segment_features,
                        sensor=sensor,
                        frequencies=fixed_frequencies,
                        output_path=plots_dir / f"segment_frequency_matrix_{safe_name(target)}_{safe_name(sensor)}.png",
                    )

            all_features.append(segment_features)

        plotted_targets += 1

    if not all_features:
        raise RuntimeError("No frequency features were generated. Check sensor names and input data.")

    print(f"Spectrograms computed: {spectra_computed:,}")
    print(f"Spectrograms skipped:  {spectra_skipped:,}")
    print()

    print_step("4/6 Building segment feature matrix")
    long_features = pd.concat(all_features, ignore_index=True)
    wide_features = pivot_sensor_features_to_segment_rows(long_features)
    quality = build_feature_quality_summary(wide_features)

    frequency_feature_count = len([c for c in wide_features.columns if "_stft_pwr_" in c])
    print(f"Feature rows:       {len(wide_features):,}")
    print(f"Frequency features: {frequency_feature_count:,}")
    print()

    print_step("5/6 Generating reference-unit plots")
    plot_reference_unit_mean_spectra(
        features=wide_features,
        sensors=sensors,
        frequencies=fixed_frequencies,
        output_dir=plots_dir,
    )
    print(f"Plots folder:       {plots_dir}")
    print()

    print_step("6/6 Saving outputs")
    analysis_config.output_dir.mkdir(parents=True, exist_ok=True)

    features_path = analysis_config.output_dir / "stft_segment_features.csv"
    quality_path = analysis_config.output_dir / "feature_quality_summary.csv"
    summary_path = analysis_config.output_dir / "analysis_summary.json"

    wide_features.to_csv(features_path, index=False)
    quality.to_csv(quality_path, index=False)

    summary = {
        "method": "STFT / spectrogram, windowed FFT along CPT depth",
        "data_dir": str(analysis_config.data_dir),
        "output_dir": str(analysis_config.output_dir),
        "boundary_source": analysis_config.boundary_source,
        "boundary_file": analysis_config.boundary_file,
        "min_segment_thickness_m": analysis_config.min_segment_thickness_m,
        "n_cpt_rows": int(len(cpt)),
        "n_strata_rows": int(len(strata)),
        "n_profiles": int(len(profiles)),
        "n_segments": int(len(segment_table)),
        "n_segment_feature_rows": int(len(wide_features)),
        "sensors": list(sensors),
        "frequency_feature_count": int(frequency_feature_count),
        "spectrograms_computed": int(spectra_computed),
        "spectrograms_skipped": int(spectra_skipped),
        "stft_config": asdict(stft_config),
        "missing_ratio_mean": float(quality["missing_ratio"].mean()) if not quality.empty else None,
        "missing_ratio_max": float(quality["missing_ratio"].max()) if not quality.empty else None,
        "near_zero_variance_count": int(quality["near_zero_variance"].sum()) if not quality.empty else None,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Features:           {features_path}")
    print(f"Quality report:     {quality_path}")
    print(f"Summary:            {summary_path}")
    print()

    print_header("Completed")
    print("What was done:")
    print("- Loaded shared CPT and strata data from the external data folder.")
    print("- Built boundary-defined segments from the selected boundary source.")
    print("- Resampled each full CPT profile to a uniform depth grid.")
    print("- Applied STFT/windowed FFT along depth for each selected sensor.")
    print("- Aggregated fixed frequency-bin powers inside every segment.")
    print("- Saved fixed-size STFT segment features, quality diagnostics, summary JSON, and plots.")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def resolve_data_dir(explicit_data_dir: Path | None) -> Path:
    if explicit_data_dir is not None:
        candidate = explicit_data_dir.resolve()
        validate_data_dir(candidate)
        return candidate

    candidates = [
        SCRIPT_DIR.parent / "data",
        SCRIPT_DIR / "data",
        Path.cwd().parent / "data",
        Path.cwd() / "data",
        Path("C:/Studies/master_thesis/data"),
    ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if is_valid_data_dir(candidate):
            return candidate

    checked = "\n".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        "Could not find the shared data directory.\n\n"
        "Expected files:\n"
        "  CPT_clean.csv\n"
        "  Input_Strata_merged_boundaries.csv\n\n"
        "Checked:\n"
        f"{checked}\n\n"
        "Run explicitly:\n"
        "  pixi run python .\\frequency_analysis.py --data-dir C:\\Studies\\master_thesis\\data"
    )


def is_valid_data_dir(data_dir: Path) -> bool:
    return (
        (data_dir / "CPT_clean.csv").exists()
        and (data_dir / "Input_Strata_merged_boundaries.csv").exists()
    )


def validate_data_dir(data_dir: Path) -> None:
    if not is_valid_data_dir(data_dir):
        raise FileNotFoundError(
            "Invalid data directory.\n\n"
            f"Received: {data_dir}\n\n"
            "Expected files:\n"
            f"  {data_dir / 'CPT_clean.csv'}\n"
            f"  {data_dir / 'Input_Strata_merged_boundaries.csv'}"
        )


def read_and_strip(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            df[col] = series.astype(str).str.strip()

    return df


def load_cpt(data_dir: Path, sensors: tuple[str, ...]) -> pd.DataFrame:
    path = data_dir / "CPT_clean.csv"
    df = read_and_strip(path)

    require_columns(df, [TARGET_COL, DEPTH_COL], path)

    df[DEPTH_COL] = pd.to_numeric(df[DEPTH_COL], errors="coerce")

    for sensor in sensors:
        if sensor in df.columns:
            df[sensor] = pd.to_numeric(df[sensor], errors="coerce")

    return (
        df.dropna(subset=[TARGET_COL, DEPTH_COL])
        .sort_values([TARGET_COL, DEPTH_COL], kind="mergesort")
        .reset_index(drop=True)
    )


def load_strata(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "Input_Strata_merged_boundaries.csv"
    df = read_and_strip(path)

    fallback_path = data_dir / "Input_Strata_merged_boundaries_loc_only.csv"
    if fallback_path.exists():
        fallback = read_and_strip(fallback_path)
        if TARGET_COL in df.columns and TARGET_COL in fallback.columns:
            existing_targets = set(df[TARGET_COL].dropna().astype(str))
            supplement = fallback[
                ~fallback[TARGET_COL].astype(str).isin(existing_targets)
            ].copy()
            if not supplement.empty:
                df = pd.concat([df, supplement], ignore_index=True)

    require_columns(df, [TARGET_COL, STRATA_TOP_COL, STRATA_BOTTOM_COL, STRATA_UNIT_COL], path)

    df[STRATA_TOP_COL] = pd.to_numeric(df[STRATA_TOP_COL], errors="coerce")
    df[STRATA_BOTTOM_COL] = pd.to_numeric(df[STRATA_BOTTOM_COL], errors="coerce")

    df = df.dropna(subset=[TARGET_COL, STRATA_TOP_COL, STRATA_BOTTOM_COL, STRATA_UNIT_COL])

    if POINT_ID_COL in df.columns:
        first_pid = df.groupby(TARGET_COL)[POINT_ID_COL].first()
        df = df[
            df.apply(lambda r: r[POINT_ID_COL] == first_pid[r[TARGET_COL]], axis=1)
        ]

    return df.reset_index(drop=True)


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}.\n"
            f"Missing: {missing}\n"
            f"Available: {list(df.columns)}"
        )


# ---------------------------------------------------------------------
# Segment construction
# ---------------------------------------------------------------------


def split_by_target(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(target): group.copy()
        for target, group in df.groupby(TARGET_COL, sort=False)
    }


def load_boundaries(
    strata: pd.DataFrame,
    config: AnalysisConfig,
) -> dict[str, list[float]]:
    if config.boundary_source == "ground_truth":
        return boundaries_from_strata(strata)

    if config.boundary_source == "exported":
        return boundaries_from_exported(config.data_dir / "boundaries" / config.boundary_file)

    raise ValueError("boundary_source must be 'ground_truth' or 'exported'.")


def boundaries_from_strata(strata: pd.DataFrame) -> dict[str, list[float]]:
    boundaries: dict[str, list[float]] = {}

    for target, group in strata.groupby(TARGET_COL, sort=False):
        tops = sorted(pd.to_numeric(group[STRATA_TOP_COL], errors="coerce").dropna().unique())
        boundaries[str(target)] = [float(v) for v in tops[1:]]

    return boundaries


def boundaries_from_exported(path: Path) -> dict[str, list[float]]:
    df = read_and_strip(path)

    col_map = {c.lower(): c for c in df.columns}
    target_col = col_map.get("target")
    depth_col = col_map.get("boundary_depth") or col_map.get("depth")

    if target_col is None or depth_col is None:
        raise ValueError(
            f"{path} must contain Target and boundary_depth/Depth columns. "
            f"Available: {list(df.columns)}"
        )

    df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
    df = df.dropna(subset=[target_col, depth_col])

    boundaries: dict[str, list[float]] = {}

    for target, group in df.groupby(target_col, sort=False):
        boundaries[str(target)] = sorted(float(v) for v in group[depth_col].unique())

    return boundaries


def build_segments(
    profiles: dict[str, pd.DataFrame],
    boundaries: dict[str, list[float]],
    min_segment_thickness_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_segments: list[pd.DataFrame] = []
    all_rows: list[pd.DataFrame] = []

    targets = boundaries.keys() if boundaries else profiles.keys()

    for target in targets:
        target = str(target)
        profile = profiles.get(target)

        if profile is None or profile.empty:
            continue

        segment_table = segments_for_profile(
            target=target,
            profile=profile,
            boundary_depths=boundaries.get(target, []),
            min_segment_thickness_m=min_segment_thickness_m,
        )

        if segment_table.empty:
            continue

        rows = assign_rows_to_segments(profile, segment_table)

        all_segments.append(segment_table)
        all_rows.append(rows)

    if not all_segments:
        raise RuntimeError("No segments were created. Check boundaries and CPT profiles.")

    return (
        pd.concat(all_segments, ignore_index=True),
        pd.concat(all_rows, ignore_index=True),
    )


def segments_for_profile(
    target: str,
    profile: pd.DataFrame,
    boundary_depths: list[float],
    min_segment_thickness_m: float,
) -> pd.DataFrame:
    depths = pd.to_numeric(profile[DEPTH_COL], errors="coerce").dropna()

    if depths.empty:
        return pd.DataFrame()

    profile_top = float(depths.min())
    profile_bottom = float(depths.max())

    edges = sorted(
        {profile_top, profile_bottom}
        | {float(d) for d in boundary_depths if profile_top < float(d) < profile_bottom}
    )

    if len(edges) < 2:
        return pd.DataFrame()

    rows = [
        {
            "target": target,
            "segment_id": idx,
            "top": float(edges[idx]),
            "bottom": float(edges[idx + 1]),
        }
        for idx in range(len(edges) - 1)
    ]

    segment_table = pd.DataFrame(rows)
    return merge_thin_segments(segment_table, min_segment_thickness_m)


def merge_thin_segments(segment_table: pd.DataFrame, min_thickness: float) -> pd.DataFrame:
    segment_table = segment_table.reset_index(drop=True)

    while len(segment_table) > 1:
        segment_table["thickness"] = segment_table["bottom"] - segment_table["top"]
        thin = segment_table.index[segment_table["thickness"] < min_thickness].tolist()

        if not thin:
            break

        idx = thin[0]

        if idx == 0:
            segment_table.loc[idx + 1, "top"] = segment_table.loc[idx, "top"]
        else:
            segment_table.loc[idx - 1, "bottom"] = segment_table.loc[idx, "bottom"]

        segment_table = segment_table.drop(idx).reset_index(drop=True)

    segment_table["segment_id"] = range(len(segment_table))
    segment_table["thickness"] = segment_table["bottom"] - segment_table["top"]

    return segment_table


def assign_rows_to_segments(profile: pd.DataFrame, segment_table: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    last_segment_id = int(segment_table["segment_id"].max())

    for seg in segment_table.itertuples(index=False):
        if int(seg.segment_id) == last_segment_id:
            mask = (profile[DEPTH_COL] >= float(seg.top)) & (profile[DEPTH_COL] <= float(seg.bottom))
        else:
            mask = (profile[DEPTH_COL] >= float(seg.top)) & (profile[DEPTH_COL] < float(seg.bottom))

        chunk = profile.loc[mask].copy()
        chunk["target"] = str(seg.target)
        chunk["segment_id"] = int(seg.segment_id)
        chunk["seg_top"] = float(seg.top)
        chunk["seg_bottom"] = float(seg.bottom)
        parts.append(chunk)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def attach_reference_units(
    segment_table: pd.DataFrame,
    strata: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for seg in segment_table.itertuples(index=False):
        target_strata = strata[strata[TARGET_COL].astype(str) == str(seg.target)]

        row = seg._asdict()
        row["reference_unit"] = best_overlap_unit(
            segment_top=float(seg.top),
            segment_bottom=float(seg.bottom),
            strata=target_strata,
        )

        rows.append(row)

    return pd.DataFrame(rows)


def best_overlap_unit(
    segment_top: float,
    segment_bottom: float,
    strata: pd.DataFrame,
) -> str | None:
    if strata.empty:
        return None

    tops = pd.to_numeric(strata[STRATA_TOP_COL], errors="coerce").to_numpy(dtype=float)
    bottoms = pd.to_numeric(strata[STRATA_BOTTOM_COL], errors="coerce").to_numpy(dtype=float)
    units = strata[STRATA_UNIT_COL].astype(str).to_numpy(dtype=object)

    overlaps = np.minimum(bottoms, segment_bottom) - np.maximum(tops, segment_top)
    overlaps = np.clip(overlaps, 0.0, None)

    if len(overlaps) == 0 or float(np.nanmax(overlaps)) <= 0:
        return None

    return str(units[int(np.nanargmax(overlaps))])


# ---------------------------------------------------------------------
# STFT feature extraction
# ---------------------------------------------------------------------


def compute_profile_spectrogram(
    profile: pd.DataFrame,
    target: str,
    sensor: str,
    stft_config: StftConfig,
) -> ProfileSpectrogram | None:
    profile = profile.sort_values(DEPTH_COL, kind="mergesort")

    depth = pd.to_numeric(profile[DEPTH_COL], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(profile[sensor], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(depth) & np.isfinite(values)

    if int(valid.sum()) < stft_config.min_profile_points:
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

    if len(depth) < stft_config.min_profile_points:
        return None

    depth_min = float(depth.min())
    depth_max = float(depth.max())

    if depth_max <= depth_min:
        return None

    uniform_depth = np.arange(
        depth_min,
        depth_max + stft_config.depth_step_m,
        stft_config.depth_step_m,
    )

    if len(uniform_depth) < stft_config.min_profile_points:
        return None

    uniform_values = np.interp(
        uniform_depth,
        depth,
        values,
        left=values[0],
        right=values[-1],
    )

    if stft_config.detrend_signal:
        uniform_values = detrend_linear(uniform_depth, uniform_values)
    else:
        uniform_values = uniform_values - np.nanmean(uniform_values)

    sampling_frequency = 1.0 / stft_config.depth_step_m

    nperseg = max(8, int(round(stft_config.window_length_m / stft_config.depth_step_m)))
    noverlap = int(round(stft_config.window_overlap_m / stft_config.depth_step_m))
    noverlap = max(0, min(noverlap, nperseg - 1))

    if len(uniform_values) < nperseg:
        return None

    raw_frequencies, window_positions, raw_power = spectrogram(
        uniform_values,
        fs=sampling_frequency,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="psd",
    )

    if raw_power.size == 0 or len(raw_frequencies) < 2:
        return None

    fixed_frequencies = frequency_grid(stft_config)
    fixed_power = np.empty((len(fixed_frequencies), raw_power.shape[1]), dtype=float)

    for window_idx in range(raw_power.shape[1]):
        fixed_power[:, window_idx] = np.interp(
            fixed_frequencies,
            raw_frequencies,
            raw_power[:, window_idx],
            left=np.nan,
            right=np.nan,
        )

    window_depths = depth_min + window_positions

    return ProfileSpectrogram(
        target=target,
        sensor=sensor,
        window_depths=window_depths,
        frequencies=fixed_frequencies,
        power=fixed_power,
    )


def aggregate_spectrogram_by_segment(
    target_segments: pd.DataFrame,
    spec: ProfileSpectrogram,
    stft_config: StftConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for seg in target_segments.itertuples(index=False):
        top = float(seg.top)
        bottom = float(seg.bottom)

        row: dict[str, object] = {
            "target": str(seg.target),
            "segment_id": int(seg.segment_id),
            "top": top,
            "bottom": bottom,
            "thickness": float(bottom - top),
            "reference_unit": getattr(seg, "reference_unit", None),
            "sensor": spec.sensor,
        }

        mask = (spec.window_depths >= top) & (spec.window_depths <= bottom)
        row[f"{spec.sensor}_stft_n_windows"] = int(mask.sum())

        if int(mask.sum()) < stft_config.min_segment_windows:
            for frequency in spec.frequencies:
                row[feature_name(spec.sensor, frequency)] = np.nan
            rows.append(row)
            continue

        segment_power = spec.power[:, mask].copy()

        if stft_config.normalize_segment_power:
            total_power = float(np.nansum(segment_power))
            if total_power > 0:
                segment_power = segment_power / total_power

        mean_power = np.nanmean(segment_power, axis=1)

        for frequency, value in zip(spec.frequencies, mean_power, strict=True):
            row[feature_name(spec.sensor, frequency)] = (
                float(value) if np.isfinite(value) else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def empty_segment_frequency_features(
    target_segments: pd.DataFrame,
    sensor: str,
    frequencies: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for seg in target_segments.itertuples(index=False):
        top = float(seg.top)
        bottom = float(seg.bottom)

        row: dict[str, object] = {
            "target": str(seg.target),
            "segment_id": int(seg.segment_id),
            "top": top,
            "bottom": bottom,
            "thickness": float(bottom - top),
            "reference_unit": getattr(seg, "reference_unit", None),
            "sensor": sensor,
            f"{sensor}_stft_n_windows": 0,
        }

        for frequency in frequencies:
            row[feature_name(sensor, frequency)] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def pivot_sensor_features_to_segment_rows(long_features: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "segment_id", "top", "bottom", "thickness", "reference_unit"]

    base = (
        long_features[keys]
        .drop_duplicates()
        .sort_values(["target", "segment_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    for _, sensor_group in long_features.groupby("sensor", sort=False):
        feature_cols = [c for c in sensor_group.columns if c not in keys and c != "sensor"]
        base = base.merge(sensor_group[keys + feature_cols], on=keys, how="left")

    return base


def build_feature_quality_summary(features: pd.DataFrame) -> pd.DataFrame:
    ignored = {"target", "segment_id", "top", "bottom", "thickness", "reference_unit"}
    rows: list[dict[str, object]] = []

    for col in features.columns:
        if col in ignored or not pd.api.types.is_numeric_dtype(features[col]):
            continue

        values = pd.to_numeric(features[col], errors="coerce")
        non_null = int(values.notna().sum())
        missing = int(values.isna().sum())
        std = float(values.std(ddof=0)) if non_null else np.nan

        rows.append(
            {
                "feature": col,
                "non_null_count": non_null,
                "missing_count": missing,
                "missing_ratio": float(missing / len(values)) if len(values) else 1.0,
                "mean": float(values.mean()) if non_null else np.nan,
                "std": std,
                "min": float(values.min()) if non_null else np.nan,
                "max": float(values.max()) if non_null else np.nan,
                "near_zero_variance": bool(std < 1e-12) if np.isfinite(std) else True,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["missing_ratio", "near_zero_variance", "feature"],
        kind="mergesort",
    )


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------


def plot_raw_signal_with_segments(
    profile: pd.DataFrame,
    target_segments: pd.DataFrame,
    target: str,
    sensor: str,
    output_path: Path,
) -> None:
    depth = pd.to_numeric(profile[DEPTH_COL], errors="coerce")
    values = pd.to_numeric(profile[sensor], errors="coerce")

    plt.figure(figsize=(6, 8))
    plt.plot(values, depth, linewidth=1.0)

    for seg in target_segments.itertuples(index=False):
        plt.axhline(float(seg.top), linestyle="--", linewidth=0.7, alpha=0.5)
        plt.axhline(float(seg.bottom), linestyle="--", linewidth=0.7, alpha=0.5)

    plt.gca().invert_yaxis()
    plt.xlabel(sensor)
    plt.ylabel("Depth [m]")
    plt.title(f"Raw CPT signal with segment boundaries\nTarget={target}, Sensor={sensor}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_spectrogram(spec: ProfileSpectrogram, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))

    power = np.log10(spec.power + 1e-12)

    plt.pcolormesh(
        spec.window_depths,
        spec.frequencies,
        power,
        shading="auto",
    )

    plt.xlabel("Depth [m]")
    plt.ylabel("Spatial frequency [cycles/m]")
    plt.title(f"STFT spectrogram\nTarget={spec.target}, Sensor={spec.sensor}")

    colorbar = plt.colorbar()
    colorbar.set_label("log10 spectral power")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_segment_frequency_matrix(
    segment_features: pd.DataFrame,
    sensor: str,
    frequencies: np.ndarray,
    output_path: Path,
) -> None:
    cols = [feature_name(sensor, frequency) for frequency in frequencies]
    cols = [col for col in cols if col in segment_features.columns]

    if not cols:
        return

    matrix = segment_features[cols].to_numpy(dtype=float)

    plt.figure(figsize=(9, max(4, 0.35 * len(segment_features))))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")

    plt.xlabel("Frequency [cycles/m]")
    plt.ylabel("Segment ID")
    plt.title(f"Fixed-size segment frequency matrix\nSensor={sensor}")

    x_positions = np.arange(len(frequencies))
    x_labels = [f"{frequency:.2f}" for frequency in frequencies]

    if len(x_positions) > 12:
        step = max(1, len(x_positions) // 10)
        x_positions = x_positions[::step]
        x_labels = x_labels[::step]

    plt.xticks(x_positions, x_labels, rotation=45, ha="right")
    plt.yticks(
        np.arange(len(segment_features)),
        segment_features["segment_id"].astype(str).tolist(),
    )

    colorbar = plt.colorbar()
    colorbar.set_label("Mean normalized spectral power")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_reference_unit_mean_spectra(
    features: pd.DataFrame,
    sensors: tuple[str, ...],
    frequencies: np.ndarray,
    output_dir: Path,
) -> None:
    if "reference_unit" not in features.columns:
        return

    work = features.dropna(subset=["reference_unit"]).copy()

    if work.empty:
        return

    for sensor in sensors:
        cols = [feature_name(sensor, frequency) for frequency in frequencies]
        cols = [col for col in cols if col in work.columns]

        if not cols:
            continue

        plt.figure(figsize=(8, 5))
        plotted_lines = 0

        for unit, group in work.groupby("reference_unit", sort=True):
            mean_values = group[cols].mean(axis=0, skipna=True).to_numpy(dtype=float)

            if np.isfinite(mean_values).sum() == 0:
                continue

            plt.plot(
                frequencies[: len(mean_values)],
                mean_values,
                label=str(unit),
                linewidth=1.5,
            )
            plotted_lines += 1

        if plotted_lines == 0:
            plt.close()
            continue

        plt.xlabel("Spatial frequency [cycles/m]")
        plt.ylabel("Mean normalized spectral power")
        plt.title(f"Mean STFT spectrum by reference unit\nSensor={sensor}")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"reference_unit_mean_spectra_{safe_name(sensor)}.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def detrend_linear(depth: np.ndarray, values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & np.isfinite(values)

    if int(valid.sum()) < 2:
        return values - np.nanmean(values)

    coefficients = np.polyfit(depth[valid], values[valid], deg=1)
    trend = np.polyval(coefficients, depth)

    return values - trend


def frequency_grid(config: StftConfig) -> np.ndarray:
    return np.round(
        np.arange(
            config.frequency_min_cpm,
            config.frequency_max_cpm + config.frequency_step_cpm / 2.0,
            config.frequency_step_cpm,
        ),
        8,
    )


def feature_name(sensor: str, frequency_cpm: float) -> str:
    label = f"{frequency_cpm:.2f}".replace(".", "_")
    return f"{sensor}_stft_pwr_{label}"


def safe_name(value: object) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def print_header(title: str) -> None:
    line = "=" * 72
    print(line)
    print(title)
    print(line)


def print_step(title: str) -> None:
    print(f"[{title}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CPT segment frequency representation using STFT/windowed FFT."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Shared data folder containing CPT_clean.csv and Input_Strata_merged_boundaries.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: segmentation_v2/outputs/frequency_analysis_stft.",
    )

    parser.add_argument(
        "--boundary-source",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "exported"],
    )

    parser.add_argument(
        "--boundary-file",
        type=str,
        default="exported_boundaries.csv",
        help="Used only when --boundary-source exported.",
    )

    parser.add_argument(
        "--sensors",
        nargs="*",
        default=None,
        help="Sensor columns. Default: SCPT_RES SCPT_FRES SCPT_PWP2.",
    )

    parser.add_argument("--min-segment-thickness-m", type=float, default=0.5)
    parser.add_argument("--depth-step-m", type=float, default=0.05)
    parser.add_argument("--window-length-m", type=float, default=1.0)
    parser.add_argument("--window-overlap-m", type=float, default=0.5)
    parser.add_argument("--frequency-min-cpm", type=float, default=0.25)
    parser.add_argument("--frequency-max-cpm", type=float, default=5.0)
    parser.add_argument("--frequency-step-cpm", type=float, default=0.25)
    parser.add_argument("--min-profile-points", type=int, default=32)
    parser.add_argument("--min-segment-windows", type=int, default=1)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--max-plotted-targets", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    main()