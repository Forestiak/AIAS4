from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FftConfig:
    cpt_path: Path
    boundaries_path: Path
    output_dir: Path = Path("fft_outputs")

    target_col: str = "Target"
    depth_col: str = "Depth"
    boundary_target_col: str = "Target"
    boundary_depth_col: str = "Boundary_depth"

    signal_cols: tuple[str, ...] = ("SCPT_RES", "SCPT_FRES", "SCPT_PWP2")

    min_segment_thickness_m: float = 0.5
    min_points_per_segment: int = 8

    # Uniform interpolation step before FFT.
    # Set to your CPT sampling interval if known.
    depth_step_m: float = 0.02

    # Thesis FFT band used for segment-level analysis.
    min_frequency_cycles_per_m: float = 0.25
    max_frequency_cycles_per_m: float = 8.0

    detrend: bool = True
    window: str = "hann"  # "hann" or "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(".."))
    parser.add_argument("--cpt-file", type=str, default="CPT_clean.csv")
    parser.add_argument("--boundary-file", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()

    config = FftConfig(
        cpt_path=project_dir / "data" / args.cpt_file,
        boundaries_path=project_dir / "data" / "boundaries" / args.boundary_file,
        output_dir=Path("fft_outputs") / args.run_name,
    )

    cpt = read_cpt(config)
    boundaries = read_boundaries(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for target, profile in cpt.groupby(config.target_col, sort=False):
        target = str(target)
        target_boundaries = boundaries.get(target, [])

        segments = build_segments_for_profile(profile, target_boundaries, config)

        for segment_id, top, bottom in segments:
            segment = profile[
                (profile[config.depth_col] >= top)
                & (profile[config.depth_col] <= bottom)
            ].copy()

            if len(segment) < config.min_points_per_segment:
                continue

            for signal_col in config.signal_cols:
                if signal_col not in segment.columns:
                    continue

                result = compute_segment_fft(segment, signal_col, config)
                if result is None:
                    continue

                freq, amp = result

                band_mask = (
                    (freq >= config.min_frequency_cycles_per_m)
                    & (freq <= config.max_frequency_cycles_per_m)
                )

                if not band_mask.any():
                    continue

                freq_band = freq[band_mask]
                amp_band = amp[band_mask]

                peak_idx = int(np.argmax(amp_band)) if len(amp_band) else 0
                peak_freq = float(freq_band[peak_idx]) if len(freq_band) else np.nan
                peak_amp = float(amp_band[peak_idx]) if len(amp_band) else np.nan

                summary_rows.append(
                    {
                        "target": target,
                        "segment_id": segment_id,
                        "top": top,
                        "bottom": bottom,
                        "thickness": bottom - top,
                        "signal": signal_col,
                        "n_points": len(segment),
                        "frequency_min_cycles_per_m": config.min_frequency_cycles_per_m,
                        "frequency_max_cycles_per_m": config.max_frequency_cycles_per_m,
                        "peak_frequency_cycles_per_m": peak_freq,
                        "peak_wavelength_m": 1.0 / peak_freq if peak_freq > 0 else np.nan,
                        "peak_amplitude": peak_amp,
                    }
                )

                save_fft_plot(
                    target=target,
                    segment_id=segment_id,
                    top=top,
                    bottom=bottom,
                    signal_col=signal_col,
                    freq=freq_band,
                    amp=amp_band,
                    output_dir=config.output_dir,
                    min_frequency=config.min_frequency_cycles_per_m,
                    max_frequency=config.max_frequency_cycles_per_m,
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.output_dir / "fft_segment_summary.csv", index=False)

    print(f"CPT file: {config.cpt_path}")
    print(f"Boundary file: {config.boundaries_path}")
    print(f"Saved plots and summary to: {config.output_dir.resolve()}")

    if summary.empty:
        print("Segments analyzed: 0")
        print("FFT rows: 0")
    else:
        print(f"Segments analyzed: {summary[['target', 'segment_id']].drop_duplicates().shape[0]}")
        print(f"FFT rows: {len(summary)}")


def read_cpt(config: FftConfig) -> pd.DataFrame:
    df = pd.read_csv(config.cpt_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required = [config.target_col, config.depth_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CPT file missing required columns: {missing}")

    df[config.depth_col] = pd.to_numeric(df[config.depth_col], errors="coerce")

    for col in config.signal_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=[config.target_col, config.depth_col])
        .sort_values([config.target_col, config.depth_col])
        .reset_index(drop=True)
    )


def read_boundaries(config: FftConfig) -> dict[str, list[float]]:
    df = pd.read_csv(config.boundaries_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required = [config.boundary_target_col, config.boundary_depth_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Boundary file missing required columns: {missing}")

    df[config.boundary_depth_col] = pd.to_numeric(
        df[config.boundary_depth_col],
        errors="coerce",
    )
    df = df.dropna(subset=[config.boundary_target_col, config.boundary_depth_col])

    return {
        str(target): sorted(group[config.boundary_depth_col].astype(float).unique())
        for target, group in df.groupby(config.boundary_target_col, sort=False)
    }


def build_segments_for_profile(
    profile: pd.DataFrame,
    boundary_depths: list[float],
    config: FftConfig,
) -> list[tuple[int, float, float]]:
    depth = profile[config.depth_col].astype(float)
    top = float(depth.min())
    bottom = float(depth.max())

    edges = sorted(
        {top, bottom}
        | {float(b) for b in boundary_depths if top < float(b) < bottom}
    )

    raw_segments = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    merged: list[tuple[float, float]] = []
    for seg_top, seg_bottom in raw_segments:
        if not merged:
            merged.append((seg_top, seg_bottom))
            continue

        if seg_bottom - seg_top < config.min_segment_thickness_m:
            prev_top, _ = merged[-1]
            merged[-1] = (prev_top, seg_bottom)
        else:
            merged.append((seg_top, seg_bottom))

    return [(i, seg_top, seg_bottom) for i, (seg_top, seg_bottom) in enumerate(merged)]


def compute_segment_fft(
    segment: pd.DataFrame,
    signal_col: str,
    config: FftConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    work = segment[[config.depth_col, signal_col]].dropna().copy()
    if len(work) < config.min_points_per_segment:
        return None

    work = work.groupby(config.depth_col, as_index=False)[signal_col].mean()
    work = work.sort_values(config.depth_col)

    depth = work[config.depth_col].to_numpy(dtype=float)
    values = work[signal_col].to_numpy(dtype=float)

    if len(depth) < config.min_points_per_segment:
        return None

    start = float(depth.min())
    stop = float(depth.max())
    if stop <= start:
        return None

    uniform_depth = np.arange(start, stop + config.depth_step_m, config.depth_step_m)
    if len(uniform_depth) < config.min_points_per_segment:
        return None

    y = np.interp(uniform_depth, depth, values)

    if config.detrend:
        x = uniform_depth - uniform_depth.mean()
        slope, intercept = np.polyfit(x, y, deg=1)
        y = y - (slope * x + intercept)

    y = y - np.mean(y)

    if config.window == "hann":
        y = y * np.hanning(len(y))
    elif config.window != "none":
        raise ValueError(f"Unsupported window: {config.window}")

    fft = np.fft.rfft(y)
    freq = np.fft.rfftfreq(len(y), d=config.depth_step_m)

    # Amplitude spectrum. Scaling keeps amplitudes comparable across segment lengths.
    amp = (2.0 / len(y)) * np.abs(fft)

    return freq, amp


def save_fft_plot(
    target: str,
    segment_id: int,
    top: float,
    bottom: float,
    signal_col: str,
    freq: np.ndarray,
    amp: np.ndarray,
    output_dir: Path,
    min_frequency: float,
    max_frequency: float,
) -> None:
    safe_target = target.replace("/", "_").replace("\\", "_")
    target_dir = output_dir / safe_target
    target_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(freq, amp, linewidth=1.2)

    ax.set_title(
        f"{target} | segment {segment_id} | {top:.2f}-{bottom:.2f} m | {signal_col}"
    )
    ax.set_xlabel("Spatial frequency [cycles/m]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=min_frequency, right=max_frequency)

    fig.tight_layout()

    filename = (
        f"{safe_target}_seg-{segment_id:03d}_"
        f"{top:.2f}-{bottom:.2f}m_{signal_col}.png"
    )
    fig.savefig(target_dir / filename, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()