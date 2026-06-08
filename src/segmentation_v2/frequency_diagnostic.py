from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Config
from src.data import load_cpt


# ─────────────────────────────────────────────────────────────────────
# Frequency diagnostic config
# ─────────────────────────────────────────────────────────────────────

SIGNAL_COLUMNS = ("SCPT_RES", "SCPT_FRES", "SCPT_PWP2")

DZ = 0.02
CUTOFF_CYCLES_PER_M = 8.0
MAX_PLOT_FREQ_CYCLES_PER_M = 25.0

OUTPUT_ROOT = Path("segmentation_v2/outputs/frequency_diagnostics")


def regularize_profile(
    profile: pd.DataFrame,
    depth_col: str,
    signal_col: str,
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    work = profile[[depth_col, signal_col]].copy()
    work[depth_col] = pd.to_numeric(work[depth_col], errors="coerce")
    work[signal_col] = pd.to_numeric(work[signal_col], errors="coerce")
    work = work.dropna().sort_values(depth_col)

    if len(work) < 4:
        raise ValueError("Profile has too few valid points for FFT.")

    # Collapse duplicate depths.
    work = work.groupby(depth_col, as_index=False)[signal_col].mean()

    depth = work[depth_col].to_numpy(dtype=float)
    signal = work[signal_col].to_numpy(dtype=float)

    grid = np.arange(depth.min(), depth.max() + dz, dz)
    values = np.interp(grid, depth, signal)

    return grid, values


def fft_low_pass(
    signal: np.ndarray,
    dz: float,
    cutoff_cycles_per_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = signal - np.nanmean(signal)

    freq = np.fft.rfftfreq(len(centered), d=dz)
    spectrum = np.fft.rfft(centered)

    filtered_spectrum = spectrum.copy()
    filtered_spectrum[freq > cutoff_cycles_per_m] = 0.0

    filtered = np.fft.irfft(filtered_spectrum, n=len(centered)) + np.nanmean(signal)
    power = np.abs(spectrum) ** 2

    return freq, power, filtered


def plot_fft_diagnostic(
    depth: np.ndarray,
    raw: np.ndarray,
    filtered: np.ndarray,
    freq: np.ndarray,
    power: np.ndarray,
    cutoff: float,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(raw, depth, linewidth=0.8, label="Raw")
    axes[0].plot(filtered, depth, linewidth=1.2, label="Low-pass")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Signal value")
    axes[0].set_ylabel("Depth [m]")
    axes[0].set_title("Depth-domain signal")
    axes[0].legend()

    positive = freq > 0
    axes[1].plot(freq[positive], power[positive], linewidth=0.8)
    axes[1].axvline(cutoff, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Spatial frequency [cycles/m]")
    axes[1].set_ylabel("Power")
    axes[1].set_title("FFT power spectrum")
    axes[1].set_yscale("log")
    axes[1].set_xlim(left=0, right=MAX_PLOT_FREQ_CYCLES_PER_M)

    fig.suptitle(title)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--cpt-file", type=str, default="CPT_clean.csv")
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    config = Config(project_dir=args.project_dir.resolve())

    # Select CPT input file. It is resolved by Config as project_dir / data / cpt_file.
    config.cpt_file = args.cpt_file
    cpt = load_cpt(config)

    output_dir = OUTPUT_ROOT / args.run_name
    targets = sorted(cpt[config.target_col].astype(str).unique())

    saved_count = 0
    skipped_count = 0

    for target in targets:
        profile = cpt[cpt[config.target_col].astype(str) == target]

        for signal_col in SIGNAL_COLUMNS:
            if signal_col not in cpt.columns:
                skipped_count += 1
                continue

            try:
                depth, raw = regularize_profile(
                    profile=profile,
                    depth_col=config.depth_col,
                    signal_col=signal_col,
                    dz=DZ,
                )

                freq, power, filtered = fft_low_pass(
                    signal=raw,
                    dz=DZ,
                    cutoff_cycles_per_m=CUTOFF_CYCLES_PER_M,
                )

                output_path = (
                    output_dir
                    / safe_name(target)
                    / f"{safe_name(target)}_{signal_col}_frequency_diagnostic.png"
                )

                plot_fft_diagnostic(
                    depth=depth,
                    raw=raw,
                    filtered=filtered,
                    freq=freq,
                    power=power,
                    cutoff=CUTOFF_CYCLES_PER_M,
                    title=f"{target} | {signal_col} | cutoff={CUTOFF_CYCLES_PER_M} cycles/m",
                    output_path=output_path,
                )

                saved_count += 1

            except ValueError:
                skipped_count += 1
                continue

    print(f"CPT file: {config.cpt_path}")
    print(f"Targets processed: {len(targets)}")
    print(f"Diagnostics saved: {saved_count}")
    print(f"Diagnostics skipped: {skipped_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()