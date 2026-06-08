"""
Run tracking.
Each pipeline run is saved to a timestamped folder inside logs/.
Tracks previous and best runs for comparison.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


LOGS_DIR_NAME = "logs"
BEST_RUN_FILE = "best_run.json"
LATEST_RUN_FILE = "latest_run.json"


def create_run_dir(output_dir: Path, model_id: str = "default") -> Path:
    """Create a timestamped run folder like logs/<model_id>/2026-04-12_14-30-00/."""
    logs_dir = output_dir / LOGS_DIR_NAME / model_id
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = logs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run(run_dir: Path, summary: dict, config_snapshot: dict, report: str) -> None:
    """Save all run artifacts to the run folder."""
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2))
    if report:
        (run_dir / "classification_report.txt").write_text(report)


def load_previous_run(output_dir: Path, model_id: str = "default") -> dict | None:
    """Load the latest_run.json pointer to get last run's summary."""
    path = output_dir / LOGS_DIR_NAME / model_id / LATEST_RUN_FILE
    if not path.exists():
        return None
    return _load_json(path)


def load_best_run(output_dir: Path, model_id: str = "default") -> dict | None:
    path = output_dir / LOGS_DIR_NAME / model_id / BEST_RUN_FILE
    if not path.exists():
        return None
    return _load_json(path)


def update_pointers(output_dir: Path, run_dir: Path, summary: dict, model_id: str = "default") -> None:
    """Update latest_run.json and best_run.json if this run is better."""
    logs_dir = output_dir / LOGS_DIR_NAME / model_id
    pointer = {"run_dir": run_dir.name, "metrics": summary.get("metrics", {})}

    # Always update latest
    (logs_dir / LATEST_RUN_FILE).write_text(json.dumps(pointer, indent=2))

    # Update best if this run has higher NMI (primary ranking metric)
    best = load_best_run(output_dir, model_id)
    current_nmi = summary.get("metrics", {}).get("nmi", 0.0)
    best_nmi = best.get("metrics", {}).get("nmi", 0.0) if best else -1.0
    if current_nmi >= best_nmi:
        (logs_dir / BEST_RUN_FILE).write_text(json.dumps(pointer, indent=2))


def compare_runs(current: dict, previous: dict | None, best: dict | None) -> str:
    """Build a short comparison table between current, previous, and best."""
    current_m = current.get("metrics", {})
    prev_m = previous.get("metrics", {}) if previous else {}
    best_m = best.get("metrics", {}) if best else {}

    lines = []
    lines.append("")
    lines.append(f"{'Metric':<20} {'Current':>10} {'vs Prev':>10} {'vs Best':>10}")
    lines.append("-" * 52)

    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "ari", "nmi"):
        cur = current_m.get(key)
        if cur is None:
            continue
        prev = prev_m.get(key)
        best_val = best_m.get(key)
        delta_prev = _delta_str(cur, prev)
        delta_best = _delta_str(cur, best_val)
        lines.append(f"{key:<20} {cur:>10.4f} {delta_prev:>10} {delta_best:>10}")

    lines.append("")
    return "\n".join(lines)


def config_to_snapshot(config) -> dict:
    """Serialize the config dataclass to a plain dict for logging."""
    return {
        "model_id": config.model_id,
        "model_type": config.model_type,
        "feature_set_id": config.feature_set_id,
        "boundary_source": config.boundary_source,
        "boundary_file": config.boundary_file or "(default)",
        "feature_columns": list(config.feature_columns),
        "segment_stats": list(config.segment_stats),
        "include_thickness": config.include_thickness,
        "extra_derived": list(config.extra_derived),
        "representation_type": config.representation_type,
        "representation_length": config.representation_length,
        "representation_bins": config.representation_bins,
        "include_derivatives": config.include_derivatives,
        "cluster_selection": config.cluster_selection,
        "n_clusters": config.n_clusters,
        "min_clusters": config.min_clusters,
        "max_clusters": config.max_clusters,
        "min_segment_thickness_m": config.min_segment_thickness_m,
        "random_state": config.random_state,
        "covariance_type": config.covariance_type,
        "pca_n_components": config.pca_n_components,
        "reg_covar": config.reg_covar,
    }


def _delta_str(current: float, reference: float | None) -> str:
    if reference is None:
        return "—"
    diff = current - reference
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.4f}"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())
