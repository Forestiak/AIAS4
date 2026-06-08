"""
Backend helpers that read/write to the core project's files.
No imports from segmentation_v2 - only JSON/CSV file operations.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from . import constants as C


# Model definitions

def normalize_model_folder(folder: str) -> str:
    """Normalize a folder path relative to models/ and reject unsafe paths."""
    raw = folder.replace("\\", "/").strip().strip("/")
    if not raw or raw == "(root)":
        return ""

    parts = [part.strip() for part in raw.split("/") if part.strip()]
    if not parts:
        return ""
    if any(part in {".", ".."} for part in parts):
        raise ValueError("Folder paths cannot contain '.' or '..'.")

    return "/".join(parts)


def _iter_model_files() -> Iterable[Path]:
    if not C.MODELS_DIR.exists():
        return []
    return sorted(
        C.MODELS_DIR.rglob("*.json"),
        key=lambda path: path.relative_to(C.MODELS_DIR).as_posix().lower(),
    )


def _model_folder_from_path(path: Path) -> str:
    parent = path.relative_to(C.MODELS_DIR).parent.as_posix()
    return "" if parent == "." else parent


def _attach_model_metadata(definition: dict, path: Path) -> dict:
    model = dict(definition)
    model["_path"] = str(path)
    model["_relative_path"] = path.relative_to(C.MODELS_DIR).as_posix()
    model["_folder"] = _model_folder_from_path(path)
    feature_set_id = model.get("feature_set_id")
    if feature_set_id:
        try:
            model.update(_prefixed_feature_set_metadata(load_feature_set(feature_set_id), prefix="_"))
        except Exception:
            pass
    return model


def _find_model_paths(model_id: str) -> list[Path]:
    if not C.MODELS_DIR.exists():
        return []
    return sorted(C.MODELS_DIR.rglob(f"{model_id}.json"))


def list_models() -> list[dict]:
    models = []

    for path in _iter_model_files():
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                print(f"[WARN] Skipping empty model JSON: {path}")
                continue

            models.append(_attach_model_metadata(json.loads(raw), path))

        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping invalid model JSON: {path} ({exc})")
            continue

    return models


def load_model(model_id: str) -> dict:
    matches = _find_model_paths(model_id)
    if not matches:
        raise FileNotFoundError(f"Model '{model_id}' not found under {C.MODELS_DIR}")
    if len(matches) > 1:
        joined = ", ".join(path.relative_to(C.MODELS_DIR).as_posix() for path in matches)
        raise ValueError(f"Multiple model files found for '{model_id}': {joined}")
    path = matches[0]
    return _attach_model_metadata(json.loads(path.read_text()), path)


def list_model_folders() -> list[str]:
    if not C.MODELS_DIR.exists():
        return []
    folders = {
        path.relative_to(C.MODELS_DIR).as_posix()
        for path in C.MODELS_DIR.rglob("*")
        if path.is_dir()
    }
    return sorted(folder for folder in folders if folder != ".")


def create_model_folder(folder: str) -> Path:
    normalized = normalize_model_folder(folder)
    if not normalized:
        raise ValueError("Folder name cannot be empty.")
    path = C.MODELS_DIR / normalized
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(definition: dict, folder: str = "") -> Path:
    normalized_folder = normalize_model_folder(folder)
    model_id = definition["model_id"]

    target_dir = C.MODELS_DIR / normalized_folder if normalized_folder else C.MODELS_DIR
    target_path = target_dir / f"{model_id}.json"

    existing_paths = _find_model_paths(model_id)
    if existing_paths and target_path not in existing_paths:
        existing = existing_paths[0].relative_to(C.MODELS_DIR).as_posix()
        raise FileExistsError(f"Model ID '{model_id}' already exists at {existing}")

    target_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(definition)
    if normalized_folder:
        payload["model_folder"] = normalized_folder
    else:
        payload.pop("model_folder", None)
    target_path.write_text(json.dumps(payload, indent=2))
    return target_path


def delete_model(model_id: str) -> None:
    for path in _find_model_paths(model_id):
        if path.exists():
            path.unlink()


# Feature-set definitions

def list_feature_sets() -> list[dict]:
    sets: list[dict] = []
    if C.FEATURE_SETS_DIR.exists():
        for f in sorted(C.FEATURE_SETS_DIR.glob("*.json")):
            definition = json.loads(f.read_text())
            definition.update(_prefixed_feature_set_metadata(definition, prefix="_"))
            sets.append(definition)
    return sets


def load_feature_set(feature_set_id: str) -> dict:
    path = C.FEATURE_SETS_DIR / f"{feature_set_id}.json"
    definition = json.loads(path.read_text())
    definition.update(_prefixed_feature_set_metadata(definition, prefix="_"))
    return definition


def save_feature_set(definition: dict) -> Path:
    C.FEATURE_SETS_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FEATURE_SETS_DIR / f"{definition['feature_set_id']}.json"
    path.write_text(json.dumps(definition, indent=2))
    return path


def delete_feature_set(feature_set_id: str) -> None:
    path = C.FEATURE_SETS_DIR / f"{feature_set_id}.json"
    if path.exists():
        path.unlink()


# Runs

def list_runs(model_id: str) -> list[dict]:
    """Return run summaries for a model, newest first."""
    model_logs = C.LOGS_DIR / model_id
    if not model_logs.exists():
        return []
    runs = []
    for run_dir in sorted(model_logs.iterdir(), reverse=True):
        summary_path = run_dir / "summary.json"
        if run_dir.is_dir() and summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["_run_dir"] = str(run_dir)
            summary["_timestamp"] = run_dir.name
            runs.append(summary)
    return runs


def load_run(run_dir: str) -> dict:
    """Load full run data from a run directory path."""
    rd = Path(run_dir)
    data: dict = {}
    summary_path = rd / "summary.json"
    if summary_path.exists():
        data.update(json.loads(summary_path.read_text()))
    report_path = rd / "classification_report.txt"
    if report_path.exists():
        data["report"] = report_path.read_text()
    data["_run_dir"] = str(rd)
    data["_plots"] = sorted(str(p) for p in (rd / "plots").glob("*.png")) if (rd / "plots").exists() else []
    return data


def best_run(model_id: str) -> dict | None:
    runs = list_runs(model_id)
    if not runs:
        return None
    return max(runs, key=lambda run: run.get("metrics", {}).get("nmi", -1.0))


def list_all_runs() -> list[dict]:
    """Return all run summaries across all models, newest first."""
    if not C.LOGS_DIR.exists():
        return []

    runs: list[dict] = []
    for model_logs in sorted(path for path in C.LOGS_DIR.iterdir() if path.is_dir()):
        model_id = model_logs.name
        for run in list_runs(model_id):
            runs.append(run)

    return sorted(runs, key=lambda run: run.get("_timestamp", ""), reverse=True)


# Boundary files

def list_boundary_files() -> list[str]:
    """Return CSV filenames from data/boundaries/."""
    boundaries_dir = C.CORE_DIR.parent / "data" / "boundaries"
    if not boundaries_dir.exists():
        return []
    return sorted(path.name for path in boundaries_dir.glob("*.csv"))


# Pipeline execution

def run_pipeline(model_id: str, boundary_file: str = "") -> subprocess.Popen:
    """Launch the pipeline as a subprocess. Returns the Popen handle."""
    cmd = [sys.executable, str(C.PIPELINE_SCRIPT), "--model", model_id]
    if boundary_file:
        cmd += ["--boundary-file", boundary_file]
    return subprocess.Popen(
        cmd,
        cwd=str(C.CORE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def summarize_feature_set(definition: dict) -> dict:
    representation = definition.get("representation") or {}
    representation_type = representation.get("type", "summary")
    representation_length = representation.get("length")
    representation_bins = representation.get("bins")
    include_derivatives = bool(representation.get("include_derivatives", False))
    segment_stats = list(definition.get("segment_stats", []))
    selected_columns_count = len(definition.get("columns", []))

    if representation_type == "summary":
        size_label = "-"
        config_label = f"stats: {', '.join(segment_stats)}" if segment_stats else "stats: -"
        comparison_label = "summary"
    elif representation_type == "resample":
        size_label = str(representation_length or "-")
        config_label = f"length={representation_length or '-'}"
        comparison_label = f"resample (length={representation_length or '-'})"
    elif representation_type == "fft":
        min_freq = representation.get("min_frequency_cycles_per_m", 0.25)
        max_freq = representation.get("max_frequency_cycles_per_m", 8.0)
        bands = representation.get("bands_cycles_per_m", [])
        size_label = str(len(bands)) if isinstance(bands, list) else "-"
        config_label = f"{min_freq}-{max_freq} cyc/m"
        comparison_label = f"fft ({min_freq}-{max_freq} cyc/m)"
    elif representation_type == "row_local":
        size_label = "-"
        config_label = "row-level local features"
        comparison_label = "row_local"
    else:
        size_label = str(representation_bins or "-")
        config_label = f"bins={representation_bins or '-'}"
        comparison_label = f"paa (bins={representation_bins or '-'})"

    return {
        "representation_type": representation_type,
        "representation_length": representation_length,
        "representation_bins": representation_bins,
        "include_derivatives": include_derivatives,
        "segment_stats": segment_stats,
        "selected_columns_count": selected_columns_count,
        "representation_size_label": size_label,
        "representation_config_label": config_label,
        "comparison_representation_label": comparison_label,
    }


def _prefixed_feature_set_metadata(definition: dict, prefix: str = "") -> dict:
    return {
        f"{prefix}{key}": value
        for key, value in summarize_feature_set(definition).items()
    }
