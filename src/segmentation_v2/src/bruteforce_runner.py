"""
Shared runner for brute-force CPT feature-representation suites.
"""
from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path

import pandas as pd

from .config import Config
from .data import boundaries_from_exported, boundaries_from_strata, load_cpt, load_strata, split_by_target
from .feature_set_registry import FEATURE_SETS_DIR
from .features import build_features, feature_matrix_columns
from .model_registry import MODELS_DIR, load_model
from .segments import build_segments


RESULT_COLUMNS = [
    "model_id",
    "feature_set_id",
    "representation_type",
    "columns_family",
    "n_input_columns",
    "bins_or_length",
    "include_derivatives",
    "pca_n_components",
    "cluster_selection",
    "n_clusters_requested",
    "n_clusters_selected",
    "covariance_type",
    "bic",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "ari",
    "nmi",
    "run_dir",
    "status",
    "error_message",
    "nan_percentage",
    "constant_column_count",
    "generated_feature_count",
]


def main(suite_dir: Path, expected_prefix: str) -> int:
    parser = argparse.ArgumentParser(description=f"Run brute-force suite from {suite_dir.name}")
    parser.add_argument("--dry-run", action="store_true", help="Print the models that would run without executing them.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of models to run.")
    parser.add_argument("--model-id", default=None, help="Run a single model by ID.")
    args = parser.parse_args()

    project_dir = suite_dir.parent.parent
    segmentation_dir = suite_dir.parent
    pipeline_module = _load_pipeline_module(segmentation_dir)

    manifest = load_suite_manifest(suite_dir, expected_prefix)
    model_entries = select_entries(manifest, args.model_id, args.limit)

    print(f"Suite: {suite_dir.name}")
    print(f"Selected {len(model_entries)} model(s).")

    if args.dry_run:
        for entry in model_entries:
            print(
                f"  {entry['model_id']} | family={entry['columns_family']} | "
                f"{entry['representation_type']}={entry['bins_or_length']} | "
                f"derivatives={entry['include_derivatives']} | pca={entry['pca_n_components']}"
            )
        return 0

    sync_counts = sync_suite_registry(suite_dir, expected_prefix)
    print(f"Refreshed {sync_counts['feature_sets']} feature sets and {sync_counts['models']} models in the main registries.")

    raw_cpt_columns = _read_csv_columns(project_dir / "data" / "CPT_clean.csv")
    results: list[dict] = []
    errors: list[dict] = []

    for entry in model_entries:
        model_id = entry["model_id"]
        print(f"\n[{model_id}] starting")
        config: Config | None = None
        try:
            definition = load_model(model_id)
            config = Config.from_model_definition(project_dir, definition)
            validation = validate_model(config, raw_cpt_columns)
            print(
                f"[{model_id}] features={validation['generated_feature_count']} "
                f"constants={validation['constant_column_count']} "
                f"nan%={validation['nan_percentage']:.2f}"
            )

            run_result = pipeline_module.run(config)
            metrics = run_result.get("metrics", {})
            row = build_result_row(
                entry=entry,
                config=config,
                validation=validation,
                status="success",
                error_message="",
                run_dir=run_result.get("run_dir", ""),
                bic=run_result.get("bic"),
                n_clusters_selected=run_result.get("n_clusters"),
                metrics=metrics,
            )
            results.append(row)
        except Exception as exc:
            message = str(exc)
            traceback.print_exc()
            if config is None:
                config = Config(
                    project_dir=project_dir,
                    model_id=entry["model_id"],
                    feature_set_id=entry["feature_set_id"],
                    model_type="gmm",
                    representation_type=entry["representation_type"],
                    include_derivatives=bool(entry["include_derivatives"]),
                    representation_bins=entry["bins_or_length"] if entry["representation_type"] == "paa" else None,
                    representation_length=entry["bins_or_length"] if entry["representation_type"] == "resample" else None,
                    pca_n_components=entry["pca_n_components"],
                    cluster_selection=entry["cluster_selection"],
                    n_clusters=entry["n_clusters_requested"],
                    covariance_type=entry["covariance_type"],
                )
            row = build_result_row(
                entry=entry,
                config=config,
                validation=None,
                status="failed",
                error_message=message,
                run_dir="",
                bic=None,
                n_clusters_selected=None,
                metrics={},
            )
            results.append(row)
            errors.append(row)
            print(f"[{model_id}] failed: {message}")

    write_results(suite_dir, results, errors)
    print(f"\nFinished {len(model_entries)} model(s). Results: {suite_dir / 'results.csv'}")
    if errors:
        print(f"Errors: {suite_dir / 'errors.csv'}")
    return 0


def sync_suite_registry(suite_dir: Path, expected_prefix: str) -> dict[str, int]:
    feature_count = _sync_json_dir(
        source_dir=suite_dir / "feature_sets",
        target_dir=FEATURE_SETS_DIR,
        id_field="feature_set_id",
        expected_prefix=expected_prefix,
    )
    model_count = _sync_json_dir(
        source_dir=suite_dir / "models",
        target_dir=MODELS_DIR,
        id_field="model_id",
        expected_prefix=expected_prefix,
    )
    return {"feature_sets": feature_count, "models": model_count}


def load_suite_manifest(suite_dir: Path, expected_prefix: str) -> list[dict]:
    manifest_path = suite_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("models", [])
    for entry in entries:
        model_id = entry["model_id"]
        feature_set_id = entry["feature_set_id"]
        if not model_id.startswith(expected_prefix):
            raise ValueError(f"Unexpected model_id in manifest: {model_id}")
        if not feature_set_id.startswith(expected_prefix):
            raise ValueError(f"Unexpected feature_set_id in manifest: {feature_set_id}")
    return entries


def select_entries(entries: list[dict], model_id: str | None, limit: int | None) -> list[dict]:
    selected = entries
    if model_id:
        selected = [entry for entry in selected if entry["model_id"] == model_id]
        if not selected:
            raise ValueError(f"Model ID not found in suite: {model_id}")
    if limit is not None:
        selected = selected[:limit]
    return selected


def validate_model(config: Config, raw_cpt_columns: set[str]) -> dict[str, float | int]:
    missing_columns = [column for column in config.feature_columns if column not in raw_cpt_columns]
    if missing_columns:
        raise ValueError(f"Missing CPT columns: {', '.join(missing_columns)}")

    cpt = load_cpt(config)
    strata = load_strata(config)
    profiles = split_by_target(cpt, config.target_col)

    if config.boundary_source == "ground_truth":
        bound_map = boundaries_from_strata(strata, config)
    elif config.boundary_source == "bocpd_estimate":
        bound_map = boundaries_from_exported(config)
    else:
        raise ValueError(f"Unknown boundary_source: {config.boundary_source}")

    _, seg_rows = build_segments(profiles, bound_map, config)
    features = build_features(seg_rows, config)
    feature_cols = feature_matrix_columns(config)
    X = features[feature_cols]

    if len(X) <= 1:
        raise ValueError("Feature matrix must contain more than one row.")

    generated_feature_count = len(feature_cols)
    constant_column_count = sum(_is_constant(X[column]) for column in X.columns)
    if constant_column_count >= len(feature_cols):
        raise ValueError("Feature matrix contains no non-constant numeric features.")

    total_values = X.shape[0] * X.shape[1]
    nan_percentage = 100.0 * float(X.isna().sum().sum()) / total_values if total_values else 100.0
    return {
        "generated_feature_count": generated_feature_count,
        "constant_column_count": constant_column_count,
        "nan_percentage": nan_percentage,
    }


def build_result_row(
    entry: dict,
    config: Config,
    validation: dict | None,
    status: str,
    error_message: str,
    run_dir: str,
    bic: float | None,
    n_clusters_selected: int | None,
    metrics: dict,
) -> dict:
    return {
        "model_id": config.model_id,
        "feature_set_id": config.feature_set_id,
        "representation_type": config.representation_type,
        "columns_family": entry["columns_family"],
        "n_input_columns": len(config.feature_columns),
        "bins_or_length": config.representation_size,
        "include_derivatives": config.include_derivatives,
        "pca_n_components": config.pca_n_components,
        "cluster_selection": config.cluster_selection,
        "n_clusters_requested": config.n_clusters,
        "n_clusters_selected": n_clusters_selected,
        "covariance_type": config.covariance_type,
        "bic": bic,
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
        "ari": metrics.get("ari"),
        "nmi": metrics.get("nmi"),
        "run_dir": run_dir,
        "status": status,
        "error_message": error_message,
        "nan_percentage": validation.get("nan_percentage") if validation else None,
        "constant_column_count": validation.get("constant_column_count") if validation else None,
        "generated_feature_count": validation.get("generated_feature_count") if validation else None,
    }


def write_results(suite_dir: Path, results: list[dict], errors: list[dict]) -> None:
    results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    results_df.to_csv(suite_dir / "results.csv", index=False)
    (suite_dir / "results.json").write_text(json.dumps(results, indent=2))

    errors_path = suite_dir / "errors.csv"
    if errors:
        pd.DataFrame(errors, columns=RESULT_COLUMNS).to_csv(errors_path, index=False)
    elif errors_path.exists():
        errors_path.unlink()


def _sync_json_dir(source_dir: Path, target_dir: Path, id_field: str, expected_prefix: str) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        item_id = payload[id_field]
        if not item_id.startswith(expected_prefix):
            raise ValueError(f"Unexpected ID in {path.name}: {item_id}")
        target_path = target_dir / f"{item_id}.json"
        shutil.copyfile(path, target_path)
        count += 1
    return count


def _read_csv_columns(path: Path) -> set[str]:
    header = pd.read_csv(path, nrows=0)
    return {str(column).strip() for column in header.columns}


def _is_constant(series: pd.Series) -> bool:
    return series.nunique(dropna=False) <= 1


def _load_pipeline_module(segmentation_dir: Path):
    import importlib.util

    module_path = segmentation_dir / "pipeline.py"
    spec = importlib.util.spec_from_file_location("segmentation_v2_pipeline_runtime", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
