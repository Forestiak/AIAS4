"""
Unsupervised CPT Segmentation Pipeline (v2)
============================================
Takes raw CPT measurements and the strata interpretation file.
Boundary depths are extracted from the strata Top/Bottom columns or from
external BOCPD/exported boundary files.

Segments are clustered using the selected model definition from models/.
Strata unit labels are used only after clustering for evaluation/naming.

Usage:
    pixi run run
    pixi run run -- --model gmm_default
    pixi run run -- --model segment_hybrid_gmm_24 --boundary-source bocpd_estimate
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config
from src.data import (
    boundaries_from_exported,
    boundaries_from_perfect_recall,
    boundaries_from_strata,
    load_cpt,
    load_strata,
    save_csv,
    split_by_target,
)
from src.evaluate import (
    apply_cluster_mapping,
    attach_reference_units,
    attach_reference_units_to_measurements,
    build_cluster_mapping,
    compute_metrics,
    per_class_report,
)
from src.features import build_features, feature_matrix_columns
from src.model import fit, predict
from src.model_registry import load_model
from src.plots import generate_profile_plots, generate_robertson_charts
from src.segments import build_segments
from src.tracker import (
    compare_runs,
    config_to_snapshot,
    create_run_dir,
    load_best_run,
    load_previous_run,
    save_run,
    update_pointers,
)


BOUNDARY_SOURCES = ["ground_truth", "bocpd_estimate", "perfect_recall"]

DATASET_PROFILES: dict[str, dict[str, str]] = {
    "default": {
        "cpt_file": "CPT_clean.csv",
        "strata_file": "Input_Strata_merged_boundaries.csv",
        "location_file": "Input_Location_clean.csv",
        "perfect_recall_file": "perfect_recall.csv",
    },
    "new_location": {
        "cpt_file": "2_CPT_clean.csv",
        "strata_file": "2_Input_Strata_only_loc_targets.csv",
        "location_file": "2_Input_Location.csv",
        "perfect_recall_file": "new_dataset_51_profiles_high_recall.csv",
    },
}


def _apply_dataset_profile(config: Config, profile: str) -> None:
    values = DATASET_PROFILES.get(profile)
    if values is None:
        raise ValueError(
            f"Unknown dataset_profile: {profile}. "
            f"Expected one of: {', '.join(DATASET_PROFILES)}"
        )

    config.data_profile = profile
    config.cpt_file = values["cpt_file"]
    config.strata_file = values["strata_file"]
    config.location_file = values["location_file"]
    config.perfect_recall_file = values["perfect_recall_file"]


def run(config: Config) -> dict:
    # 0. Load inputs
    cpt = load_cpt(config)
    strata = load_strata(config)

    profiles = split_by_target(cpt, config.target_col)

    if config.boundary_source == "ground_truth":
        bound_map = boundaries_from_strata(strata, config)
    elif config.boundary_source == "bocpd_estimate":
        bound_map = boundaries_from_exported(config)
    elif config.boundary_source == "perfect_recall":
        bound_map = boundaries_from_perfect_recall(config)
    else:
        raise ValueError(f"Unknown boundary_source: {config.boundary_source}")

    # 1. Build boundary-defined segments
    seg_table, seg_rows = build_segments(profiles, bound_map, config)

    if seg_table.empty or seg_rows.empty:
        raise ValueError(
            f"No segments were built for boundary_source={config.boundary_source}, "
            f"boundary_file={config.boundary_file or '(default)'}."
        )

    # 2. Compute per-segment features
    features = build_features(seg_rows, config)
    feature_cols = feature_matrix_columns(config)

    missing_cols = [col for col in feature_cols if col not in features.columns]
    if missing_cols:
        raise ValueError(
            "Feature matrix column mismatch. "
            f"Missing columns for representation '{config.representation_type}': {missing_cols[:20]}"
            + (" ..." if len(missing_cols) > 20 else "")
        )

    X = features[feature_cols]

    # 3. Fit unsupervised model — no labels used
    result = fit(X, config)
    features["cluster_id"] = predict(result, X)

    # 3b. Optional segment-level spatial MRF refinement
    if config.use_spatial_mrf:
        from src.spatial import refine_labels_mrf

        features["cluster_id"] = refine_labels_mrf(features, result, X, config)

    # 4. Evaluate against strata only after clustering
    evaluated_segments = attach_reference_units(features, strata, config)

    measurement_rows = seg_rows.merge(
        features[["target", "segment_id", "cluster_id"]],
        on=["target", "segment_id"],
        how="left",
        validate="many_to_one",
    )

    evaluated_measurements = attach_reference_units_to_measurements(
        measurement_rows,
        strata,
        config,
    )

    # Cluster naming is based on measurement-level majority reference unit.
    cluster_map = build_cluster_mapping(evaluated_measurements)

    mapped = apply_cluster_mapping(evaluated_segments, cluster_map)
    mapped_measurements = apply_cluster_mapping(evaluated_measurements, cluster_map)

    # Primary metrics are measurement-weighted.
    metrics = compute_metrics(mapped_measurements)
    report = per_class_report(mapped_measurements)

    summary = {
        "model_id": config.model_id,
        "model_type": config.model_type,
        "pipeline_type": config.pipeline_type,
        "feature_set_id": config.feature_set_id,
        "representation_type": config.representation_type,
        "representation_length": config.representation_length,
        "representation_bins": config.representation_bins,
        "include_derivatives": config.include_derivatives,
        "feature_columns_count": len(config.feature_columns),
        "feature_matrix_columns_count": len(feature_cols),
        "cluster_selection": config.cluster_selection,
        "n_clusters_requested": config.n_clusters,
        "pca_n_components": config.pca_n_components,
        "reg_covar": config.reg_covar,
        "covariance_type": config.covariance_type,
        "use_spatial_mrf": config.use_spatial_mrf,
        "boundary_source": config.boundary_source,
        "boundary_file": config.boundary_file or "(default)",
        "data_profile": config.data_profile,
        "cpt_file": config.cpt_file,
        "strata_file": config.strata_file,
        "location_file": config.location_file,
        "perfect_recall_file": config.perfect_recall_file,
        "n_segments": int(len(features)),
        "n_measurements": int(len(mapped_measurements)),
        "n_clusters": result.n_clusters,
        "bic": result.bic,
        "cluster_map": {str(k): v for k, v in cluster_map.items()},
        "evaluation_level": "measurement",
        "evaluation_rows": int(len(mapped_measurements)),
        "metrics": metrics,
    }

    # 5. Save artifacts
    previous = load_previous_run(config.output_dir, config.model_id)
    best = load_best_run(config.output_dir, config.model_id)

    run_dir = create_run_dir(config.output_dir, config.model_id)

    save_csv(seg_table, run_dir / "segments.csv")
    save_csv(features, run_dir / "segment_features.csv")
    save_csv(mapped, run_dir / "clustered_segments.csv")
    save_csv(mapped_measurements, run_dir / "clustered_measurements.csv")

    generate_profile_plots(mapped, strata, config, run_dir)
    generate_robertson_charts(seg_rows, mapped, run_dir)

    save_run(run_dir, summary, config_to_snapshot(config), report)
    update_pointers(config.output_dir, run_dir, summary, config.model_id)

    comparison = compare_runs(summary, previous, best)

    return {
        **summary,
        "comparison": comparison,
        "report": report,
        "run_dir": str(run_dir),
    }


def _representation_descriptor(config: Config) -> str:
    if config.representation_type == "summary":
        return f"stats: {', '.join(config.segment_stats)}"

    if config.representation_type == "resample":
        return (
            f"resample length={config.representation_length or 32}, "
            f"derivatives={config.include_derivatives}"
        )

    if config.representation_type == "paa":
        return (
            f"paa bins={config.representation_bins or 8}, "
            f"derivatives={config.include_derivatives}"
        )

    if config.representation_type == "hybrid":
        return (
            f"hybrid length={config.representation_length or 24}, "
            f"shape=True, stats=True, derivatives=True, geometry=True"
        )

    return config.representation_type


def _print_result(config: Config, result: dict) -> None:
    print(f"  Dataset:  {config.data_profile}")
    print(f"  Boundary: {config.boundary_source}")
    print(f"  Clusters: {result['n_clusters']}")

    if result["bic"] is not None:
        print(f"  BIC:      {result['bic']:.1f}")

    m = result.get("metrics", {})
    print(
        f"  Accuracy: {m.get('accuracy', 0):.4f}   "
        f"F1: {m.get('f1_macro', 0):.4f}   "
        f"NMI: {m.get('nmi', 0):.4f}   "
        f"ARI: {m.get('ari', 0):.4f}"
    )


def _print_comparison_table(results: dict[str, dict]) -> None:
    source_labels = {
        "ground_truth": "Ground Truth",
        "bocpd_estimate": "BOCPD Est.",
        "perfect_recall": "Perfect Recall",
    }

    ordered_sources = [s for s in BOUNDARY_SOURCES if s in results]
    if len(ordered_sources) < 2:
        return

    label_width = max(len(source_labels.get(s, s)) for s in ordered_sources)
    metric_width = 20
    col_width = max(12, label_width + 2)
    total_width = metric_width + (col_width * len(ordered_sources)) + col_width

    print("\n" + "=" * total_width)

    header = f"{'Metric':<{metric_width}}"
    for source in ordered_sources:
        header += f" {source_labels.get(source, source):>{col_width}}"
    header += f" {'Delta':>{col_width}}"

    print(header)
    print("-" * total_width)

    baseline_source = ordered_sources[0]
    compare_source = ordered_sources[-1]

    baseline_metrics = results.get(baseline_source, {}).get("metrics", {})
    compare_metrics = results.get(compare_source, {}).get("metrics", {})

    for key in (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "ari",
        "nmi",
    ):
        row = f"{key:<{metric_width}}"

        for source in ordered_sources:
            value = results.get(source, {}).get("metrics", {}).get(key)
            value_str = f"{value:.4f}" if value is not None else "-"
            row += f" {value_str:>{col_width}}"

        baseline_val = baseline_metrics.get(key)
        compare_val = compare_metrics.get(key)

        if baseline_val is not None and compare_val is not None:
            diff = compare_val - baseline_val
            delta = f"{'+' if diff >= 0 else ''}{diff:.4f}"
        else:
            delta = "-"

        row += f" {delta:>{col_width}}"
        print(row)

    print("=" * total_width)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unsupervised CPT segmentation")

    parser.add_argument(
        "--model",
        default="gmm_default",
        help="Model ID, filename in models/ without .json",
    )

    parser.add_argument(
        "--boundary-source",
        default="both",
        choices=["both", "ground_truth", "bocpd_estimate", "perfect_recall"],
        help="Boundary source to run. Default runs all available boundary sources.",
    )

    parser.add_argument(
        "--boundary-file",
        default="",
        help="Boundary CSV filename inside data/boundaries/, e.g. detected_boundaries.csv",
    )

    parser.add_argument(
        "--dataset-profile",
        default="default",
        choices=sorted(DATASET_PROFILES),
        help="Input dataset profile to use.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    project_dir = Path(__file__).resolve().parent.parent
    definition = load_model(args.model)

    if args.boundary_file and args.boundary_source in {"ground_truth", "perfect_recall"}:
        raise ValueError(
            "--boundary-file can only be used with --boundary-source bocpd_estimate or both."
        )

    if args.boundary_source == "both":
        sources = ["bocpd_estimate"] if args.boundary_file else BOUNDARY_SOURCES
    else:
        sources = [args.boundary_source]

    all_results: dict[str, dict] = {}

    for source in sources:
        config = Config.from_model_definition(project_dir, definition)
        _apply_dataset_profile(config, args.dataset_profile)
        config.boundary_source = source

        if source == "bocpd_estimate" and args.boundary_file:
            config.boundary_file = args.boundary_file

        print(f"\n{'-' * 40}")
        print(f"Model: {config.model_id} ({config.model_type})")

        descriptor = _representation_descriptor(config)
        print(
            f"  Feature set: {config.feature_set_id}  "
            f"({len(config.feature_columns)} columns, {descriptor})"
        )

        result = run(config)

        _print_result(config, result)

        print(f"\nPer-class report ({source}):")
        print(result["report"])

        all_results[source] = result

    if len(all_results) > 1:
        _print_comparison_table(all_results)