"""
Configuration for the unsupervised segmentation pipeline.

Single dataclass holding:
- paths
- feature-set settings
- model settings
- spatial MRF settings
- hybrid feature settings
- optional row-level legacy/experimental settings
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_dir: Path

    # ─────────────────────────────────────────────────────────────────
    # Model identity
    # ─────────────────────────────────────────────────────────────────

    model_id: str = "gmm_default"
    model_type: str = "gmm"

    # Pipeline type:
    # - "segment": main boundary-first thesis pipeline
    # - "row_level": experimental legacy path
    pipeline_type: str = "segment"

    # ─────────────────────────────────────────────────────────────────
    # Feature-set identity
    # ─────────────────────────────────────────────────────────────────

    feature_set_id: str = "raw_sensors"
    feature_columns: tuple[str, ...] = ("SCPT_RES", "SCPT_FRES", "SCPT_PWP2")

    segment_stats: tuple[str, ...] = ("mean", "std", "median")
    include_thickness: bool = True
    extra_derived: tuple[str, ...] = ()

    representation_type: str = "summary"
    representation_length: int | None = None
    representation_bins: int | None = None
    include_derivatives: bool = False

    # Hybrid segment representation
    hybrid_shape_length: int = 24
    hybrid_include_shape: bool = True
    hybrid_include_derivative_shape: bool = True
    hybrid_include_distribution: bool = True
    hybrid_include_engineered: bool = True
    hybrid_include_geometry: bool = True
    hybrid_include_coordinates: bool = False

    # ─────────────────────────────────────────────────────────────────
    # Cluster/model settings
    # ─────────────────────────────────────────────────────────────────

    cluster_selection: str = "bic"
    n_clusters: int | None = None
    min_clusters: int = 2
    max_clusters: int = 8

    min_segment_thickness_m: float = 0.5
    random_state: int = 42

    covariance_type: str = "full"
    pca_n_components: int | None = None
    reg_covar: float = 1e-6

    scaler_type: str = "robust"  # robust or standard
    gmm_n_init: int = 3
    gmm_max_iter: int = 300

    # ─────────────────────────────────────────────────────────────────
    # Boundary source
    # ─────────────────────────────────────────────────────────────────

    boundary_source: str = "ground_truth"
    boundary_file: str = ""
    data_profile: str = "default"

    # ─────────────────────────────────────────────────────────────────
    # Segment-level spatial MRF
    # ─────────────────────────────────────────────────────────────────

    use_spatial_mrf: bool = False
    spatial_lambda: float = 1.0
    spatial_max_distance_m: float = 5000.0
    spatial_sigma_m: float = 2000.0
    location_file: str = "Input_Location_clean.csv"
    cpt_file: str = "CPT_clean.csv"
    strata_file: str = "Input_Strata_merged_boundaries.csv"
    perfect_recall_file: str = "perfect_recall.csv"

    # ─────────────────────────────────────────────────────────────────
    # Row-level experimental settings
    # Keep these for backward compatibility if old row-level files exist.
    # They are not required for the main hybrid segment pipeline.
    # ─────────────────────────────────────────────────────────────────

    row_representation: str = "measurement"
    depth_grid_step_m: float = 0.05
    row_feature_windows_m: tuple[float, ...] = (0.10, 0.25, 0.50)
    row_include_absolute_depth: bool = True
    row_include_relative_depth: bool = True
    row_include_coordinates: bool = False
    row_include_derivatives: bool = True
    row_include_rolling_stats: bool = True
    row_include_engineered_cpt: bool = True

    vertical_lambda: float = 1.0
    vertical_sigma_m: float = 0.15
    spatial_row_lambda: float = 0.5
    spatial_depth_tolerance_m: float = 0.15
    spatial_k_neighbors: int = 4

    row_postprocess_min_layer_thickness_m: float = 0.20
    row_output_segment_merge: bool = True

    # ─────────────────────────────────────────────────────────────────
    # Input column names
    # ─────────────────────────────────────────────────────────────────

    target_col: str = "Target"
    depth_col: str = "Depth"
    point_id_col: str = "PointID"
    strata_top_col: str = "Top"
    strata_bottom_col: str = "Bottom"
    strata_unit_col: str = "UNIT"

    # ─────────────────────────────────────────────────────────────────
    # Paths
    # ─────────────────────────────────────────────────────────────────

    @property
    def cpt_path(self) -> Path:
        return self.project_dir / "data" / self.cpt_file

    @property
    def strata_path(self) -> Path:
        return self.project_dir / "data" / self.strata_file

    @property
    def boundaries_dir(self) -> Path:
        return self.project_dir / "data" / "boundaries"

    @property
    def exported_boundaries_path(self) -> Path:
        if self.boundary_file:
            return self.boundaries_dir / self.boundary_file
        return self.boundaries_dir / "exported_boundaries.csv"

    @property
    def perfect_recall_boundaries_path(self) -> Path:
        return self.boundaries_dir / self.perfect_recall_file

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "segmentation_v2" / "outputs"

    @property
    def representation_size(self) -> int | None:
        if self.representation_type == "resample":
            return self.representation_length
        if self.representation_type == "paa":
            return self.representation_bins
        if self.representation_type == "hybrid":
            return self.representation_length or self.hybrid_shape_length
        return None

    # ─────────────────────────────────────────────────────────────────
    # Factory from model JSON
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def from_model_definition(cls, project_dir: Path, definition: dict) -> "Config":
        params = definition.get("parameters", {})
        spatial = definition.get("spatial_parameters", {})
        row_params = definition.get("row_parameters", {})

        # ── Resolve feature set ──────────────────────────────────────

        feature_set_id = definition.get("feature_set_id", "")

        if feature_set_id:
            from .feature_set_registry import load_feature_set

            fs = load_feature_set(feature_set_id)

            feature_columns = tuple(c["name"] for c in fs.get("columns", []))
            segment_stats = tuple(fs.get("segment_stats", ["mean", "std", "median"]))
            include_thickness = bool(fs.get("include_thickness", True))
            extra_derived = tuple(fs.get("extra_derived", []))

            representation = fs.get("representation") or {}
            representation_type = representation.get("type", "summary")
            representation_length = representation.get("length")
            representation_bins = representation.get("bins")
            include_derivatives = bool(representation.get("include_derivatives", False))

            hybrid_shape_length = int(
                representation.get("shape_length")
                or representation.get("length")
                or 24
            )
            hybrid_include_shape = bool(representation.get("include_shape", True))
            hybrid_include_derivative_shape = bool(
                representation.get("include_derivative_shape", True)
            )
            hybrid_include_distribution = bool(
                representation.get("include_distribution", True)
            )
            hybrid_include_engineered = bool(
                representation.get("include_engineered", True)
            )
            hybrid_include_geometry = bool(
                representation.get("include_geometry", True)
            )
            hybrid_include_coordinates = bool(
                representation.get("include_coordinates", False)
            )

        else:
            # Backward compatibility for old inline model definitions.
            feature_columns = tuple(
                definition.get(
                    "feature_columns",
                    ("SCPT_RES", "SCPT_FRES", "SCPT_PWP2"),
                )
            )

            use_derived = definition.get("use_derived_features", False)
            if use_derived:
                segment_stats = ("mean", "std", "median", "min", "max")
                extra_derived = ("Rf", "log_qc")
            else:
                segment_stats = ("mean", "std", "median")
                extra_derived = ()

            include_thickness = True
            feature_set_id = "(inline)"
            representation_type = "summary"
            representation_length = None
            representation_bins = None
            include_derivatives = False

            hybrid_shape_length = 24
            hybrid_include_shape = True
            hybrid_include_derivative_shape = True
            hybrid_include_distribution = True
            hybrid_include_engineered = True
            hybrid_include_geometry = True
            hybrid_include_coordinates = False

        return cls(
            project_dir=project_dir,

            # Identity
            model_id=definition["model_id"],
            model_type=definition.get("model_type", "gmm"),
            pipeline_type=definition.get("pipeline_type", "segment"),

            # Features
            feature_set_id=feature_set_id,
            feature_columns=feature_columns,
            segment_stats=segment_stats,
            include_thickness=include_thickness,
            extra_derived=extra_derived,
            representation_type=representation_type,
            representation_length=representation_length,
            representation_bins=representation_bins,
            include_derivatives=include_derivatives,

            # Hybrid
            hybrid_shape_length=hybrid_shape_length,
            hybrid_include_shape=hybrid_include_shape,
            hybrid_include_derivative_shape=hybrid_include_derivative_shape,
            hybrid_include_distribution=hybrid_include_distribution,
            hybrid_include_engineered=hybrid_include_engineered,
            hybrid_include_geometry=hybrid_include_geometry,
            hybrid_include_coordinates=hybrid_include_coordinates,

            # Clustering
            cluster_selection=params.get("cluster_selection", "bic"),
            n_clusters=params.get("n_clusters"),
            min_clusters=params.get("min_clusters", 2),
            max_clusters=params.get("max_clusters", 8),
            random_state=params.get("random_state", 42),
            min_segment_thickness_m=definition.get("min_segment_thickness_m", 0.5),
            covariance_type=params.get("covariance_type", "full"),
            pca_n_components=params.get("pca_n_components"),
            reg_covar=params.get("reg_covar", 1e-6),
            scaler_type=params.get("scaler_type", "robust"),
            gmm_n_init=params.get("gmm_n_init", 3),
            gmm_max_iter=params.get("gmm_max_iter", 300),

            # Boundaries
            boundary_source=definition.get("boundary_source", "ground_truth"),
            boundary_file=definition.get("boundary_file", ""),
            data_profile=definition.get("data_profile", "default"),

            # Segment spatial MRF
            use_spatial_mrf=definition.get("use_spatial_mrf", False),
            spatial_lambda=spatial.get("lambda", 1.0),
            spatial_max_distance_m=spatial.get("max_distance_m", 5000.0),
            spatial_sigma_m=spatial.get("sigma_m", 2000.0),
            location_file=definition.get("location_file", "Input_Location_clean.csv"),
            cpt_file=definition.get("cpt_file", "CPT_clean.csv"),
            strata_file=definition.get("strata_file", "Input_Strata_merged_boundaries.csv"),
            perfect_recall_file=definition.get("perfect_recall_file", "perfect_recall.csv"),

            # Row-level experimental compatibility
            row_representation=row_params.get("row_representation", "measurement"),
            depth_grid_step_m=row_params.get("depth_grid_step_m", 0.05),
            row_feature_windows_m=tuple(
                row_params.get("feature_windows_m", (0.10, 0.25, 0.50))
            ),
            row_include_absolute_depth=row_params.get("include_absolute_depth", True),
            row_include_relative_depth=row_params.get("include_relative_depth", True),
            row_include_coordinates=row_params.get("include_coordinates", False),
            row_include_derivatives=row_params.get("include_derivatives", True),
            row_include_rolling_stats=row_params.get("include_rolling_stats", True),
            row_include_engineered_cpt=row_params.get("include_engineered_cpt", True),
            vertical_lambda=spatial.get("vertical_lambda", 1.0),
            vertical_sigma_m=spatial.get("vertical_sigma_m", 0.15),
            spatial_row_lambda=spatial.get("spatial_lambda", 0.5),
            spatial_depth_tolerance_m=spatial.get("depth_tolerance_m", 0.15),
            spatial_k_neighbors=spatial.get("spatial_k_neighbors", 4),
            row_postprocess_min_layer_thickness_m=row_params.get(
                "postprocess_min_layer_thickness_m",
                0.20,
            ),
            row_output_segment_merge=row_params.get("output_segment_merge", True),
        )