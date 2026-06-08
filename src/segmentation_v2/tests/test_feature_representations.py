from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEGMENTATION_DIR = ROOT / "segmentation_v2"
if str(SEGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_DIR))

from src.config import Config
from src.feature_set_registry import load_feature_set
from src.features import build_features, feature_matrix_columns
from src.model_registry import load_model


def _make_segmented_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"target": "T1", "segment_id": 0, "seg_top": 0.0, "seg_bottom": 2.0, "Depth": 0.0, "SCPT_RES": 1.0, "SCPT_FRES": 10.0, "SCPT_PWP2": 100.0},
            {"target": "T1", "segment_id": 0, "seg_top": 0.0, "seg_bottom": 2.0, "Depth": 1.0, "SCPT_RES": 2.0, "SCPT_FRES": 20.0, "SCPT_PWP2": 200.0},
            {"target": "T1", "segment_id": 0, "seg_top": 0.0, "seg_bottom": 2.0, "Depth": 2.0, "SCPT_RES": 3.0, "SCPT_FRES": 30.0, "SCPT_PWP2": 300.0},
            {"target": "T1", "segment_id": 1, "seg_top": 2.0, "seg_bottom": 5.0, "Depth": 2.0, "SCPT_RES": 4.0, "SCPT_FRES": 40.0, "SCPT_PWP2": 400.0},
            {"target": "T1", "segment_id": 1, "seg_top": 2.0, "seg_bottom": 5.0, "Depth": 3.5, "SCPT_RES": 5.0, "SCPT_FRES": 50.0, "SCPT_PWP2": 500.0},
            {"target": "T1", "segment_id": 1, "seg_top": 2.0, "seg_bottom": 5.0, "Depth": 5.0, "SCPT_RES": 6.0, "SCPT_FRES": 60.0, "SCPT_PWP2": 600.0},
        ]
    )


def _make_edge_case_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"target": "T1", "segment_id": 0, "seg_top": 0.0, "seg_bottom": 2.0, "Depth": 1.0, "SCPT_RES": 10.0, "SCPT_FRES": np.nan, "SCPT_PWP2": 30.0},
            {"target": "T1", "segment_id": 1, "seg_top": 2.0, "seg_bottom": 4.0, "Depth": 2.5, "SCPT_RES": 1.0, "SCPT_FRES": np.nan, "SCPT_PWP2": 5.0},
            {"target": "T1", "segment_id": 1, "seg_top": 2.0, "seg_bottom": 4.0, "Depth": 2.5, "SCPT_RES": 3.0, "SCPT_FRES": np.nan, "SCPT_PWP2": 7.0},
            {"target": "T1", "segment_id": 1, "seg_top": 2.0, "seg_bottom": 4.0, "Depth": 3.5, "SCPT_RES": 5.0, "SCPT_FRES": np.nan, "SCPT_PWP2": 9.0},
        ]
    )


class FeatureRepresentationTests(unittest.TestCase):
    def test_every_base_feature_set_has_resample_and_paa_variants_with_models(self):
        base_ids = [
            "raw_sensors",
            "raw_sensors_extended",
            "robertson_core",
            "raw_plus_robertson",
            "extended_derived",
            "dataset_full",
        ]
        for base_id in base_ids:
            base_feature_set = load_feature_set(base_id)
            for suffix, folder in (
                ("resampled_32_derivatives", "resample_gmm"),
                ("paa_8_derivatives", "paa_gmm"),
            ):
                feature_set_id = f"{base_id}_{suffix}"
                feature_set = load_feature_set(feature_set_id)
                self.assertEqual(
                    [column["name"] for column in feature_set["columns"]],
                    [column["name"] for column in base_feature_set["columns"]],
                )
                self.assertEqual(feature_set.get("include_thickness", True), base_feature_set.get("include_thickness", True))
                self.assertEqual(feature_set.get("extra_derived", []), base_feature_set.get("extra_derived", []))

                model_id = f"gmm_{base_id}_{suffix}"
                model = load_model(model_id)
                self.assertEqual(model["feature_set_id"], feature_set_id)
                self.assertTrue((SEGMENTATION_DIR / "models" / folder / f"{model_id}.json").exists())

    def test_legacy_summary_defaults_and_column_names_are_unchanged(self):
        config = Config.from_model_definition(ROOT, load_model("gmm_default"))
        self.assertEqual(config.representation_type, "summary")
        self.assertEqual(config.segment_stats, ("mean", "std", "median"))
        self.assertEqual(
            feature_matrix_columns(config),
            [
                "SCPT_RES_mean",
                "SCPT_RES_std",
                "SCPT_RES_median",
                "SCPT_FRES_mean",
                "SCPT_FRES_std",
                "SCPT_FRES_median",
                "SCPT_PWP2_mean",
                "SCPT_PWP2_std",
                "SCPT_PWP2_median",
                "thickness",
            ],
        )

    def test_resampled_representation_has_fixed_feature_count_and_metadata(self):
        config = Config(
            project_dir=ROOT,
            feature_columns=("SCPT_RES", "SCPT_FRES", "SCPT_PWP2"),
            representation_type="resample",
            representation_length=32,
            include_derivatives=True,
            include_thickness=True,
        )
        features = build_features(_make_segmented_rows(), config)
        expected_feature_count = (3 * 32) + (3 * 32) + 1

        self.assertEqual(features.shape[0], 2)
        self.assertEqual(len(feature_matrix_columns(config)), expected_feature_count)
        self.assertEqual(list(features.columns[:6]), ["target", "segment_id", "top", "bottom", "thickness", "n_rows"])
        self.assertTrue(all(name in features.columns for name in ["SCPT_RES_r00", "SCPT_RES_r31", "SCPT_RES_d00", "SCPT_RES_d31"]))

    def test_paa_representation_has_fixed_feature_count(self):
        config = Config(
            project_dir=ROOT,
            feature_columns=("SCPT_RES", "SCPT_FRES", "SCPT_PWP2"),
            representation_type="paa",
            representation_bins=8,
            include_derivatives=True,
            include_thickness=True,
        )
        features = build_features(_make_segmented_rows(), config)
        expected_feature_count = (3 * 8) + (3 * 8) + 1

        self.assertEqual(features.shape[0], 2)
        self.assertEqual(len(feature_matrix_columns(config)), expected_feature_count)
        self.assertTrue(all(name in features.columns for name in ["SCPT_RES_paa00", "SCPT_RES_paa07", "SCPT_RES_paa_d00", "SCPT_RES_paa_d07"]))

    def test_edge_cases_are_handled_for_shape_features(self):
        config = Config(
            project_dir=ROOT,
            feature_columns=("SCPT_RES", "SCPT_FRES", "SCPT_PWP2"),
            representation_type="resample",
            representation_length=32,
            include_derivatives=True,
        )
        features = build_features(_make_edge_case_rows(), config)

        single_row = features.loc[features["segment_id"] == 0].iloc[0]
        self.assertEqual(single_row["SCPT_RES_r00"], 10.0)
        self.assertEqual(single_row["SCPT_RES_r31"], 10.0)
        self.assertEqual(single_row["SCPT_RES_d00"], 0.0)
        self.assertTrue(np.isnan(single_row["SCPT_FRES_r00"]))

        duplicate_depth = features.loc[features["segment_id"] == 1].iloc[0]
        self.assertTrue(np.isfinite(duplicate_depth["SCPT_RES_r00"]))
        self.assertTrue(np.isfinite(duplicate_depth["SCPT_RES_r31"]))
        self.assertTrue(duplicate_depth.filter(like="SCPT_FRES_r").isna().all())


if __name__ == "__main__":
    unittest.main()
