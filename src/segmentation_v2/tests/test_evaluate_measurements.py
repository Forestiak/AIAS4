from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SEGMENTATION_DIR = ROOT / "segmentation_v2"
if str(SEGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_DIR))

from src.config import Config
from src.evaluate import apply_cluster_mapping, attach_reference_units_to_measurements, build_cluster_mapping, compute_metrics


class MeasurementEvaluationTests(unittest.TestCase):
    def test_cluster_mapping_and_metrics_are_based_on_measurements(self):
        config = Config(project_dir=ROOT)
        strata = pd.DataFrame(
            [
                {"Target": "T1", "PointID": "P1", "Top": 0.0, "Bottom": 2.0, "UNIT": "Unit A"},
                {"Target": "T1", "PointID": "P1", "Top": 2.0, "Bottom": 12.0, "UNIT": "Unit B"},
            ]
        )
        rows = pd.DataFrame(
            [
                {"target": "T1", "segment_id": 0, "Depth": 0.5, "cluster_id": 0},
                {"target": "T1", "segment_id": 1, "Depth": 1.5, "cluster_id": 0},
                {"target": "T1", "segment_id": 2, "Depth": 2.0, "cluster_id": 0},
                {"target": "T1", "segment_id": 2, "Depth": 3.0, "cluster_id": 0},
                {"target": "T1", "segment_id": 2, "Depth": 4.0, "cluster_id": 0},
                {"target": "T1", "segment_id": 2, "Depth": 5.0, "cluster_id": 0},
                {"target": "T1", "segment_id": 2, "Depth": 6.0, "cluster_id": 0},
                {"target": "T1", "segment_id": 2, "Depth": 7.0, "cluster_id": 0},
                {"target": "T1", "segment_id": 2, "Depth": 8.0, "cluster_id": 0},
                {"target": "T1", "segment_id": 2, "Depth": 9.0, "cluster_id": 0},
                {"target": "T1", "segment_id": 2, "Depth": 10.0, "cluster_id": 0},
                {"target": "T1", "segment_id": 2, "Depth": 11.0, "cluster_id": 0},
            ]
        )

        measured = attach_reference_units_to_measurements(rows, strata, config)
        mapping = build_cluster_mapping(measured)
        mapped = apply_cluster_mapping(measured, mapping)
        metrics = compute_metrics(mapped)

        self.assertEqual(mapping, {0: "Unit B"})
        self.assertAlmostEqual(metrics["accuracy"], 10.0 / 12.0)
        self.assertEqual(mapped.loc[mapped["Depth"] == 2.0, "reference_unit"].iloc[0], "Unit B")


if __name__ == "__main__":
    unittest.main()