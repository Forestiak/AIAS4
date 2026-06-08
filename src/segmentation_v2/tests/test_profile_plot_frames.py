from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
SEGMENTATION_DIR = ROOT / "segmentation_v2"
if str(SEGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_DIR))

from src.config import Config
from src.plots import build_profile_plot_frames


class ProfilePlotFrameTests(unittest.TestCase):
    def test_profile_plot_uses_true_strata_on_left_and_predicted_segments_on_right(self):
        config = Config(project_dir=Path("."))
        strata = pd.DataFrame(
            [
                {"Target": "T1", "PointID": "P1", "Top": 0.0, "Bottom": 2.0, "UNIT": "Unit A"},
                {"Target": "T1", "PointID": "P1", "Top": 2.0, "Bottom": 4.0, "UNIT": "Unit B"},
            ]
        )
        mapped = pd.DataFrame(
            [
                {"target": "T1", "top": 0.0, "bottom": 1.0, "predicted_unit": "Unit A"},
                {"target": "T1", "top": 1.0, "bottom": 3.0, "predicted_unit": "Unit B"},
                {"target": "T1", "top": 3.0, "bottom": 4.0, "predicted_unit": "Unit B"},
            ]
        )

        frames = build_profile_plot_frames(mapped, strata, config)

        reference = frames["T1"]["reference"]
        predicted = frames["T1"]["predicted"]

        self.assertEqual(reference[["top", "bottom", "unit"]].to_dict("records"), [
            {"top": 0.0, "bottom": 2.0, "unit": "Unit A"},
            {"top": 2.0, "bottom": 4.0, "unit": "Unit B"},
        ])
        self.assertEqual(predicted[["top", "bottom", "unit"]].to_dict("records"), [
            {"top": 0.0, "bottom": 1.0, "unit": "Unit A"},
            {"top": 1.0, "bottom": 3.0, "unit": "Unit B"},
            {"top": 3.0, "bottom": 4.0, "unit": "Unit B"},
        ])


if __name__ == "__main__":
    unittest.main()