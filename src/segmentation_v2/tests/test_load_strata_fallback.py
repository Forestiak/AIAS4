from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEGMENTATION_DIR = ROOT / "segmentation_v2"
if str(SEGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_DIR))

from src.config import Config
from src.data import load_strata


class LoadStrataFallbackTests(unittest.TestCase):
    def test_load_strata_supplements_missing_targets_from_loc_only_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            data_dir = project_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"PointID": "BH-02", "Target": "Loc-02", "Top": 0.0, "Bottom": 5.0, "UNIT": "Unit A"},
                ]
            ).to_csv(data_dir / "Input_Strata_merged_boundaries.csv", index=False)

            pd.DataFrame(
                [
                    {"PointID": "Loc-02", "Target": "Loc-02", "Top": 0.0, "Bottom": 5.0, "UNIT": "Unit A"},
                    {"PointID": "Loc-04", "Target": "Loc-04", "Top": 0.0, "Bottom": 6.0, "UNIT": "Unit B"},
                ]
            ).to_csv(data_dir / "Input_Strata_merged_boundaries_loc_only.csv", index=False)

            config = Config(project_dir=project_dir)
            strata = load_strata(config)

            self.assertCountEqual(strata["Target"].unique().tolist(), ["Loc-02", "Loc-04"])
            loc04 = strata[strata["Target"] == "Loc-04"]
            self.assertEqual(len(loc04), 1)
            self.assertEqual(loc04.iloc[0]["UNIT"], "Unit B")


if __name__ == "__main__":
    unittest.main()