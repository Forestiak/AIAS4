from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

if str(SEGMENTATION_DIR := (Path(__file__).resolve().parents[2] / "segmentation_v2")) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_DIR))

from src.feature_set_registry import load_feature_set
from src.model_registry import load_model
ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_COLUMNS = {
    "HEADING",
    "PointID",
    "Target",
    "SCPG_TESN",
    "Depth",
    "File source",
    "EXCLUDE",
    "EXCLUDE_REASON",
    "Target_number",
    "UNIT",
    "GROUP",
    "SCPT_CPO",
    "SCPT_CPOD",
    "SCPT_ISPP",
    "BDEN",
}


class BruteForceSuiteTests(unittest.TestCase):
    def test_suite_manifests_and_file_counts(self):
        for suite_name, prefix in (("bruteforce_paa", "bf_paa_"), ("bruteforce_sample", "bf_sample_")):
            suite_dir = SEGMENTATION_DIR / suite_name
            manifest = json.loads((suite_dir / "manifest.json").read_text())
            model_files = sorted((suite_dir / "models").glob("*.json"))
            feature_files = sorted((suite_dir / "feature_sets").glob("*.json"))

            self.assertEqual(manifest["model_count"], 20)
            self.assertEqual(len(manifest["models"]), 20)
            self.assertEqual(len(model_files), 20)
            self.assertEqual(len(feature_files), 20)

            model_ids = set()
            for entry in manifest["models"]:
                self.assertTrue(entry["model_id"].startswith(prefix))
                self.assertTrue(entry["feature_set_id"].startswith(prefix))
                self.assertFalse(entry["use_spatial_mrf"])
                self.assertEqual(entry["cluster_selection"], "fixed")
                self.assertEqual(entry["n_clusters_requested"], 20)
                self.assertEqual(entry["covariance_type"], "diag")
                self.assertNotIn(entry["model_id"], model_ids)
                self.assertTrue(set(entry["columns"]).isdisjoint(FORBIDDEN_COLUMNS))
                model_ids.add(entry["model_id"])

    def test_dry_run_scripts_exit_successfully(self):
        for suite_name in ("bruteforce_paa", "bruteforce_sample"):
            script = SEGMENTATION_DIR / suite_name / "run_bruteforce.py"
            result = subprocess.run(
                [sys.executable, str(script), "--dry-run", "--limit", "2"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("Selected 2 model(s).", result.stdout)

    def test_bruteforce_models_are_available_in_main_registries(self):
        for model_id, feature_set_id in (
            ("bf_paa_raw3_b4_d0", "bf_paa_raw3_b4_d0"),
            ("bf_sample_hybrid_core_l32_d1_pca30", "bf_sample_hybrid_core_l32_d1_pca30"),
        ):
            model = load_model(model_id)
            feature_set = load_feature_set(feature_set_id)
            self.assertEqual(model["feature_set_id"], feature_set_id)
            self.assertEqual(feature_set["feature_set_id"], feature_set_id)


if __name__ == "__main__":
    unittest.main()
