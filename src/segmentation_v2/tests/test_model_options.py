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
from src.model import fit


class ModelOptionTests(unittest.TestCase):
    def test_config_parses_fixed_cluster_and_pca_options(self):
        definition = {
            "model_id": "test_fixed_gmm",
            "model_type": "gmm",
            "feature_columns": ["SCPT_RES", "SCPT_FRES", "SCPT_PWP2"],
            "parameters": {
                "cluster_selection": "fixed",
                "n_clusters": 20,
                "covariance_type": "diag",
                "pca_n_components": 30,
                "reg_covar": 1e-5,
                "random_state": 42,
            },
        }
        config = Config.from_model_definition(ROOT, definition)
        self.assertEqual(config.cluster_selection, "fixed")
        self.assertEqual(config.n_clusters, 20)
        self.assertEqual(config.pca_n_components, 30)
        self.assertEqual(config.reg_covar, 1e-5)
        self.assertEqual(config.covariance_type, "diag")

    def test_fit_uses_fixed_clusters_and_caps_pca_components(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(5, 6)), columns=[f"f{i}" for i in range(6)])
        config = Config(
            project_dir=ROOT,
            cluster_selection="fixed",
            n_clusters=2,
            pca_n_components=20,
            covariance_type="diag",
            reg_covar=1e-5,
        )

        result = fit(X, config)

        self.assertEqual(result.n_clusters, 2)
        self.assertIn("pca", result.pipeline.named_steps)
        self.assertEqual(result.pipeline.named_steps["pca"].n_components, 4)
        self.assertEqual(result.estimator.n_components, 2)
        self.assertEqual(result.estimator.reg_covar, 1e-5)

    def test_fit_skips_pca_when_not_requested(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame(rng.normal(size=(6, 4)), columns=[f"f{i}" for i in range(4)])
        config = Config(project_dir=ROOT)

        result = fit(X, config)

        self.assertNotIn("pca", result.pipeline.named_steps)


if __name__ == "__main__":
    unittest.main()
