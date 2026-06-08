from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEGMENTATION_DIR = ROOT / "segmentation_v2"
if str(SEGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_DIR))

from src.config import Config
from src.model_registry import load_model


def _load_pipeline_module():
    module_path = SEGMENTATION_DIR / "pipeline.py"
    spec = importlib.util.spec_from_file_location("segmentation_v2_pipeline", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_smoke_inputs(project_dir: Path) -> None:
    data_dir = project_dir / "data"
    (project_dir / "segmentation_v2" / "outputs").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cpt_rows: list[dict] = []
    strata_rows: list[dict] = []
    units = ["Unit 1A siSa", "Unit 2A siSa", "Unit 3A siCl", "Unit 3C Sa"]

    for target_idx in range(1, 7):
        target = f"T{target_idx}"
        for depth in np.arange(0.0, 4.51, 0.5):
            cpt_rows.append(
                {
                    "Target": target,
                    "PointID": f"P{target_idx}",
                    "Depth": depth,
                    "SCPT_RES": 1.0 + target_idx + depth,
                    "SCPT_FRES": 0.2 + (target_idx * 0.1) + (depth * 0.05),
                    "SCPT_PWP2": 0.5 + (target_idx * 0.2) + (depth * 0.1),
                    "SCPT_QT": 1.2 + target_idx + depth,
                    "SCPT_QNET": 0.8 + target_idx + depth,
                    "SCPT_FRR": 0.4 + target_idx + depth,
                    "SCPT_NFR": 0.6 + target_idx + depth,
                    "SCPT_BQ": 0.3 + target_idx + depth,
                    "SCPT_NU2": 0.7 + target_idx + depth,
                    "delta_u2": 0.1 + target_idx + depth,
                    "SCPT_NQT": 1.5 + target_idx + depth,
                    "SCPT_ICBE": 2.0 + target_idx + depth,
                }
            )

        boundaries = [0.0, 1.0, 2.0, 3.0, 4.5]
        for unit_idx, (top, bottom) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=0):
            strata_rows.append(
                {
                    "Target": target,
                    "PointID": f"P{target_idx}",
                    "Top": top,
                    "Bottom": bottom,
                    "UNIT": units[unit_idx],
                }
            )

    pd.DataFrame(cpt_rows).to_csv(data_dir / "CPT_clean.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(data_dir / "Input_Strata_merged_boundaries.csv", index=False)


class PipelineSmokeTests(unittest.TestCase):
    def test_stats_and_new_representation_models_complete_and_write_outputs(self):
        pipeline = _load_pipeline_module()
        summary_definition = {
            "model_id": "smoke_summary",
            "model_type": "gmm",
            "feature_columns": ["SCPT_RES", "SCPT_FRES", "SCPT_PWP2"],
            "parameters": {
                "min_clusters": 2,
                "max_clusters": 8,
                "random_state": 42,
            },
        }

        def fake_fit(feature_matrix, config):
            self.assertGreater(feature_matrix.shape[1], 0)
            return SimpleNamespace(n_clusters=2, bic=123.45, model_type=config.model_type)

        def fake_predict(result, feature_matrix):
            return np.arange(len(feature_matrix)) % result.n_clusters

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            _write_smoke_inputs(project_dir)

            with patch.object(pipeline, "fit", side_effect=fake_fit), patch.object(pipeline, "predict", side_effect=fake_predict):
                for definition in (
                    summary_definition,
                    load_model("gmm_raw_sensors_resampled_32_derivatives"),
                    load_model("gmm_raw_sensors_extended_resampled_32_derivatives"),
                    load_model("gmm_raw_sensors_paa_8_derivatives"),
                    load_model("gmm_raw_sensors_extended_paa_8_derivatives"),
                ):
                    model_id = definition["model_id"]
                    config = Config.from_model_definition(project_dir, definition)
                    config.boundary_source = "ground_truth"
                    result = pipeline.run(config)
                    self.assertEqual(result["model_id"], model_id)

                    model_log_dir = project_dir / "segmentation_v2" / "outputs" / "logs" / model_id
                    run_dirs = [path for path in model_log_dir.iterdir() if path.is_dir()]
                    self.assertTrue(run_dirs)
                    latest_run = max(run_dirs)
                    self.assertTrue((latest_run / "summary.json").exists())
                    self.assertTrue((latest_run / "config.json").exists())
                    self.assertTrue((latest_run / "segment_features.csv").exists())
                    self.assertTrue((latest_run / "clustered_measurements.csv").exists())
                    self.assertTrue(any((latest_run / "plots").glob("robertson_class_*.png")))
                    self.assertFalse((latest_run / "plots" / "robertson_all.png").exists())


if __name__ == "__main__":
    unittest.main()
