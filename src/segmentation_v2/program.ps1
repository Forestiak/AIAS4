$featureSetsDir = ".\feature_sets"
New-Item -ItemType Directory -Force -Path $featureSetsDir | Out-Null

@'
{
  "feature_set_id": "fc_summary_core",
  "display_name": "FC Summary Core",
  "description": "Segment-level statistical baseline using CPT core and Robertson features.",
  "columns": [
    { "name": "SCPT_RES", "description": "Cone resistance qc", "category": "raw" },
    { "name": "SCPT_FRES", "description": "Sleeve friction fs", "category": "raw" },
    { "name": "SCPT_PWP2", "description": "Porewater pressure u2", "category": "raw" },
    { "name": "SCPT_NQT", "description": "Normalised cone resistance Qtn", "category": "robertson" },
    { "name": "SCPT_NFR", "description": "Normalised friction ratio Fr", "category": "robertson" },
    { "name": "SCPT_ICBE", "description": "Soil behaviour type index Ic", "category": "robertson" }
  ],
  "segment_stats": ["mean", "std", "median", "min", "max"],
  "include_thickness": true,
  "extra_derived": ["Rf", "log_qc"],
  "representation": {
    "type": "summary"
  }
}
'@ | Set-Content "$featureSetsDir\fc_summary_core.json" -Encoding UTF8

@'
{
  "feature_set_id": "fc_paa4_core",
  "display_name": "FC PAA-4 Core",
  "description": "Compact 4-bin PAA segment representation. Strong denoising baseline.",
  "columns": [
    { "name": "SCPT_RES", "description": "Cone resistance qc", "category": "raw" },
    { "name": "SCPT_FRES", "description": "Sleeve friction fs", "category": "raw" },
    { "name": "SCPT_PWP2", "description": "Porewater pressure u2", "category": "raw" },
    { "name": "SCPT_NQT", "description": "Normalised cone resistance Qtn", "category": "robertson" },
    { "name": "SCPT_NFR", "description": "Normalised friction ratio Fr", "category": "robertson" },
    { "name": "SCPT_ICBE", "description": "Soil behaviour type index Ic", "category": "robertson" }
  ],
  "include_thickness": true,
  "extra_derived": ["Rf", "log_qc"],
  "representation": {
    "type": "paa",
    "bins": 4,
    "include_derivatives": false
  }
}
'@ | Set-Content "$featureSetsDir\fc_paa4_core.json" -Encoding UTF8

@'
{
  "feature_set_id": "fc_paa8_core",
  "display_name": "FC PAA-8 Core",
  "description": "Moderately compact 8-bin PAA segment representation. Main compromise candidate.",
  "columns": [
    { "name": "SCPT_RES", "description": "Cone resistance qc", "category": "raw" },
    { "name": "SCPT_FRES", "description": "Sleeve friction fs", "category": "raw" },
    { "name": "SCPT_PWP2", "description": "Porewater pressure u2", "category": "raw" },
    { "name": "SCPT_NQT", "description": "Normalised cone resistance Qtn", "category": "robertson" },
    { "name": "SCPT_NFR", "description": "Normalised friction ratio Fr", "category": "robertson" },
    { "name": "SCPT_ICBE", "description": "Soil behaviour type index Ic", "category": "robertson" }
  ],
  "include_thickness": true,
  "extra_derived": ["Rf", "log_qc"],
  "representation": {
    "type": "paa",
    "bins": 8,
    "include_derivatives": false
  }
}
'@ | Set-Content "$featureSetsDir\fc_paa8_core.json" -Encoding UTF8

@'
{
  "feature_set_id": "fc_paa16_core",
  "display_name": "FC PAA-16 Core",
  "description": "Higher-detail 16-bin PAA representation for testing whether more vertical detail improves clustering.",
  "columns": [
    { "name": "SCPT_RES", "description": "Cone resistance qc", "category": "raw" },
    { "name": "SCPT_FRES", "description": "Sleeve friction fs", "category": "raw" },
    { "name": "SCPT_PWP2", "description": "Porewater pressure u2", "category": "raw" },
    { "name": "SCPT_NQT", "description": "Normalised cone resistance Qtn", "category": "robertson" },
    { "name": "SCPT_NFR", "description": "Normalised friction ratio Fr", "category": "robertson" },
    { "name": "SCPT_ICBE", "description": "Soil behaviour type index Ic", "category": "robertson" }
  ],
  "include_thickness": true,
  "extra_derived": ["Rf", "log_qc"],
  "representation": {
    "type": "paa",
    "bins": 16,
    "include_derivatives": false
  }
}
'@ | Set-Content "$featureSetsDir\fc_paa16_core.json" -Encoding UTF8

@'
{
  "feature_set_id": "fc_resample16_core",
  "display_name": "FC Resample-16 Core",
  "description": "16-point interpolated segment shape representation. Tests time-series shape without PAA averaging.",
  "columns": [
    { "name": "SCPT_RES", "description": "Cone resistance qc", "category": "raw" },
    { "name": "SCPT_FRES", "description": "Sleeve friction fs", "category": "raw" },
    { "name": "SCPT_PWP2", "description": "Porewater pressure u2", "category": "raw" },
    { "name": "SCPT_NQT", "description": "Normalised cone resistance Qtn", "category": "robertson" },
    { "name": "SCPT_NFR", "description": "Normalised friction ratio Fr", "category": "robertson" },
    { "name": "SCPT_ICBE", "description": "Soil behaviour type index Ic", "category": "robertson" }
  ],
  "include_thickness": true,
  "extra_derived": ["Rf", "log_qc"],
  "representation": {
    "type": "resample",
    "length": 16,
    "include_derivatives": false
  }
}
'@ | Set-Content "$featureSetsDir\fc_resample16_core.json" -Encoding UTF8

@'
{
  "feature_set_id": "fc_resample32_core",
  "display_name": "FC Resample-32 Core",
  "description": "32-point interpolated segment shape representation. Higher-resolution time-series shape test.",
  "columns": [
    { "name": "SCPT_RES", "description": "Cone resistance qc", "category": "raw" },
    { "name": "SCPT_FRES", "description": "Sleeve friction fs", "category": "raw" },
    { "name": "SCPT_PWP2", "description": "Porewater pressure u2", "category": "raw" },
    { "name": "SCPT_NQT", "description": "Normalised cone resistance Qtn", "category": "robertson" },
    { "name": "SCPT_NFR", "description": "Normalised friction ratio Fr", "category": "robertson" },
    { "name": "SCPT_ICBE", "description": "Soil behaviour type index Ic", "category": "robertson" }
  ],
  "include_thickness": true,
  "extra_derived": ["Rf", "log_qc"],
  "representation": {
    "type": "resample",
    "length": 32,
    "include_derivatives": false
  }
}
'@ | Set-Content "$featureSetsDir\fc_resample32_core.json" -Encoding UTF8

@'
{
  "feature_set_id": "fc_hybrid24_core",
  "display_name": "FC Hybrid-24 Core",
  "description": "Full hybrid experimental representation. Included for comparison, not expected to be the strongest model.",
  "columns": [
    { "name": "SCPT_RES", "description": "Cone resistance qc", "category": "raw" },
    { "name": "SCPT_FRES", "description": "Sleeve friction fs", "category": "raw" },
    { "name": "SCPT_PWP2", "description": "Porewater pressure u2", "category": "raw" },
    { "name": "SCPT_NQT", "description": "Normalised cone resistance Qtn", "category": "robertson" },
    { "name": "SCPT_NFR", "description": "Normalised friction ratio Fr", "category": "robertson" },
    { "name": "SCPT_ICBE", "description": "Soil behaviour type index Ic", "category": "robertson" }
  ],
  "segment_stats": ["mean", "std", "median", "min", "max"],
  "include_thickness": true,
  "extra_derived": ["Rf", "log_qc"],
  "representation": {
    "type": "hybrid",
    "length": 24,
    "include_derivatives": true,
    "include_shape": true,
    "include_derivative_shape": true,
    "include_distribution": true,
    "include_engineered": true,
    "include_geometry": true,
    "include_coordinates": false
  }
}
'@ | Set-Content "$featureSetsDir\fc_hybrid24_core.json" -Encoding UTF8