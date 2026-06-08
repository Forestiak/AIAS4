"""
Paths and constants shared across the UI.
Points to the segmentation_v2 project on disk.
"""
from pathlib import Path

# The core pipeline lives one folder up from segmentation_ui
CORE_DIR = Path(__file__).resolve().parent.parent.parent / "segmentation_v2"
MODELS_DIR = CORE_DIR / "models"
FEATURE_SETS_DIR = CORE_DIR / "feature_sets"
LOGS_DIR = CORE_DIR / "outputs" / "logs"
PIPELINE_SCRIPT = CORE_DIR / "pipeline.py"
PIXI_EXE = "pixi"

# ── CPT column catalog (for the feature-set column picker) ───────────
# Each entry: (column_name, description, category)
CPT_COLUMN_CATALOG: list[tuple[str, str, str]] = [
    ("SCPT_RES",  "Cone resistance qc (MPa)", "raw"),
    ("SCPT_FRES", "Sleeve friction fs (MPa)", "raw"),
    ("SCPT_PWP2", "Shoulder porewater pressure u2 (MPa)", "raw"),
    ("SCPT_QT",   "Corrected cone resistance qt (MPa)", "derived"),
    ("SCPT_QNET", "Net cone resistance qn (MPa)", "derived"),
    ("SCPT_FRR",  "Friction ratio Rf (%)", "derived"),
    ("SCPT_NFR",  "Normalised friction ratio Fr (%)", "robertson"),
    ("SCPT_BQ",   "Pore pressure ratio Bq (-)", "robertson"),
    ("SCPT_NU2",  "Normalized pore pressure U2 (-)", "robertson"),
    ("delta_u2",  "Excess penetration pore pressure delta_u2 (MPa)", "derived"),
    ("SCPT_NQT",  "Normalised cone resistance Qtn (-)", "robertson"),
    ("SCPT_ICBE", "Soil behaviour type index Ic (-)", "robertson"),
    ("n_exp",     "Stress exponent n (-)", "robertson"),
    ("SCPT_CPO",  "Total vertical stress (kPa)", "stress"),
    ("SCPT_CPOD", "Effective vertical stress (kPa)", "stress"),
    ("SCPT_ISPP", "Hydrostatic pressure (MPa)", "stress"),
]

SEGMENT_STATS_OPTIONS = ["mean", "std", "median", "min", "max"]
FEATURE_REPRESENTATION_OPTIONS = [
    ("summary", "Summary statistics"),
    ("resample", "Resampled shape"),
    ("paa", "PAA shape"),
    ("row_local", "Row-level local features"),
]

DARK_STYLE = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QLabel {
    color: #cdd6f4;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 16px;
}
QPushButton:hover {
    background-color: #45475a;
}
QPushButton:pressed {
    background-color: #585b70;
}
QPushButton#accent {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
}
QPushButton#accent:hover {
    background-color: #74c7ec;
}
QTableWidget {
    background-color: #181825;
    alternate-background-color: #1e1e2e;
    gridline-color: #313244;
    border: 1px solid #313244;
    border-radius: 6px;
}
QTableWidget::item {
    padding: 6px 4px;
}
QHeaderView::section {
    background-color: #313244;
    color: #cdd6f4;
    padding: 6px;
    border: none;
    font-weight: bold;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
}
QComboBox::drop-down {
    border: none;
}
QScrollBar:vertical {
    background-color: #181825;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
    min-height: 30px;
}
QGroupBox {
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
"""
