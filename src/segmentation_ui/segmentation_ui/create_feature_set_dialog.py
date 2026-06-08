"""
Create / edit feature-set dialog.
Supports both legacy summary-stat feature sets and the newer shape
representations driven by the shared JSON schema.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import constants as C


class CreateFeatureSetDialog(QDialog):
    def __init__(self, parent=None, existing: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("New Feature Set" if existing is None else "Edit Feature Set")
        self.setMinimumSize(560, 640)
        self.result_definition: dict | None = None
        self._existing = existing
        self._col_checks: dict[str, QCheckBox] = {}
        self._stat_checks: dict[str, QCheckBox] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("e.g. raw_plus_robertson")
        form.addRow("Feature Set ID:", self.id_edit)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Raw + Robertson")
        form.addRow("Display Name:", self.name_edit)

        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Description of this feature set...")
        self.desc_edit.setMaximumHeight(60)
        form.addRow("Description:", self.desc_edit)
        layout.addLayout(form)

        col_group = QGroupBox("CPT Columns")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_inner = QWidget()
        col_layout = QVBoxLayout(scroll_inner)

        current_category = ""
        for col_name, description, category in C.CPT_COLUMN_CATALOG:
            if category != current_category:
                current_category = category
                cat_label = QLabel(f"-- {category.upper()} --")
                cat_label.setStyleSheet("font-weight: bold; color: #89b4fa; margin-top: 6px;")
                col_layout.addWidget(cat_label)

            cb = QCheckBox(f"{col_name} - {description}")
            self._col_checks[col_name] = cb
            col_layout.addWidget(cb)

        col_layout.addStretch()
        scroll.setWidget(scroll_inner)

        group_layout = QVBoxLayout(col_group)
        group_layout.addWidget(scroll)
        layout.addWidget(col_group)

        self.stat_group = QGroupBox("Segment Statistics")
        stat_layout = QHBoxLayout(self.stat_group)
        for stat in C.SEGMENT_STATS_OPTIONS:
            cb = QCheckBox(stat)
            if stat in ("mean", "std", "median"):
                cb.setChecked(True)
            self._stat_checks[stat] = cb
            stat_layout.addWidget(cb)
        layout.addWidget(self.stat_group)

        repr_group = QGroupBox("Representation")
        repr_form = QFormLayout(repr_group)

        self.repr_combo = QComboBox()
        for key, label in C.FEATURE_REPRESENTATION_OPTIONS:
            self.repr_combo.addItem(label, key)
        self.repr_combo.currentIndexChanged.connect(self._sync_representation_ui)
        repr_form.addRow("Type:", self.repr_combo)

        self.fft_cb = QCheckBox("Use FFT representation")
        self.fft_cb.toggled.connect(self._sync_representation_ui)
        repr_form.addRow("", self.fft_cb)

        self.length_label = QLabel("Resample Length:")
        self.length_spin = QSpinBox()
        self.length_spin.setRange(1, 512)
        self.length_spin.setValue(32)
        repr_form.addRow(self.length_label, self.length_spin)

        self.bins_label = QLabel("PAA Bins:")
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(1, 128)
        self.bins_spin.setValue(8)
        repr_form.addRow(self.bins_label, self.bins_spin)

        self.derivatives_cb = QCheckBox("Include first-derivative features")
        repr_form.addRow("", self.derivatives_cb)
        layout.addWidget(repr_group)

        self.thickness_cb = QCheckBox("Include segment thickness as feature")
        self.thickness_cb.setChecked(True)
        layout.addWidget(self.thickness_cb)

        btns = QHBoxLayout()
        btn_ok = QPushButton("Create" if self._existing is None else "Save")
        btn_ok.setObjectName("accent")
        btn_ok.clicked.connect(self._accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)
        layout.addLayout(btns)

        if self._existing:
            self._populate(self._existing)
        else:
            self._sync_representation_ui()

    def _populate(self, definition: dict):
        self.id_edit.setText(definition.get("feature_set_id", ""))
        self.id_edit.setReadOnly(True)
        self.name_edit.setText(definition.get("display_name", ""))
        self.desc_edit.setPlainText(definition.get("description", ""))

        selected_cols = {column["name"] for column in definition.get("columns", [])}
        for col_name, cb in self._col_checks.items():
            cb.setChecked(col_name in selected_cols)

        selected_stats = set(definition.get("segment_stats", []))
        for stat, cb in self._stat_checks.items():
            cb.setChecked(stat in selected_stats)

        representation = definition.get("representation") or {}
        representation_type = representation.get("type", "summary")
        combo_representation_type = "summary" if representation_type == "fft" else representation_type
        idx = self.repr_combo.findData(combo_representation_type)
        if idx >= 0:
            self.repr_combo.setCurrentIndex(idx)
        self.fft_cb.setChecked(representation_type == "fft")
        self.length_spin.setValue(int(representation.get("length", 32) or 32))
        self.bins_spin.setValue(int(representation.get("bins", 8) or 8))
        self.derivatives_cb.setChecked(bool(representation.get("include_derivatives", False)))
        self.thickness_cb.setChecked(definition.get("include_thickness", True))
        self._sync_representation_ui()

    def _sync_representation_ui(self):
        representation_type = self.repr_combo.currentData()
        is_fft = self.fft_cb.isChecked()
        is_summary = representation_type == "summary"
        is_resample = representation_type == "resample"
        is_paa = representation_type == "paa"

        self.repr_combo.setEnabled(not is_fft)
        self.stat_group.setVisible(is_summary and not is_fft)
        self.length_label.setVisible(is_resample and not is_fft)
        self.length_spin.setVisible(is_resample and not is_fft)
        self.bins_label.setVisible(is_paa and not is_fft)
        self.bins_spin.setVisible(is_paa and not is_fft)
        self.derivatives_cb.setVisible((not is_summary) and (not is_fft))

    def _accept(self):
        feature_set_id = self.id_edit.text().strip()
        if not feature_set_id:
            return

        columns = []
        catalog_map = {name: (description, category) for name, description, category in C.CPT_COLUMN_CATALOG}
        for col_name, cb in self._col_checks.items():
            if cb.isChecked():
                description, category = catalog_map[col_name]
                columns.append({"name": col_name, "description": description, "category": category})

        if not columns:
            return

        definition = {
            "feature_set_id": feature_set_id,
            "display_name": self.name_edit.text().strip() or feature_set_id,
            "description": self.desc_edit.toPlainText().strip(),
            "columns": columns,
            "include_thickness": self.thickness_cb.isChecked(),
        }

        representation_type = self.repr_combo.currentData()
        if self.fft_cb.isChecked():
            definition["representation"] = {
                "type": "fft",
                "depth_step_m": 0.02,
                "min_points_per_segment": 8,
                "min_frequency_cycles_per_m": 0.25,
                "max_frequency_cycles_per_m": 8.0,
                "bands_cycles_per_m": [
                    [0.25, 1.0],
                    [1.0, 2.0],
                    [2.0, 4.0],
                    [4.0, 8.0],
                ],
                "detrend": True,
                "window": "hann",
            }
        elif representation_type == "summary":
            stats = [stat for stat, cb in self._stat_checks.items() if cb.isChecked()]
            if not stats:
                return
            definition["segment_stats"] = stats
        elif representation_type == "resample":
            definition["representation"] = {
                "type": "resample",
                "length": self.length_spin.value(),
                "include_derivatives": self.derivatives_cb.isChecked(),
            }
        else:
            definition["representation"] = {
                "type": "paa",
                "bins": self.bins_spin.value(),
                "include_derivatives": self.derivatives_cb.isChecked(),
            }

        self.result_definition = definition
        self.accept()
