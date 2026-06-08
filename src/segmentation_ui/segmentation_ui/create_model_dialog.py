"""
Create / edit model dialog.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import backend


class CreateModelDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, existing: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("New Model" if existing is None else "Edit Model")
        self.setMinimumWidth(480)
        self.result_definition: dict | None = None
        self.result_folder: str = ""
        self._existing = existing
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.folder_combo = QComboBox()
        self.folder_combo.setEditable(True)
        self.folder_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._populate_model_folders()

        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_combo, 1)

        btn_folder = QPushButton("New Folder")
        btn_folder.clicked.connect(self._create_folder)
        folder_row.addWidget(btn_folder)
        form.addRow("Folder:", folder_row)

        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("e.g. gmm_k12_robertson")
        form.addRow("Model ID:", self.id_edit)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. GMM k12 Robertson")
        form.addRow("Display Name:", self.name_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["gmm", "kmeans", "agglomerative"])
        form.addRow("Model Type:", self.type_combo)

        self.fs_combo = QComboBox()
        self._populate_feature_sets()
        form.addRow("Feature Set:", self.fs_combo)

        self.fs_info = QLabel("")
        self.fs_info.setWordWrap(True)
        self.fs_info.setStyleSheet("color: #a6adc8; font-size: 11px; margin-left: 4px;")
        form.addRow("", self.fs_info)
        self.fs_combo.currentIndexChanged.connect(self._on_fs_changed)
        self._on_fs_changed()

        self.min_k = QSpinBox()
        self.min_k.setRange(2, 30)
        self.min_k.setValue(2)
        form.addRow("Min Clusters:", self.min_k)

        self.max_k = QSpinBox()
        self.max_k.setRange(2, 30)
        self.max_k.setValue(8)
        form.addRow("Max Clusters:", self.max_k)

        self.seed = QSpinBox()
        self.seed.setRange(0, 99999)
        self.seed.setValue(42)
        form.addRow("Random State:", self.seed)

        self.thickness = QDoubleSpinBox()
        self.thickness.setRange(0.1, 5.0)
        self.thickness.setSingleStep(0.1)
        self.thickness.setValue(0.5)
        form.addRow("Min Segment (m):", self.thickness)

        layout.addLayout(form)

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

    def _populate_feature_sets(self):
        self.fs_combo.clear()
        for fs in backend.list_feature_sets():
            fsid = fs.get("feature_set_id", "?")
            label = fs.get("display_name", fsid)
            self.fs_combo.addItem(f"{label} ({fsid})", fsid)

    def _populate_model_folders(self, selected: str = ""):
        normalized_selected = backend.normalize_model_folder(selected)
        self.folder_combo.blockSignals(True)
        self.folder_combo.clear()
        self.folder_combo.addItem("(root)", "")
        for folder in backend.list_model_folders():
            self.folder_combo.addItem(folder, folder)

        index = self.folder_combo.findData(normalized_selected)
        if index >= 0:
            self.folder_combo.setCurrentIndex(index)
        else:
            self.folder_combo.setEditText(normalized_selected or "(root)")
        self.folder_combo.blockSignals(False)

    def _selected_folder(self) -> str:
        text = self.folder_combo.currentText().strip()
        return backend.normalize_model_folder(text)

    def _create_folder(self):
        initial = self._selected_folder()
        folder, ok = QInputDialog.getText(
            self,
            "New Model Folder",
            "Folder path under models/:",
            text=initial,
        )
        if not ok:
            return

        try:
            normalized = backend.normalize_model_folder(folder)
            if not normalized:
                raise ValueError("Folder name cannot be empty.")
            backend.create_model_folder(normalized)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Folder", str(exc))
            return

        self._populate_model_folders(normalized)

    def _on_fs_changed(self):
        fsid = self.fs_combo.currentData()
        if not fsid:
            self.fs_info.setText("")
            return
        try:
            fs = backend.load_feature_set(fsid)
            cols = [c["name"] for c in fs.get("columns", [])]
            desc = fs.get("description", "")
            representation = fs.get("_representation_type", "summary")
            config_label = fs.get("_representation_config_label", "")
            derivatives = "yes" if fs.get("_include_derivatives", False) else "no"
            self.fs_info.setText(
                f"{desc}\n"
                f"Columns: {', '.join(cols)}\n"
                f"Representation: {representation} ({config_label})\n"
                f"Derivatives: {derivatives}"
            )
        except Exception:
            self.fs_info.setText("")

    def _populate(self, d: dict):
        self._populate_model_folders(d.get("_folder", d.get("model_folder", "")))
        self.id_edit.setText(d.get("model_id", ""))
        self.id_edit.setReadOnly(True)
        self.name_edit.setText(d.get("display_name", ""))
        idx = self.type_combo.findText(d.get("model_type", "gmm"))
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)

        fsid = d.get("feature_set_id", "raw_sensors")
        for i in range(self.fs_combo.count()):
            if self.fs_combo.itemData(i) == fsid:
                self.fs_combo.setCurrentIndex(i)
                break

        params = d.get("parameters", {})
        self.min_k.setValue(params.get("min_clusters", 2))
        self.max_k.setValue(params.get("max_clusters", 8))
        self.seed.setValue(params.get("random_state", 42))
        self.thickness.setValue(d.get("min_segment_thickness_m", 0.5))

    def _accept(self):
        mid = self.id_edit.text().strip()
        if not mid:
            QMessageBox.warning(self, "Missing Model ID", "Model ID is required.")
            return

        try:
            folder = self._selected_folder()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Folder", str(exc))
            return

        fsid = self.fs_combo.currentData()
        if not fsid:
            QMessageBox.warning(self, "Missing Feature Set", "Select a feature set for the model.")
            return

        self.result_definition = {
            "model_id": mid,
            "model_type": self.type_combo.currentText(),
            "display_name": self.name_edit.text().strip() or mid,
            "feature_set_id": fsid,
            "parameters": {
                "min_clusters": self.min_k.value(),
                "max_clusters": self.max_k.value(),
                "random_state": self.seed.value(),
            },
            "min_segment_thickness_m": self.thickness.value(),
        }
        self.result_folder = folder
        self.accept()
