"""
Feature Sets management page – list, create, edit, delete feature sets.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import backend
from .create_feature_set_dialog import CreateFeatureSetDialog


class FeatureSetsPage(QWidget):
    def __init__(self, on_back):
        super().__init__()
        self._on_back = on_back
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.clicked.connect(self._on_back)
        header.addWidget(btn_back)

        title = QLabel("Feature Sets")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        btn_new = QPushButton("+ New Feature Set")
        btn_new.setObjectName("accent")
        btn_new.clicked.connect(self._create)
        header.addWidget(btn_new)

        layout.addLayout(header)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["ID", "Display Name", "Columns", "Representation", "Config", "Description", ""])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(48)
        layout.addWidget(self.table)

    def refresh(self):
        sets = backend.list_feature_sets()
        self.table.setRowCount(len(sets))
        for row, fs in enumerate(sets):
            fsid = fs.get("feature_set_id", "?")
            cols = [c["name"] for c in fs.get("columns", [])]
            self.table.setItem(row, 0, QTableWidgetItem(fsid))
            self.table.setItem(row, 1, QTableWidgetItem(fs.get("display_name", "")))
            self.table.setItem(row, 2, QTableWidgetItem(", ".join(cols)))
            self.table.setItem(row, 3, QTableWidgetItem(fs.get("_representation_type", "summary")))
            self.table.setItem(row, 4, QTableWidgetItem(fs.get("_representation_config_label", "")))
            self.table.setItem(row, 5, QTableWidgetItem(fs.get("description", "")))

            btns = QWidget()
            btn_layout = QHBoxLayout(btns)
            btn_layout.setContentsMargins(2, 2, 2, 2)

            btn_edit = QPushButton("Edit")
            btn_edit.clicked.connect(lambda checked, fid=fsid: self._edit(fid))
            btn_layout.addWidget(btn_edit)

            btn_del = QPushButton("Delete")
            btn_del.setStyleSheet("color: #f38ba8;")
            btn_del.clicked.connect(lambda checked, fid=fsid: self._delete(fid))
            btn_layout.addWidget(btn_del)

            self.table.setCellWidget(row, 6, btns)

    def _create(self):
        dialog = CreateFeatureSetDialog(self)
        if dialog.exec() and dialog.result_definition:
            backend.save_feature_set(dialog.result_definition)
            self.refresh()

    def _edit(self, fsid: str):
        try:
            existing = backend.load_feature_set(fsid)
        except Exception:
            return
        dialog = CreateFeatureSetDialog(self, existing=existing)
        if dialog.exec() and dialog.result_definition:
            backend.save_feature_set(dialog.result_definition)
            self.refresh()

    def _delete(self, fsid: str):
        reply = QMessageBox.question(
            self, "Delete Feature Set",
            f"Delete feature set '{fsid}'?\nModels referencing it will need updating.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            backend.delete_feature_set(fsid)
            self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
