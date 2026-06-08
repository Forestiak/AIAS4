"""
Dashboard page - lists all models, grouped by folder, and highlights the global best by NMI.
"""
from __future__ import annotations

from collections import defaultdict

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import backend

GOLD = QColor("#f9e2af")
GOLD_BG = QColor(249, 226, 175, 30)
FOLDER_BG = QColor("#232336")
FOLDER_FG = QColor("#cdd6f4")


class DashboardPage(QWidget):
    def __init__(self, on_model_selected, on_create_model, on_create_folder, on_manage_feature_sets, on_run_models, on_view_results):
        super().__init__()
        self._on_model_selected = on_model_selected
        self._on_create_model = on_create_model
        self._on_create_folder = on_create_folder
        self._on_manage_feature_sets = on_manage_feature_sets
        self._on_run_models = on_run_models
        self._on_view_results = on_view_results
        self._row_model_ids: dict[int, str] = {}
        self._folder_rows: dict[int, str] = {}
        self._folder_children: dict[str, list[int]] = {}
        self._collapsed_folders: dict[str, bool] = {}
        self._model_checks: dict[str, QCheckBox] = {}
        self._folder_checks: dict[str, QCheckBox] = {}
        self._updating_checks = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        title = QLabel("Models")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        btn_new = QPushButton("+ New Model")
        btn_new.setObjectName("accent")
        btn_new.clicked.connect(self._on_create_model)
        header.addWidget(btn_new)

        btn_folder = QPushButton("+ New Folder")
        btn_folder.clicked.connect(self._on_create_folder)
        header.addWidget(btn_folder)

        btn_fs = QPushButton("Feature Sets")
        btn_fs.clicked.connect(self._on_manage_feature_sets)
        header.addWidget(btn_fs)

        btn_results = QPushButton("All Results")
        btn_results.clicked.connect(self._on_view_results)
        header.addWidget(btn_results)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        header.addWidget(btn_refresh)

        btn_run_selected = QPushButton("Run Selected")
        btn_run_selected.setObjectName("accent")
        btn_run_selected.clicked.connect(self._run_selected_models)
        header.addWidget(btn_run_selected)

        layout.addLayout(header)

        self.best_label = QLabel()
        self.best_label.setWordWrap(True)
        self.best_label.setStyleSheet(
            "background-color: #181825; padding: 10px; border-radius: 6px; border-left: 3px solid #f9e2af;"
        )
        layout.addWidget(self.best_label)

        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels(
            ["Run", "Model ID", "Type", "Feature Set", "Representation", "Size", "Derivatives", "Columns", "MRF", "Best NMI", ""]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(48)
        self.table.setColumnWidth(0, 44)
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        layout.addWidget(self.table)

    def refresh(self):
        models = backend.list_models()
        rows_data: list[tuple[dict, dict | None]] = []
        for model in models:
            model_id = model.get("model_id", "?")
            rows_data.append((model, backend.best_run(model_id)))

        global_best_nmi = -1.0
        global_best_id = ""
        for model, best in rows_data:
            if best and "metrics" in best:
                nmi = best["metrics"].get("nmi", 0.0)
                if nmi > global_best_nmi:
                    global_best_nmi = nmi
                    global_best_id = model.get("model_id", "")

        if global_best_id:
            best_data = next(best for model, best in rows_data if model.get("model_id") == global_best_id)
            assert best_data is not None
            metrics = best_data["metrics"]
            boundary_file = best_data.get("boundary_file", "")
            boundary_suffix = f"  |  File: <b>{boundary_file}</b>" if boundary_file and boundary_file != "(default)" else ""
            self.best_label.setText(
                f"<b>Global Best:</b> {global_best_id}  |  "
                f"NMI: <b>{metrics.get('nmi', 0):.4f}</b>  |  "
                f"F1: <b>{metrics.get('f1_macro', 0):.4f}</b>  |  "
                f"Acc: <b>{metrics.get('accuracy', 0):.4f}</b>  |  "
                f"ARI: <b>{metrics.get('ari', 0):.4f}</b>  |  "
                f"Boundaries: <b>{best_data.get('boundary_source', '?')}</b>{boundary_suffix}"
            )
            self.best_label.setVisible(True)
        else:
            self.best_label.setVisible(False)

        grouped: dict[str, list[tuple[dict, dict | None]]] = defaultdict(list)
        for model, best in rows_data:
            grouped[model.get("_folder", "")].append((model, best))

        total_rows = len(rows_data) + len(grouped)
        self.table.clearSpans()
        self.table.setRowCount(total_rows)
        self._row_model_ids.clear()
        self._folder_rows.clear()
        self._folder_children.clear()
        self._model_checks.clear()
        self._folder_checks.clear()

        folder_font = QFont()
        folder_font.setBold(True)

        row = 0
        for folder in sorted(grouped.keys(), key=lambda item: (item != "", item.lower())):
            folder_label = "Ungrouped" if not folder else folder
            self._collapsed_folders.setdefault(folder, False)
            folder_check = QCheckBox()
            folder_check.setTristate(True)
            folder_check.setStyleSheet("margin-left: 12px;")
            folder_check.checkStateChanged.connect(lambda state, folder_name=folder: self._on_folder_check_changed(folder_name, state))
            self.table.setCellWidget(row, 0, folder_check)
            self._folder_checks[folder] = folder_check

            folder_item = QTableWidgetItem()
            folder_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            folder_item.setFont(folder_font)
            folder_item.setBackground(QBrush(FOLDER_BG))
            folder_item.setForeground(QBrush(FOLDER_FG))
            self.table.setItem(row, 1, folder_item)
            for col in range(2, self.table.columnCount()):
                filler = QTableWidgetItem("")
                filler.setFlags(Qt.ItemFlag.ItemIsEnabled)
                filler.setBackground(QBrush(FOLDER_BG))
                filler.setForeground(QBrush(FOLDER_FG))
                self.table.setItem(row, col, filler)
            self._folder_rows[row] = folder
            self._folder_children[folder] = []
            self._set_folder_label(row, folder_label, len(grouped[folder]))
            row += 1

            for model, best in grouped[folder]:
                model_id = model.get("model_id", "?")
                is_best = model_id == global_best_id
                metrics = best.get("metrics", {}) if best else {}

                checkbox = QCheckBox()
                checkbox.setStyleSheet("margin-left: 12px;")
                checkbox.checkStateChanged.connect(lambda state, folder_name=folder: self._sync_folder_checkbox(folder_name))
                self.table.setCellWidget(row, 0, checkbox)
                self._model_checks[model_id] = checkbox

                model_item = QTableWidgetItem(model_id)
                model_item.setToolTip(model.get("_relative_path", model_id))
                self.table.setItem(row, 1, model_item)
                self.table.setItem(row, 2, QTableWidgetItem(model.get("model_type", "?")))
                self.table.setItem(row, 3, QTableWidgetItem(model.get("feature_set_id", "-")))
                self.table.setItem(row, 4, QTableWidgetItem(model.get("_representation_type", "summary")))
                self.table.setItem(row, 5, QTableWidgetItem(model.get("_representation_size_label", "-")))
                self.table.setItem(row, 6, QTableWidgetItem("yes" if model.get("_include_derivatives", False) else "no"))
                self.table.setItem(row, 7, QTableWidgetItem(str(model.get("_selected_columns_count", 0))))
                self.table.setItem(row, 8, QTableWidgetItem("on" if model.get("use_spatial_mrf", False) else "off"))
                self.table.setItem(row, 9, QTableWidgetItem(f"{metrics.get('nmi', 0):.4f}" if metrics else "-"))

                btn = QPushButton("Open")
                btn.clicked.connect(lambda checked, mid=model_id: self._on_model_selected(mid))
                self.table.setCellWidget(row, 10, btn)
                self._row_model_ids[row] = model_id
                self._folder_children[folder].append(row)

                if is_best:
                    for col in range(1, self.table.columnCount() - 1):
                        item = self.table.item(row, col)
                        if item:
                            item.setBackground(QBrush(GOLD_BG))
                            item.setForeground(QBrush(GOLD))

                row += 1

            self._sync_folder_checkbox(folder)

        self._apply_folder_visibility()

    def _set_folder_label(self, row: int, folder_label: str, count: int):
        item = self.table.item(row, 1)
        if item is None:
            return
        folder = self._folder_rows.get(row, "")
        marker = ">" if self._collapsed_folders.get(folder, False) else "v"
        item.setText(f"{marker}  {folder_label} ({count})")

    def _apply_folder_visibility(self):
        for row, folder in self._folder_rows.items():
            collapsed = self._collapsed_folders.get(folder, False)
            self.table.setRowHidden(row, False)
            for child_row in self._folder_children.get(folder, []):
                self.table.setRowHidden(child_row, collapsed)

    def _toggle_folder(self, row: int):
        folder = self._folder_rows[row]
        self._collapsed_folders[folder] = not self._collapsed_folders.get(folder, False)
        folder_label = "Ungrouped" if not folder else folder
        self._set_folder_label(row, folder_label, len(self._folder_children.get(folder, [])))
        self._apply_folder_visibility()

    def _on_cell_clicked(self, row: int, column: int):
        if row in self._folder_rows:
            if column != 0:
                self._toggle_folder(row)

    def _on_cell_double_clicked(self, row: int, column: int):
        if row in self._folder_rows:
            return
        model_id = self._row_model_ids.get(row)
        if model_id:
            self._on_model_selected(model_id)

    def _run_selected_models(self):
        selected = [model_id for model_id, checkbox in self._model_checks.items() if checkbox.isChecked()]
        self._on_run_models(selected)

    def _on_folder_check_changed(self, folder: str, state: int):
        if self._updating_checks:
            return
        check_state = Qt.CheckState(state)
        if check_state == Qt.CheckState.PartiallyChecked:
            return
        self._updating_checks = True
        try:
            for row in self._folder_children.get(folder, []):
                model_id = self._row_model_ids.get(row)
                if not model_id:
                    continue
                checkbox = self._model_checks.get(model_id)
                if checkbox is not None:
                    checkbox.setChecked(check_state == Qt.CheckState.Checked)
        finally:
            self._updating_checks = False
        self._sync_folder_checkbox(folder)

    def _sync_folder_checkbox(self, folder: str):
        folder_checkbox = self._folder_checks.get(folder)
        if folder_checkbox is None:
            return
        child_checks = []
        for row in self._folder_children.get(folder, []):
            model_id = self._row_model_ids.get(row)
            if model_id and model_id in self._model_checks:
                child_checks.append(self._model_checks[model_id])
        if not child_checks:
            return

        checked_count = sum(1 for checkbox in child_checks if checkbox.isChecked())
        self._updating_checks = True
        try:
            if checked_count == 0:
                folder_checkbox.setCheckState(Qt.CheckState.Unchecked)
            elif checked_count == len(child_checks):
                folder_checkbox.setCheckState(Qt.CheckState.Checked)
            else:
                folder_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        finally:
            self._updating_checks = False

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
