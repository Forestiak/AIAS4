"""
Results page - full comparison table across all runs with filters.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
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


class ResultsPage(QWidget):
    def __init__(self, on_back, on_run_selected):
        super().__init__()
        self._on_back = on_back
        self._on_run_selected = on_run_selected
        self._all_runs: list[dict] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.clicked.connect(self._on_back)
        header.addWidget(btn_back)

        title = QLabel("All Results")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        header.addWidget(btn_refresh)
        layout.addLayout(header)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Model:"))
        self.model_filter = QComboBox()
        self.model_filter.currentIndexChanged.connect(self._apply_filters)
        filters.addWidget(self.model_filter)

        filters.addWidget(QLabel("Input file:"))
        self.input_file_filter = QComboBox()
        self.input_file_filter.currentIndexChanged.connect(self._apply_filters)
        filters.addWidget(self.input_file_filter)

        filters.addStretch()
        layout.addLayout(filters)

        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color: #a6adc8; margin: 4px 0;")
        layout.addWidget(self.summary_label)

        self.table = QTableWidget()
        self.table.setColumnCount(13)
        self.table.setHorizontalHeaderLabels(
            [
                "Timestamp",
                "Model ID",
                "Feature Set",
                "Representation",
                "Boundaries",
                "Input File",
                "Clusters",
                "BIC",
                "Accuracy",
                "F1 Macro",
                "F1 Weighted",
                "NMI",
                "ARI",
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 135)
        self.table.setColumnWidth(5, 280)
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(self._open_selected_run)
        layout.addWidget(self.table)

        actions = QHBoxLayout()
        actions.addStretch()
        btn_open = QPushButton("Open Selected Run")
        btn_open.setObjectName("accent")
        btn_open.clicked.connect(self._open_current_run)
        actions.addWidget(btn_open)
        layout.addLayout(actions)

    def refresh(self):
        self._all_runs = backend.list_all_runs()
        self._populate_filters()
        self._apply_filters()

    def _populate_filters(self):
        model_value = self.model_filter.currentData()
        input_file_value = self.input_file_filter.currentData()

        models = sorted({run.get("model_id", "") for run in self._all_runs if run.get("model_id")})
        input_files = sorted({self._input_file_label(run) for run in self._all_runs})

        self.model_filter.blockSignals(True)
        self.input_file_filter.blockSignals(True)
        try:
            self.model_filter.clear()
            self.model_filter.addItem("All models", "")
            for model_id in models:
                self.model_filter.addItem(model_id, model_id)

            self.input_file_filter.clear()
            self.input_file_filter.addItem("All input files", "")
            for input_file in input_files:
                self.input_file_filter.addItem(input_file, input_file)

            self._restore_combo_value(self.model_filter, model_value)
            self._restore_combo_value(self.input_file_filter, input_file_value)
        finally:
            self.model_filter.blockSignals(False)
            self.input_file_filter.blockSignals(False)

    def _restore_combo_value(self, combo: QComboBox, value: str | None):
        if not value:
            combo.setCurrentIndex(0)
            return
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _apply_filters(self):
        model_filter = self.model_filter.currentData() or ""
        input_file_filter = self.input_file_filter.currentData() or ""

        filtered = []
        for run in self._all_runs:
            if model_filter and run.get("model_id") != model_filter:
                continue
            if input_file_filter and self._input_file_label(run) != input_file_filter:
                continue
            filtered.append(run)

        self._populate_table(filtered)
        self.summary_label.setText(f"Showing {len(filtered)} of {len(self._all_runs)} runs")

    def _populate_table(self, runs: list[dict]):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(runs))

        for row, run in enumerate(runs):
            metrics = run.get("metrics", {})
            representation = self._representation_label(run)

            self.table.setItem(row, 0, self._text_item(run.get("_timestamp", "-"), run.get("_run_dir", "")))
            self.table.setItem(row, 1, self._text_item(run.get("model_id", "-")))
            self.table.setItem(row, 2, self._text_item(run.get("feature_set_id", "-")))
            self.table.setItem(row, 3, self._text_item(representation))
            self.table.setItem(row, 4, self._text_item(run.get("boundary_source", "-")))
            self.table.setItem(row, 5, self._text_item(self._input_file_label(run)))
            self.table.setItem(row, 6, self._number_item(run.get("n_clusters")))
            self.table.setItem(row, 7, self._number_item(run.get("bic"), decimals=4))
            self.table.setItem(row, 8, self._number_item(metrics.get("accuracy"), decimals=4))
            self.table.setItem(row, 9, self._number_item(metrics.get("f1_macro"), decimals=4))
            self.table.setItem(row, 10, self._number_item(metrics.get("f1_weighted"), decimals=4))
            self.table.setItem(row, 11, self._number_item(metrics.get("nmi"), decimals=4))
            self.table.setItem(row, 12, self._number_item(metrics.get("ari"), decimals=4))

        self.table.setSortingEnabled(True)
        self.table.sortItems(0, Qt.SortOrder.DescendingOrder)

    def _representation_label(self, run: dict) -> str:
        representation_type = run.get("representation_type", "summary")
        if representation_type == "summary":
            return "summary"
        if representation_type == "resample":
            length = run.get("representation_length")
            suffix = ", deriv" if run.get("include_derivatives") else ""
            return f"resample ({length}{suffix})"
        if representation_type == "paa":
            bins = run.get("representation_bins")
            suffix = ", deriv" if run.get("include_derivatives") else ""
            return f"paa ({bins}{suffix})"
        if representation_type == "fft":
            min_freq = run.get("fft_min_frequency_cycles_per_m", 0.25)
            max_freq = run.get("fft_max_frequency_cycles_per_m", 8.0)
            return f"fft ({min_freq}-{max_freq} cyc/m)"
        if representation_type == "row_local":
            return "row-level local"
        return representation_type

    def _input_file_label(self, run: dict) -> str:
        boundary_file = str(run.get("boundary_file", "") or "").strip()
        return boundary_file if boundary_file else "(default)"

    def _text_item(self, value: str, tooltip: str = "") -> QTableWidgetItem:
        item = QTableWidgetItem(value)
        if tooltip:
            item.setToolTip(tooltip)
        return item

    def _number_item(self, value, decimals: int = 0) -> QTableWidgetItem:
        if value is None or value == "":
            return QTableWidgetItem("-")
        if isinstance(value, int) and decimals == 0:
            item = QTableWidgetItem(str(value))
            item.setData(Qt.ItemDataRole.EditRole, value)
            return item
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return QTableWidgetItem(str(value))

        text = f"{numeric:.{decimals}f}" if decimals else str(int(numeric))
        item = QTableWidgetItem(text)
        item.setData(Qt.ItemDataRole.EditRole, numeric)
        return item

    def _open_current_run(self):
        row = self.table.currentRow()
        if row < 0:
            return
        self._open_selected_run(row, 0)

    def _open_selected_run(self, row: int, _column: int):
        item = self.table.item(row, 0)
        if item is None:
            return
        run_dir = item.toolTip()
        if run_dir:
            self._on_run_selected(run_dir)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()