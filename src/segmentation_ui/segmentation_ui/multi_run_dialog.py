"""
Dialog for running multiple model definitions sequentially and comparing results.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from . import backend


class MultiRunDialog(QDialog):
    def __init__(self, model_ids: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Selected Models")
        self.setMinimumSize(1100, 720)
        self._model_refs = model_ids
        self._models_by_ref = {model_ref: backend.load_model(model_ref) for model_ref in model_ids}
        self._row_by_model_ref = {model_ref: idx for idx, model_ref in enumerate(self._model_refs)}
        self._pending: list[str] = []
        self._current_model_ref: str | None = None
        self._current_model_id: str | None = None
        self._process: QProcess | None = None
        self._build_ui()
        self._populate_table()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        selected_labels = [
            self._models_by_ref[model_ref].get("model_id", model_ref)
            for model_ref in self._model_refs
        ]
        title = QLabel(f"Selected models: {', '.join(selected_labels)}")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        controls = QHBoxLayout()
        form = QFormLayout()

        self.dataset_profile_combo = QComboBox()
        self.dataset_profile_combo.addItem("Default", "default")
        self.dataset_profile_combo.addItem("New location (2_* files)", "new_location")
        form.addRow("Dataset:", self.dataset_profile_combo)

        self.boundary_source_combo = QComboBox()
        self.boundary_source_combo.addItem("Ground truth", "ground_truth")
        self.boundary_source_combo.addItem("BOCPD estimate", "bocpd_estimate")
        self.boundary_source_combo.addItem("Perfect recall", "perfect_recall")
        self.boundary_source_combo.currentIndexChanged.connect(self._sync_boundary_controls)
        form.addRow("Boundary source:", self.boundary_source_combo)

        self.boundary_file_combo = QComboBox()
        self.boundary_file_combo.addItem("(default) boundaries/exported_boundaries.csv", "")
        for fname in backend.list_boundary_files():
            if fname != "exported_boundaries.csv":
                self.boundary_file_combo.addItem(fname, fname)
        form.addRow("Boundary file:", self.boundary_file_combo)
        controls.addLayout(form)
        controls.addStretch()

        self.status_label = QLabel("Ready.")
        controls.addWidget(self.status_label)

        self.start_button = QPushButton("Start Runs")
        self.start_button.setObjectName("accent")
        self.start_button.clicked.connect(self._start_runs)
        controls.addWidget(self.start_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        controls.addWidget(close_button)

        layout.addLayout(controls)

        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels(
            ["Model ID", "Feature Set", "Representation", "Clusters", "BIC", "Accuracy", "F1 Macro", "F1 Weighted", "ARI", "NMI", "Output / Run Folder"]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #11111b; color: #a6e3a1; font-family: Consolas;")
        layout.addWidget(self.console)

        self._sync_boundary_controls()

    def _populate_table(self):
        self.table.setRowCount(len(self._model_refs))
        for row, model_ref in enumerate(self._model_refs):
            model = self._models_by_ref[model_ref]
            self.table.setItem(row, 0, QTableWidgetItem(model.get("model_id", "")))
            self.table.setItem(row, 1, QTableWidgetItem(model.get("feature_set_id", "")))
            representation = model.get("_comparison_representation_label", model.get("_representation_type", "summary"))
            if model.get("_include_derivatives", False):
                representation = f"{representation}, derivatives"
            self.table.setItem(row, 2, QTableWidgetItem(representation))
            for col in range(3, self.table.columnCount()):
                self.table.setItem(row, col, QTableWidgetItem("-"))

    def _sync_boundary_controls(self):
        use_bocpd = self.boundary_source_combo.currentData() == "bocpd_estimate"
        self.boundary_file_combo.setEnabled(use_bocpd)

    def _start_runs(self):
        self.console.clear()
        self._pending = list(self._model_refs)
        self.start_button.setEnabled(False)
        self.dataset_profile_combo.setEnabled(False)
        self.boundary_source_combo.setEnabled(False)
        self.boundary_file_combo.setEnabled(False)
        self._start_next_run()

    def _start_next_run(self):
        if not self._pending:
            self.status_label.setText("All runs finished.")
            self.start_button.setEnabled(True)
            self.dataset_profile_combo.setEnabled(True)
            self.boundary_source_combo.setEnabled(True)
            self.boundary_file_combo.setEnabled(self.boundary_source_combo.currentData() == "bocpd_estimate")
            return

        self._current_model_ref = self._pending.pop(0)
        model = self._models_by_ref[self._current_model_ref]
        self._current_model_id = model.get("model_id", "")
        dataset_profile = self.dataset_profile_combo.currentData() or "default"
        boundary_source = self.boundary_source_combo.currentData()
        boundary_file = self.boundary_file_combo.currentData() or ""

        self.status_label.setText(f"Running {self._current_model_id}...")
        self.console.append(
            f"Running {self._current_model_id} [{boundary_source}] (dataset={dataset_profile})"
        )

        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(backend.C.CORE_DIR))
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._on_finished)

        args = [
            "run",
            "python",
            "pipeline.py",
            "--model",
            self._current_model_ref,
            "--boundary-source",
            boundary_source,
            "--dataset-profile",
            dataset_profile,
        ]
        if boundary_source == "bocpd_estimate" and boundary_file:
            args += ["--boundary-file", boundary_file]
        self._process.start("pixi", args)

    def _read_output(self):
        if self._process is None:
            return
        data = self._process.readAllStandardOutput().data().decode()
        self.console.append(data.rstrip())

    def _on_finished(self, exit_code, exit_status):
        model_ref = self._current_model_ref
        model_id = self._current_model_id
        if model_ref is None or model_id is None:
            return

        if exit_code == 0:
            self.console.append(f"Completed: {model_id}\n")
            self._load_latest_result(model_ref)
        else:
            self.console.append(f"Failed: {model_id} (exit code {exit_code})\n")
            row = self._row_by_model_ref[model_ref]
            self.table.setItem(row, 3, QTableWidgetItem("FAILED"))

        self._process = None
        self._current_model_ref = None
        self._current_model_id = None
        self._start_next_run()

    def _load_latest_result(self, model_ref: str):
        model = self._models_by_ref[model_ref]
        model_id = model.get("model_id", "")
        runs = backend.list_runs(model_id)
        if not runs:
            return

        latest = runs[0]
        metrics = latest.get("metrics", {})
        row = self._row_by_model_ref[model_ref]
        self.table.setItem(row, 3, QTableWidgetItem(str(latest.get("n_clusters", "-"))))
        bic = latest.get("bic")
        self.table.setItem(row, 4, QTableWidgetItem(f"{bic:.4f}" if bic is not None else "-"))
        self.table.setItem(row, 5, QTableWidgetItem(f"{metrics.get('accuracy', 0):.4f}" if metrics else "-"))
        self.table.setItem(row, 6, QTableWidgetItem(f"{metrics.get('f1_macro', 0):.4f}" if metrics else "-"))
        self.table.setItem(row, 7, QTableWidgetItem(f"{metrics.get('f1_weighted', 0):.4f}" if metrics else "-"))
        self.table.setItem(row, 8, QTableWidgetItem(f"{metrics.get('ari', 0):.4f}" if metrics else "-"))
        self.table.setItem(row, 9, QTableWidgetItem(f"{metrics.get('nmi', 0):.4f}" if metrics else "-"))
        self.table.setItem(row, 10, QTableWidgetItem(str(Path(latest.get("_run_dir", "-")))))
