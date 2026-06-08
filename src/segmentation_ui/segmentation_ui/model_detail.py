"""
Model detail page – shows model config + runs table + run button.
"""
from __future__ import annotations

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import backend


class ModelDetailPage(QWidget):
    def __init__(self, on_back, on_run_selected):
        super().__init__()
        self._on_back = on_back
        self._on_run_selected = on_run_selected
        self._model_id: str = ""
        self._model_ref: str = ""
        self._process: QProcess | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.clicked.connect(self._on_back)
        header.addWidget(btn_back)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(self.title_label)
        header.addStretch()

        # Boundary source selector
        header.addWidget(QLabel("Dataset:"))
        self.dataset_profile_combo = QComboBox()
        self.dataset_profile_combo.addItem("Default", "default")
        self.dataset_profile_combo.addItem("New location (2_* files)", "new_location")
        header.addWidget(self.dataset_profile_combo)

        header.addWidget(QLabel("Boundary source:"))
        self.boundary_source_combo = QComboBox()
        self.boundary_source_combo.addItem("Both", "both")
        self.boundary_source_combo.addItem("Ground truth", "ground_truth")
        self.boundary_source_combo.addItem("BOCPD estimate", "bocpd_estimate")
        self.boundary_source_combo.addItem("Perfect recall", "perfect_recall")
        self.boundary_source_combo.currentIndexChanged.connect(self._sync_boundary_controls)
        header.addWidget(self.boundary_source_combo)

        # Boundary file selector
        header.addWidget(QLabel("Boundary file:"))
        self.boundary_combo = QComboBox()
        self.boundary_combo.setMinimumWidth(200)
        header.addWidget(self.boundary_combo)

        btn_run = QPushButton("▶ Run Pipeline")
        btn_run.setObjectName("accent")
        btn_run.clicked.connect(self._run_pipeline)
        header.addWidget(btn_run)

        layout.addLayout(header)

        # Config info
        self.config_label = QLabel()
        self.config_label.setStyleSheet("color: #a6adc8; margin: 4px 0;")
        layout.addWidget(self.config_label)

        # Runs table
        group = QGroupBox("Runs")
        group_layout = QVBoxLayout(group)
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Boundaries", "Boundary File", "Clusters", "Accuracy", "F1 Macro", "NMI", "ARI", ""])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(48)
        self.table.setColumnWidth(0, 135)
        self.table.setColumnWidth(2, 280)
        group_layout.addWidget(self.table)
        layout.addWidget(group)

        # Console output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        self.console.setStyleSheet("background-color: #11111b; color: #a6e3a1; font-family: Consolas;")
        self.console.setVisible(False)
        layout.addWidget(self.console)

    def load_model(self, model_id: str):
        model = backend.load_model(model_id)
        self._model_id = model.get("model_id", model_id)
        self._model_ref = model.get("_relative_path", f"{self._model_id}.json")
        self.title_label.setText(f"{model.get('display_name', model_id)}")
        params = model.get("parameters", {})
        fsid = model.get("feature_set_id", "-")
        folder = model.get("_folder", "")
        representation = model.get("_representation_type", "summary")
        repr_config = model.get("_representation_config_label", "")
        include_derivatives = "on" if model.get("_include_derivatives", False) else "off"
        selected_columns = model.get("_selected_columns_count", 0)
        mrf = "on" if model.get("use_spatial_mrf", False) else "off"
        pipeline_type = model.get("pipeline_type", "segment")
        folder_str = f"Folder: {folder or '(root)'}  |  "
        info = (
            f"{folder_str}"
            f"Type: {model.get('model_type', '?')}  |  "
            f"Pipeline: {pipeline_type}  |  "
            f"Feature Set: {fsid}  |  "
            f"Representation: {representation} ({repr_config})  |  "
            f"Derivatives: {include_derivatives}  |  "
            f"Columns: {selected_columns}  |  "
            f"Spatial MRF: {mrf}  |  "
            f"Clusters: {params.get('min_clusters', '?')}--{params.get('max_clusters', '?')}  |  "
            f"Seed: {params.get('random_state', '?')}"
        )
        self.config_label.setText(info)

        # Populate boundary file dropdown
        self.boundary_combo.clear()
        self.boundary_combo.addItem("(default) boundaries/exported_boundaries.csv", "")
        for fname in backend.list_boundary_files():
            if fname != "exported_boundaries.csv":
                self.boundary_combo.addItem(fname, fname)

        self._sync_boundary_controls()

        self._refresh_runs()

    def _sync_boundary_controls(self):
        source = self.boundary_source_combo.currentData()
        use_bocpd = source in {"both", "bocpd_estimate"}
        self.boundary_combo.setEnabled(use_bocpd)

    def _refresh_runs(self):
        runs = backend.list_runs(self._model_id)
        self.table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            m = run.get("metrics", {})
            self.table.setItem(row, 0, QTableWidgetItem(run.get("_timestamp", "?")))
            self.table.setItem(row, 1, QTableWidgetItem(run.get("boundary_source", "?")))
            self.table.setItem(row, 2, QTableWidgetItem(run.get("boundary_file", "-")))
            self.table.setItem(row, 3, QTableWidgetItem(str(run.get("n_clusters", "?"))))
            self.table.setItem(row, 4, QTableWidgetItem(f"{m.get('accuracy', 0):.4f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{m.get('f1_macro', 0):.4f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{m.get('nmi', 0):.4f}"))
            self.table.setItem(row, 7, QTableWidgetItem(f"{m.get('ari', 0):.4f}"))

            btn = QPushButton("View")
            btn.clicked.connect(lambda checked, rd=run["_run_dir"]: self._on_run_selected(rd))
            self.table.setCellWidget(row, 8, btn)

    def _run_pipeline(self):
        self.console.clear()
        self.console.setVisible(True)

        boundary_source = self.boundary_source_combo.currentData() or "both"
        dataset_profile = self.dataset_profile_combo.currentData() or "default"
        boundary_file = self.boundary_combo.currentData() or ""
        label = self.boundary_combo.currentText()
        if boundary_source == "both":
            self.console.append(f"Running pipeline for model: {self._model_id} (all boundary sources)")
        else:
            self.console.append(f"Running pipeline for model: {self._model_id} ({boundary_source})")
        self.console.append(f"  Dataset profile: {dataset_profile}")
        if boundary_file and boundary_source in {"both", "bocpd_estimate"}:
            self.console.append(f"  BOCPD boundary file: {label}")

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
            self._model_ref,
            "--boundary-source",
            boundary_source,
            "--dataset-profile",
            dataset_profile,
        ]
        if boundary_file and boundary_source in {"both", "bocpd_estimate"}:
            args += ["--boundary-file", boundary_file]
        self._process.start("pixi", args)

    def _read_output(self):
        if self._process:
            data = self._process.readAllStandardOutput().data().decode()
            self.console.append(data.rstrip())

    def _on_finished(self, exit_code, exit_status):
        if exit_code == 0:
            self.console.append("\n✓ Pipeline finished successfully.")
        else:
            self.console.append(f"\n✗ Pipeline exited with code {exit_code}.")
        self._refresh_runs()
