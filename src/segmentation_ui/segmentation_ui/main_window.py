"""
Main window - stacked pages: Dashboard -> Model Detail -> Run Detail, plus Feature Sets page.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QInputDialog, QMainWindow, QMessageBox, QStackedWidget

from . import backend
from .constants import DARK_STYLE
from .create_model_dialog import CreateModelDialog
from .dashboard import DashboardPage
from .feature_sets_page import FeatureSetsPage
from .model_detail import ModelDetailPage
from .multi_run_dialog import MultiRunDialog
from .results_page import ResultsPage
from .run_detail import RunDetailPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CPT Segmentation Manager")
        self.setMinimumSize(900, 650)
        self.setStyleSheet(DARK_STYLE)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.dashboard = DashboardPage(
            on_model_selected=self._open_model,
            on_create_model=self._create_model,
            on_create_folder=self._create_model_folder,
            on_manage_feature_sets=self._open_feature_sets,
            on_run_models=self._run_models,
            on_view_results=self._open_results,
        )
        self.model_page = ModelDetailPage(
            on_back=self._go_dashboard,
            on_run_selected=self._open_run,
        )
        self.run_page = RunDetailPage(
            on_back=self._go_back_from_run,
        )
        self.feature_sets_page = FeatureSetsPage(
            on_back=self._go_dashboard,
        )
        self.results_page = ResultsPage(
            on_back=self._go_dashboard,
            on_run_selected=self._open_run_from_results,
        )

        self.stack.addWidget(self.dashboard)
        self.stack.addWidget(self.model_page)
        self.stack.addWidget(self.run_page)
        self.stack.addWidget(self.feature_sets_page)
        self.stack.addWidget(self.results_page)

        self._current_model_id: str = ""
        self._return_page_index = 0

    def _go_dashboard(self):
        self._return_page_index = 0
        self.stack.setCurrentIndex(0)
        self.dashboard.refresh()

    def _open_model(self, model_id: str):
        self._current_model_id = model_id
        self.model_page.load_model(model_id)
        self._return_page_index = 1
        self.stack.setCurrentIndex(1)

    def _open_run(self, run_dir: str):
        self._return_page_index = 1
        self.run_page.load_run(run_dir)
        self.stack.setCurrentIndex(2)

    def _open_run_from_results(self, run_dir: str):
        self._return_page_index = 4
        self.run_page.load_run(run_dir)
        self.stack.setCurrentIndex(2)

    def _go_back_from_run(self):
        if self._return_page_index == 4:
            self._open_results()
        elif self._current_model_id:
            self._open_model(self._current_model_id)
        else:
            self._go_dashboard()

    def _open_feature_sets(self):
        self.feature_sets_page.refresh()
        self.stack.setCurrentIndex(3)

    def _open_results(self):
        self.results_page.refresh()
        self._return_page_index = 4
        self.stack.setCurrentIndex(4)

    def _create_model_folder(self):
        folder, ok = QInputDialog.getText(self, "New Model Folder", "Folder path under models/:")
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
        self.dashboard.refresh()

    def _create_model(self):
        dialog = CreateModelDialog(self)
        if dialog.exec() and dialog.result_definition:
            try:
                backend.save_model(dialog.result_definition, folder=dialog.result_folder)
            except (FileExistsError, ValueError) as exc:
                QMessageBox.warning(self, "Could Not Save Model", str(exc))
                return
            self.dashboard.refresh()

    def _run_models(self, model_ids: list[str]):
        if not model_ids:
            QMessageBox.information(self, "No Models Selected", "Select at least one model to run.")
            return
        dialog = MultiRunDialog(model_ids, self)
        dialog.exec()
        self.dashboard.refresh()
