"""
Run detail page – shows metrics, classification report, and profile plots.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import backend


class RunDetailPage(QWidget):
    def __init__(self, on_back):
        super().__init__()
        self._on_back = on_back
        self._profile_plots: list[str] = []
        self._robertson_plots: list[str] = []
        self._profile_idx = 0
        self._robertson_idx = 0
        self._profile_pixmap: QPixmap | None = None
        self._robertson_pixmap: QPixmap | None = None
        self._current_run_dir: str = ""
        self._map_dialog = None
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
        self.btn_3d_map = QPushButton("3D Map")
        self.btn_3d_map.setObjectName("accent")
        self.btn_3d_map.clicked.connect(self._open_3d_map)
        header.addWidget(self.btn_3d_map)
        layout.addLayout(header)

        # Horizontal splitter: left = metrics+report, right = tabbed plots
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: metrics + classification report ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.metrics_label = QLabel()
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("background-color: #181825; padding: 10px; border-radius: 6px;")
        left_layout.addWidget(self.metrics_label)

        report_group = QGroupBox("Classification Report")
        rg_layout = QVBoxLayout(report_group)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet("background-color: #11111b; color: #cdd6f4; font-family: Consolas;")
        rg_layout.addWidget(self.report_text)
        left_layout.addWidget(report_group)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # ── Right panel: tab widget ──
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabBar::tab {
                background: #313244;
                color: #cdd6f4;
                padding: 6px 18px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
        """)

        self.tab_widget.addTab(self._build_plot_panel("profile"), "Depth Profiles")
        self.tab_widget.addTab(self._build_plot_panel("robertson"), "Robertson SBT")
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        splitter.addWidget(self.tab_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

    def _build_plot_panel(self, kind: str) -> QWidget:
        """Build a navigator + image viewer panel. kind is 'profile' or 'robertson'."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        nav = QHBoxLayout()
        btn_prev = QPushButton("← Prev")
        btn_next = QPushButton("Next →")
        name_label = QLabel()
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(btn_prev)
        nav.addWidget(name_label, 1)
        nav.addWidget(btn_next)
        layout.addLayout(nav)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(image_label, 1)

        if kind == "profile":
            self.profile_btn_prev = btn_prev
            self.profile_btn_next = btn_next
            self.profile_name_label = name_label
            self.profile_image = image_label
            btn_prev.clicked.connect(self._prev_profile)
            btn_next.clicked.connect(self._next_profile)
        else:
            self.robertson_btn_prev = btn_prev
            self.robertson_btn_next = btn_next
            self.robertson_name_label = name_label
            self.robertson_image = image_label
            btn_prev.clicked.connect(self._prev_robertson)
            btn_next.clicked.connect(self._next_robertson)

        return widget

    def load_run(self, run_dir: str):
        self._current_run_dir = run_dir
        data = backend.load_run(run_dir)
        ts = Path(run_dir).name
        model_id = data.get("model_id", "?")
        boundary_source = data.get("boundary_source", "?")
        self.title_label.setText(f"Run: {ts}")

        # Metrics
        m = data.get("metrics", {})
        parts = [f"<b>{k}:</b> {v:.4f}" for k, v in m.items()]
        clusters = data.get("n_clusters", "?")
        bic = data.get("bic")
        bic_str = f"{bic:.1f}" if bic is not None else "-"
        boundary_file = data.get("boundary_file", "")
        bf_str = f"  |  <b>File:</b> {boundary_file}" if boundary_file and boundary_file != "(default)" else ""
        self.metrics_label.setText(
            f"<b>Model:</b> {model_id}  |  <b>Boundaries:</b> {boundary_source}{bf_str}  |  <b>Clusters:</b> {clusters}  |  <b>BIC:</b> {bic_str}<br>"
            + "  |  ".join(parts)
        )

        # Report
        self.report_text.setText(data.get("report", "No report available."))

        # Split plots into profile and Robertson groups
        all_plots = data.get("_plots", [])
        self._robertson_plots = [p for p in all_plots if Path(p).stem.startswith("robertson_")]
        self._profile_plots   = [p for p in all_plots if not Path(p).stem.startswith("robertson_")]

        self._profile_idx = 0
        self._robertson_idx = 0
        self._show_profile()
        self._show_robertson()

    def _show_profile(self):
        self._show(
            self._profile_plots, self._profile_idx,
            self.profile_image, self.profile_name_label,
            self.profile_btn_prev, self.profile_btn_next,
            "_profile_pixmap",
        )

    def _show_robertson(self):
        self._show(
            self._robertson_plots, self._robertson_idx,
            self.robertson_image, self.robertson_name_label,
            self.robertson_btn_prev, self.robertson_btn_next,
            "_robertson_pixmap",
        )

    def _show(self, plots, idx, image_lbl, name_lbl, btn_prev, btn_next, pixmap_attr):
        if not plots:
            image_lbl.setText("No plots available.")
            name_lbl.setText("")
            btn_prev.setEnabled(False)
            btn_next.setEnabled(False)
            setattr(self, pixmap_attr, None)
            return

        path = plots[idx]
        px = QPixmap(path)
        setattr(self, pixmap_attr, px if not px.isNull() else None)

        if not px.isNull():
            QTimer.singleShot(0, lambda: self._fit(image_lbl, pixmap_attr))
        else:
            image_lbl.setText(f"Could not load: {path}")

        name_lbl.setText(f"{Path(path).stem}  ({idx + 1}/{len(plots)})")
        btn_prev.setEnabled(idx > 0)
        btn_next.setEnabled(idx < len(plots) - 1)

    def _fit(self, image_lbl: QLabel, pixmap_attr: str):
        px = getattr(self, pixmap_attr, None)
        if px is None or px.isNull():
            return
        h = image_lbl.height()
        if h < 100:
            h = self.height() - 80
        scaled = px.scaledToHeight(max(h, 200), Qt.TransformationMode.SmoothTransformation)
        image_lbl.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit(self.profile_image, "_profile_pixmap")
        self._fit(self.robertson_image, "_robertson_pixmap")

    def _on_tab_changed(self, _index: int):
        # Re-fit whichever tab just became visible
        QTimer.singleShot(0, lambda: self._fit(self.profile_image, "_profile_pixmap"))
        QTimer.singleShot(0, lambda: self._fit(self.robertson_image, "_robertson_pixmap"))

    def _prev_profile(self):
        if self._profile_idx > 0:
            self._profile_idx -= 1
            self._show_profile()

    def _next_profile(self):
        if self._profile_idx < len(self._profile_plots) - 1:
            self._profile_idx += 1
            self._show_profile()

    def _prev_robertson(self):
        if self._robertson_idx > 0:
            self._robertson_idx -= 1
            self._show_robertson()

    def _next_robertson(self):
        if self._robertson_idx < len(self._robertson_plots) - 1:
            self._robertson_idx += 1
            self._show_robertson()

    def _open_3d_map(self):
        if not self._current_run_dir:
            return
        from .map_3d import Map3DDialog
        self._map_dialog = Map3DDialog(self._current_run_dir, parent=self)
        self._map_dialog.show()
