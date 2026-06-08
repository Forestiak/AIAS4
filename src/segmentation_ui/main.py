"""
Entry point for the segmentation UI.
"""
import sys

from PyQt6.QtWidgets import QApplication

from segmentation_ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
