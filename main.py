from __future__ import annotations
import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.controllers import AppController

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    _controller = AppController(w)  # keep reference
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
