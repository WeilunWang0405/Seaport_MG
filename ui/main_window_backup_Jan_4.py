from __future__ import annotations
from pathlib import Path
import yaml

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QLabel, QSlider,
    QGroupBox, QPlainTextEdit, QFileDialog
)

from ui.topology_view import TopologyView
from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import QToolBar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Port Operations GUI (Template)")

        cfg_path = Path(__file__).resolve().parents[1] / "config" / "app.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        default_dir = (Path(__file__).resolve().parents[1] / cfg.get("default_data_dir", "sysdata")).resolve()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        root.addLayout(left, 0)

        gb = QGroupBox("Inputs")
        left.addWidget(gb)
        form = QFormLayout(gb)

        self.leDataDir = QLineEdit(str(default_dir))
        self.btnBrowse = QPushButton("Browse…")
        row = QWidget()
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0,0,0,0)
        row_l.addWidget(self.leDataDir, 1)
        row_l.addWidget(self.btnBrowse, 0)
        form.addRow("Data dir", row)

        self.spnT = QSpinBox()
        self.spnT.setRange(1, 2000)
        self.spnT.setValue(int(cfg.get("default_T", 48)))
        form.addRow("T (periods)", self.spnT)

        self.dspnDtHours = QDoubleSpinBox()
        self.dspnDtHours.setRange(1e-6, 24.0)
        self.dspnDtHours.setDecimals(4)
        self.dspnDtHours.setValue(float(cfg.get("default_dt_hours", 0.5)))
        form.addRow("dt (hours)", self.dspnDtHours)

        self.btnLoadData = QPushButton("Load Data")
        self.btnRunOpt = QPushButton("Run")
        form.addRow(self.btnLoadData, self.btnRunOpt)

        self.txtStatus = QPlainTextEdit()
        self.txtStatus.setReadOnly(True)
        self.txtStatus.setPlaceholderText("Status messages…")
        left.addWidget(self.txtStatus, 1)

        mid = QVBoxLayout()
        root.addLayout(mid, 1)

        self.topoView = TopologyView()
        mid.addWidget(self.topoView, 1)

        time_row = QHBoxLayout()
        mid.addLayout(time_row, 0)

        self.lblTime = QLabel("t=0")
        self.sldTime = QSlider(Qt.Horizontal)
        self.sldTime.setEnabled(False)

        time_row.addWidget(self.lblTime, 0)
        time_row.addWidget(self.sldTime, 1)

        self.btnBrowse.clicked.connect(self._browse_dir)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select data directory", self.leDataDir.text())
        if d:
            self.leDataDir.setText(d)

    def log(self, msg: str):
        self.txtStatus.appendPlainText(msg)
