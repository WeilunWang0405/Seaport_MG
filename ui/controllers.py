from __future__ import annotations
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from data.io import load_system
from engine.runner import run_optimization

class AppController:
    def __init__(self, w):
        self.w = w
        self.system = None
        self.result = None
        self.data_dir = None

        self.w.btnLoadData.clicked.connect(self.on_load_data)
        self.w.btnRunOpt.clicked.connect(self.on_run)
        self.w.sldTime.valueChanged.connect(self.on_time_changed)

    def on_load_data(self):
        try:
            self.data_dir = Path(self.w.leDataDir.text()).resolve()
            T = int(self.w.spnT.value())
            dt = float(self.w.dspnDtHours.value())

            self.system = load_system(self.data_dir, T=T, dt_hours=dt)
            self.w.topoView.load_topology_from_system(self.system)
            self.w.log("Loaded and validated system data.")
            self.w.sldTime.setEnabled(False)

            QMessageBox.information(self.w, "OK", "System data loaded and validated.")
        except Exception as e:
            QMessageBox.critical(self.w, "Load Error", str(e))
            self.w.log(f"[Load Error] {e}")

    def on_run(self):
        if self.system is None:
            QMessageBox.warning(self.w, "Warning", "Please load data first.")
            return
        try:
            result, meta = run_optimization(self.system, data_dir=self.data_dir)
            self.result = result

            self.w.topoView.set_results(self.system, self.result)

            self.w.sldTime.setEnabled(True)
            self.w.sldTime.setRange(0, self.result.T - 1)
            self.w.sldTime.setValue(0)
            self.w.lblTime.setText(f"t=0 / {self.result.T-1}")
            self.w.topoView.update_time(0)

            self.w.log(f"Run finished. Engine={meta.get('engine')}. {meta.get('message','')}")
            QMessageBox.information(self.w, "OK", f"Run finished (engine={meta.get('engine')}).")
        except Exception as e:
            QMessageBox.critical(self.w, "Run Error", str(e))
            self.w.log(f"[Run Error] {e}")

    def on_time_changed(self, t: int):
        if self.result is None:
            return
        self.w.lblTime.setText(f"t={t} / {self.result.T-1}")
        self.w.topoView.update_time(t)
