from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QLabel, QSlider,
    QGroupBox, QPlainTextEdit, QFileDialog, QToolBar, QDockWidget,
    QScrollArea, QColorDialog
)

from ui.topology_view import TopologyView
import numpy as np


# ------------------------- Developer style editor -------------------------
StyleItem = Tuple[str, str, str, Dict[str, Any]]
# (attr_name, label, kind, meta) where kind in {"int","float","color"}


def _qcolor_to_hex(value: Any) -> str:
    try:
        return QColor(value).name(QColor.HexArgb)
    except Exception:
        return ""


def _hex_to_qcolor(s: str) -> QColor | None:
    try:
        c = QColor(s.strip())
        return c if c.isValid() else None
    except Exception:
        return None


class StyleEditor(QWidget):
    """Scroll-based form editor that maps widgets to TopologyView style attributes."""

    def __init__(self, grouped_spec: Dict[str, List[StyleItem]], parent: QWidget | None = None):
        super().__init__(parent)
        self._grouped_spec = grouped_spec
        self._widgets: Dict[str, QWidget] = {}
        self._color_edits: Dict[str, QLineEdit] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, 1)

        inner = QWidget()
        scroll.setWidget(inner)
        inner_l = QVBoxLayout(inner)
        inner_l.setContentsMargins(6, 6, 6, 6)
        inner_l.setSpacing(8)

        for group_title, items in self._grouped_spec.items():
            gb = QGroupBox(group_title)
            gb_l = QFormLayout(gb)
            gb_l.setLabelAlignment(Qt.AlignLeft)
            gb_l.setFormAlignment(Qt.AlignTop)
            inner_l.addWidget(gb)

            for attr, label, kind, meta in items:
                w = self._make_widget(attr, kind, meta)
                gb_l.addRow(label, w)

        inner_l.addStretch(1)

        self._cost_inst = None  # np.ndarray (T,)
        self._cost_cum = None  # np.ndarray (T,)
        self._last_result = None
        self._last_system = None

    def _make_widget(self, attr: str, kind: str, meta: Dict[str, Any]) -> QWidget:
        if kind == "int":
            sp = QSpinBox()
            sp.setRange(int(meta.get("min", -10**9)), int(meta.get("max", 10**9)))
            sp.setSingleStep(int(meta.get("step", 1)))
            self._widgets[attr] = sp
            return sp

        if kind == "float":
            sp = QDoubleSpinBox()
            sp.setRange(float(meta.get("min", -1e18)), float(meta.get("max", 1e18)))
            sp.setSingleStep(float(meta.get("step", 0.1)))
            sp.setDecimals(int(meta.get("decimals", 3)))
            self._widgets[attr] = sp
            return sp

        if kind == "color":
            row = QWidget()
            row_l = QHBoxLayout(row)
            row_l.setContentsMargins(0, 0, 0, 0)

            le = QLineEdit()
            le.setPlaceholderText("#RRGGBB or #AARRGGBB")
            btn = QPushButton("Pick…")
            btn.setMaximumWidth(80)

            def _pick():
                cur = _hex_to_qcolor(le.text()) or QColor(Qt.black)
                c = QColorDialog.getColor(cur, self, f"Select {attr}")
                if c.isValid():
                    le.setText(c.name(QColor.HexArgb))

            btn.clicked.connect(_pick)

            row_l.addWidget(le, 1)
            row_l.addWidget(btn, 0)

            self._widgets[attr] = le
            self._color_edits[attr] = le
            return row

        # Fallback: editable text
        le = QLineEdit()
        self._widgets[attr] = le
        return le

    def set_values(self, values: Dict[str, Any]) -> None:
        for attr, w in self._widgets.items():
            if attr not in values:
                continue
            v = values[attr]

            if isinstance(w, QSpinBox):
                try:
                    w.setValue(int(v))
                except Exception:
                    pass
            elif isinstance(w, QDoubleSpinBox):
                try:
                    w.setValue(float(v))
                except Exception:
                    pass
            elif isinstance(w, QLineEdit):
                # Colors arrive as hex; allow raw string too
                if isinstance(v, (QColor, Qt.GlobalColor)):
                    w.setText(_qcolor_to_hex(v))
                else:
                    w.setText(str(v))

    def values(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for attr, w in self._widgets.items():
            if isinstance(w, QSpinBox):
                out[attr] = int(w.value())
            elif isinstance(w, QDoubleSpinBox):
                out[attr] = float(w.value())
            elif isinstance(w, QLineEdit):
                out[attr] = w.text().strip()
        return out


    def _prepare_cost_series(self, system, res) -> None:
        import numpy as np

        T = int(res.T)
        dt = float(getattr(system, "dt_hours", 1.0))

        price = getattr(system, "price", None)
        if price is None:
            price = np.ones(T, dtype=float)
        else:
            price = np.asarray(price, dtype=float).reshape(-1)
            if price.size != T:
                price = np.resize(price, T)

        p_imp = np.asarray(getattr(res, "p_grid_pos", np.zeros(T)), dtype=float).reshape(-1)
        p_exp = np.asarray(getattr(res, "p_grid_neg", np.zeros(T)), dtype=float).reshape(-1)

        # ===== 成本计算：默认假设 price 单位是 SGD/kWh，p 是 kW =====
        # energy(kWh) = p(kW) * dt(h)
        cost_inst = (price * (p_imp - p_exp)) * dt

        self._cost_inst = cost_inst
        self._cost_cum = np.cumsum(cost_inst)


# ------------------------------ Main window ------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Port Operations GUI (Template)")

        # QSettings: persists style between launches
        self._settings = QSettings("PortMG", "PortOperationsGUI")

        cfg_path = Path(__file__).resolve().parents[1] / "config" / "app.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        default_dir = (Path(__file__).resolve().parents[1] / cfg.get("default_data_dir", "sysdata")).resolve()

        # ---- Central layout ----
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
        row_l.setContentsMargins(0, 0, 0, 0)
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

        # ---- Cost panel (left column) ----
        gbCost = QGroupBox("Cost")
        left.addWidget(gbCost)

        cost_form = QFormLayout(gbCost)
        cost_form.setLabelAlignment(Qt.AlignLeft)
        cost_form.setFormAlignment(Qt.AlignTop)

        self.lblCostNow = QLabel("Cost @ t=0: -")
        self.lblCostCum = QLabel("Cumulative: -")
        self.lblCostNow.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lblCostCum.setTextInteractionFlags(Qt.TextSelectableByMouse)

        cost_form.addRow("Instant", self.lblCostNow)
        cost_form.addRow("Cumulative", self.lblCostCum)


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

        # ---- Toolbar: Developer Mode button ----
        tb = QToolBar("Toolbar")
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        self.btnDevMode = QPushButton("Developer Mode")
        self.btnDevMode.setCheckable(True)
        tb.addWidget(self.btnDevMode)

        # ---- Developer panel (dock) ----
        self._style_spec = self._build_style_spec()
        self._style_keys = [it[0] for grp in self._style_spec.values() for it in grp]

        self._style_defaults = self._capture_style_values()  # baseline defaults from TopologyView

        self.devDock = QDockWidget("Developer Mode", self)
        self.devDock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.TopDockWidgetArea, self.devDock)
        self.devDock.setVisible(False)

        dev_root = QWidget()
        self.devDock.setWidget(dev_root)
        dev_l = QVBoxLayout(dev_root)
        dev_l.setContentsMargins(6, 6, 6, 6)

        self.styleEditor = StyleEditor(self._style_spec)
        dev_l.addWidget(self.styleEditor, 1)

        btn_row = QHBoxLayout()
        dev_l.addLayout(btn_row, 0)

        self.btnApplyStyle = QPushButton("Update Plot")
        self.btnSaveStyle = QPushButton("Save")
        self.btnDefaultStyle = QPushButton("Default")

        btn_row.addWidget(self.btnApplyStyle, 0)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btnSaveStyle, 0)
        btn_row.addWidget(self.btnDefaultStyle, 0)

        # Populate editor with defaults then apply persisted style (if any)
        self.styleEditor.set_values(self._style_defaults)
        self._load_style_from_settings(apply_now=True)

        # ---- Signals ----
        self.btnBrowse.clicked.connect(self._browse_dir)
        self.btnDevMode.toggled.connect(self._toggle_dev_mode)
        self.btnApplyStyle.clicked.connect(self._apply_style_from_editor)
        self.btnSaveStyle.clicked.connect(self._save_style_to_settings)
        self.btnDefaultStyle.clicked.connect(self._restore_default_style)
        self.sldTime.valueChanged.connect(self._on_time_changed)

    # --------------------------- Style spec ---------------------------
    def _build_style_spec(self) -> Dict[str, List[StyleItem]]:
        # The "attr" names must match TopologyView attributes.
        return {
            "Layout / Zoom": [
                ("X_STEP", "X_STEP", "float", {"min": 50, "max": 4000, "step": 10, "decimals": 1}),
                ("Y_STEP", "Y_STEP", "float", {"min": 50, "max": 3000, "step": 10, "decimals": 1}),
                ("FIT_MARGIN", "FIT_MARGIN", "float", {"min": 0, "max": 600, "step": 10, "decimals": 1}),
                ("_zoom_step", "zoom step", "float", {"min": 1.01, "max": 2.0, "step": 0.01, "decimals": 3}),
                ("_zoom_min", "zoom min (steps)", "int", {"min": -80, "max": 0, "step": 1}),
                ("_zoom_max", "zoom max (steps)", "int", {"min": 0, "max": 80, "step": 1}),
            ],
            "HV Grid / Transformer": [
                ("HV_GAP", "HV_GAP", "float", {"min": 0, "max": 600, "step": 5, "decimals": 1}),
                ("HV_TEXT_DY", "HV_TEXT_DY", "float", {"min": -200, "max": 200, "step": 2, "decimals": 1}),
                ("HV_LINE_W", "HV_LINE_W", "int", {"min": 1, "max": 30, "step": 1}),
                ("HV_HALF_LEN", "HV_HALF_LEN", "float", {"min": 50, "max": 5000, "step": 20, "decimals": 1}),
                ("HV_COLOR", "HV_COLOR", "color", {}),
            ],
            "Buses / Busbars": [
                ("BUSBAR_LEN", "BUSBAR_LEN", "float", {"min": 50, "max": 2000, "step": 10, "decimals": 1}),
                ("BUSBAR_W", "BUSBAR_W", "int", {"min": 1, "max": 40, "step": 1}),
                ("SLACK_DROP", "SLACK_DROP", "float", {"min": 0, "max": 800, "step": 10, "decimals": 1}),
                ("SLACK_PORT_OFFSET", "SLACK_PORT_OFFSET", "float", {"min": -800, "max": 800, "step": 10, "decimals": 1}),
            ],
            "Edges": [
                ("EDGE_W", "EDGE_W", "int", {"min": 1, "max": 20, "step": 1}),
                ("TRACK_STEP", "TRACK_STEP", "float", {"min": 1, "max": 50, "step": 1, "decimals": 1}),
                ("PFLOW_TEXT_OFFSET", "PFLOW_TEXT_OFFSET", "float", {"min": 0, "max": 400, "step": 5, "decimals": 1}),
            ],
            "Devices": [
                ("DEV_BOX_W", "DEV_BOX_W", "float", {"min": 40, "max": 600, "step": 5, "decimals": 1}),
                ("DEV_BOX_H", "DEV_BOX_H", "float", {"min": 20, "max": 200, "step": 2, "decimals": 1}),
                ("DEV_BOX_GAP_X", "DEV_BOX_GAP_X", "float", {"min": 0, "max": 300, "step": 5, "decimals": 1}),
                ("DEV_BOX_VOFFSET", "DEV_BOX_VOFFSET", "float", {"min": -300, "max": 300, "step": 5, "decimals": 1}),
                ("DEV_BOX_VGAP", "DEV_BOX_VGAP", "float", {"min": 0, "max": 200, "step": 2, "decimals": 1}),
            ],
            "Fonts": [
                ("FONT_BUS_NAME", "FONT_BUS_NAME", "int", {"min": 5, "max": 40, "step": 1}),
                ("FONT_BUS_ID", "FONT_BUS_ID", "int", {"min": 5, "max": 40, "step": 1}),
                ("FONT_EDGE", "FONT_EDGE", "int", {"min": 5, "max": 40, "step": 1}),
                ("FONT_DEVICE", "FONT_DEVICE", "int", {"min": 5, "max": 40, "step": 1}),
                ("FONT_LOAD", "FONT_LOAD", "int", {"min": 5, "max": 40, "step": 1}),
            ],
            "Bus Info Box": [
                ("BUS_INFO_BG", "BUS_INFO_BG", "color", {}),
                ("BUS_INFO_BORDER", "BUS_INFO_BORDER", "color", {}),
                ("BUS_INFO_PAD", "BUS_INFO_PAD", "float", {"min": 0, "max": 60, "step": 1, "decimals": 1}),
                ("BUS_INFO_RADIUS", "BUS_INFO_RADIUS", "float", {"min": 0, "max": 50, "step": 1, "decimals": 1}),
                ("BUS_INFO_TEXT_W", "BUS_INFO_TEXT_W", "float", {"min": 80, "max": 800, "step": 10, "decimals": 1}),
                ("BUS_INFO_OFFSET_X", "BUS_INFO_OFFSET_X", "float", {"min": -500, "max": 500, "step": 5, "decimals": 1}),
                ("BUS_INFO_OFFSET_Y", "BUS_INFO_OFFSET_Y", "float", {"min": -500, "max": 500, "step": 5, "decimals": 1}),
            ],
            "Alerts / Blinking": [
                ("V_LOW_TH", "V_LOW_TH", "float", {"min": 0.5, "max": 1.2, "step": 0.01, "decimals": 3}),
                ("V_LOW_COLOR", "V_LOW_COLOR", "color", {}),
                ("LOAD_GREEN_TH", "LOAD_GREEN_TH", "float", {"min": 0, "max": 300, "step": 1, "decimals": 1}),
                ("LOAD_YELLOW_TH", "LOAD_YELLOW_TH", "float", {"min": 0, "max": 300, "step": 1, "decimals": 1}),
                ("LOAD_RED_TH", "LOAD_RED_TH", "float", {"min": 0, "max": 300, "step": 1, "decimals": 1}),
                ("COL_LOAD_GREEN", "COL_LOAD_GREEN", "color", {}),
                ("COL_LOAD_YELLOW", "COL_LOAD_YELLOW", "color", {}),
                ("COL_LOAD_RED", "COL_LOAD_RED", "color", {}),
                ("BLINK_INTERVAL_MS", "BLINK_INTERVAL_MS", "int", {"min": 50, "max": 5000, "step": 10}),
                ("BLINK_DIM_ALPHA", "BLINK_DIM_ALPHA", "int", {"min": 0, "max": 255, "step": 5}),
            ],
        }

    def _capture_style_values(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in self._style_keys:
            if not hasattr(self.topoView, k):
                continue
            v = getattr(self.topoView, k)
            if isinstance(v, (QColor, Qt.GlobalColor)) or "COLOR" in k or k.endswith("_BG") or k.endswith("_BORDER"):
                out[k] = _qcolor_to_hex(v)
            else:
                out[k] = v
        return out

    # --------------------------- Developer mode ---------------------------
    def _toggle_dev_mode(self, checked: bool) -> None:
        self.devDock.setVisible(bool(checked))

    def _apply_style_from_editor(self) -> None:
        values = self.styleEditor.values()

        for k, raw in values.items():
            if not hasattr(self.topoView, k):
                continue

            # Color fields
            if "COLOR" in k or k.endswith("_BG") or k.endswith("_BORDER"):
                c = _hex_to_qcolor(str(raw))
                if c is not None:
                    setattr(self.topoView, k, c)
                continue

            # Numeric fields
            cur = getattr(self.topoView, k)
            try:
                if isinstance(cur, int):
                    setattr(self.topoView, k, int(raw))
                else:
                    setattr(self.topoView, k, float(raw))
            except Exception:
                # ignore invalid
                continue

        # Blink interval needs a live timer update
        if hasattr(self.topoView, "_blink_timer") and hasattr(self.topoView, "BLINK_INTERVAL_MS"):
            try:
                self.topoView._blink_timer.setInterval(int(self.topoView.BLINK_INTERVAL_MS))
            except Exception:
                pass

        self._refresh_topology_view()

    def _refresh_topology_view(self) -> None:
        """Rebuild scene using the existing system/result objects, without restarting the GUI."""
        sys = getattr(self.topoView, "_system", None)
        if sys is None:
            # Nothing loaded yet; still update viewport
            self.topoView.viewport().update()


            return

        res = getattr(self.topoView, "_result", None)
        t = int(self.sldTime.value()) if self.sldTime.isEnabled() else 0

        # Rebuild geometry and items
        self.topoView.load_topology_from_system(sys)

        # Restore results/time if available
        if res is not None:
            self.topoView.set_results(sys, res)

            self._prepare_cost_series(sys, res)
            self._update_cost_panel(int(self.sldTime.value()) if self.sldTime.isEnabled() else 0)

            try:
                T = int(getattr(res, "T"))
                t = max(0, min(t, T - 1))
            except Exception:
                pass
            self.topoView.update_time(t)
            self._update_cost_panel(t)

    def _save_style_to_settings(self) -> None:
        values = self.styleEditor.values()
        # Persist only known keys, in stable order
        data: Dict[str, Any] = {k: values.get(k) for k in self._style_keys if k in values}
        y = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        self._settings.setValue("dev/style_yaml", y)
        self._settings.sync()
        self.log("Developer style saved.")

    def _load_style_from_settings(self, *, apply_now: bool) -> None:
        raw = self._settings.value("dev/style_yaml", "")
        if not raw:
            return
        try:
            data = yaml.safe_load(str(raw))
            if not isinstance(data, dict):
                return
        except Exception:
            return

        # Update editor
        self.styleEditor.set_values(data)

        if apply_now:
            self._apply_style_from_editor()
            self.log("Developer style loaded from previous session.")

    def _restore_default_style(self) -> None:
        self.styleEditor.set_values(self._style_defaults)
        self._apply_style_from_editor()
        self.log("Developer style restored to defaults (not saved).")

    def _get_cost_series_best_effort(self):
        """
        Return (cost_t, cost_cum, unit) where:
          - cost_t: np.ndarray shape (T,)
          - cost_cum: np.ndarray shape (T,)
          - unit: str
        If unavailable, return (None, None, unit).
        """
        view = self.topoView
        r = getattr(view, "_result", None)
        sys = getattr(view, "_system", None)

        unit = "SGD"  # 你需要可以改成 USD / $ 等

        if r is None:
            return None, None, unit

        # 1) 优先：result 直接给了每时段 cost
        for attr in ("cost_t", "cost_time", "time_cost", "total_cost_t", "obj_t", "objective_t"):
            if hasattr(r, attr):
                try:
                    ct = np.asarray(getattr(r, attr), dtype=float).reshape(-1)
                    return ct, np.cumsum(ct), unit
                except Exception:
                    pass

        # 2) 次优：result 只给了总成本（显示到 cumulative）
        for attr in ("total_cost", "obj", "objective", "cost_total"):
            if hasattr(r, attr):
                try:
                    total = float(getattr(r, attr))
                    # 用常数序列占位：instant 仍显示 ?，cumulative 显示 total
                    return None, np.asarray([total], dtype=float), unit
                except Exception:
                    pass

        # 3) 兜底：用 price * max(P_import,0) * dt 估算
        # price：常见字段名（按你的工程适配）
        price = None
        if sys is not None:
            for attr in ("price_E", "price", "price_sgd_per_kwh", "price_usd_per_kwh"):
                if hasattr(sys, attr):
                    try:
                        price = np.asarray(getattr(sys, attr), dtype=float).reshape(-1)
                        break
                    except Exception:
                        pass

        # grid power：尽量找“总购电功率”字段
        p = None
        for attr in ("P_grid_total", "p_grid_total", "P_MG_total", "p_MG_total", "P_grid", "p_grid", "P_slack", "p_slack"):
            if hasattr(r, attr):
                try:
                    arr = np.asarray(getattr(r, attr), dtype=float)
                    # 兼容 (T,), (T,1)
                    if arr.ndim == 2 and 1 in arr.shape:
                        arr = arr.reshape(-1)
                    # 兼容 (T, n_bus) ：取 slack bus 列（如果能定位）
                    if arr.ndim == 2:
                        slack = getattr(view, "_slack_bus", None)
                        bus_ids = getattr(r, "bus_ids", None)
                        if slack is not None and bus_ids is not None:
                            try:
                                bi = {bid: i for i, bid in enumerate(bus_ids)}[int(slack)]
                                if arr.shape[1] == len(bus_ids):
                                    arr = arr[:, bi]
                                elif arr.shape[0] == len(bus_ids):
                                    arr = arr[bi, :]
                                else:
                                    arr = arr.reshape(-1)
                            except Exception:
                                arr = arr.reshape(-1)
                        else:
                            arr = arr.reshape(-1)
                    p = arr.reshape(-1)
                    break
                except Exception:
                    pass

        if price is None or p is None:
            return None, None, unit

        # dt (hours)：用 GUI 当前值（比 sys 里字段更可靠）
        dt_hr = float(self.dspnDtHours.value())

        T = int(getattr(r, "T", min(len(price), len(p))))
        price = price[:T]
        p = p[:T]

        p_import = np.maximum(p, 0.0)  # 只对购电计费
        ct = price * p_import * dt_hr
        return ct, np.cumsum(ct), unit

    def _update_cost_panel(self, t: int) -> None:
        ct, cc, unit = self._get_cost_series_best_effort()

        # 默认显示
        self.lblCostNow.setText(f"Cost @ t={t}: ? {unit}")
        self.lblCostCum.setText(f"Cumulative: ? {unit}")

        # 只有总成本（cc 只有一个元素）也能显示
        if cc is not None and len(cc) == 1 and (ct is None):
            self.lblCostCum.setText(f"Cumulative: {float(cc[0]):.2f} {unit}")
            return

        if ct is None or cc is None or len(ct) == 0 or len(cc) == 0:
            return

        t = int(max(0, min(t, len(ct) - 1)))
        self.lblCostNow.setText(f"Cost @ t={t}: {float(ct[t]):.2f} {unit}")
        self.lblCostCum.setText(f"Cumulative: {float(cc[t]):.2f} {unit}")


    # --------------------------- Time slider ---------------------------
    def _on_time_changed(self, t: int) -> None:
        self.lblTime.setText(f"t={t+1}")
        try:
            self.topoView.update_time(int(t))
        except Exception:
            pass

        try:
            self._update_cost_panel(int(t))
        except Exception:
            pass

        # NEW: update left cost panel
        self._update_cost_panel(int(t))

    # --------------------------- Misc helpers ---------------------------
    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select data directory", self.leDataDir.text())
        if d:
            self.leDataDir.setText(d)

    def log(self, msg: str) -> None:
        self.txtStatus.appendPlainText(msg)
