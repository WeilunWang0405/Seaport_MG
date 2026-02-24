# ui/topology_view.py
from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import numpy as np
import networkx as nx

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QPen, QBrush, QPolygonF, QPainterPath, QPainter, QFont, QColor

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene,
    QGraphicsLineItem, QGraphicsPathItem,
    QGraphicsTextItem, QGraphicsPolygonItem,
    QGraphicsRectItem, QGraphicsEllipseItem,
    QDialog, QVBoxLayout, QLabel
)
import math
from PySide6.QtCore import QPointF

# ---------------- interactive graphics items ----------------
class _InteractiveMixin:
    """Small helper to attach an object identity and a double-click callback."""
    def __init__(self, object_type: str, object_id, on_dbl_click=None):
        # super().__init__()
        self._object_type = object_type
        self._object_id = object_id
        self._on_dbl_click = on_dbl_click

    def mouseDoubleClickEvent(self, event):
        if callable(self._on_dbl_click):
            try:
                self._on_dbl_click(self._object_type, self._object_id)
            except Exception:
                pass
        event.accept()

class InteractiveRectItem(_InteractiveMixin, QGraphicsRectItem):
    def __init__(self, rect: QRectF, object_type: str, object_id, on_dbl_click=None):
        QGraphicsRectItem.__init__(self, rect)
        _InteractiveMixin.__init__(self, object_type, object_id, on_dbl_click)

class InteractivePathItem(_InteractiveMixin, QGraphicsPathItem):
    def __init__(self, path: QPainterPath, object_type: str, object_id, on_dbl_click=None):
        QGraphicsPathItem.__init__(self, path)
        _InteractiveMixin.__init__(self, object_type, object_id, on_dbl_click)

class InteractiveTextItem(_InteractiveMixin, QGraphicsTextItem):
    def __init__(self, text: str, object_type: str, object_id, on_dbl_click=None):
        QGraphicsTextItem.__init__(self, text)
        _InteractiveMixin.__init__(self, object_type, object_id, on_dbl_click)


class TopologyView(QGraphicsView):
    """
    Single-line-diagram style topology view (matches your screenshot style):
      - MG bus = thick horizontal busbar
      - HV grid = long grey bar on top
      - Transformer = two grey circles vertically aligned (each passes through the other's center)
      - Lines = orthogonal polyline + arrow
      - Devices = labeled rectangle, connected from busbar quarter point via orth elbow
      - Load = short elbow arrow (placed opposite side of devices)
      - Slack bus (GRID) centered on x=0
      - Slack outgoing edges: first drop down by same length, then go left/right (consistent drop)
    """

    # ---------------- init ----------------
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # ---- interaction: zoom + pan ----
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)

        self._zoom = 0
        self._zoom_step = 1.15
        self._zoom_min = -15
        self._zoom_max = 25

        self._panning = False
        self._pan_start = None

        # ----- graphics caches -----
        self.busbar_item: Dict[int, QGraphicsLineItem] = {}
        self.bus_id_text: Dict[int, QGraphicsTextItem] = {}
        self.node_text: Dict[int, QGraphicsTextItem] = {}

        self.edge_items: Dict[Tuple[int, int], QGraphicsPathItem] = {}
        self.edge_text: Dict[Tuple[int, int], QGraphicsTextItem] = {}
        self.edge_arrow: Dict[Tuple[int, int], QGraphicsPolygonItem] = {}

        self.device_boxes: Dict[Tuple[int, str, int], QGraphicsRectItem] = {}
        self.device_text: Dict[Tuple[int, str, int], QGraphicsTextItem] = {}
        self.device_link: Dict[Tuple[int, str, int], QGraphicsPathItem] = {}

        self.load_arrow_item: Dict[int, QGraphicsPathItem] = {}
        self.load_arrow_head: Dict[int, QGraphicsPolygonItem] = {}

        self.hv_items: List[object] = []
        self._device_rects: List[QRectF] = []
        # ---- HV grid styling ----
        self.HV_GAP = 340.0  # HV线 到 slack busbar 的垂直距离（加大=距离变大）
        self.HV_TEXT_DY = -40.0  # “HV GRID”文字相对HV线的竖直偏移（负数=上移）
        self.HV_LINE_W = 10.0  # HV线宽（你现在用的也行）

        # ----- data refs -----
        self._system = None
        self._result = None
        self._edge_keys: List[Tuple[int, int]] = []
        self._pos: Dict[int, QPointF] = {}
        self._grid_xy: Dict[int, Tuple[float, float]] = {}
        self._slack_bus: Optional[int] = None

        # =======================
        #  Layout knobs (你要改间距就改这里)
        # =======================
        self.X_STEP = 520     # 水平节点间距（越大越“展开”，越不乱，但fitInView会更缩小）
        self.Y_STEP = 400     # 垂直层间距（越大越不乱）

        # busbar style
        self.BUSBAR_LEN = 180   # 母线长度（截图中是比较长的）
        self.BUSBAR_W = 10

        # edge style
        self.EDGE_W = 2
        self.TRACK_STEP = 18    # 线路避让偏移（并行段错开）

        # slack outgoing “same drop then go sideways”
        self.SLACK_DROP = 55
        self.SLACK_PORT_OFFSET = 16  # slack不同出线在busbar上轻微错开，避免同点重叠

        # device box style
        self.DEV_BOX_W = 160
        self.DEV_BOX_H = 80
        self.DEV_GAP = 8
        self.DEV_OFFSET = 120          # 设备框离母线的垂直距离（你觉得长就减小）
        self.DEV_SHIFT_STEP = 40      # 设备框重叠时，向外“推开”的步长
        self.DEV_BOX_VOFFSET = 30

        # load arrow style (elbow)
        self.LOAD_DROP = 18
        self.LOAD_RUN = 22

        # HV + transformer style
        self.HV_COLOR = Qt.lightGray
        self.HV_W = 10
        self.HV_HALF_LEN = 150
        self.HV_Y_OFFSET = 55
        self.TR_R = 30  # transformer circle radius

        # ----- routing occupancy (simple overlap prevention) -----
        self._used_h: Dict[int, List[Tuple[float, float]]] = {}  # y -> [x1,x2]
        self._used_v: Dict[int, List[Tuple[float, float]]] = {}  # x -> [y1,y2]

        # adjacency + per-parent routing slots (for clean multi-branch routing)
        self._adj = {}  # {bus_id: [neighbor_bus_id, ...]}
        self._pc_slot = {}  # {(parent, child): slot_index}
        self._pc_slot_count = {}  # {parent: number_of_children_slots}

        # ----- font sizes (increase here) -----
        self.FONT_BUS_NAME = 20  # bus name + V,theta
        self.FONT_BUS_ID = 20  # bus numeric id
        self.FONT_EDGE = 20  # line flow text P=...
        self.FONT_DEVICE = 20  # device box text
        self.FONT_LOAD = 20  # load label text (if any)

        self.load_text: Dict[int, QGraphicsTextItem] = {}  # bus_id -> load label text item
        self._bus_index: Dict[int, int] = {}  # bus_id -> row index in load/pv matrices

        self._slack_trunk_child = None  # the "main downward" child of slack (draw from center)

        # ---- bus side info box (name + V + theta) ----
        self.BUS_INFO_BG = QColor(235, 235, 235)  # 浅灰背景
        self.BUS_INFO_BORDER = QColor(190, 190, 190)  # 边框灰
        self.BUS_INFO_PAD = 6.0  # 文本到边框的内边距
        self.BUS_INFO_RADIUS = 8.0  # 圆角半径
        self.BUS_INFO_TEXT_W = 160  # 文本区域宽度(px)，决定框的宽
        self.BUS_INFO_OFFSET_X = 12.0  # 框距离母线右端的水平距离
        self.BUS_INFO_OFFSET_Y = -6.0  # 竖直微调：负数=向上

        self.bus_info_box: Dict[int, QGraphicsPathItem] = {}  # bus_id -> rounded rect background
        self.bus_info_text: Dict[int, QGraphicsTextItem] = {}  # bus_id -> text inside box
        self.bus_disp_name: Dict[int, str] = {}  # bus_id -> display name (fixed)

        self.bus_left_text = {}  # bus_id -> QGraphicsTextItem (name + id)
        self.bus_right_text = {}  # bus_id -> QGraphicsTextItem (V + theta)

        self.DEV_NAME_ALIAS = {
            "Reefer_group": "Reefer",
        }
        # 线路潮流文字离线路的距离（像素），越大越远
        self.PFLOW_TEXT_OFFSET = -35

        # ----- load box (rectangle around load label) -----
        self.load_box_item: Dict[int, QGraphicsRectItem] = {}

        # ----- line loading percent display -----
        self.edge_loading_text: Dict[Tuple[int, int], QGraphicsTextItem] = {}
        self.edge_loading_box: Dict[Tuple[int, int], QGraphicsRectItem] = {}

        # ----- transformer loading percent display -----
        self.tx_loading_box: Optional[QGraphicsRectItem] = None
        self.tx_loading_text: Optional[QGraphicsTextItem] = None

        # ----- blinking control (overload / undervoltage) -----
        self._blink_phase = True
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(600)  # ~0.8-1 Hz perceived blink
        self._blink_timer.timeout.connect(self._toggle_blink_phase)
        self._blink_timer.start()

    # ---------------- public API ----------------
    def clear(self):
        self.scene.clear()

        self.busbar_item.clear()
        self.bus_id_text.clear()
        self.node_text.clear()

        self.bus_info_box.clear()
        self.bus_info_text.clear()
        self.bus_disp_name.clear()

        self.load_text.clear()

        self.edge_items.clear()
        self.edge_text.clear()
        self.edge_arrow.clear()
        self.edge_loading_text.clear()
        self.edge_loading_box.clear()

        self.device_boxes.clear()
        self.device_text.clear()
        self.device_link.clear()

        self.load_arrow_item.clear()
        self.load_arrow_head.clear()
        self.load_box_item.clear()

        self.hv_items.clear()
        self._device_rects.clear()
        self.tx_loading_box = None
        self.tx_loading_text = None

        self._used_h.clear()
        self._used_v.clear()

    def load_topology_from_system(self, system):
        self._system = system
        self._result = None
        self.clear()

        # ---- slack/grid bus ----
        slack_candidates = [b.bus_id for b in system.buses if int(getattr(b, "is_slack", 0)) == 1]
        if not slack_candidates:
            raise ValueError("No slack bus found in sys_bus.csv (is_slack=1).")
        self._slack_bus = slack_candidates[0]

        # ---- edges in sys_line order ----
        self._edge_keys = [(ln.from_bus, ln.to_bus) for ln in system.lines]

        from collections import defaultdict
        adj = defaultdict(list)
        for (u, v) in self._edge_keys:
            adj[u].append(v)
            adj[v].append(u)
        self._adj = dict(adj)

        # ---- compute layout ----
        self._pos = self._compute_single_line_layout(system)
        self._build_row_index()

        self._assign_parent_child_slots()

        self._bus_index = {b.bus_id: i for i, b in enumerate(self._system.buses)}

        # ---- draw HV grid + transformer ----
        self._draw_hv_grid_and_transformer(self._slack_bus)

        # ---- draw edges behind busbars ----
        for (u, v) in self._edge_keys:
            self._draw_edge_routed(u, v)

        # ---- draw busbars ----
        for b in system.buses:
            self._draw_busbar(b.bus_id, b.bus_name)

        # ---- draw devices & load arrows ----
        device_map = self._collect_devices_by_bus(system)

        # 预先定位：第二行左/中/右
        row2_left = self._pick_row_bus(1, "left")
        row2_mid = self._pick_row_bus(1, "mid")
        row2_right = self._pick_row_bus(1, "right")

        # 第三行：按x排序后的第1..第5
        row3 = []
        if hasattr(self, "_row_keys") and len(self._row_keys) >= 3:
            gy3 = self._row_keys[2]
            row3 = list(self._row_buses.get(gy3, []))

        row3_no_load = set()
        row3_center_load = set()
        row3_fifth = None
        if len(row3) >= 1: row3_no_load.add(row3[0])  # 第1不画load
        if len(row3) >= 4: row3_no_load.add(row3[3])  # 第4不画load
        if len(row3) >= 5:
            row3_no_load.add(row3[4])  # 第5不画load
            row3_fifth = row3[4]
        if len(row3) >= 2: row3_center_load.add(row3[1])  # 第2 load从中点
        if len(row3) >= 3: row3_center_load.add(row3[2])  # 第3 load从中点

        for b in system.buses:
            bid = b.bus_id
            devs = device_map.get(bid, [])

            # -------- devices --------
            for k, dev_name in enumerate(devs):
                disp_name = self.DEV_NAME_ALIAS.get(dev_name, dev_name)

                # 默认：设备左侧、下方、母线左1/4引出、箭头指向设备
                side = "down"
                lr = "left"
                anchor_frac = None
                arrow_to_bus = False

                # (4) 第二行左1节点：Reefer 从母线中点向上引出
                if row2_left is not None and bid == row2_left and dev_name == "Reefer":
                    side = "up"
                    lr = "left"
                    anchor_frac = 0.0

                # (6) 第二行右节点：PV 从中点向上，引出到右侧；箭头指向母线
                if row2_right is not None and bid == row2_right and dev_name == "PV":
                    side = "up"
                    lr = "right"
                    anchor_frac = 0.0
                    arrow_to_bus = True

                # (7) 第三行第5节点：Cold-Ironing 从右1/4点向下引出（设备放右侧）
                if row3_fifth is not None and bid == row3_fifth and dev_name == "Cold-Ironing":
                    side = "down"
                    lr = "right"
                    anchor_frac = 0.5

                self._draw_device_box(
                    bid, disp_name, k,
                    side=side, lr=lr,
                    anchor_frac=anchor_frac,
                    arrow_toward_bus=arrow_to_bus
                )

            # -------- load arrows --------
            # (3) 全时段load=0在 _draw_load_arrow 里也会自动跳过
            # (7) 第三行第1/4/5节点：强制不画load
            if bid in row3_no_load:
                continue

            # 默认：右1/4
            frac = 0.5
            # (4) 第二行左1节点：load从 1/3 处向下
            if row2_left is not None and bid == row2_left:
                frac = 0.33
            # (5) 第二行中间节点：load从中点向下
            if row2_mid is not None and bid == row2_mid:
                frac = 0.0
            # (7) 第三行第2/3节点：load从中点向下
            if bid in row3_center_load:
                frac = 0.0

            self._draw_load_arrow(bid, anchor_frac=frac)

        # ---- fit view (保持你截图那种：图在中间偏下、整体不占满) ----
        self.fitInView(self.scene.itemsBoundingRect().adjusted(-120, -120, 120, 120), Qt.KeepAspectRatio)
        self._zoom = 0

    def set_results(self, system, result):
        self._system = system
        self._result = result

        # bus id -> row index
        self._bus_to_i = {bid: i for i, bid in enumerate(result.bus_ids)}

        # ------- 兜底：预计算每个bus的reefer总功率 P_reefer_bus[bus_i, t] -------
        self._P_reefer_bus = None

        # 1) 优先用 solver 已经算好的
        if hasattr(result, "P_reefer_bus") and result.P_reefer_bus is not None:
            self._P_reefer_bus = np.asarray(result.P_reefer_bus, dtype=float)

        else:
            # 2) 否则尝试从 “每个reefer的功率矩阵” 汇总
            # 你工程里这个字段名字可能是 p_rf / P_rf / p_R 等，你自己看一下 result 里是什么
            p_rf = None
            for cand in ("p_rf", "P_rf", "p_R", "p_reefer"):
                if hasattr(result, cand):
                    p_rf = getattr(result, cand)
                    break

            if p_rf is not None:
                p_rf = np.asarray(p_rf, dtype=float)  # shape: (n_reefers, T)
                nb = len(result.bus_ids)
                T = result.T
                Pbus = np.zeros((nb, T), dtype=float)

                # system.reefers 必须是按 p_rf 行顺序对应的列表
                for r, rf in enumerate(system.reefers):
                    bid = int(rf.node_number)
                    if bid in self._bus_to_i:
                        Pbus[self._bus_to_i[bid], :] += p_rf[r, :]

                self._P_reefer_bus = Pbus

    def _build_row_index(self):
        """
        根据 sys_bus 的 (x,y) 网格坐标把节点分行，并按 x 排序。
        依赖：self._grid_xy[bus_id] = (gx, gy)  或你已有的类似字典
        """
        rows = {}
        for bid, (gx, gy) in self._grid_xy.items():
            rows.setdefault(gy, []).append(bid)
        for gy in rows:
            rows[gy].sort(key=lambda b: self._grid_xy[b][0])  # 按x排序

        self._row_keys = sorted(rows.keys())  # 从上到下
        self._row_buses = rows  # gy -> [bus_id...]

    def _pick_row_bus(self, row_idx: int, pos: str) -> int | None:
        """row_idx=0表示最上面一行。pos in {'left','mid','right'}"""
        if not hasattr(self, "_row_keys") or row_idx >= len(self._row_keys):
            return None
        gy = self._row_keys[row_idx]
        lst = self._row_buses[gy]
        if not lst:
            return None
        if pos == "left":
            return lst[0]
        if pos == "right":
            return lst[-1]
        return lst[len(lst) // 2]  # mid

    def update_time(self, t: int):
        if self._result is None:
            return

        r = self._result
        bus_to_idx = {bid: i for i, bid in enumerate(r.bus_ids)}
        edge_to_idx = {ek: i for i, ek in enumerate(r.edge_keys)}

        # update bus text
        # update bus right text (V/theta); left text (name/id) is fixed
        # update bus info box text (name + V/theta)
        for bid in r.bus_ids:
            i = bus_to_idx[bid]
            V = float(r.V_bus[t, i])
            th = float(r.theta_bus[t, i])

            if bid in self.bus_info_text:
                name = self.bus_disp_name.get(bid, f"Bus {bid}")
                self.bus_info_text[bid].setHtml(
                    f"<div align='left'>"
                    f"<b>{name}</b><br/>"
                    f"V={V:.4f}<br/>&theta;={th:.4f}"
                    f"</div>"
                )


            # voltage styling (rule: <0.95 pu -> red + slow blink; else black)
            if bid in self.bus_info_text:
                if V < 0.95:
                    col = QColor(200, 0, 0)
                    if not self._blink_phase:
                        col.setAlpha(80)
                    self.bus_info_text[bid].setDefaultTextColor(col)
                else:
                    self.bus_info_text[bid].setDefaultTextColor(Qt.black)

        # update line text
        for ek in r.edge_keys:
            e = edge_to_idx[ek]
            p = float(r.P_line[t, e])

            # P label
            if ek in self.edge_text:
                self.edge_text[ek].setPlainText(f"P={p:.1f} kW")
                self.edge_text[ek].setDefaultTextColor(Qt.red if p < 0 else Qt.darkBlue)

            # loading percent (best-effort: use line capacity if available; otherwise hide)
            cap = self._line_capacity_kw(*ek)
            if cap is not None and cap > 1e-9:
                pct = abs(p) / cap * 100.0
                col, blink = self._loading_style(pct)
                if blink and not self._blink_phase:
                    col.setAlpha(80)

                if ek in self.edge_loading_text:
                    self.edge_loading_text[ek].setPlainText(f"{pct:.0f}%")
                    self.edge_loading_text[ek].setDefaultTextColor(col)

                if ek in self.edge_loading_box:
                    self.edge_loading_box[ek].setPen(QPen(col, 1.6))
            else:
                if ek in self.edge_loading_text:
                    self.edge_loading_text[ek].setPlainText("")


        # -------- update transformer loading (best-effort) --------
        if self.tx_loading_text is not None and self.tx_loading_box is not None:
            pct = None
            # prefer result-provided percent
            for attr in ("tx_loading_pct", "transformer_loading_pct"):
                if hasattr(r, attr):
                    try:
                        arr = getattr(r, attr)
                        if hasattr(arr, "__len__"):
                            pct = float(arr[t])
                        else:
                            pct = float(arr)
                    except Exception:
                        pct = None
                    break

            # fallback: if you later add P_tx/Q_tx and S_rated
            if pct is not None:
                col, blink = self._loading_style(pct)
                if blink and not self._blink_phase:
                    col.setAlpha(80)
                self.tx_loading_text.setHtml(f"<div align='center'><b>TX</b><br/>Load={pct:.0f}%</div>")
                self.tx_loading_text.setDefaultTextColor(col)
                self.tx_loading_box.setPen(QPen(col, 1.8))

        # -------- update load labels (always from system load matrix) --------
        load_mat = None
        for attr in ("load_kw", "loads_kw", "p_load_kw", "P_load_kw"):
            if hasattr(self._system, attr):
                load_mat = getattr(self._system, attr)
                break

        for bus_id, txt in self.load_text.items():
            i = self._bus_index.get(bus_id, None)
            if i is None:
                continue
            val = self._value_from_matrix(load_mat, i, t)
            if val is None:
                txt.setPlainText("Load=? kW")
            else:
                txt.setPlainText(f"Load={val:.1f} kW")
            br = txt.boundingRect()
            if bus_id in self.load_box_item:
                self.load_box_item[bus_id].setRect(QRectF(txt.x() - 6, txt.y() - 4, br.width() + 12, br.height() + 8))

        # -------- update device power text --------
        pv_mat = getattr(self._system, "pv_kw", None)

        # pre-compute cold-ironing fixed demand per bus at time t (fallback if no result)
        ci_fixed_by_bus = {}
        if getattr(self._system, "cold_ironing", None) is not None:
            for task in self._system.cold_ironing:
                # 假设 start_time/end_time 在文件中是 1..T（常见）
                tt = t + 1
                if int(task.start_time) <= tt <= int(task.end_time):
                    ci_fixed_by_bus[task.node_number] = ci_fixed_by_bus.get(task.node_number, 0.0) + float(
                        task.demand_fix_kw)

        # ESS power fallback
        ess_bus = getattr(getattr(self._system, "ess", None), "node_number", None)
        ess_p = None
        if self._result is not None:
            if hasattr(self._result, "P_ess_bus"):
                ess_p = getattr(self._result, "P_ess_bus")
            elif hasattr(self._result, "P_ess"):
                ess_p = getattr(self._result, "P_ess")

        for (bus_id, dev_name, idx), txt in self.device_text.items():
            pval = None

            if dev_name == "PV":
                i = self._bus_index.get(bus_id, None)
                if i is not None:
                    pval = self._value_from_matrix(pv_mat, i, t)

            elif dev_name == "ESS":
                if ess_bus is not None and int(bus_id) == int(ess_bus):
                    # if P_ess_bus exists -> matrix; else P_ess maybe (T,)
                    if hasattr(ess_p, "shape"):
                        if len(getattr(ess_p, "shape", ())) == 2:
                            i = self._bus_index.get(bus_id, None)
                            if i is not None:
                                pval = self._value_from_matrix(ess_p, i, t)
                        elif len(getattr(ess_p, "shape", ())) == 1:
                            try:
                                pval = float(ess_p[t])
                            except Exception:
                                pval = None

            elif dev_name == "Cold-Ironing":
                # fallback: fixed demand sum at this bus
                pval = float(ci_fixed_by_bus.get(bus_id, 0.0))

            elif dev_name in ("Reefer", "Reefer_group"):
            # 1) 优先使用 set_results() 里汇总好的 self._P_reefer_bus（按 result.bus_ids 顺序）
                if getattr(self, "_P_reefer_bus", None) is not None:
                    bi = bus_to_idx.get(bus_id, None)
                    if bi is not None:
                        try:
                            pval = float(self._P_reefer_bus[bi, t])  # shape: (n_bus, T)
                        except Exception:
                            pval = None

            # 2) 其次：如果 result 直接给了 P_reefer_bus，就用它
                elif self._result is not None and hasattr(self._result, "P_reefer_bus"):
                    mat = getattr(self._result, "P_reefer_bus")
                    bi = bus_to_idx.get(bus_id, None)
                    if bi is not None:
                        try:
                        # 兼容 (n_bus,T) 或 (T,n_bus)
                            if hasattr(mat, "shape") and len(mat.shape) == 2:
                                if mat.shape[0] == len(r.bus_ids):
                                    pval = float(mat[bi, t])
                                elif mat.shape[1] == len(r.bus_ids):
                                    pval = float(mat[t, bi])
                                else:
                                    pval = None
                            else:
                                pval = None
                        except Exception:
                            pval = None
                    else:
                        pval = None

    # update box text (HTML) - enforce display alias
            disp = self.DEV_NAME_ALIAS.get(dev_name, dev_name)

            if pval is None:
                txt.setHtml(f"<div align='center'><b>{disp}</b><br/>P=? kW</div>")
            else:
                txt.setHtml(f"<div align='center'><b>{disp}</b><br/>P={pval:.1f} kW</div>")


    # ---------------- status styling helpers ----------------
    def _toggle_blink_phase(self):
        self._blink_phase = not self._blink_phase
        # force repaint; we also rely on update_time() to set current styles
        self.viewport().update()

    @staticmethod
    def _loading_style(pct: float) -> Tuple[QColor, bool]:
        """Return (color, needs_blink) for a loading percentage."""
        if pct is None or not np.isfinite(pct):
            return QColor(0, 0, 0), False
        if pct < 70.0:
            return QColor(0, 140, 0), False
        if pct < 90.0:
            return QColor(200, 160, 0), False
        # >=90
        if pct <= 100.0:
            return QColor(200, 0, 0), False
        return QColor(200, 0, 0), True

    def _open_detail_window(self, object_type: str, object_id):
        """Open a small time-series window for the given object."""
        try:
            dlg = TimeSeriesDialog(parent=self, view=self, object_type=object_type, object_id=object_id)
            dlg.exec()
        except Exception:
            # Never crash GUI on detail failures
            return

    def _line_capacity_kw(self, u: int, v: int) -> Optional[float]:
        """Best-effort to retrieve a line capacity (kW) from system.lines."""
        if self._system is None:
            return None
        # locate line object in sys_line order
        for ln in getattr(self._system, "lines", []):
            if (int(ln.from_bus), int(ln.to_bus)) == (int(u), int(v)) or (int(ln.from_bus), int(ln.to_bus)) == (int(v), int(u)):
                for attr in ("capacity_kw", "rate_a_kw", "thermal_limit_kw", "s_max_kw", "p_max_kw"):
                    if hasattr(ln, attr):
                        try:
                            val = float(getattr(ln, attr))
                            if val > 1e-9:
                                return val
                        except Exception:
                            pass
                # common: kVA rating
                for attr in ("s_max_kva", "capacity_kva", "rate_a_kva", "thermal_limit_kva"):
                    if hasattr(ln, attr):
                        try:
                            val = float(getattr(ln, attr)) * 1.0  # treat kVA ~= kW for display if Q unavailable
                            if val > 1e-9:
                                return val
                        except Exception:
                            pass
        return None

    # ---------------- layout ----------------
    def _compute_single_line_layout(self, system) -> Dict[int, QPointF]:
        """
        If sys_bus provides valid x,y for all buses:
          - treat (x,y) as grid coordinates
          - map to scene coordinates using X_STEP/Y_STEP
          - x in {-1,0,1} -> three columns
          - y in {0..16}  -> 17 rows (top to bottom)
        Otherwise fallback to auto layout (leaf-order DFS).
        """
        # --------- 1) Try user-defined grid coordinates from sys_bus ---------
        use_xy = True
        bus_xy: Dict[int, Tuple[float, float]] = {}

        for b in system.buses:
            bid = b.bus_id
            x = getattr(b, "x", None)
            y = getattr(b, "y", None)

            # missing / NaN check
            if x is None or y is None:
                use_xy = False
                break
            try:
                xf = float(x)
                yf = float(y)
            except Exception:
                use_xy = False
                break

            # NaN check
            if xf != xf or yf != yf:  # NaN != NaN
                use_xy = False
                break

            bus_xy[bid] = (xf, yf)

        if use_xy and len(bus_xy) == len(system.buses):
            slack = self._slack_bus
            if slack is not None and slack in bus_xy:
                slack_x, slack_y = bus_xy[slack]
            else:
                slack_x, slack_y = (0.0, 0.0)

            # 保存“相对 slack 的网格坐标”，用于第2/3行定位
            self._grid_xy = {bid: (gx - slack_x, gy - slack_y) for bid, (gx, gy) in bus_xy.items()}

            pos: Dict[int, QPointF] = {}
            for bid, (gx, gy) in self._grid_xy.items():
                sx = gx * self.X_STEP
                sy = gy * self.Y_STEP
                pos[bid] = QPointF(sx, sy)

            return pos

        # --------- 2) Fallback: auto layout (your previous behavior) ---------
        G = nx.Graph()
        for ln in system.lines:
            G.add_edge(ln.from_bus, ln.to_bus)

        slack = self._slack_bus
        assert slack is not None

        parent: Dict[int, Optional[int]] = {slack: None}
        depth: Dict[int, int] = {slack: 0}
        q = [slack]
        while q:
            u = q.pop(0)
            for v in G.neighbors(u):
                if v in parent:
                    continue
                parent[v] = u
                depth[v] = depth[u] + 1
                q.append(v)

        for bobj in system.buses:
            if bobj.bus_id not in parent:
                parent[bobj.bus_id] = slack
                depth[bobj.bus_id] = 1

        children: Dict[int, List[int]] = {k: [] for k in parent.keys()}
        for v, p in parent.items():
            if p is not None:
                children[p].append(v)

        x_index: Dict[int, float] = {}
        next_leaf = 0

        def dfs(u: int):
            nonlocal next_leaf
            if len(children.get(u, [])) == 0:
                x_index[u] = float(next_leaf)
                next_leaf += 1
                return
            for c in children[u]:
                dfs(c)
            xs = [x_index[c] for c in children[u]]
            x_index[u] = sum(xs) / len(xs)

        dfs(slack)

        pos: Dict[int, QPointF] = {}
        for node in parent.keys():
            x = x_index.get(node, 0.0) * self.X_STEP
            y = float(depth.get(node, 0)) * self.Y_STEP
            pos[node] = QPointF(x, y)

        dx = pos[slack].x()
        for k in pos:
            pos[k] = QPointF(pos[k].x() - dx, pos[k].y())
        return pos

    def _assign_parent_child_slots(self):
        """
        Build a rooted tree (BFS from slack) and assign each parent->child edge a unique slot index.
        The slot index is used to offset y_mid so sibling branches do not overlap.
        """
        self._pc_slot.clear()
        self._pc_slot_count.clear()

        if self._slack_bus is None:
            return

        # BFS from slack to define parent-child direction (by connectivity)
        from collections import deque
        slack = self._slack_bus

        parent = {slack: None}
        depth = {slack: 0}

        q = deque([slack])
        while q:
            u = q.popleft()
            for v in self._adj.get(u, []):
                if v in parent:
                    continue
                parent[v] = u
                depth[v] = depth[u] + 1
                q.append(v)

        # Build children lists
        children = {}
        for v, p in parent.items():
            if p is None:
                continue
            children.setdefault(p, []).append(v)

        # For each parent, only consider "downward" children in the drawing (y larger)
        # and assign slots ordered by child x (left->right) for stable appearance.
        eps_y = 0.5
        for p, childs in children.items():
            yp = self._pos[p].y()
            down = [c for c in childs if self._pos[c].y() > yp + eps_y]

            if len(down) <= 1:
                # still set count for completeness
                if len(down) == 1:
                    self._pc_slot[(p, down[0])] = 0
                    self._pc_slot_count[p] = 1
                continue

            down_sorted = sorted(down, key=lambda c: self._pos[c].x())
            self._pc_slot_count[p] = len(down_sorted)
            # choose slack trunk child: the downward child closest to slack x (draw from center)
            if self._slack_bus is not None and p == self._slack_bus and len(down_sorted) >= 1:
                sx = self._pos[self._slack_bus].x()
                self._slack_trunk_child = min(down_sorted, key=lambda c: abs(self._pos[c].x() - sx))

            for idx, c in enumerate(down_sorted):
                self._pc_slot[(p, c)] = idx

    # ---------------- HV + transformer ----------------
    def _draw_hv_grid_and_transformer(self, slack_bus: int):
        p_slack = self._pos[slack_bus]
        hv_y = p_slack.y() - self.HV_GAP

        pen_hv = QPen(self.HV_COLOR, self.HV_W, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        pen_tr = QPen(self.HV_COLOR, 2.0)

        # HV busbar (auto length)
        max_abs_x = max(abs(p.x()) for p in self._pos.values())
        half_len = max(float(self.HV_HALF_LEN), max_abs_x + self.BUSBAR_LEN)

        hv_line = QGraphicsLineItem(-half_len, hv_y, half_len, hv_y)
        hv_line.setPen(pen_hv)
        self.scene.addItem(hv_line)
        self.hv_items.append(hv_line)

        hv_text = QGraphicsTextItem("HV GRID")
        hv_text.setDefaultTextColor(self.HV_COLOR)
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        hv_text.setFont(f)

        # always align to HV left end
        hv_text.setPos(-half_len + 10, hv_y + self.HV_TEXT_DY)

        self.scene.addItem(hv_text)
        self.hv_items.append(hv_text)

        # transformer circles (vertical), centers separated by exactly r
        x_c = 0.0
        r = float(self.TR_R)
        y_mid = (hv_y + p_slack.y()) / 2.0

        y_top = y_mid - r / 2.0
        y_bot = y_mid + r / 2.0

        # connection: HV -> top circle top
        conn1 = QGraphicsLineItem(x_c, hv_y, x_c, y_top - r - 2)
        conn1.setPen(QPen(self.HV_COLOR, 2.0))
        self.scene.addItem(conn1)
        self.hv_items.append(conn1)

        c1 = QGraphicsEllipseItem(QRectF(x_c - r, y_top - r, 2 * r, 2 * r))
        c1.setPen(pen_tr)
        c1.setBrush(QBrush(Qt.transparent))
        self.scene.addItem(c1)
        self.hv_items.append(c1)

        c2 = QGraphicsEllipseItem(QRectF(x_c - r, y_bot - r, 2 * r, 2 * r))
        c2.setPen(pen_tr)
        c2.setBrush(QBrush(Qt.transparent))
        self.scene.addItem(c2)
        self.hv_items.append(c2)

        # connection: bottom circle bottom -> MG slack bus
        conn2 = QGraphicsLineItem(x_c, y_bot + r + 2, x_c, p_slack.y())
        conn2.setPen(QPen(self.HV_COLOR, 2.0))
        self.scene.addItem(conn2)
        self.hv_items.append(conn2)

        # transformer loading box (updated in update_time)
        tx_w, tx_h = 140.0, 44.0
        tx_x0 = x_c + r + 18.0
        tx_y0 = y_mid - tx_h / 2.0
        tx_box = InteractiveRectItem(QRectF(tx_x0, tx_y0, tx_w, tx_h), object_type="transformer", object_id="TX1",
                                     on_dbl_click=self._open_detail_window)
        tx_box.setPen(QPen(Qt.black, 1.4))
        tx_box.setBrush(QBrush(Qt.white))
        self.scene.addItem(tx_box)

        tx_txt = InteractiveTextItem("TX Load=?%", object_type="transformer", object_id="TX1",
                                     on_dbl_click=self._open_detail_window)
        ff = QFont()
        ff.setPointSize(18)
        ff.setBold(True)
        tx_txt.setFont(ff)
        tx_txt.setDefaultTextColor(Qt.black)
        tx_txt.setTextWidth(tx_w)
        tx_txt.setHtml("<div align='center'><b>TX</b><br/>Load=?%</div>")
        brt = tx_txt.boundingRect()
        tx_txt.setPos(tx_x0 + (tx_w - brt.width()) / 2.0, tx_y0 + (tx_h - brt.height()) / 2.0)
        self.scene.addItem(tx_txt)

        self.tx_loading_box = tx_box
        self.tx_loading_text = tx_txt

    # ---------------- drawing primitives ----------------
    def _draw_busbar(self, bus_id: int, bus_name: str = ""):
        c = self._pos[bus_id]
        half = self.BUSBAR_LEN / 2.0

        pen = QPen(Qt.black, self.BUSBAR_W, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        bar = QGraphicsLineItem(c.x() - half, c.y(), c.x() + half, c.y())
        bar.setPen(pen)
        self.scene.addItem(bar)
        self.busbar_item[bus_id] = bar

        # -------- side info box: NAME + V + theta (light grey rounded rect) --------
        name = bus_name if bus_name else ("GRID" if bus_id == self._slack_bus else f"Bus {bus_id}")
        self.bus_disp_name[bus_id] = name

        # text (inner)
        info_txt = InteractiveTextItem('', object_type='bus', object_id=bus_id, on_dbl_click=self._open_detail_window)
        info_txt.setDefaultTextColor(Qt.black)
        f = QFont()
        f.setPointSize(self.FONT_BUS_NAME)
        info_txt.setFont(f)
        info_txt.setTextWidth(float(self.BUS_INFO_TEXT_W))
        info_txt.setHtml(
            f"<div align='left'>"
            f"<b>{name}</b><br/>"
            f"V=?<br/>&theta;=?"
            f"</div>"
        )

        # background rounded rect (outer)
        pad = float(self.BUS_INFO_PAD)
        br = info_txt.boundingRect()
        box_w = float(self.BUS_INFO_TEXT_W) + 2 * pad
        box_h = float(br.height()) + 2 * pad

        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, box_w, box_h), self.BUS_INFO_RADIUS, self.BUS_INFO_RADIUS)
        box = InteractivePathItem(path, object_type='bus', object_id=bus_id, on_dbl_click=self._open_detail_window)
        box.setPen(QPen(self.BUS_INFO_BORDER, 1))
        box.setBrush(QBrush(self.BUS_INFO_BG))

        # position: to the right of busbar, slightly upward
        x0 = c.x() + half + float(self.BUS_INFO_OFFSET_X)
        y0 = c.y() - box_h / 2.0 + float(self.BUS_INFO_OFFSET_Y)

        box.setPos(x0, y0)
        box.setZValue(9)
        self.scene.addItem(box)

        info_txt.setPos(x0 + pad, y0 + pad)
        info_txt.setZValue(10)
        self.scene.addItem(info_txt)

        self.bus_info_box[bus_id] = box
        self.bus_info_text[bus_id] = info_txt

    # ---------------- edge drawing ----------------
    def _draw_edge_routed(self, u: int, v: int):
        p1 = self._pos[u]
        p2 = self._pos[v]

        pts = self._route_orth_avoid(u, v, p1, p2)

        path = QPainterPath(pts[0])
        for pt in pts[1:]:
            path.lineTo(pt)

        item = QGraphicsPathItem(path)
        item.setPen(QPen(Qt.black, self.EDGE_W))
        self.scene.addItem(item)
        self.edge_items[(u, v)] = item

        arrow = self._make_arrow_at_end(pts)
        self.scene.addItem(arrow)
        self.edge_arrow[(u, v)] = arrow

        # ---------- label position (use PFLOW_TEXT_OFFSET) ----------
        mid_pt, mid_dir = self._polyline_midpoint_and_dir(pts)

        # normal direction
        nx_, ny_ = (-mid_dir[1], mid_dir[0])

        # 同一竖直列：强制把潮流文字放到“右侧”，距离由 PFLOW_TEXT_OFFSET 控制
        same_column = abs(self._pos[u].x() - self._pos[v].x()) < 1e-6
        if same_column:
            label_pos = QPointF(mid_pt.x() + float(self.PFLOW_TEXT_OFFSET), mid_pt.y())
        else:
            off = float(self.PFLOW_TEXT_OFFSET)
            label_pos = QPointF(mid_pt.x() + off * nx_, mid_pt.y() + off * ny_)

        etxt = InteractiveTextItem("P=? kW", object_type='line', object_id=(u, v), on_dbl_click=self._open_detail_window)
        etxt.setDefaultTextColor(Qt.darkBlue)
        f = QFont()
        f.setPointSize(self.FONT_EDGE)
        etxt.setFont(f)

        br = etxt.boundingRect()
        etxt.setPos(label_pos.x() - br.width() / 2.0, label_pos.y() - br.height() / 2.0)

        # light background box for readability + interaction
        bg = InteractiveRectItem(QRectF(etxt.x() - 6, etxt.y() - 4, br.width() + 12, br.height() + 8),
                                 object_type='line', object_id=(u, v), on_dbl_click=self._open_detail_window)
        bg.setPen(QPen(QColor(210, 210, 210), 1))
        bg.setBrush(QBrush(QColor(255, 255, 255)))
        bg.setZValue(etxt.zValue() - 1)

        self.scene.addItem(bg)
        self.scene.addItem(etxt)
        self.edge_text[(u, v)] = etxt
        self.edge_loading_box[(u, v)] = bg

        # loading percent text (placed slightly below the P label)
        ltxt = InteractiveTextItem("0%", object_type='line', object_id=(u, v), on_dbl_click=self._open_detail_window)
        lf = QFont()
        lf.setPointSize(self.FONT_EDGE)
        ltxt.setFont(lf)
        ltxt.setDefaultTextColor(Qt.darkBlue)

        lbr = ltxt.boundingRect()
        ltxt.setPos(label_pos.x() - lbr.width() / 2.0, label_pos.y() - lbr.height() / 2.0 + br.height() + 2)

        self.scene.addItem(ltxt)
        self.edge_loading_text[(u, v)] = ltxt



        f = QFont()
        f.setPointSize(self.FONT_EDGE)
        etxt.setFont(f)

        self._reserve_polyline(pts)

    def _route_orth_avoid(self, u: int, v: int, p1: QPointF, p2: QPointF) -> List[QPointF]:
        """
        Routing policy update:
          - If two buses are on the same y-level: connect via inner quarter points,
            and route down -> horizontal -> up to avoid overlapping with busbars.
          - If u is slack: keep "same drop then go sideways", but also use quarter anchors.
          - Otherwise: default HV/VH with offset-avoid.
        """
        eps_y = 0.5  # tolerance for "same y-level"

        # ---------- 1) Same-level buses: use quarter anchors and go DOWN then H then UP ----------
        if abs(p1.y() - p2.y()) < eps_y:
            a1 = self._bus_inner_quarter_anchor(u, toward_x=p2.x())
            a2 = self._bus_inner_quarter_anchor(v, toward_x=p1.x())

            base_drop = max(20.0, 0.20 * self.Y_STEP)  # go below the busbar
            y_mid0 = a1.y() + base_drop

            for k in range(0, 10):
                if k == 0:
                    off = 0.0
                else:
                    sgn = 1.0 if (k % 2 == 1) else -1.0
                    off = sgn * ((k + 1) // 2) * self.TRACK_STEP

                y_mid = y_mid0 + off
                pts = self._compress([
                    a1,
                    QPointF(a1.x(), y_mid),
                    QPointF(a2.x(), y_mid),
                    a2,
                ])
                if not self._polyline_overlaps(pts):
                    return pts

            return self._compress([a1, QPointF(a1.x(), y_mid0), QPointF(a2.x(), y_mid0), a2])

        # ---------- 2) Slack outgoing ----------
        if self._slack_bus is not None and u == self._slack_bus:
            # trunk child from CENTER; other children use port offset
            if self._slack_trunk_child is not None and v == self._slack_trunk_child:
                start_x = self._pos[u].x()  # center of slack busbar
            else:
                # keep your previous separation logic for non-trunk edges
                sign = -1.0 if p2.x() < p1.x() else 1.0
                start_x = self._pos[u].x() + sign * self.SLACK_PORT_OFFSET

            y_mid = p1.y() + self.SLACK_DROP

            # end point: use CHILD MIDPOINT (busbar center) for clean vertical landing
            end_x = self._pos[v].x()
            end_y = self._pos[v].y()

            pts0 = self._compress([
                QPointF(start_x, p1.y()),
                QPointF(start_x, y_mid),  # first go down
                QPointF(end_x, y_mid),  # then go sideways
                QPointF(end_x, end_y),  # then go down/up to child
            ])

            if not self._polyline_overlaps(pts0):
                return pts0

            # try y_mid shifts if needed
            for k in range(1, 8):
                y_try = y_mid + k * self.TRACK_STEP
                cand = self._compress([
                    QPointF(start_x, p1.y()),
                    QPointF(start_x, y_try),
                    QPointF(end_x, y_try),
                    QPointF(end_x, end_y),
                ])
                if not self._polyline_overlaps(cand):
                    return cand

            return pts0

        # ---------- Multi-child parent -> child (different column): use slots + down-half, horizontal, down-half ----------
        eps_y = 0.5
        eps_x = 0.5

        def down_children_count(parent_id: int) -> int:
            y0 = self._pos[parent_id].y()
            cnt = 0
            for nb in self._adj.get(parent_id, []):
                if self._pos[nb].y() > y0 + eps_y:
                    cnt += 1
            return cnt

        # Identify parent (higher y) and child (lower y) geometrically for routing
        if p2.y() > p1.y() + eps_y:
            parent_id, child_id = u, v
            pp, pc = p1, p2
        elif p1.y() > p2.y() + eps_y:
            parent_id, child_id = v, u
            pp, pc = p2, p1
        else:
            parent_id, child_id = None, None
            pp, pc = None, None

        if parent_id is not None:
            # apply ONLY when parent has >=2 downward children AND child not in same column
            if down_children_count(parent_id) >= 2 and abs(pp.x() - pc.x()) > eps_x:
                # parent anchor: inner quarter toward child
                a1 = self._bus_inner_quarter_anchor(parent_id, toward_x=pc.x())

                # child anchor: MIDPOINT (as you requested)
                a2 = QPointF(self._pos[child_id].x(), self._pos[child_id].y())

                # base mid level: half of vertical distance
                dy = a2.y() - a1.y()
                y_mid_base = a1.y() + 0.5 * dy

                # slot offset: separate sibling branches into different "channels"
                slot = self._pc_slot.get((parent_id, child_id), 0)
                m = self._pc_slot_count.get(parent_id, 1)
                center = (m - 1) / 2.0
                slot_offset = (slot - center) * (1.6 * self.TRACK_STEP)  # 1.6 can be tuned

                y_mid0 = y_mid_base + slot_offset

                # try small additional offsets to avoid overlapping with existing routes
                for k in range(0, 8):
                    if k == 0:
                        extra = 0.0
                    else:
                        sgn = 1.0 if (k % 2 == 1) else -1.0
                        extra = sgn * ((k + 1) // 2) * self.TRACK_STEP

                    y_mid = y_mid0 + extra

                    pts = self._compress([
                        a1,
                        QPointF(a1.x(), y_mid),
                        QPointF(a2.x(), y_mid),
                        a2,
                    ])

                    if not self._polyline_overlaps(pts):
                        return pts

                # fallback without overlap avoidance
                return self._compress([a1, QPointF(a1.x(), y_mid0), QPointF(a2.x(), y_mid0), a2])

        # ---------- 3) Non-slack default: try HV with offsets ----------
        best = self._try_route_with_offsets(p1, p2, mode="HV")
        if best is not None:
            return best

        best = self._try_route_with_offsets(p1, p2, mode="VH")
        if best is not None:
            return best

        return [p1, QPointF(p2.x(), p1.y()), p2]

    def _try_route_with_offsets(self, p1: QPointF, p2: QPointF, mode: str) -> Optional[List[QPointF]]:
        if abs(p1.x() - p2.x()) < 1e-6 or abs(p1.y() - p2.y()) < 1e-6:
            pts = [p1, p2]
            if not self._polyline_overlaps(pts):
                return pts

        max_k = 10
        for k in range(0, max_k + 1):
            if k == 0:
                off = 0
            else:
                sgn = 1 if k % 2 == 1 else -1
                off = sgn * ((k + 1) // 2) * self.TRACK_STEP

            if mode == "HV":
                x_mid = p2.x() + off
                pts = self._compress([p1, QPointF(x_mid, p1.y()), QPointF(x_mid, p2.y()), p2])
            else:
                y_mid = p2.y() + off
                pts = self._compress([p1, QPointF(p1.x(), y_mid), QPointF(p2.x(), y_mid), p2])

            if not self._polyline_overlaps(pts):
                return pts

        return None

    def _compress(self, pts: List[QPointF]) -> List[QPointF]:
        if not pts:
            return pts
        out = [pts[0]]
        for p in pts[1:]:
            if abs(p.x() - out[-1].x()) < 1e-6 and abs(p.y() - out[-1].y()) < 1e-6:
                continue
            out.append(p)

        if len(out) <= 2:
            return out

        res = [out[0]]
        for i in range(1, len(out) - 1):
            a, b, c = res[-1], out[i], out[i + 1]
            if (abs(a.x() - b.x()) < 1e-6 and abs(b.x() - c.x()) < 1e-6) or \
               (abs(a.y() - b.y()) < 1e-6 and abs(b.y() - c.y()) < 1e-6):
                continue
            res.append(b)
        res.append(out[-1])
        return res

    def _load_profile_kw(self, bus_id: int) -> np.ndarray | None:
        """返回该bus全时段load序列（kW）。取不到就返回None。"""
        if self._system is None:
            return None
        i = self._bus_index.get(bus_id, None)
        if i is None:
            return None

        # 兼容不同字段名
        for attr in ("load_kw", "loads_kw", "load_p_kw", "p_load_kw", "P_load_kw"):
            mat = getattr(self._system, attr, None)
            if mat is None:
                continue
            try:
                mat = np.asarray(mat, dtype=float)
                # (n_bus, T)
                if mat.ndim == 2 and mat.shape[0] == len(self._system.buses):
                    return mat[i, :]
                # (T, n_bus)
                if mat.ndim == 2 and mat.shape[1] == len(self._system.buses):
                    return mat[:, i]
            except Exception:
                pass

        return None

    def _polyline_midpoint_and_dir(self, pts: List[QPointF]) -> Tuple[QPointF, Tuple[float, float]]:
        """
        Return geometric midpoint of a polyline and the local direction unit vector at that midpoint.
        """
        if len(pts) < 2:
            return pts[0], (1.0, 0.0)

        segs = []
        total = 0.0
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            dx, dy = (b.x() - a.x()), (b.y() - a.y())
            L = (dx * dx + dy * dy) ** 0.5
            if L > 1e-9:
                segs.append((a, b, L))
                total += L

        if total < 1e-9:
            return pts[0], (1.0, 0.0)

        half = total / 2.0
        acc = 0.0
        for (a, b, L) in segs:
            if acc + L >= half:
                t = (half - acc) / L
                x = a.x() + t * (b.x() - a.x())
                y = a.y() + t * (b.y() - a.y())
                dx = (b.x() - a.x()) / L
                dy = (b.y() - a.y()) / L
                return QPointF(x, y), (dx, dy)
            acc += L

        # fallback
        a, b, L = segs[-1]
        dx = (b.x() - a.x()) / L
        dy = (b.y() - a.y()) / L
        return QPointF(b.x(), b.y()), (dx, dy)

    def _polyline_overlaps(self, pts: List[QPointF]) -> bool:
        if len(pts) < 2:
            return False
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if abs(a.y() - b.y()) < 1e-6:
                yk = int(round(a.y()))
                x1, x2 = sorted([a.x(), b.x()])
                if self._interval_overlaps(self._used_h.get(yk, []), (x1, x2)):
                    return True
            elif abs(a.x() - b.x()) < 1e-6:
                xk = int(round(a.x()))
                y1, y2 = sorted([a.y(), b.y()])
                if self._interval_overlaps(self._used_v.get(xk, []), (y1, y2)):
                    return True
            else:
                return True
        return False

    @staticmethod
    def _interval_overlaps(existing: List[Tuple[float, float]], seg: Tuple[float, float]) -> bool:
        x1, x2 = seg
        if x2 - x1 < 1e-6:
            return False
        for a, b in existing:
            if not (x2 <= a or x1 >= b):
                return True
        return False

    def _reserve_polyline(self, pts: List[QPointF]):
        if len(pts) < 2:
            return
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if abs(a.y() - b.y()) < 1e-6:
                yk = int(round(a.y()))
                x1, x2 = sorted([a.x(), b.x()])
                self._used_h.setdefault(yk, []).append((x1, x2))
            elif abs(a.x() - b.x()) < 1e-6:
                xk = int(round(a.x()))
                y1, y2 = sorted([a.y(), b.y()])
                self._used_v.setdefault(xk, []).append((y1, y2))

    def _make_arrow_at_end(self, pts: List[QPointF]) -> QGraphicsPolygonItem:
        if len(pts) < 2:
            return QGraphicsPolygonItem(QPolygonF())

        p_end = pts[-1]
        p_prev = pts[-2]
        dx = p_end.x() - p_prev.x()
        dy = p_end.y() - p_prev.y()
        L = (dx * dx + dy * dy) ** 0.5
        if L < 1e-6:
            return QGraphicsPolygonItem(QPolygonF())

        ux, uy = dx / L, dy / L
        arrow_len = 12
        arrow_w = 6

        tip = QPointF(p_end.x() - 6 * ux, p_end.y() - 6 * uy)
        base = QPointF(tip.x() - arrow_len * ux, tip.y() - arrow_len * uy)
        px, py = -uy, ux
        left = QPointF(base.x() + arrow_w * px, base.y() + arrow_w * py)
        right = QPointF(base.x() - arrow_w * px, base.y() - arrow_w * py)

        poly = QPolygonF([tip, left, right])
        item = QGraphicsPolygonItem(poly)
        item.setBrush(QBrush(Qt.black))
        item.setPen(QPen(Qt.black, 1.0))
        return item

    # ---------------- devices & load symbols ----------------
    def _preferred_device_lr(self, bus_id: int) -> str:
        x = self._pos[bus_id].x()
        if abs(x) < 1e-6:
            return "left"
        return "right" if x > 0 else "left"

    def _draw_device_box(
            self,
            bus_id: int,
            name: str,
            idx: int,
            side: str = "down",  # "down" or "up"
            lr: str = "left",  # "left" or "right"
            anchor_frac: float | None = None,  # None=按lr默认；0=中点；0.5=右1/4；-0.5=左1/4
            arrow_toward_bus: bool = False  # True: 箭头指向母线（如PV注入）
    ):
        c = self._pos[bus_id]
        half = self.BUSBAR_LEN / 2.0

        w = float(getattr(self, "DEV_BOX_W", 160))
        h = float(getattr(self, "DEV_BOX_H", 80))
        gap_x = float(getattr(self, "DEV_BOX_GAP_X", 18))
        v0 = float(getattr(self, "DEV_BOX_VOFFSET", 40))
        vgap = float(getattr(self, "DEV_BOX_VGAP", 12))
        line_w = float(getattr(self, "EDGE_W", 2.0))

        # ---- anchor on busbar ----
        if anchor_frac is None:
            anchor_frac = (-0.5 if lr == "left" else 0.5)

        ax = c.x() + anchor_frac * half
        anchor = QPointF(ax, c.y())

        # ---- box position ----
        if lr == "left":
            x0 = (c.x() - half) - gap_x - w
        else:
            x0 = (c.x() + half) + gap_x

        if side == "down":
            y0 = c.y() + v0 + idx * (h + vgap)
        else:
            y0 = c.y() - v0 - h - idx * (h + vgap)

        rect = InteractiveRectItem(QRectF(x0, y0, w, h), object_type='device', object_id=(bus_id, name, idx), on_dbl_click=self._open_detail_window)
        rect.setPen(QPen(Qt.black, 1.6))
        rect.setBrush(QBrush(Qt.white))
        self.scene.addItem(rect)

        # ---- text ----
        txt = InteractiveTextItem('', object_type='device', object_id=(bus_id, name, idx), on_dbl_click=self._open_detail_window)
        f = QFont()
        f.setPointSize(getattr(self, "FONT_DEVICE", 11))
        f.setBold(True)
        txt.setFont(f)
        txt.setDefaultTextColor(Qt.black)
        txt.setHtml(f"<div align='center'><b>{name}</b><br/>P=? kW</div>")
        txt.setTextWidth(w - 8)
        br = txt.boundingRect()
        txt.setPos(x0 + (w - br.width()) / 2.0, y0 + (h - br.height()) / 2.0)
        self.scene.addItem(txt)
        self.device_text[(bus_id, name, idx)] = txt

        # ---- connector endpoints ----
        if lr == "left":
            box_attach = QPointF(x0 + w, y0 + h / 2.0)  # 连接到盒子右中点
        else:
            box_attach = QPointF(x0, y0 + h / 2.0)  # 连接到盒子左中点

        p_vert = QPointF(anchor.x(), box_attach.y())

        # polyline path
        path = QPainterPath(anchor)
        path.lineTo(p_vert)
        path.lineTo(box_attach)

        conn = QGraphicsPathItem(path)
        conn.setPen(QPen(Qt.black, line_w))
        self.scene.addItem(conn)

        # ---- arrow head ----
        def add_triangle(tip: QPointF, dx: float, dy: float, length=12.0, width=6.0):
            # (dx,dy) should be unit-ish direction of arrow pointing "toward tip"
            L = math.hypot(dx, dy)
            if L < 1e-9:
                return
            ux, uy = dx / L, dy / L
            base = QPointF(tip.x() - ux * length, tip.y() - uy * length)
            px, py = -uy, ux
            left = QPointF(base.x() + px * width, base.y() + py * width)
            right = QPointF(base.x() - px * width, base.y() - py * width)
            poly = QPolygonF([tip, left, right])
            head = QGraphicsPolygonItem(poly)
            head.setBrush(QBrush(Qt.black))
            head.setPen(QPen(Qt.black, 1.0))
            self.scene.addItem(head)

        if arrow_toward_bus:
            # 箭头指向母线：放在 anchor 处，沿最后一段（p_vert -> anchor）方向
            dx = anchor.x() - p_vert.x()
            dy = anchor.y() - p_vert.y()
            add_triangle(anchor, dx, dy)
        else:
            # 默认：箭头指向设备：放在 box_attach，沿最后一段（p_vert -> box_attach）方向
            dx = box_attach.x() - p_vert.x()
            dy = box_attach.y() - p_vert.y()
            add_triangle(box_attach, dx, dy)

    def _draw_load_arrow(self, bus_id: int, anchor_frac: float = 0.5):
        """
        Load arrow: always vertical downward.
        anchor_frac:
          0.5  -> 右1/4点（默认）
          0.33 -> 右1/3处
          0.0  -> 中点
        """
        prof = self._load_profile_kw(bus_id)
        if prof is not None and np.allclose(prof, 0.0):
            return  # 全为0则不画load

        c = self._pos[bus_id]
        half = self.BUSBAR_LEN / 2.0

        anchor_x = c.x() + float(anchor_frac) * half

        L = self.Y_STEP / 3.0  # 你要求：延长至 1/3 垂直间隔
        p1 = QPointF(anchor_x, c.y())
        p2 = QPointF(anchor_x, c.y() + L)

        path = QPainterPath(p1)
        path.lineTo(p2)

        item = QGraphicsPathItem(path)
        item.setPen(QPen(Qt.black, 1.4))
        self.scene.addItem(item)
        self.load_arrow_item[bus_id] = item

        # arrow head at p2 pointing down
        arrow_len = 12
        arrow_w = 6
        tip = p2
        base = QPointF(tip.x(), tip.y() - arrow_len)
        left = QPointF(base.x() - arrow_w, base.y())
        right = QPointF(base.x() + arrow_w, base.y())

        poly = QPolygonF([tip, left, right])
        head = QGraphicsPolygonItem(poly)
        head.setBrush(QBrush(Qt.black))
        head.setPen(QPen(Qt.black, 1.0))
        self.scene.addItem(head)
        self.load_arrow_head[bus_id] = head

        # load text to the RIGHT of arrow head
        txt = InteractiveTextItem("Load=? kW", object_type="load", object_id=bus_id, on_dbl_click=self._open_detail_window)
        f = QFont()
        f.setPointSize(getattr(self, "FONT_LOAD", 11))
        txt.setFont(f)
        txt.setDefaultTextColor(Qt.darkBlue)

        br = txt.boundingRect()
        txt.setPos(tip.x() + 8.0, tip.y() - br.height() / 2.0)

        # background rectangle (load must be boxed)
        box = InteractiveRectItem(QRectF(txt.x() - 6, txt.y() - 4, br.width() + 12, br.height() + 8),
                                  object_type="load", object_id=bus_id, on_dbl_click=self._open_detail_window)
        box.setPen(QPen(Qt.black, 1.2))
        box.setBrush(QBrush(Qt.white))
        box.setZValue(txt.zValue() - 1)

        self.scene.addItem(box)
        self.scene.addItem(txt)
        self.load_box_item[bus_id] = box
        self.load_text[bus_id] = txt

    # ---------------- device collection ----------------
    def _collect_devices_by_bus(self, system) -> Dict[int, List[str]]:
        dev: Dict[int, List[str]] = {b.bus_id: [] for b in system.buses}

        # ESS
        if getattr(system, "ess", None) is not None:
            dev[system.ess.node_number].append("ESS")

        # Reefer (aggregate per bus)
        if getattr(system, "reefers", None) is not None:
            reefer_by_bus: Dict[int, int] = {}
            for rf in system.reefers:
                reefer_by_bus[rf.node_number] = reefer_by_bus.get(rf.node_number, 0) + 1
            for bid in reefer_by_bus.keys():
                dev[bid].append("Reefer")

        # Cold-ironing
        if getattr(system, "cold_ironing", None) is not None:
            ci_by_bus: Dict[int, int] = {}
            for tk in system.cold_ironing:
                # NOTE: your task schema must include node_number
                ci_by_bus[tk.node_number] = ci_by_bus.get(tk.node_number, 0) + 1
            for bid in ci_by_bus.keys():
                dev[bid].append("Cold-Ironing")

        # PV
        if getattr(system, "pv_kw", None) is not None:
            for i, b in enumerate(system.buses):
                try:
                    if (system.pv_kw[i, :] > 1e-6).any():
                        dev[b.bus_id].append("PV")
                except Exception:
                    pass

        dev = {k: v for k, v in dev.items() if len(v) > 0}
        return dev

    def wheelEvent(self, event):
        """Mouse wheel zoom (Ctrl not required)."""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        if delta > 0:
            if self._zoom >= self._zoom_max:
                return
            factor = self._zoom_step
            self._zoom += 1
        else:
            if self._zoom <= self._zoom_min:
                return
            factor = 1.0 / self._zoom_step
            self._zoom -= 1

        self.scale(factor, factor)


    def mousePressEvent(self, event):
        """Middle button (or left+Space if you want later) to start panning."""
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)


    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()

            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

            event.accept()
            return
        super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _bus_inner_quarter_anchor(self, bus_id: int, toward_x: float) -> QPointF:
        """Return the inner quarter-point on the busbar, pointing toward toward_x."""
        c = self._pos[bus_id]
        half = self.BUSBAR_LEN / 2.0
        q = half / 2.0  # quarter offset
        if toward_x >= c.x():
            return QPointF(c.x() + q, c.y())
        else:
            return QPointF(c.x() - q, c.y())

    def _value_from_matrix(self, mat, bus_i: int, t: int):
        """Return mat value robustly for shapes (n_bus, T) or (T, n_bus)."""
        if mat is None:
            return None
        try:
            n0, n1 = mat.shape
            if n0 == len(self._bus_index):  # (n_bus, T)
                return float(mat[bus_i, t])
            if n1 == len(self._bus_index):  # (T, n_bus)
                return float(mat[t, bus_i])
        except Exception:
            pass
        # fallback try common indexing
        try:
            return float(mat[bus_i, t])
        except Exception:
            try:
                return float(mat[t, bus_i])
            except Exception:
                return None

    def _polyline_label_pos(self, pts: list[QPointF], offset: float) -> QPointF:
        """
        给折线 pts 找一个“在线中点附近、垂直偏移 offset”的 label 位置。
        pts: [p0, p1, p2, ...]
        """
        if len(pts) < 2:
            return pts[0] if pts else QPointF(0, 0)

        # 找“最长的一段”作为参考方向（更稳定）
        best_i = 0
        best_len = -1.0
        for i in range(len(pts) - 1):
            dx = pts[i + 1].x() - pts[i].x()
            dy = pts[i + 1].y() - pts[i].y()
            L = dx * dx + dy * dy
            if L > best_len:
                best_len = L
                best_i = i

        p0, p1 = pts[best_i], pts[best_i + 1]
        dx = p1.x() - p0.x()
        dy = p1.y() - p0.y()
        L = math.hypot(dx, dy)
        if L < 1e-9:
            L = 1.0

        # 法向量（单位）
        nx, ny = -dy / L, dx / L

        # 线中点（用整条 polyline 的几何中点近似：取首尾中点也行）
        mid = QPointF((pts[0].x() + pts[-1].x()) * 0.5, (pts[0].y() + pts[-1].y()) * 0.5)

        # 垂直偏移
        return QPointF(mid.x() + nx * offset, mid.y() + ny * offset)

    def _busbar_point(self, bus_id: int, frac: float):
        """
        frac=0 表示母线中点
        frac=+0.5 表示右1/4点（更靠右端）
        frac=-0.5 表示左1/4点
        frac=+0.33 表示右1/3点
        """
        c = self._pos[bus_id]
        half = self.BUSBAR_LEN / 2.0
        return c.x() + frac * half, c.y()


# ---------------- time-series detail dialog ----------------
class TimeSeriesDialog(QDialog):
    """Simple, self-contained detail window. Uses matplotlib if available."""
    def __init__(self, parent=None, view: TopologyView | None = None, object_type: str = "", object_id=None):
        super().__init__(parent)
        self.setWindowTitle(f"Details: {object_type} {object_id}")
        self.resize(900, 520)
        self._view = view
        self._object_type = object_type
        self._object_id = object_id

        layout = QVBoxLayout(self)
        self._label = QLabel(self._summary_text())
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        # Try matplotlib
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
        except Exception:
            layout.addWidget(QLabel("Matplotlib is not available; cannot plot time-series in this environment."))
            return

        fig = Figure()
        self._canvas = FigureCanvas(fig)
        layout.addWidget(self._canvas)

        ax = fig.add_subplot(111)
        series = self._get_series()

        if not series:
            ax.text(0.5, 0.5, "No time-series data available for this object.", ha="center", va="center")
            self._canvas.draw()
            return

        T = None
        for k, y in series.items():
            if y is None:
                continue
            y = np.asarray(y, dtype=float).reshape(-1)
            if T is None:
                T = len(y)
            x = np.arange(len(y))
            ax.plot(x, y, label=str(k))

        ax.set_xlabel("Time index")
        ax.grid(True)
        ax.legend()
        self._canvas.draw()

    def _summary_text(self) -> str:
        if self._view is None or self._view._system is None:
            return ""
        sys = self._view._system
        ot = self._object_type
        oid = self._object_id
        if ot == "bus" and isinstance(oid, int):
            name = ""
            for b in sys.buses:
                if int(b.bus_id) == int(oid):
                    name = getattr(b, "bus_name", "") or ""
                    break
            return f"Bus {oid} {('('+name+')') if name else ''}"
        return f"{ot}: {oid}"

    def _get_series(self) -> Dict[str, np.ndarray]:
        view = self._view
        if view is None or view._system is None:
            return {}
        sys = view._system
        r = view._result

        ot = self._object_type
        oid = self._object_id
        out: Dict[str, np.ndarray] = {}

        if ot == "bus" and r is not None and isinstance(oid, int):
            try:
                bi = {bid: i for i, bid in enumerate(r.bus_ids)}[oid]
                out["V(pu)"] = np.asarray(r.V_bus[:, bi], dtype=float)
                out["theta"] = np.asarray(r.theta_bus[:, bi], dtype=float)
            except Exception:
                pass
            # load/pv
            try:
                i = view._bus_index.get(oid, None)
                if i is not None and getattr(sys, "load_kw", None) is not None:
                    out["Load(kW)"] = np.asarray(sys.load_kw[i, :], dtype=float)
                if i is not None and getattr(sys, "pv_kw", None) is not None:
                    out["PV(kW)"] = np.asarray(sys.pv_kw[i, :], dtype=float)
            except Exception:
                pass
            return out

        if ot == "load" and isinstance(oid, int):
            # load profile
            prof = view._load_profile_kw(oid)
            if prof is not None:
                out["Load(kW)"] = np.asarray(prof, dtype=float)
            return out

        if ot == "device" and isinstance(oid, tuple) and len(oid) == 3:
            bus_id, name, idx = oid
            # PV profile
            if str(name) == "PV":
                i = view._bus_index.get(bus_id, None)
                if i is not None and getattr(sys, "pv_kw", None) is not None:
                    out["P_pv(kW)"] = np.asarray(sys.pv_kw[i, :], dtype=float)
            # ESS (if result includes)
            if str(name) == "ESS" and r is not None:
                for cand in ("P_ess_bus", "P_ess"):
                    if hasattr(r, cand):
                        arr = getattr(r, cand)
                        arr = np.asarray(arr, dtype=float)
                        if arr.ndim == 2:
                            i = view._bus_index.get(bus_id, None)
                            if i is not None:
                                out["P_ess(kW)"] = arr[i, :]
                        elif arr.ndim == 1:
                            out["P_ess(kW)"] = arr
                        break
                if hasattr(r, "SOC"):
                    out["SOC"] = np.asarray(r.SOC, dtype=float)
            # Cold-Ironing: use fixed demand profile if available
            if str(name) == "Cold-Ironing":
                T = getattr(sys, "T", None)
                if T is not None:
                    y = np.zeros(int(T), dtype=float)
                    for task in getattr(sys, "cold_ironing", []) or []:
                        if int(task.node_number) != int(bus_id):
                            continue
                        st = int(task.start_time)
                        en = int(task.end_time)
                        y[st:en] += float(getattr(task, "demand_fix_kw", 0.0))
                    out["P_fix(kW)"] = y
            # Reefer: if pre-aggregated by bus exists
            if str(name) in ("Reefer", "Reefer_group") and getattr(view, "_P_reefer_bus", None) is not None and r is not None:
                try:
                    bi = {bid: i for i, bid in enumerate(r.bus_ids)}[bus_id]
                    out["P_reefer(kW)"] = np.asarray(view._P_reefer_bus[bi, :], dtype=float)
                except Exception:
                    pass
            return out

        if ot == "line" and r is not None and isinstance(oid, tuple) and len(oid) == 2:
            ek = (int(oid[0]), int(oid[1]))
            try:
                ei = {k: i for i, k in enumerate(r.edge_keys)}[ek]
            except Exception:
                # try reversed
                try:
                    ei = {k: i for i, k in enumerate(r.edge_keys)}[(ek[1], ek[0])]
                    ek = (ek[1], ek[0])
                except Exception:
                    return out
            out["P_line(kW)"] = np.asarray(r.P_line[:, ei], dtype=float)
            cap = view._line_capacity_kw(*ek)
            if cap is not None and cap > 1e-9:
                out["Loading(%)"] = np.abs(out["P_line(kW)"]) / cap * 100.0
            return out

        if ot == "transformer":
            # currently only percent if result provides it
            if r is not None:
                for attr in ("tx_loading_pct", "transformer_loading_pct"):
                    if hasattr(r, attr):
                        out["TX Loading(%)"] = np.asarray(getattr(r, attr), dtype=float).reshape(-1)
                        break
            return out

        return out
