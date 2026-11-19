# -*- coding: utf-8 -*-
"""
app_gui.py —— PySide6 桌面版：无人机照片面积量算

三栏布局
--------
左：照片列表（● 已标 n 块）   中：图像画布   右：本图地块 + 当前对象

两类状态（重要）
----------------
    草稿：识别完立刻进右侧列表，可改名可删；只在内存里，本次运行内跨图保留。
    保存：按 Ctrl+S 写进照片文件夹的 project.json；关闭软件时若未保存会提示。

按键（画布获得焦点时）
----------------------
    左键 点目标    右键 排除该点
    R/回车 结束当前对象（准备标下一个）    U 撤销一个点
    Delete 删除选中地块    F2 改名
    A/D 上一张/下一张    Ctrl+S 保存工程

用法
----
    python app_gui.py                        # 启动后「打开文件夹」
    python app_gui.py "照片文件夹"            # 直接打开（也支持拖到 bat 上）
    python app_gui.py --selftest 照片 --click u,v --shot out.png   # 无人值守自检
"""
from __future__ import annotations

import dataclasses
import math
import os
import shutil
import sys
import tempfile

import PySide6  # noqa: F401  必须先于 cv2/torch 导入 Qt
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QIcon, QImage,
                           QKeySequence, QPainter, QPen, QPixmap, QPolygonF,
                           QShortcut)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame,
                               QGraphicsEllipseItem, QGraphicsItem,
                               QGraphicsPixmapItem, QGraphicsPolygonItem,
                               QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QToolBar, QVBoxLayout, QWidget)

import cv2                  # 必须早于 torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_geo import (  # noqa: E402
    Camera, DjiMeta, imread_unicode, m2_to_mu, read_dji_meta,
)
from measure_area import (  # noqa: E402
    DEFAULT_CONFIG, DEFAULT_WEIGHT, Segmenter, build_field, pick_mask,
)
from project_store import FieldRec, ProjectStore  # noqa: E402

APP_NAME = "无人机照片面积量算"
ITEM_NO_SCALE = QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations

# 每个地块一个固定颜色：画布多边形与右侧表格圆点同色，靠颜色对号入座
PALETTE = ["#0B84F3", "#12B886", "#F59F00", "#E8590C",
           "#9C36B5", "#1C7ED6", "#C2255C", "#2B8A3E"]

STYLE = """
* { font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI"; }
QMainWindow { background: #F8FAFC; }
QToolBar { background: #FFFFFF; border: none; border-bottom: 1px solid #E4E7EB;
           padding: 8px; spacing: 4px; }
QToolButton { padding: 6px 12px; border-radius: 6px; color: #0F172A;
              font-size: 13px; background: transparent; }
QToolButton:hover { background: #F1F3F5; }
QToolButton:pressed { background: #E2E8F0; }
QToolButton:disabled { color: #94A3B8; }
QToolButton#primary { background: #2563EB; color: #FFFFFF; }
QToolButton#primary:hover { background: #1D4ED8; }

QListWidget { background: #FFFFFF; border: none; outline: 0; font-size: 13px; }
QListWidget::item { padding: 8px; margin: 2px 8px; border-radius: 6px; color: #334155; }
QListWidget::item:hover { background: #F1F3F5; }
QListWidget::item:selected { background: #EFF4FF; color: #1E3A5F; }

QTableWidget { background: #FFFFFF; border: none; outline: 0; font-size: 13px;
               gridline-color: transparent; }
QTableWidget::item { padding: 6px 8px; border-bottom: 1px solid #F1F3F5; color: #0F172A; }
QTableWidget::item:selected { background: #EFF4FF; color: #1E3A5F; }
QHeaderView::section { background: #FFFFFF; border: none;
                       border-bottom: 1px solid #E4E7EB; padding: 8px;
                       color: #64748B; font-size: 12px; font-weight: 500; }

QStatusBar { background: #FFFFFF; border-top: 1px solid #E4E7EB;
             color: #64748B; font-size: 12px; }
QStatusBar::item { border: none; }

QLabel#panelTitle { color: #64748B; font-size: 12px; padding: 12px 12px 4px 12px; }
QLabel#hint { color: #64748B; font-size: 12px; }
QLabel#chip { font-size: 12px; padding: 2px 8px; border-radius: 9px; }

QFrame#card { background: #FFFFFF; border: 1px solid #E4E7EB; border-radius: 8px; }
QFrame#cardActive { background: #FFFFFF; border: 1px solid #2563EB; border-radius: 8px; }

QPushButton { background: #FFFFFF; border: 1px solid #D7DCE2; border-radius: 6px;
              padding: 6px 12px; color: #0F172A; font-size: 12px; }
QPushButton:hover { background: #F8FAFC; border-color: #B6BFCC; }
QPushButton:disabled { color: #94A3B8; border-color: #E4E7EB; }
QPushButton#primary { background: #2563EB; border-color: #2563EB; color: #FFFFFF; }
QPushButton#danger { color: #DC2626; border-color: #F3C6C6; }
QPushButton#danger:hover { background: #FEF2F2; }

QSplitter::handle { background: #E4E7EB; width: 1px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 5px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: #94A3B8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""



# =========================================================
# 1. 推理引擎（普通线程 + 队列）
# =========================================================
class InferEngine:
    """
    为什么不用 QThread：实测 QThread 里跑 SAM2 的 predict 会卡死
    （同代码用 threading.Thread 正常）。这里用线程 + 双队列，主线程用 QTimer 轮询。
    """

    def __init__(self):
        import queue
        import threading

        self.jobs = queue.Queue()
        self.out = queue.Queue()
        self.cfg = {"target_z": 0.0, "use_lrf": False, "n_samples": 300,
                    "clean": True, "fill_holes": True}
        self._seg = None
        self._cam = None
        threading.Thread(target=self._loop, daemon=True).start()

    def submit_load(self, rid, path):
        self.jobs.put(("load", rid, {"path": path}))

    def submit_predict(self, rid, points, labels):
        self.jobs.put(("predict", rid, {"points": list(points), "labels": list(labels)}))

    def configure(self, **kw):
        self.jobs.put(("config", -1, kw))

    def poll(self):
        import queue

        items = []
        while True:
            try:
                items.append(self.out.get_nowait())
            except queue.Empty:
                return items

    def _loop(self):
        while True:
            kind, rid, kw = self.jobs.get()
            try:
                if kind == "config":
                    self.cfg.update(kw)
                elif kind == "load":
                    self._do_load(rid, kw["path"])
                elif kind == "predict":
                    self._do_predict(rid, kw["points"], kw["labels"])
            except Exception as exc:
                self.out.put(("failed", rid, f"{type(exc).__name__}: {exc}".strip()))

    def _do_load(self, rid, path):
        import time

        t0 = time.time()
        if self._seg is None:
            print("[引擎] 加载 SAM2 模型…", flush=True)
            self._seg = Segmenter(DEFAULT_WEIGHT, DEFAULT_CONFIG)
            print(f"[引擎] 模型就绪 {time.time() - t0:.1f}s device={self._seg.device}",
                  flush=True)
        img = imread_unicode(path)
        meta = read_dji_meta(path)
        cfg = self.cfg
        cam = Camera(meta, target_z=float(cfg["target_z"]),
                     height_m=meta.lrf_drop_m if cfg["use_lrf"] else None)
        self._seg.set_image(img)
        self._cam = cam
        print(f"[引擎] 编码完成 {time.time() - t0:.1f}s", flush=True)
        self.out.put(("loaded", rid, {
            "path": path, "img": img, "meta": dataclasses.asdict(meta), "h": cam.h}))

    def _do_predict(self, rid, points, labels):
        if self._cam is None or self._seg is None:
            self.out.put(("failed", rid, "图像还没准备好"))
            return
        masks, scores = self._seg.predict(points, labels)
        idx = pick_mask(masks, scores)
        cfg = self.cfg
        f = build_field(self._cam, masks[idx], 1, int(cfg["n_samples"]),
                        score=float(scores[idx]), clean=bool(cfg["clean"]),
                        fill_holes=bool(cfg["fill_holes"]))
        self.out.put(("predicted", rid, {
            "contour": f.contour_px, "area_m2": f.area_m2, "area_mu": f.area_mu,
            "score": float(scores[idx]), "warnings": list(f.warnings),
            "center": f.center_latlon, "pixel_area": f.pixel_area, "h": self._cam.h,
        }))


# =========================================================
# 2. 画布
# =========================================================
class Canvas(QGraphicsView):
    """场景坐标 = 原图像素坐标，避免二次换算。"""

    pointClicked = Signal(float, float, int)

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(38, 40, 43)))
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFrameShape(QFrame.NoFrame)
        self._pix_item = None
        self._items = []
        self._point_items = []
        self._press = None
        self._pan = False

    def show_image(self, bgr: np.ndarray):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._scene.clear()
        self._pix_item = QGraphicsPixmapItem(QPixmap.fromImage(img))
        self._scene.addItem(self._pix_item)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self._items, self._point_items = [], []
        self.fitInView(self._pix_item, Qt.KeepAspectRatio)

    def show_message(self, text: str, color=QColor(255, 120, 120)):
        if self._pix_item is None:
            self._scene.setSceneRect(QRectF(0, 0, 1000, 600))
        t = QGraphicsSimpleTextItem(text)
        f = QFont()
        f.setPointSize(13)
        t.setFont(f)
        t.setBrush(QBrush(color))
        t.setFlag(ITEM_NO_SCALE)
        r = self.sceneRect()
        t.setPos(r.center().x() - 240, r.center().y() - 16)
        self._scene.addItem(t)
        self._items.append(t)

    def set_fields(self, items, selected_fid=None):
        for it in self._items:
            self._scene.removeItem(it)
        self._items = []
        for fid, name, contour, area_m2, mu, color in items:
            poly = QPolygonF([QPointF(float(u), float(v)) for u, v in contour])
            sel = (fid == selected_fid)
            item = QGraphicsPolygonItem(poly)
            pen = QPen(QColor(color), 3 if sel else 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(95 if sel else 55)
            item.setBrush(QBrush(fill))
            self._scene.addItem(item)
            self._items.append(item)

            title = name if str(name).startswith("#") else f"#{fid} {name}"
            lab = QGraphicsSimpleTextItem(f"{title}  {area_m2:.1f} m2 / {mu:.3f} mu")
            f = QFont()
            f.setPointSize(10)
            f.setBold(True)
            lab.setFont(f)
            lab.setBrush(QBrush(QColor(255, 250, 230)))
            lab.setPen(QPen(QColor(0, 0, 0), 1))
            lab.setFlag(ITEM_NO_SCALE)
            lab.setPos(poly.boundingRect().center())
            self._scene.addItem(lab)
            self._items.append(lab)

    def set_points(self, pts):
        for it in self._point_items:
            self._scene.removeItem(it)
        self._point_items = []
        for u, v, lab in pts:
            r = 5
            item = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            item.setPen(QPen(QColor(255, 255, 255), 1))
            item.setBrush(QBrush(QColor(0, 230, 90) if lab == 1 else QColor(255, 70, 70)))
            item.setFlag(ITEM_NO_SCALE)
            item.setPos(float(u), float(v))
            self._scene.addItem(item)
            self._point_items.append(item)

    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mousePressEvent(self, ev):
        self.setFocus()
        if ev.button() == Qt.MiddleButton:
            self._pan = True
            self._press = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        self._press = ev.position()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan and self._press is not None:
            d = ev.position() - self._press
            self._press = ev.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(d.y()))
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._pan:
            self._pan = False
            self.setCursor(Qt.ArrowCursor)
            return
        moved = 999.0
        if self._press is not None:
            d = ev.position() - self._press
            moved = math.hypot(d.x(), d.y())
        self._press = None
        if moved < 5 and self._pix_item is not None:
            p = self.mapToScene(ev.position().toPoint())
            if ev.button() == Qt.LeftButton:
                self.pointClicked.emit(float(p.x()), float(p.y()), 1)
            elif ev.button() == Qt.RightButton:
                self.pointClicked.emit(float(p.x()), float(p.y()), 0)
        super().mouseReleaseEvent(ev)


# =========================================================
# 3. 高度基准对话框
# =========================================================
class HeightDialog(QDialog):
    def __init__(self, target_z: float, use_lrf: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("高度基准")
        self.setMinimumWidth(430)
        form = QFormLayout(self)

        tip = QLabel("目标地面比起飞点高多少米（高填正、低填负）。\n"
                     "面积与高度平方成正比：高差 10 m / 航高 100 m → 面积差约 20%。")
        tip.setWordWrap(True)
        tip.setObjectName("hint")
        form.addRow(tip)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(-500.0, 500.0)
        self.spin.setDecimals(1)
        self.spin.setSuffix(" m")
        self.spin.setValue(float(target_z))
        form.addRow("目标地面相对起飞点：", self.spin)

        self.chk = QCheckBox("用照片自带激光测距（LRF）自动修正")
        self.chk.setChecked(bool(use_lrf))
        form.addRow(self.chk)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self):
        return self.spin.value(), self.chk.isChecked()


class EnterEndsEdit(QObject):
    """
    接住名称输入框里的回车。

    为什么不用 QLineEdit.returnPressed：QTableWidget 的 item delegate 会在编辑器上装
    事件过滤器，回车先被它拦去提交并关闭编辑器，returnPressed 根本不会发出
    （实测：名字改成功，但"结束对象"没执行，下一个点仍落回原来那块）。
    事件过滤器按"后装先跑"的顺序执行，我们在编辑器创建之后安装，所以能抢先拿到回车。
    """

    def __init__(self, parent, callback):
        super().__init__(parent)
        self._cb = callback

    def install(self, editor):
        """装到编辑器上；过滤器以 editor 为父对象，编辑器销毁时一起回收。"""
        self.setParent(editor)
        editor.installEventFilter(self)
        return self

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress and ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._cb()
            return True
        return False


def _dot(color_hex: str) -> QPixmap:
    pm = QPixmap(10, 10)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color_hex))
    p.drawEllipse(1, 1, 8, 8)
    p.end()
    return pm


# =========================================================
# 4. 主窗口
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setStyleSheet(STYLE)

        self.proj: ProjectStore | None = None
        self.draft: dict[str, list] = {}      # 内存草稿：照片名 → [FieldRec]
        self.dirty = False
        self.active_fid = None                # 当前接收点选的地块（=表格选中行）
        self._fresh = set()                   # 本次运行新建的地块 (name, fid)
        self.idx = -1
        self.rid = 0
        self.prid = 0
        self.img = None
        self.meta = None
        self.cam = None
        self.points, self.labels = [], []
        self._bad = set()
        self._loading = False
        self._st = None

        self._build_ui()
        self._start_engine()

    # ---------- 界面 ----------
    def _build_ui(self):
        # 按屏幕可用区域定窗口大小：小屏（如 1366×768）不硬塞 1480
        scr = QApplication.primaryScreen().availableGeometry()
        win_w = max(1024, min(1480, scr.width() - 60))
        win_h = max(640, min(920, scr.height() - 60))
        self.resize(win_w, win_h)

        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(text, fn, shortcut=None, tip=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
            a.triggered.connect(fn)
            tb.addAction(a)
            return a

        act("打开文件夹", self.on_open, "Ctrl+O")
        self.act_save = act("保存工程", self.save_project, "Ctrl+S",
                            "把本次所有标注写入照片文件夹的 project.json")
        btn_save = tb.widgetForAction(self.act_save)
        if btn_save is not None:
            btn_save.setObjectName("primary")
        tb.addSeparator()
        self.act_prev = act("◀ 上一张", lambda: self.goto(self.idx - 1))
        self.act_next = act("下一张 ▶", lambda: self.goto(self.idx + 1))
        tb.addSeparator()
        act("高度基准", self.on_height)
        act("导出 CSV", lambda: self.on_export("csv"))
        act("导出 GeoJSON", lambda: self.on_export("geojson"))

        # ---- 左栏 ----
        self.lbl_list = QLabel("照片 0 张")
        self.lbl_list.setObjectName("panelTitle")
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.on_row_changed)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        lv.addWidget(self.lbl_list)
        lv.addWidget(self.list, 1)

        # ---- 中栏 ----
        self.canvas = Canvas()
        self.canvas.pointClicked.connect(self.on_point)

        # ---- 右栏 ----
        self.lbl_sum = QLabel("本图 0 块")
        self.lbl_sum.setObjectName("panelTitle")
        self.lbl_meta = QLabel("")
        self.lbl_meta.setObjectName("hint")
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setContentsMargins(10, 0, 10, 6)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["名称", "㎡", "亩"])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked
                                   | QAbstractItemView.EditKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 68)
        self.table.setColumnWidth(2, 58)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.itemSelectionChanged.connect(self.on_table_select)

        self.card_cur = QFrame()
        self.card_cur.setObjectName("card")
        cv = QVBoxLayout(self.card_cur)
        cv.setContentsMargins(10, 8, 10, 10)
        self.lbl_cur = QLabel("当前对象：无\n左键点一下目标开始标注")
        self.lbl_cur.setWordWrap(True)
        self.lbl_cur.setObjectName("hint")
        cv.addWidget(self.lbl_cur)
        row = QHBoxLayout()
        self.btn_finish = QPushButton("结束此对象 (R)")
        self.btn_finish.clicked.connect(self.finish_object)
        self.btn_del = QPushButton("删除选中")
        self.btn_del.setObjectName("danger")
        self.btn_del.clicked.connect(self.delete_selected_field)
        row.addWidget(self.btn_finish)
        row.addWidget(self.btn_del)
        cv.addLayout(row)
        self.card_cur.setLayout(cv)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(8, 0, 8, 8)
        bl.addWidget(self.card_cur)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(4)
        rv.addWidget(self.lbl_sum)
        rv.addWidget(self.lbl_meta)
        rv.addWidget(self.table, 1)
        rv.addWidget(bottom)
        right.setMinimumWidth(280)
        right.setMaximumWidth(430)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.canvas)
        split.addWidget(right)
        split.setSizes([int(win_w * 0.15), int(win_w * 0.60), int(win_w * 0.25)])
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        self.statusBar()
        self.lbl_status = QLabel("就绪　·　打开一个照片文件夹开始（Ctrl+O）")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_counts = QLabel("")
        self.lbl_counts.setStyleSheet("color:#64748B; padding-right:10px;")
        self.statusBar().addPermanentWidget(self.lbl_counts)
        self.lbl_state = QLabel("未打开工程")
        self.lbl_state.setObjectName("chip")
        self.lbl_state.setStyleSheet(
            "color:#64748B; background:#F1F3F5; border-radius:9px; padding:2px 8px;")
        self.statusBar().addPermanentWidget(self.lbl_state)

        self._bind_shortcuts()

    def _bind_shortcuts(self):
        def sc(key, fn, widget=None):
            s = QShortcut(QKeySequence(key), widget or self.canvas)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)

        sc("R", self.finish_object)
        sc("Return", self.finish_object)
        sc("U", self.undo_point)
        sc("Ctrl+Z", self.undo_point)
        sc("D", lambda: self.goto(self.idx + 1))
        sc("A", lambda: self.goto(self.idx - 1))
        sc("Delete", self.delete_selected_field, widget=self.table)
        sc("F2", self.rename_selected, widget=self.table)

    # ---------- 引擎 ----------
    def _start_engine(self):
        self.engine = InferEngine()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pump)
        self._timer.start(50)

    def _pump(self):
        for kind, rid, data in self.engine.poll():
            if kind == "loaded":
                self.on_loaded(rid, data)
            elif kind == "predicted":
                self.on_predicted(rid, data)
            else:
                self.on_failed(rid, data)

    def closeEvent(self, ev):
        if self.dirty:
            box = QMessageBox(self)
            box.setWindowTitle("还有未保存的标注")
            box.setText("本次标注还没写入 project.json，关闭后草稿会丢失。")
            box.setInformativeText("要现在保存吗？")
            box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard
                                   | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Save)
            r = box.exec()
            if r == QMessageBox.Cancel:
                ev.ignore()
                return
            if r == QMessageBox.Save:
                self.save_project()
        self._timer.stop()
        super().closeEvent(ev)

    # ---------- 工程 ----------
    def on_open(self):
        d = QFileDialog.getExistingDirectory(self, "选择照片文件夹")
        if d:
            self.open_project(d)

    def open_project(self, folder: str):
        try:
            proj = ProjectStore(folder)
        except Exception as exc:
            QMessageBox.critical(self, "打开失败", str(exc))
            return
        if not proj.images:
            QMessageBox.warning(self, "空文件夹", "这个文件夹里没有图片。")
            return
        self.proj = proj
        self.draft = {n: proj.fields(n) for n in proj.images}
        self.dirty = False
        self._bad.clear()
        self.active_fid = None
        self._fresh = set()
        self._refresh_list()
        start = 0
        for i, n in enumerate(self.proj.images):
            if self.draft[n]:
                start = i
                break
        self.idx = -1
        self.goto(start, force=True)
        self._update_title()
        self._update_state()

    def _update_title(self):
        if not self.proj:
            self.setWindowTitle(APP_NAME)
            return
        star = " *" if self.dirty else ""
        self.setWindowTitle(f"{APP_NAME} — {os.path.basename(self.proj.root)}{star}")

    def _update_state(self):
        if not self.proj:
            self.lbl_state.setText("未打开工程")
            self.lbl_state.setStyleSheet(
                "color:#64748B; background:#F1F3F5; border-radius:9px; padding:2px 8px;")
            self.lbl_counts.setText("")
            return
        n_lab = sum(1 for n in self.proj.images if self.draft.get(n))
        n_fld = sum(len(v) for v in self.draft.values())
        self.lbl_counts.setText(f"已标 {n_lab} 张 · {n_fld} 块")
        if self.dirty:
            self.lbl_state.setText("● 未保存")
            self.lbl_state.setStyleSheet(
                "color:#B45309; background:#FEF3C7; border-radius:9px; padding:2px 8px;")
        else:
            self.lbl_state.setText("已保存")
            self.lbl_state.setStyleSheet(
                "color:#047857; background:#ECFDF5; border-radius:9px; padding:2px 8px;")

    def _refresh_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for n in self.proj.images:
            k = len(self.draft.get(n) or [])
            it = QListWidgetItem(f"{'●' if k else '○'}  {n}    {k if k else ''}".rstrip())
            if n in self._bad:
                it.setForeground(QBrush(QColor(200, 0, 0)))
            self.list.addItem(it)
        self.list.blockSignals(False)
        n_lab = sum(1 for n in self.proj.images if self.draft.get(n))
        self.lbl_list.setText(f"照片 {len(self.proj.images)} 张 · 已标 {n_lab} 张")

    @staticmethod
    def _display_name(rec):
        n = str(rec.name)
        return n if n.startswith("#") else f"#{rec.fid} {n}"

    def _current_name(self):
        return self.proj.images[self.idx] if self.proj and self.idx >= 0 else None

    def _current_fields(self):
        n = self._current_name()
        return self.draft.setdefault(n, []) if n else []

    # ---------- 切图 ----------
    def goto(self, i, force=False):
        if not self.proj:
            return
        i = max(0, min(i, len(self.proj.images) - 1))
        if i == self.idx and not force:
            return
        self.idx = i
        self.list.blockSignals(True)
        self.list.setCurrentRow(i)
        self.list.blockSignals(False)
        self._load_current()

    def on_row_changed(self, row):
        if row >= 0 and row != self.idx:
            self.idx = row
            self._load_current()

    def _load_current(self):
        if not self.proj or self.idx < 0:
            return
        name = self._current_name()
        self.points, self.labels = [], []
        self.active_fid = None
        self.img, self.meta, self.cam = None, None, None
        self._loading = True
        self.canvas.set_points([])
        self._refresh_fields()
        self._refresh_cur_label()
        self.lbl_status.setText(f"{name}：读取参数并编码…")
        self.lbl_meta.setText("")
        self.canvas.show_message("编码中…", QColor(220, 220, 220))
        d = self.proj.defaults
        self.engine.configure(target_z=float(d.get("target_z", 0.0)),
                              use_lrf=bool(d.get("use_lrf", False)),
                              n_samples=300, clean=True, fill_holes=True)
        self.rid += 1
        self.engine.submit_load(self.rid, os.path.join(self.proj.root, name))

    def on_loaded(self, rid, p):
        if rid != self.rid:
            return
        self._loading = False
        self.img = p["img"]
        self.meta = p["meta"]
        self.cam = self._cam_from_meta(self.meta)
        self.canvas.show_image(self.img)
        self._refresh_fields()
        self.canvas.setFocus()
        self.lbl_status.setText(self._current_name())
        self.lbl_meta.setText(
            f"{self.img.shape[1]}×{self.img.shape[0]}　俯角 {self.meta['gimbal_pitch']:.1f}°\n"
            f"高度基准 {self.cam.h:.1f} m　GSD {self.cam.gsd_at_center() * 100:.2f} cm/px")
        if self._st and self._st.get("phase") == "load":
            self._st["phase"] = "click"
            QTimer.singleShot(200, self._st_click)

    def on_failed(self, rid, msg):
        print(f"[失败 rid={rid}] {msg}", flush=True)
        if rid == self.rid:
            self._loading = False
            name = self._current_name()
            self._bad.add(name)
            self._refresh_list()
            self.canvas.show_message("这张照片用不了：\n" + msg[:300])
            self.lbl_status.setText(f"{name}：无法读取 DJI 参数（按 D 跳过）")
        else:
            self.lbl_status.setText("分割失败：" + msg[:160])

    # ---------- 相机 ----------
    def _cam_from_meta(self, meta_dict):
        m = DjiMeta(**meta_dict)
        d = self.proj.defaults
        h = m.lrf_drop_m if d.get("use_lrf") else None
        return Camera(m, target_z=float(d.get("target_z", 0.0)), height_m=h)

    def _cam_builder(self, name, meta_dict):
        return self._cam_from_meta(meta_dict)

    # ---------- 右侧表格 ----------
    def _color_of(self, idx):
        return PALETTE[idx % len(PALETTE)]

    def _field_items(self):
        recs = self._current_fields()
        return [(r.fid, r.name, r.contour_px, r.area_m2, r.area_mu, self._color_of(i))
                for i, r in enumerate(recs)]

    def _refresh_fields(self):
        recs = self._current_fields()
        self.canvas.set_fields(self._field_items(), self._selected_fid())

        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for i, r in enumerate(recs):
            row = self.table.rowCount()
            self.table.insertRow(row)
            it = QTableWidgetItem(r.name)
            it.setData(Qt.UserRole, r.fid)
            it.setIcon(QIcon(_dot(self._color_of(i))))
            self.table.setItem(row, 0, it)
            for c, txt in ((1, f"{r.area_m2:.1f}"), (2, f"{r.area_mu:.3f}")):
                cell = QTableWidgetItem(txt)
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, c, cell)

        self.table.blockSignals(False)
        self._refresh_sum()

    def _refresh_sum(self):
        recs = self._current_fields()
        m2 = sum(r.area_m2 for r in recs)
        self.lbl_sum.setText(f"本图 {len(recs)} 块 · {m2:.1f} m² · {m2_to_mu(m2):.3f} 亩")

    def _refresh_cur_label(self):
        rec = self._active_rec()
        if rec is None:
            self.card_cur.setObjectName("card")
            self.card_cur.style().unpolish(self.card_cur)
            self.card_cur.style().polish(self.card_cur)
            self.lbl_cur.setText("当前对象：无\n下一次点击将新建一块；"
                                 "要继续改旧的，点右侧它的那一行")
            self.btn_finish.setEnabled(False)
            return
        self.card_cur.setObjectName("cardActive")
        self.card_cur.style().unpolish(self.card_cur)
        self.card_cur.style().polish(self.card_cur)
        warn = ("\n⚠ " + rec.warnings[0]) if rec.warnings else ""
        self.lbl_cur.setText(
            f"当前对象：{self._display_name(rec)}　"
            f"{rec.area_m2:.1f} m² / {rec.area_mu:.3f} 亩\n"
            f"左键补点、右键排除；按回车/R 结束；点右侧其他行可切换{warn}")
        self.btn_finish.setEnabled(True)

    def _active_rec(self):
        if self.active_fid is None:
            return None
        for r in self._current_fields():
            if r.fid == self.active_fid:
                return r
        return None

    def _selected_fid(self):
        sm = self.table.selectionModel()
        rows = sm.selectedRows() if sm else []
        if not rows:
            return None
        it = self.table.item(rows[0].row(), 0)
        return int(it.data(Qt.UserRole)) if it else None

    def on_table_select(self):
        fid = self._selected_fid()
        self.active_fid = fid
        self.canvas.set_fields(self._field_items(), fid)
        rec = self._active_rec()
        if rec is not None:
            self.lbl_status.setText(
                f"已选中 {self._display_name(rec)} —— 接下来点的点都针对它　"
                f"{rec.area_m2:.1f} m² / {rec.area_mu:.3f} 亩")
        self._refresh_cur_label()

    def on_item_changed(self, item):
        if item.column() != 0 or not self.proj:
            return
        fid = item.data(Qt.UserRole)
        if fid is None:
            return
        rec = None
        for r in self._current_fields():
            if r.fid == int(fid):
                rec = r
        if rec is None:
            return
        new = item.text().strip()
        if not new:
            self.table.blockSignals(True)
            item.setText(rec.name)
            self.table.blockSignals(False)
            return
        if new != rec.name:
            rec.name = new
            self.dirty = True
            self._update_title()
            self._update_state()
            self.canvas.set_fields(self._field_items(), self._selected_fid())
            self._refresh_cur_label()
            self.lbl_status.setText(f"已改名为「{new}」")

    def _delete_field(self, fid):
        recs = self._current_fields()
        recs[:] = [r for r in recs if r.fid != fid]
        self._fresh.discard((self._current_name(), fid))
        if self.active_fid == fid:
            self.active_fid = None
            self.points, self.labels = [], []
            self.canvas.set_points([])
        self.dirty = True
        self._refresh_fields()
        self._refresh_list()
        self._refresh_cur_label()
        self._update_title()
        self._update_state()
        self.lbl_status.setText(f"已删除 #{fid}")

    def delete_selected_field(self):
        fid = self._selected_fid()
        if fid is not None:
            self._delete_field(fid)

    def rename_selected(self):
        sm = self.table.selectionModel()
        rows = sm.selectedRows() if sm else []
        if not rows:
            return
        self.table.setFocus()
        self.table.editItem(self.table.item(rows[0].row(), 0))

    # ---------- 点选 ----------
    def on_point(self, u, v, label):
        if self.cam is None or self._loading:
            return
        self.points.append((u, v))
        self.labels.append(label)
        self.canvas.set_points([(a, b, c) for (a, b), c in zip(self.points, self.labels)])
        self.prid += 1
        self.lbl_status.setText("分割中…")
        self.engine.submit_predict(self.prid, self.points, self.labels)

    def on_predicted(self, rid, p):
        if rid != self.prid:
            return
        recs = self._current_fields()
        rec = self._active_rec()
        is_new = rec is None
        if is_new:
            fid = max([r.fid for r in recs], default=0) + 1
            rec = FieldRec(fid=fid, name=f"#{fid}",
                           contour_px=[list(map(float, q)) for q in p["contour"]],
                           area_m2=float(p["area_m2"]), area_mu=float(p["area_mu"]),
                           score=float(p["score"]), center=p.get("center"),
                           warnings=list(p.get("warnings") or []),
                           h_used=float(p["h"]))
            recs.append(rec)
            self._fresh.add((self._current_name(), fid))
        else:
            rec.contour_px = [list(map(float, q)) for q in p["contour"]]
            rec.area_m2 = float(p["area_m2"])
            rec.area_mu = float(p["area_mu"])
            rec.score = float(p["score"])
            rec.warnings = list(p.get("warnings") or [])
            rec.center = p.get("center")
            rec.h_used = float(p["h"])

        self.dirty = True
        self._refresh_fields()
        row = next(i for i, r in enumerate(recs) if r.fid == rec.fid)
        self.table.setCurrentCell(row, 0)      # 选中 = 激活，后续点都归它
        self._refresh_list()
        self._refresh_cur_label()
        self._update_title()
        self._update_state()
        self.lbl_status.setText(
            f"{'新对象' if is_new else '已更新'} {self._display_name(rec)}　"
            f"{rec.area_m2:.1f} m² / {rec.area_mu:.3f} 亩　置信度 {rec.score:.2f}")

        if is_new:
            self._start_rename(row)
            if self._st and self._st.get("phase") == "click":
                self._st["phase"] = "done"
                QTimer.singleShot(500, self._st_finish)

    def _start_rename(self, row):
        it = self.table.item(row, 0)
        if it is None:
            return
        self.table.setFocus()
        self.table.editItem(it)

        def _sel():
            ed = self.table.findChild(QLineEdit)
            if ed is None:
                return
            ed.selectAll()
            # 过滤器挂在编辑器上，编辑器销毁时一起消失，不会重复累积
            EnterEndsEdit(ed, self._on_name_committed).install(ed)

        QTimer.singleShot(0, _sel)

    def _on_name_committed(self):
        """名称框里按下回车：接受名称并结束当前对象。"""
        self.table.setFocus()      # 让委托提交文本、关闭编辑器（改名走 on_item_changed）
        self.finish_object()

    def undo_point(self):
        if not self.points:
            return
        self.points.pop()
        self.labels.pop()
        self.canvas.set_points([(a, b, c) for (a, b), c in zip(self.points, self.labels)])
        if self.points:
            self.prid += 1
            self.engine.submit_predict(self.prid, self.points, self.labels)
            return
        key = (self._current_name(), self.active_fid)
        if self.active_fid is not None and key in self._fresh:
            self._delete_field(self.active_fid)     # 本次新建的：连地块一起撤
        else:
            self.finish_object()                    # 老地块：只结束，不删
        self.lbl_status.setText("已撤销所有点")

    def finish_object(self):
        """结束当前对象：接受名称，下一次点击就是新的一块。"""
        if self.active_fid is None:
            return
        fid = self.active_fid
        self.active_fid = None
        self.table.clearSelection()          # 会触发 on_table_select → active=None
        self.points, self.labels = [], []
        self.canvas.set_points([])
        self._refresh_cur_label()
        self.lbl_status.setText(f"{fid} 已结束，下一次点击将新建一块")

    # ---------- 保存 ----------
    def _sync_doc(self):
        for n in self.proj.images:
            self.proj.replace_fields(n, self.draft.get(n) or [])

    def save_project(self):
        if not self.proj:
            return
        self._sync_doc()
        try:
            self.proj.save()
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        self.dirty = False
        self._update_title()
        self._update_state()
        self.lbl_status.setText(
            "已保存到 " + os.path.join(self.proj.root, "project.json"))

    # ---------- 高度基准 ----------
    def on_height(self):
        if not self.proj:
            return
        d = self.proj.defaults
        dlg = HeightDialog(float(d.get("target_z", 0.0)),
                           bool(d.get("use_lrf", False)), self)
        if dlg.exec() != QDialog.Accepted:
            return
        tz, lrf = dlg.values()
        self.proj.set_defaults(tz, lrf)
        self._sync_doc()
        ok, bad = self.proj.recompute(self._cam_builder)
        self.draft = {n: self.proj.fields(n) for n in self.proj.images}
        self.dirty = True
        self.active_fid = None
        self.points, self.labels = [], []
        self.canvas.set_points([])
        self._refresh_fields()
        self._refresh_cur_label()
        self._update_title()
        self._update_state()
        self.lbl_status.setText(
            f"已按新基准重算 {ok} 张（失败 {bad} 张），记得 Ctrl+S 保存")

    # ---------- 导出 ----------
    def on_export(self, kind):
        if not self.proj or not any(self.draft.values()):
            QMessageBox.information(self, "没有数据", "还没有标注任何地块。")
            return
        self._sync_doc()
        base = os.path.basename(self.proj.root)
        if kind == "csv":
            p, _ = QFileDialog.getSaveFileName(
                self, "导出 CSV", os.path.join(self.proj.root, f"{base}_面积.csv"),
                "CSV (*.csv)")
            if not p:
                return
            self.proj.export_csv(p)
            self.lbl_status.setText(f"已导出 CSV：{p}")
        else:
            p, _ = QFileDialog.getSaveFileName(
                self, "导出 GeoJSON",
                os.path.join(self.proj.root, f"{base}_地块.geojson"),
                "GeoJSON (*.geojson *.json)")
            if not p:
                return
            p, n = self.proj.export_geojson(p, self._cam_builder)
            self.lbl_status.setText(f"已导出 GeoJSON：{p}（{n} 块）")

    # ---------- 自检 ----------
    def start_selftest(self, click, shot):
        self._st = {"phase": "load", "click": click, "shot": shot}

    def _st_click(self):
        if not self._st:
            return
        u, v = self._st["click"]
        self.on_point(float(u), float(v), 1)

    def _st_finish(self):
        shot = self._st.get("shot")
        if shot:
            self.grab().save(shot)
            print(f"截图已保存：{shot}")
        print("自检完成，草稿地块数：",
              sum(len(v) for v in self.draft.values()) if self.draft else 0)
        QApplication.instance().quit()


# =========================================================
# 5. 入口
# =========================================================
def _install_excepthook():
    import traceback

    def hook(t, v, tb):
        print("[未捕获异常]", flush=True)
        traceback.print_exception(t, v, tb)

    sys.excepthook = hook


def main():
    import argparse

    ap = argparse.ArgumentParser(description="无人机照片面积量算 · 桌面版")
    ap.add_argument("target", nargs="?", help="要打开的文件夹（也可拖到 bat 上）")
    ap.add_argument("--folder", help="直接打开的文件夹")
    ap.add_argument("--selftest", help="自检：给一张照片，自动点一下并截图后退出")
    ap.add_argument("--click", default="1810,1810", help="自检用的像素点 u,v")
    ap.add_argument("--shot", default="selftest.png", help="自检截图路径")
    args = ap.parse_args()

    _install_excepthook()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    win = MainWindow()

    if args.selftest:
        tmp = tempfile.mkdtemp(prefix="uavarea_")
        shutil.copy2(args.selftest, os.path.join(tmp, os.path.basename(args.selftest)))
        u, v = (float(x) for x in args.click.split(","))
        win.start_selftest((u, v), args.shot)
        win.open_project(tmp)
        win.show()
        return app.exec()

    win.show()
    folder = args.folder or (args.target if args.target and os.path.isdir(args.target)
                             else None)
    if folder:
        win.open_project(folder)
    else:
        win.canvas.show_message("打开一个照片文件夹开始（Ctrl+O）\n"
                                "也可以把文件夹拖到「启动界面.bat」上",
                                QColor(150, 160, 175))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
