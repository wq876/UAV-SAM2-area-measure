# -*- coding: utf-8 -*-
"""
measure_area.py —— 单张 / 整个文件夹 的无人机照片面积量算（点击分割 + 真实面积）

流程：读 DJI 参数 → 点击 → SAM2 分割出掩膜（绿色亮斑）→ 轮廓投影到地面 → 面积（㎡ / 亩）

用法
----
单张：
    python measure_area.py "照片.JPG"

整个文件夹（逐张处理）：
    python measure_area.py --folder "照片目录"

无窗口（自动化）：
    python measure_area.py "照片.JPG" --click 1810,1810

窗口按键
--------
    左键      点目标（同一块地可连点多下，表示这些点都属于它）
    右键      明确排除该点（剔掉隔壁地块/道路/屋顶）
    Enter     确认当前这块地 → 面积进表
    U         撤销最近一个点
    Z         撤销上一块已确认的地块
    C         清空当前所有点
    D         下一张照片
    A         上一张照片
    Q / Esc   结束（当前这张会自动保存）
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass, field

import numpy as np

# ⚠️ 必须在 torch 之前导入 cv2。
# 本环境（torch 2.5.1+cu118 / opencv 4.11）里，先加载 torch 再 import cv2 会报
# "ImportError: DLL load failed while importing cv2: 操作系统无法运行 %1"。
# 顶层导入一次即可避开，后面各处直接用。
import cv2  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_geo import (  # noqa: E402
    Camera, MetadataError, imread_unicode, imwrite_unicode, m2_to_mu, read_dji_meta,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # 项目根目录
# 权重默认放在项目根的 weights/ 下；也可用 --weight 指定
DEFAULT_WEIGHT = os.path.join(_ROOT, "weights", "sam2.1_hiera_small.pt")
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 900
EDGE_MARGIN_RATIO = 0.06          # 距画面边缘 6% 以内视为"贴边"
DEPTH_RATIO_WARN = 2.0            # 近远深度比超过该值提示地形风险

_CJK_FONT_CANDIDATES = [
    r"C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
    r"C:/Windows/Fonts/simhei.ttf",    # 黑体
    r"C:/Windows/Fonts/simsun.ttc",    # 宋体
]


# =========================================================
# 1. 结果结构
# =========================================================
@dataclass
class Field:
    idx: int
    contour_px: np.ndarray          # (N,2) 采样后的轮廓点（原图像素）
    pixel_area: int
    area_m2: float
    center_latlon: tuple
    depth_min: float
    depth_max: float
    warnings: list = field(default_factory=list)
    score: float = float("nan")
    filled_px: int = 0
    removed_px: int = 0

    @property
    def area_mu(self) -> float:
        return m2_to_mu(self.area_m2)


# =========================================================
# 2. 轮廓处理
# =========================================================
def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """取掩膜的最大外轮廓，返回 (N,2) 像素点。"""
    import cv2

    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("掩膜为空，无法提取轮廓")
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).astype(np.float64)


def resample_contour(contour: np.ndarray, n: int) -> np.ndarray:
    """按弧长等距重采样闭合轮廓为 n 个点。"""
    pts = np.asarray(contour, dtype=np.float64)
    if pts.shape[0] < 3:
        raise RuntimeError("轮廓点太少")
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        raise RuntimeError("轮廓长度为 0")
    targets = np.linspace(0.0, total, n, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.stack([xs, ys], axis=1)


def contour_to_mask(contour: np.ndarray, shape_hw) -> np.ndarray:
    """
    把轮廓栅格化成掩膜。
    清理过的掩膜 = 最大连通域 + 填洞，等价于它自己的外轮廓围成的区域，
    所以用轮廓重建掩膜和原掩膜几乎一致（差 1 像素级），但只占几百字节。
    """
    import cv2

    m = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(m, [np.round(contour).astype(np.int32)], 1)
    return m.astype(bool)


def clean_mask(mask: np.ndarray, fill_holes: bool = True) -> tuple:
    """
    掩膜清理：只保留最大连通域 + 填内部孔洞。

    为什么必须做：
      - SAM2 常在地块内部留下"孔洞"（树影、灌溉设施、裸斑），面积会系统性偏小；
      - 也会甩出零星小岛（相邻地块的一角），面积会系统性偏大。
    返回 (清理后的掩膜, 填洞像素数, 去除小岛像素数)。
    """
    import cv2

    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if m.sum() == 0:
        return mask.astype(bool), 0, 0

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    removed = 0
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = int(np.argmax(areas)) + 1
        removed = int(areas.sum() - areas[keep - 1])
        m = (lab == keep).astype(np.uint8)

    filled = 0
    if fill_holes:
        h, w = m.shape
        flood = m.copy()
        mask_ff = np.zeros((h + 2, w + 2), np.uint8)
        for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            if flood[sy, sx] == 0:
                mask_ff[:] = 0
                cv2.floodFill(flood, mask_ff, (sx, sy), 1)
        holes = (flood == 0)
        filled = int(holes.sum())
        m = (m | holes.astype(np.uint8)).astype(np.uint8)

    return m.astype(bool), filled, removed


def contour_quality_notes(contour: np.ndarray, shape_hw, depth_min: float,
                          depth_max: float, pitch_deg: float = -90.0) -> list:
    """几何可靠性提示（不阻断，只提醒）。"""
    import cv2

    h, w = shape_hw
    notes = []

    x, y, bw, bh = cv2.boundingRect(contour.astype(np.int32))
    if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
        notes.append("目标贴到画面边界，可能被截断，面积会偏小")

    mx = EDGE_MARGIN_RATIO * min(w, h)
    if x < mx or y < mx or (w - (x + bw)) < mx or (h - (y + bh)) < mx:
        notes.append("目标靠近画面边缘，畸变与斜视误差都会放大，建议靠画面中部")

    if depth_min > 0 and depth_max / depth_min >= DEPTH_RATIO_WARN:
        notes.append(f"近远深度比 {depth_max / depth_min:.1f}，地形起伏造成的误差会被放大")

    if pitch_deg > -75.0 and y > h * 0.75:
        notes.append("目标位于画面下部（近处），个别像素抖动对面积影响较大")

    return notes


def mask_quality_notes(contour: np.ndarray, score: float) -> list:
    """分割本身的可靠性提示。"""
    import cv2

    notes = []
    if np.isfinite(score):
        # SAM2 的分数偏保守，实测很准的分割也常在 0.7~0.9，所以阈值放低
        if score < 0.65:
            notes.append(f"分割置信度偏低（{score:.2f}），建议加背景点或换位置再点")
        elif score < 0.80:
            notes.append(f"分割置信度一般（{score:.2f}），请核对绿色亮斑有没有溢出")

    c = np.round(contour).astype(np.int32).reshape(-1, 1, 2)
    per = float(cv2.arcLength(c, True))
    area = float(cv2.contourArea(c))
    if area > 0:
        compact = per * per / (4.0 * np.pi * area)
        if compact > 8.0:
            notes.append(f"边界非常破碎（紧凑度 {compact:.1f}），面积对细节敏感，建议放大核对")
    return notes


# =========================================================
# 3. SAM2 分割器
# =========================================================
class Segmenter:
    """懒加载 SAM2，一次载入反复分割。"""

    def __init__(self, weight: str, config: str, device: str | None = None):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not os.path.isfile(weight):
            raise FileNotFoundError(f"找不到 SAM2 权重：{weight}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if device == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        self.device = device
        model = build_sam2(config, weight, device=device)
        self.predictor = SAM2ImagePredictor(model)
        self._image_set = False
        self.shape_hw = None

    def set_image(self, bgr: np.ndarray):
        self.predictor.set_image(bgr[:, :, ::-1].copy())
        self._image_set = True
        self.shape_hw = bgr.shape[:2]

    def predict(self, points, labels):
        if not self._image_set:
            raise RuntimeError("请先 set_image")
        masks, scores, _ = self.predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True)
        return masks, scores


def pick_mask(masks, scores) -> int:
    """
    从 SAM2 的 3 个候选里挑一个给用户看。
    规则：分数最高，且不占满整幅画面（避免选中"整个场景"那种答案）。
    """
    h, w = masks.shape[-2:]
    total = h * w
    best, best_score = 0, -1.0
    for i, m in enumerate(masks):
        if float(m.sum()) / total > 0.75:
            continue
        if float(scores[i]) > best_score:
            best, best_score = i, float(scores[i])
    return best


# =========================================================
# 4. 量算
# =========================================================
def build_field(cam: Camera, mask: np.ndarray, idx: int, n_samples: int,
                score: float = float("nan"), clean: bool = True,
                fill_holes: bool = True) -> Field:
    filled = removed = 0
    if clean:
        mask, filled, removed = clean_mask(mask, fill_holes=fill_holes)

    contour = mask_to_contour(mask)
    sampled = resample_contour(contour, n_samples)
    area = cam.area_of_pixels(sampled)
    dmin, dmax = cam.depth_range(sampled)
    latlon = cam.center_latlon(sampled) if np.isfinite(cam.meta.latitude) else (float("nan"),) * 2
    notes = contour_quality_notes(sampled, mask.shape[:2], dmin, dmax,
                                  cam.meta.gimbal_pitch)
    notes += mask_quality_notes(sampled, score)
    return Field(idx=idx, contour_px=sampled, pixel_area=int(mask.sum()), area_m2=area,
                 center_latlon=latlon, depth_min=dmin, depth_max=dmax,
                 warnings=notes, score=score, filled_px=filled, removed_px=removed)


# =========================================================
# 5. 结果落盘
# =========================================================
def _load_cjk_font(size: int):
    from PIL import ImageFont

    for path in _CJK_FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


def draw_text_cn(img_bgr, lines, x: int, y: int, size: int = 30,
                 color=(255, 255, 255), line_gap: int = 8):
    """在 BGR 图上写中文（cv2.putText 只能写 ASCII）。"""
    import cv2
    from PIL import Image, ImageDraw

    font = _load_cjk_font(size)
    if font is None:
        for i, line in enumerate(lines):
            cv2.putText(img_bgr, line.encode("ascii", "replace").decode("ascii"),
                        (x, y + size + i * (size + line_gap)),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 34.0, color, 2, cv2.LINE_AA)
        return img_bgr

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    rgb = (color[2], color[1], color[0])
    for i, line in enumerate(lines):
        draw.text((x, y + i * (size + line_gap)), line, font=font, fill=rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_hud_cn(img_bgr, items, x: int, y: int, size: int, gap: int):
    """
    一次绘完多行 HUD（支持中文）。
    items: [(文字, BGR颜色, 字号倍率), ...]
    cv2.putText 只能画 ASCII，中文路径/文件名会变问号，所以这里走 PIL。
    """
    import cv2
    from PIL import Image, ImageDraw

    fonts = {}
    ok = True
    for _, _, mult in items:
        s = max(10, int(size * mult))
        if s not in fonts:
            f = _load_cjk_font(s)
            if f is None:
                ok = False
                break
            fonts[s] = f

    if not ok:
        yy = y
        for text, color, mult in items:
            cv2.putText(img_bgr, text.encode("ascii", "replace").decode("ascii"),
                        (x, yy), cv2.FONT_HERSHEY_SIMPLEX, size * mult / 34.0,
                        color, 1, cv2.LINE_AA)
            yy += int(gap * mult)
        return img_bgr

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    yy = y
    for text, color, mult in items:
        draw.text((x, yy), text, font=fonts[max(10, int(size * mult))],
                  fill=(color[2], color[1], color[0]))
        yy += int(gap * mult)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_result_image(img, cam: Camera, fields: list, photo_name: str):
    """带标注的结果图：填充半透明 + 轮廓 + 编号面积 + 顶部汇总。"""
    import cv2

    vis = img.copy()
    shade = vis.copy()
    for f in fields:
        pts = np.round(f.contour_px).astype(np.int32)
        cv2.fillPoly(shade, [pts], (60, 200, 255))
    if fields:
        vis = cv2.addWeighted(shade, 0.35, vis, 0.65, 0)

    fs = max(1.0, vis.shape[1] / 4000.0 * 1.15)
    for f in fields:
        pts = f.contour_px.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 220, 255), max(3, int(fs * 2)))
        cx, cy = pts.mean(axis=0)
        text = f"#{f.idx} {f.area_m2:.1f} m2 / {f.area_mu:.3f} mu"
        org = (int(cx) - int(90 * fs), int(cy))
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), int(fs * 5), cv2.LINE_AA)
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 255, 255), max(2, int(fs * 2)), cv2.LINE_AA)

    total = sum(f.area_m2 for f in fields)
    head = [f"{photo_name}   地块数 {len(fields)}",
            f"合计 {total:.1f} m2 / {m2_to_mu(total):.3f} mu    高度基准 {cam.h:.1f} m"]
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 108), (0, 0, 0), -1)
    return draw_text_cn(vis, head, 18, 12, size=34)


def export_results(img, cam: Camera, fields: list, photo_path: str, out_dir: str):
    """
    落盘三样：result_<名>.jpg 标注图 / fields_<名>.csv 面积表 / masks/<名>_field_NN.png
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(photo_path))[0]

    vis = draw_result_image(img, cam, fields, os.path.basename(photo_path))
    img_path = os.path.join(out_dir, f"result_{base}.jpg")
    imwrite_unicode(img_path, vis, quality=90)

    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    for f in fields:
        m = contour_to_mask(f.contour_px, img.shape[:2])
        imwrite_unicode(os.path.join(mask_dir, f"{base}_field_{f.idx:02d}.png"),
                        (m.astype(np.uint8) * 255))

    csv_path = os.path.join(out_dir, f"fields_{base}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["编号", "面积_m2", "面积_亩", "像素面积", "中心经度", "中心纬度",
                    "高度基准_m", "置信度", "提示"])
        for f in fields:
            w.writerow([f.idx, f"{f.area_m2:.2f}", f"{f.area_mu:.4f}", f.pixel_area,
                        f"{f.center_latlon[1]:.8f}" if np.isfinite(f.center_latlon[1]) else "",
                        f"{f.center_latlon[0]:.8f}" if np.isfinite(f.center_latlon[0]) else "",
                        f"{cam.h:.2f}", f"{f.score:.3f}" if np.isfinite(f.score) else "",
                        "；".join(f.warnings)])
    return img_path, csv_path, mask_dir


def write_summary(rows: list, out_dir: str):
    """整个文件夹的汇总表：每张照片一行。"""
    if not rows or not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary_全部照片.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["照片", "地块数", "合计_m2", "合计_亩", "备注"])
        for name, n, a_m2, note in rows:
            w.writerow([name, n, f"{a_m2:.2f}", f"{m2_to_mu(a_m2):.4f}", note])
    return path


# =========================================================
# 6. 交互会话（单张 / 整个文件夹）
# =========================================================
class Session:
    def __init__(self, files: list, seg: Segmenter, out_dir: str | None,
                 n_samples: int = 300, clean: bool = True, fill_holes: bool = True,
                 cam_kwargs: dict | None = None, target_z: float = 0.0,
                 use_lrf: bool = False, calib: str | None = None):
        import cv2

        self.cv2 = cv2
        self.files = files
        self.seg = seg
        self.out_dir = out_dir
        self.n_samples = n_samples
        self.clean = clean
        self.fill_holes = fill_holes
        self.cam_kwargs = cam_kwargs or {}
        self.target_z = target_z
        self.use_lrf = use_lrf
        self.calib = calib

        self.results = {p: [] for p in files}     # 每张照片已确认的地块
        self.errors = {}                           # 读不了的图记原因
        self.idx = 0

        self.img = None
        self.cam = None
        self.disp = None
        self.scale = 1.0
        self.points = []
        self.labels = []
        self.mask = None
        self.cur_area = float("nan")
        self.score = float("nan")
        self.WIN = "UAV Area Measure"

        self._open(0)

    # ---------- 打开一张 ----------
    def _open(self, i: int):
        import cv2

        self.idx = max(0, min(i, len(self.files) - 1))
        path = self.files[self.idx]
        self.points, self.labels = [], []
        self.mask, self.cur_area, self.score = None, float("nan"), float("nan")

        print(f"\n[{self.idx + 1}/{len(self.files)}] {os.path.basename(path)}")
        try:
            img = imread_unicode(path)
            meta = read_dji_meta(path, **self.cam_kwargs)
            cam = Camera(meta, target_z=self.target_z,
                         height_m=meta.lrf_drop_m if self.use_lrf else None)
            if self.calib:
                parts = [x for x in self.calib.replace(" ", "").split(",") if x]
                if len(parts) != 5:
                    raise RuntimeError("--calib 需要 5 个数：u1,v1,u2,v2,已知长度m")
                u1, v1, u2, v2, known = map(float, parts)
                cam = cam.with_height(cam.implied_height((u1, v1), (u2, v2), known))
        except (MetadataError, FileNotFoundError, IOError, RuntimeError) as exc:
            reason = str(exc).splitlines()[0]
            self.errors[path] = reason
            self.img, self.cam, self.disp = None, None, None
            print(f"   ✗ 跳过：{reason}")
            return

        self.img, self.cam = img, cam
        h, w = img.shape[:2]
        self.scale = min(1.0, MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h)
        self.disp = cv2.resize(img, (int(w * self.scale), int(h * self.scale)),
                               interpolation=cv2.INTER_AREA)
        print(f"   {w}x{h}  h={cam.h:.2f}m  pitch={meta.gimbal_pitch:.1f}°  "
              f"GSD={cam.gsd_at_center() * 100:.2f}cm/px")
        print("   编码图像中……")
        self.seg.set_image(img)
        print("   就绪，在窗口里点目标。")

    # ---------- 保存当前 ----------
    def _finish_current(self):
        path = self.files[self.idx]
        fields = self.results[path]
        if not fields or self.img is None:
            return
        if not self.out_dir:
            return
        try:
            img_path, csv_path, _ = export_results(self.img, self.cam, fields, path,
                                                   self.out_dir)
            print(f"   已保存：{os.path.basename(img_path)}  {os.path.basename(csv_path)}")
        except Exception as exc:
            print(f"   [警告] 保存失败：{exc}")

    # ---------- 绘制 ----------
    def _fill(self, canvas, contour_scaled, color, alpha=0.38):
        cv2 = self.cv2
        shade = canvas.copy()
        cv2.fillPoly(shade, [contour_scaled.astype(np.int32)], color)
        return cv2.addWeighted(shade, alpha, canvas, 1 - alpha, 0)

    def compose_canvas(self):
        cv2 = self.cv2
        canvas = self.disp.copy() if self.disp is not None else \
            np.zeros((600, 1000, 3), np.uint8)

        for f in self.results[self.files[self.idx]]:
            pts = (f.contour_px * self.scale)
            canvas = self._fill(canvas, pts, (60, 200, 255))
            cv2.polylines(canvas, [pts.astype(np.int32)], True, (0, 220, 255), 2)
            cx, cy = pts.mean(axis=0)
            text = f"#{f.idx} {f.area_m2:.1f}m2 {f.area_mu:.3f}mu"
            org = (int(cx) - 60, int(cy))
            cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)

        if self.mask is not None and int(self.mask.sum()) > 0:
            try:
                c = mask_to_contour(self.mask) * self.scale
                canvas = self._fill(canvas, c, (0, 255, 0), 0.45)
                cv2.polylines(canvas, [c.astype(np.int32)], True, (0, 255, 0), 2)
            except RuntimeError:
                pass

        for (u, v), lab in zip(self.points, self.labels):
            color = (0, 255, 0) if lab == 1 else (0, 0, 255)
            p = (int(u * self.scale), int(v * self.scale))
            cv2.circle(canvas, p, 4, color, -1)
            cv2.circle(canvas, p, 7, (255, 255, 255), 1)

        hfs = max(0.38, 0.58 * min(1.0, canvas.shape[1] / 1500.0))
        bar = int(132 * hfs / 0.58)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar), (0, 0, 0), -1)

        fields = self.results[self.files[self.idx]]
        total = sum(f.area_m2 for f in fields)
        n_fg = sum(1 for x in self.labels if x == 1)
        n_bg = sum(1 for x in self.labels if x == 0)

        hud = [
            (f"photo {self.idx + 1}/{len(self.files)}: {os.path.basename(self.files[self.idx])}",
             (0, 255, 0)),
            (f"clicked points FG={n_fg} EX={n_bg}   |   this photo: {len(fields)} fields, "
             f"{total:.1f} m2 / {m2_to_mu(total):.3f} mu", (0, 220, 255)),
        ]
        if self.cam is None:
            hud.append(("META ERROR - this photo has no usable DJI data, press D to skip",
                        (0, 0, 255)))
        elif np.isfinite(self.cur_area):
            hud.append((f"GREEN AREA: {self.cur_area:.1f} m2 / {m2_to_mu(self.cur_area):.3f} mu"
                        f"    (ENTER to accept)", (0, 255, 0)))
        else:
            hud.append(("left click the target   |   right click = exclude   |   "
                        "ENTER = accept   |   A/D = prev/next photo   |   Q = quit",
                        (150, 150, 150)))

        items = []
        for line, color in hud:
            mult = 1.25 if line.startswith("GREEN AREA") else 1.0
            items.append((line, color, mult))
        items.append((f"save to: {self.out_dir}" if self.out_dir else "", (150, 150, 150), 0.9))
        canvas = draw_hud_cn(canvas, items, 12, int(14 * hfs / 0.58),
                             max(11, int(26 * hfs / 0.58)), max(18, int(30 * hfs / 0.58)))

        return canvas

    def redraw(self):
        self.cv2.imshow(self.WIN, self.compose_canvas())

    # ---------- 预测 ----------
    def run_predict(self):
        if self.cam is None:
            return
        if not self.points:
            self.mask, self.cur_area, self.score = None, float("nan"), float("nan")
            self.redraw()
            return
        masks, scores = self.seg.predict(self.points, self.labels)
        idx = pick_mask(masks, scores)
        self.mask, self.score = masks[idx], float(scores[idx])
        try:
            f = build_field(self.cam, self.mask, 1, self.n_samples,
                            score=self.score, clean=self.clean,
                            fill_holes=self.fill_holes)
            self.cur_area = f.area_m2
            # 绿色区域用清理后的轮廓显示，和最终面积一致
            self.mask = contour_to_mask(f.contour_px, self.mask.shape[:2])
            print(f"   亮斑面积：{f.area_m2:.1f} m² / {f.area_mu:.3f} 亩"
                  f"（score={self.score:.2f}）  回车确认")
        except Exception as exc:
            self.cur_area, self.mask = float("nan"), None
            print(f"   这一点没分割出有效区域（{exc}），换个位置点")
        self.redraw()

    # ---------- 事件 ----------
    def on_mouse(self, event, x, y, flags, param):
        if self.cam is None:
            return
        if event == self.cv2.EVENT_LBUTTONDOWN:
            self.points.append((x / self.scale, y / self.scale))
            self.labels.append(1)
            self.run_predict()
        elif event == self.cv2.EVENT_RBUTTONDOWN:
            self.points.append((x / self.scale, y / self.scale))
            self.labels.append(0)
            self.run_predict()

    def confirm(self):
        if self.cam is None or not self.points:
            print("   还没有点，先左键点一下目标。")
            return
        if self.mask is None or not np.isfinite(self.cur_area):
            print("   当前没有有效亮斑，换个位置点。")
            return
        f = build_field(self.cam, self.mask, len(self.results[self.files[self.idx]]) + 1,
                        self.n_samples, score=self.score, clean=False)
        self.results[self.files[self.idx]].append(f)
        print(f"   ✓ 地块 #{f.idx}：{f.area_m2:.1f} m² / {f.area_mu:.3f} 亩")
        for w in f.warnings:
            print("     提示：" + w)
        self.points, self.labels = [], []
        self.mask, self.cur_area, self.score = None, float("nan"), float("nan")
        self.redraw()

    def undo_point(self):
        if not self.points:
            print("   当前没有点可撤销。")
            return
        self.points.pop()
        self.labels.pop()
        print(f"   撤销一个点，剩 {len(self.points)} 个，重新分割……")
        self.run_predict()

    def undo_field(self):
        fields = self.results[self.files[self.idx]]
        if fields:
            gone = fields.pop()
            print(f"   已撤销地块 #{gone.idx}")
            self.redraw()

    def goto(self, delta: int):
        self._finish_current()
        target = self.idx + delta
        if target < 0 or target >= len(self.files):
            print("   已经到头了。")
            return
        self._open(target)
        self.redraw()

    # ---------- 主循环 ----------
    def run(self):
        cv2 = self.cv2
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WIN, self.on_mouse)
        self.redraw()

        print("\n操作：左键=点目标  右键=排除  Enter=确认这块  U=撤点  Z=撤地块")
        print("      A=上一张  D=下一张  C=清空当前点  Q=退出（当前这张自动保存）")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (13, 32):
                self.confirm()
            elif key in (ord("u"), ord("U"), 8):
                self.undo_point()
            elif key in (ord("z"), ord("Z")):
                self.undo_field()
            elif key in (ord("c"), ord("C")):
                self.points, self.labels = [], []
                self.mask, self.cur_area, self.score = None, float("nan"), float("nan")
                self.redraw()
            elif key in (ord("d"), ord("D")):
                self.goto(+1)
            elif key in (ord("a"), ord("A")):
                self.goto(-1)
        cv2.destroyAllWindows()
        self._finish_current()
        return self.summary_rows()

    def summary_rows(self):
        rows = []
        for path in self.files:
            fields = self.results[path]
            note = self.errors.get(path, "")
            if not fields and not note:
                note = "未标注"
            rows.append((os.path.basename(path), len(fields),
                         sum(f.area_m2 for f in fields), note))
        return rows


# =========================================================
# 7. 终端报告
# =========================================================
def print_report(files: list, results: dict, errors: dict):
    print("\n" + "=" * 82)
    print("量 算 结 果")
    print("=" * 82)
    any_field = False
    grand = 0.0
    for path in files:
        fields = results.get(path, [])
        name = os.path.basename(path)
        if not fields:
            if path in errors:
                print(f"{name}   [跳过] {errors[path]}")
            continue
        any_field = True
        s = sum(f.area_m2 for f in fields)
        grand += s
        print(f"\n{name}   地块数 {len(fields)}   合计 {s:.1f} m² / {m2_to_mu(s):.4f} 亩")
        for f in fields:
            print(f"   #{f.idx}  {f.area_m2:9.1f} m²  {f.area_mu:7.4f} 亩"
                  f"   {'; '.join(f.warnings) if f.warnings else ''}")
    if not any_field:
        print("没有确认任何地块。")
        return
    print("\n" + "-" * 82)
    print(f"全部合计：{grand:.1f} m² / {m2_to_mu(grand):.4f} 亩")
    print("口径提醒：面积按目标位于水平平面计算；相对高度是相对起飞点，"
          "目标地面有高差时会整体偏差（误差≈高差/高度，面积误差≈2倍）。")
    print("=" * 82)


# =========================================================
# 8. 入口
# =========================================================
def parse_uv(text: str):
    u, v = text.replace("x", ",").replace(" ", "").split(",")
    return float(u), float(v)


def collect_folder(folder: str) -> list:
    files = []
    for pat in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        files += glob.glob(os.path.join(folder, pat))
    seen, uniq = set(), []
    for f in sorted(files):
        key = os.path.basename(f).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def main():
    ap = argparse.ArgumentParser(
        description="无人机照片面积量算：点击分割 → 真实面积（㎡ / 亩），支持单张或整个文件夹")
    ap.add_argument("photo", nargs="?", help="单张照片路径；用 --folder 时可省略")
    ap.add_argument("--folder", metavar="目录", help="整个文件夹逐张处理（A 上一张 / D 下一张）")
    ap.add_argument("--out", metavar="目录", help="结果落盘目录（标注图 / CSV / 掩膜）")
    ap.add_argument("--weight", default=DEFAULT_WEIGHT, help="SAM2 权重路径")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="SAM2 模型配置")
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    ap.add_argument("--samples", type=int, default=300, help="轮廓等距采样点数（默认 300）")
    ap.add_argument("--no-clean", action="store_true",
                    help="不做掩膜清理（默认去掉零星小岛并填内部孔洞）")
    ap.add_argument("--no-fill-holes", action="store_true",
                    help="只去小岛、不填内部孔洞")
    ap.add_argument("--click", action="append", metavar="u,v",
                    help="无窗口模式：直接给前景点，可重复")
    ap.add_argument("--neg", action="append", metavar="u,v",
                    help="无窗口模式：背景点（排除），可重复")
    ap.add_argument("--target-z", type=float, default=0.0,
                    help="目标地面相对起飞点的高度(m)，高填正、低填负")
    ap.add_argument("--use-lrf", action="store_true",
                    help="用照片自带的激光测距修正高度基准")
    ap.add_argument("--calib", metavar="u1,v1,u2,v2,已知长度m",
                    help="用画面里一段已知长度反推真实高度，如 1075,2204,1075,2330,4.65")
    ap.add_argument("--altitude", type=float, help="手工指定相对高度(m)")
    ap.add_argument("--pitch", type=float, help="手工指定云台俯仰(°)")
    ap.add_argument("--yaw", type=float, help="手工指定云台偏航(°)")
    ap.add_argument("--roll", type=float, help="手工指定云台横滚(°)")
    ap.add_argument("--hfov", type=float, help="手工指定水平 FOV(°)")
    ap.add_argument("--sensor-width", type=float, help="手工指定传感器物理宽度(mm)")
    args = ap.parse_args()

    # 组装待处理清单
    if args.folder:
        files = collect_folder(args.folder)
        if not files:
            print(f"目录里没有照片：{args.folder}")
            raise SystemExit(2)
        print(f"目录 {args.folder}  共 {len(files)} 张")
    elif args.photo:
        files = [args.photo]
    else:
        ap.error("需要给一张照片，或用 --folder 指定目录")

    override = dict(altitude=args.altitude, pitch=args.pitch, yaw=args.yaw,
                    roll=args.roll, hfov=args.hfov, sensor_width=args.sensor_width)

    print("加载 SAM2 模型……")
    seg = Segmenter(args.weight, args.config, args.device)
    print(f"模型就绪（device={seg.device}）")

    # ---- 无窗口模式（单张，直接出结果）----
    if args.click:
        photo = files[0]
        try:
            meta = read_dji_meta(photo, **override)
        except MetadataError as exc:
            print("参数读取失败，已中止（不会用默认值硬算）：\n" + str(exc))
            raise SystemExit(2)
        cam = Camera(meta, target_z=args.target_z,
                     height_m=meta.lrf_drop_m if args.use_lrf else None)
        if args.calib:
            parts = [p for p in args.calib.replace(" ", "").split(",") if p]
            if len(parts) != 5:
                ap.error("--calib 需要 5 个数：u1,v1,u2,v2,已知长度m")
            u1, v1, u2, v2, known = map(float, parts)
            cam = cam.with_height(cam.implied_height((u1, v1), (u2, v2), known))

        img = imread_unicode(photo)
        seg.set_image(img)
        points = [parse_uv(t) for t in args.click]
        labels = [1] * len(points)
        for t in (args.neg or []):
            points.append(parse_uv(t))
            labels.append(0)
        masks, scores = seg.predict(points, labels)
        idx = pick_mask(masks, scores)
        f = build_field(cam, masks[idx], 1, args.samples, score=float(scores[idx]),
                        clean=not args.no_clean, fill_holes=not args.no_fill_holes)
        print(f"\n亮斑面积：{f.area_m2:.1f} m² / {f.area_mu:.4f} 亩   score={scores[idx]:.3f}")
        print_report([photo], {photo: [f]}, {})
        if args.out:
            img_path, csv_path, mask_dir = export_results(img, cam, [f], photo, args.out)
            print(f"已保存：\n  结果图 {img_path}\n  CSV   {csv_path}\n  掩膜  {mask_dir}")
        return

    # ---- 交互模式 ----
    session = Session(files, seg, args.out, n_samples=args.samples,
                      clean=not args.no_clean, fill_holes=not args.no_fill_holes,
                      cam_kwargs=override, target_z=args.target_z,
                      use_lrf=args.use_lrf, calib=args.calib)
    rows = session.run()
    print_report(files, session.results, session.errors)
    if args.out:
        p = write_summary(rows, args.out)
        if p:
            print(f"\n汇总表：{p}")


if __name__ == "__main__":
    main()
