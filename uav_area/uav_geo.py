# -*- coding: utf-8 -*-
"""
uav_geo.py —— 单张无人机照片的几何内核（读参数 / 建相机模型 / 像素→地面 / 面积）

设计原则
--------
1. 不静默兜底：任何关键参数缺失都直接抛 MetadataError，绝不写死默认值。
2. FOV 口径可追溯：优先用 EXIF 的 FocalLengthIn35mmFilm 反推水平 FOV，
   并可用物理焦距 + 传感器宽做交叉校验；也可用 --hfov 手工指定。
3. 只做"平坦水平地面"假设下的射线-平面求交，不做正射、不做空三。

几何函数（build_camera_intrinsic / build_camera_to_enu_rotation /
pixel_to_ground_enu / polygon_area_xy / enu_to_latlon）为本项目独立实现。

早期版本这几个函数参考过开源项目 UAVGeoMeasure（github.com/Thamkench/UAVGeoMeasure，
其仓库未声明许可证）。为避免许可证问题，已按公开的摄影测量公式重写：
  - 旋转矩阵改为"三个基本旋转连乘"（Rz(-yaw)·Rx(pitch-90°)·Rz(roll)）；
  - 投影改为向量化批处理；
  - 面积改用叉积求和的鞋带公式。
等价性由 _geom_regress.py 用 2000 组随机姿态验证（差 < 1e-15），
真实数据基线（车长 5.79 m、坝体 1890.8 ㎡）也保持一致。
在此基础上另实现了：XMP/EXIF/GPS 读取、FOV 口径推导（含 16:9 裁切修正）、批量投影、经纬度换算。

命令行用法（单独跑可看参数说明）
-------------------------------
    python uav_geo.py 照片路径            # 打印该照片解析出的全部参数
    python uav_geo.py 照片路径 --json     # 以 JSON 输出
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict

import numpy as np

# 35mm 全画幅对角线长度（mm），用于把等效焦距换算成对角 FOV
DIAG_35MM = math.hypot(36.0, 24.0)  # 43.2666 mm

# 已知机型的传感器宽度（mm）。有这张表时用"物理焦距 × 传感器宽"算内参，
# 它不受画幅裁切（如 16:9 裁切）影响，比等效焦距反推更可靠。
SENSOR_WIDTH_MM = {
    "FC6310": 13.2,    # Phantom 4 Pro       1" 13.2x8.8
    "FC6310R": 13.2,   # Phantom 4 Pro V2.0
    "FC6310S": 13.2,   # Phantom 4 Pro RTK
    "FC330": 6.17,     # Phantom 3 Standard/4K   1/2.3"
    "FC300S": 6.17,    # Phantom 3 Advanced
    "FC300X": 6.17,    # Phantom 3 Professional
    "FC300C": 6.17,    # Phantom 3 4K
}

# 关键参数缺失时的填空提示
_MANUAL_HINT = (
    "可用命令行手工指定后重试：\n"
    "  --altitude 相对高度(m) --pitch 云台俯仰(°) --yaw 云台偏航(°) "
    "--roll 云台横滚(°) --hfov 水平视场角(°) --sensor-width 传感器宽(mm)"
)


class MetadataError(RuntimeError):
    """元数据缺失或不可用。永远不要用默认值掩盖它。"""


# =========================================================
# 1. 图像读写（兼容中文路径）
# =========================================================
def imread_unicode(path: str, flags=None):
    """cv2.imread 在 Windows 中文路径下会失败，这里用 imdecode 绕过。"""
    import cv2

    if flags is None:
        flags = cv2.IMREAD_COLOR
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"读不到文件：{path}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"图像解码失败：{path}")
    return img


def imwrite_unicode(path: str, img, quality: int = 92) -> bool:
    """cv2.imwrite 的中文路径安全版本。"""
    import cv2

    ext = os.path.splitext(path)[1].lower() or ".jpg"
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(path)
    return True


# =========================================================
# 2. 元数据读取：Pillow 优先，exiftool 兜底
# =========================================================
@dataclass
class DjiMeta:
    """一张 DJI 照片的测量所需参数（角度单位：度；高度单位：米）。"""

    path: str
    width: int
    height: int
    latitude: float
    longitude: float
    rel_altitude: float
    gimbal_yaw: float
    gimbal_pitch: float
    gimbal_roll: float
    focal_35: float | None
    focal_mm: float | None
    digital_zoom: float
    hfov_deg: float
    hfov_source: str
    model: str
    reader: str
    absolute_alt: float | None = None
    lrf_distance: float | None = None
    lrf_target_abs_alt: float | None = None
    zoom_assumed: bool = False
    fov_note: str = ""

    @property
    def lrf_drop_m(self) -> float | None:
        """激光测距给出的"相机到目标点"的垂直落差（米）。"""
        if self.absolute_alt is None or self.lrf_target_abs_alt is None:
            return None
        if self.lrf_target_abs_alt == 0.0 and (self.lrf_distance or 0.0) == 0.0:
            return None
        drop = self.absolute_alt - self.lrf_target_abs_alt
        return drop if drop > 1.0 else None

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(
            math.tan(math.radians(self.hfov_deg) / 2.0) * self.height / self.width))

    def summary(self) -> str:
        lines = [
            f"文件            : {os.path.basename(self.path)}",
            f"机型            : {self.model or '未知'}",
            f"图像尺寸        : {self.width} x {self.height}",
            f"GPS             : {self.latitude:.8f}, {self.longitude:.8f}",
            f"相对高度        : {self.rel_altitude:.3f} m  （相对起飞点，不是相对地面！）",
            f"云台 yaw/pitch/roll : {self.gimbal_yaw:.2f} / "
            f"{self.gimbal_pitch:.2f} / {self.gimbal_roll:.2f}",
            f"水平 FOV        : {self.hfov_deg:.2f}°  （来源：{self.hfov_source}）",
            f"竖直 FOV        : {self.vfov_deg:.2f}°",
            f"等效焦距/物理焦距: "
            f"{self.focal_35 if self.focal_35 else '—'} mm / "
            f"{self.focal_mm if self.focal_mm else '—'} mm",
            f"数码变焦倍率    : {self.digital_zoom:g}"
            + ("（元数据为空/NaN，按 1 处理）" if self.zoom_assumed else ""),
            f"元数据来源      : {self.reader}",
        ]
        if self.lrf_drop_m is not None:
            lines.append(f"激光测距   : 目标点比相机低 {self.lrf_drop_m:.1f} m "
                         f"（可作高度基准修正参考）")
        if self.fov_note:
            lines.append(f"FOV 备注   : {self.fov_note}")
        return "\n".join(lines)


def _to_float(value, name: str) -> float:
    try:
        return float(str(value).strip())
    except Exception as exc:
        raise MetadataError(f"字段 {name} 无法转成数值：{value!r}") from exc


def _read_xmp_pillow(path: str) -> dict:
    """返回 XMP 里的 drone-dji:* 平铺字典。读不到返回空字典。"""
    from PIL import Image

    try:
        with Image.open(path) as im:
            xmp = im.getxmp()
    except Exception:
        return {}

    if not xmp:
        return {}

    desc = {}
    try:
        desc = xmp.get("xmpmeta", {}).get("RDF", {}).get("Description", {})
        if isinstance(desc, list):
            desc = desc[0] if desc else {}
    except Exception:
        desc = {}

    # Pillow 12 起属性名带前缀（如 "drone-dji:RelativeAltitude"）；
    # 老版本可能去掉前缀，这里做一次归一化。
    flat = {}
    for key, value in desc.items():
        short = key.split(":")[-1]
        flat[short] = value
    return flat


def _read_exif_pillow(path: str) -> dict:
    """读 EXIF 主 IFD + ExifIFD 的焦距信息 + GPS IFD 的经纬度。

    注意：老机型（如 Phantom 4 Pro）的经纬度只写在 EXIF 的 GPS 子段里，
    XMP 段没有 drone-dji:GpsLatitude，必须单独读，否则会得到 NaN。
    """
    from PIL import Image
    from PIL.ExifTags import TAGS

    out = {}
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769)
            gps = exif.get_ifd(0x8825)
    except Exception:
        return out

    for ifd in (exif, sub):
        for tag_id, value in (ifd or {}).items():
            name = TAGS.get(tag_id, tag_id)
            if name in ("FocalLengthIn35mmFilm", "FocalLength",
                        "DigitalZoomRatio", "Make", "Model"):
                out.setdefault(name, value)

    def dms_to_deg(dms):
        try:
            d, m, s = (float(x) for x in tuple(dms)[:3])
        except Exception:
            return None
        return d + m / 60.0 + s / 3600.0

    if gps:
        lat = dms_to_deg(gps.get(2)) if 2 in gps else None
        lon = dms_to_deg(gps.get(4)) if 4 in gps else None
        lat_ref = str(gps.get(1, "N")).strip().upper()[:1] or "N"
        lon_ref = str(gps.get(3, "E")).strip().upper()[:1] or "E"
        if lat is not None:
            out["GPSLatitude"] = -lat if lat_ref == "S" else lat
        if lon is not None:
            out["GPSLongitude"] = -lon if lon_ref == "W" else lon
    return out


def _read_exiftool(path: str) -> dict:
    """exiftool 兜底：返回 -G1 -s 形式的扁平日志信息。"""
    exe = shutil.which("exiftool")
    if not exe:
        return {}
    try:
        proc = subprocess.run(
            [exe, "-j", "-n", "-a", "-G1", "-s", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60,
        )
    except Exception:
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        data = json.loads(proc.stdout)[0]
    except Exception:
        return {}

    # 统一成短名，便于和 Pillow 结果合并
    flat = {}
    for full_key, value in data.items():
        flat[full_key.split(":")[-1]] = value
    flat["__exiftool_full__"] = data
    return flat


def hfov_from_focal_35(focal_35: float, width: int, height: int) -> float:
    """
    由 35mm 等效焦距推水平 FOV。

    35mm 等效焦距按"对角视场"定义（等效画幅 36×24mm），
    所以先算对角 FOV，再按画幅比例拆到水平方向。

    ⚠️ 只在"图像画幅 = 传感器全画幅"时准确。若相机做 16:9 裁切，
    水平视场其实没变，但这里会按裁切后的比例算，得到偏大的 FOV
    （P4P 实测：3:2 得 73.74° 正确，16:9 得 76.31° 偏大约 4.8% 的尺度）。
    有传感器宽度信息时请优先走 hfov_from_focal_sensor()。
    """
    tan_half_diag = (DIAG_35MM / 2.0) / float(focal_35)
    return math.degrees(2.0 * math.atan(tan_half_diag * width / math.hypot(width, height)))


def hfov_from_focal_sensor(focal_mm: float, sensor_width_mm: float,
                           width: int) -> float:
    """
    由物理焦距 + 传感器物理宽度推水平 FOV。与画面是否被裁切无关：
    fx = focal(mm) × 图像宽(px) / 传感器宽(mm)。
    """
    fx = float(focal_mm) * float(width) / float(sensor_width_mm)
    return math.degrees(2.0 * math.atan(width / (2.0 * fx)))


def read_dji_meta(path: str,
                  altitude: float | None = None,
                  pitch: float | None = None,
                  yaw: float | None = None,
                  roll: float | None = None,
                  hfov: float | None = None,
                  sensor_width: float | None = None) -> DjiMeta:
    """
    读取一张 DJI 照片的测量参数。

    顺序：Pillow(XMP+EXIF) → exiftool。任何关键字段读不到且没有手工指定，
    都抛 MetadataError，不会静默用默认值。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"照片不存在：{path}")

    img = imread_unicode(path)
    height, width = img.shape[:2]

    xmp = _read_xmp_pillow(path)
    exif = _read_exif_pillow(path)
    reader = "Pillow(XMP+EXIF)"

    if not xmp and not exif:
        # Pillow 什么都没读到，换 exiftool
        et = _read_exiftool(path)
        if et:
            xmp = et
            exif = et
            reader = "exiftool"

    def pick(*names):
        for n in names:
            for src in (xmp, exif):
                if n in src and str(src[n]).strip() not in ("", "None"):
                    return src[n]
        return None

    lat_raw = pick("GPSLatitude", "GpsLatitude")
    lon_raw = pick("GPSLongitude", "GpsLongitude")
    alt_raw = pick("RelativeAltitude")
    pitch_raw = pick("GimbalPitchDegree")
    yaw_raw = pick("GimbalYawDegree")
    roll_raw = pick("GimbalRollDegree")
    f35_raw = pick("FocalLengthIn35mmFilm")
    fmm_raw = pick("FocalLength")
    zoom_raw = pick("DigitalZoomRatio")
    model = pick("Model", "UniqueCameraModel") or ""

    missing = []
    if alt_raw is None and altitude is None:
        missing.append("RelativeAltitude(相对高度)")
    if pitch_raw is None and pitch is None:
        missing.append("GimbalPitchDegree(云台俯仰)")
    if yaw_raw is None and yaw is None:
        missing.append("GimbalYawDegree(云台偏航)")
    if roll_raw is None and roll is None:
        missing.append("GimbalRollDegree(云台横滚)")
    if f35_raw is None and hfov is None and fmm_raw is None:
        missing.append("FocalLengthIn35mmFilm / FocalLength(焦距，用于推 FOV)")
    if missing:
        raise MetadataError(
            "照片里缺少关键参数：" + "、".join(missing) + "\n"
            "可能原因：这不是原始 DJI 照片（例如从视频抽的帧、或被修图软件重存过）。\n"
            + _MANUAL_HINT
        )

    # --- FOV 口径 ---
    # 优先级：手工指定 > 物理焦距×传感器宽（不受裁切影响）> 等效焦距反推
    fov_note = ""
    sw = sensor_width
    if sw is None:
        sw = SENSOR_WIDTH_MM.get(str(model).strip().upper())

    if hfov is not None:
        hfov_deg, hfov_src = float(hfov), "命令行 --hfov 指定"
    elif sw is not None and fmm_raw is not None:
        focal_mm_v = _to_float(fmm_raw, "FocalLength")
        hfov_deg = hfov_from_focal_sensor(focal_mm_v, sw, width)
        hfov_src = f"物理焦距 {focal_mm_v:g}mm × 传感器宽 {sw:g}mm"
        if f35_raw is not None:
            alt = hfov_from_focal_35(_to_float(f35_raw, "FocalLengthIn35mmFilm"),
                                     width, height)
            if abs(alt - hfov_deg) > 0.5:
                fov_note = (f"画面被裁切过（当前 {width}×{height}），等效焦距反推会得到 "
                            f"{alt:.2f}°，已按物理焦距修正为 {hfov_deg:.2f}°")
    elif f35_raw is not None:
        hfov_deg = hfov_from_focal_35(_to_float(f35_raw, "FocalLengthIn35mmFilm"),
                                      width, height)
        hfov_src = f"由等效焦距 {float(f35_raw):g}mm 反推"
        aspect = width / float(height)
        if all(abs(aspect - a) > 0.02 for a in (4 / 3, 3 / 2, 16 / 9)):
            fov_note = f"画面长宽比 {aspect:.2f} 非标准，FOV 反推可能不准"
        elif abs(aspect - 16 / 9) < 0.02:
            fov_note = ("画面是 16:9，多数机型由 4:3/3:2 裁切而来；"
                        "若已知传感器尺寸，可用 --sensor-width 提高精度")
    else:
        raise MetadataError(
            "只有物理焦距、没有等效焦距和传感器尺寸，无法推 FOV。\n"
            "可加 --sensor-width 传感器宽(mm) 指定。\n"
            + _MANUAL_HINT
        )

    # 数码变焦：NaN / 0 是 DJI 常见的"未定义"值，按 1 处理并标记出来
    zoom_val, zoom_assumed = 1.0, False
    if zoom_raw is not None:
        try:
            z = float(str(zoom_raw).strip())
        except Exception:
            z = float("nan")
        if math.isfinite(z) and 0.5 <= z <= 20.0:
            zoom_val = z
        else:
            zoom_assumed = True

    meta = DjiMeta(
        path=path,
        width=width,
        height=height,
        latitude=_to_float(lat_raw, "GPSLatitude") if lat_raw is not None else float("nan"),
        longitude=_to_float(lon_raw, "GPSLongitude") if lon_raw is not None else float("nan"),
        rel_altitude=float(altitude) if altitude is not None else _to_float(alt_raw, "RelativeAltitude"),
        gimbal_yaw=float(yaw) if yaw is not None else _to_float(yaw_raw, "GimbalYawDegree"),
        gimbal_pitch=float(pitch) if pitch is not None else _to_float(pitch_raw, "GimbalPitchDegree"),
        gimbal_roll=float(roll) if roll is not None else _to_float(roll_raw, "GimbalRollDegree"),
        focal_35=_to_float(f35_raw, "FocalLengthIn35mmFilm") if f35_raw is not None else None,
        focal_mm=_to_float(fmm_raw, "FocalLength") if fmm_raw is not None else None,
        digital_zoom=zoom_val,
        hfov_deg=hfov_deg,
        hfov_source=hfov_src,
        model=str(model),
        reader=reader,
        absolute_alt=(_to_float(pick("AbsoluteAltitude"), "AbsoluteAltitude")
                      if pick("AbsoluteAltitude") is not None else None),
        lrf_distance=(_to_float(pick("LRFTargetDistance"), "LRFTargetDistance")
                      if pick("LRFTargetDistance") is not None else None),
        lrf_target_abs_alt=(_to_float(pick("LRFTargetAbsAlt"), "LRFTargetAbsAlt")
                            if pick("LRFTargetAbsAlt") is not None else None),
        zoom_assumed=zoom_assumed,
        fov_note=fov_note,
    )

    if meta.rel_altitude <= 1.0:
        raise MetadataError(f"相对高度异常：{meta.rel_altitude} m，无法用于地面投影。")
    if meta.gimbal_pitch > -5.0:
        raise MetadataError(
            f"云台俯仰 {meta.gimbal_pitch:.1f}° 接近水平，画面里几乎没有可用的地面，"
            "请用俯角更大的照片（建议 -30° 以上俯视）。"
        )
    return meta


# =========================================================
# 3. 几何内核：单像摄影测量（针孔模型 + 射线与水平面求交）
# =========================================================
# 公式都是公开的摄影测量基础（针孔投影、旋转矩阵、鞋带公式），本文件为独立实现。
def _rot_z(angle_rad: float) -> np.ndarray:
    """绕 ENU 天轴（Z，向上）旋转。"""
    c, s_ = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s_, 0.0],
                     [s_, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_x(angle_rad: float) -> np.ndarray:
    """绕 ENU 东轴（X）旋转。"""
    c, s_ = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s_],
                     [0.0, s_, c]], dtype=np.float64)


def build_camera_intrinsic(width: int, height: int, hfov_deg: float,
                           digital_zoom_ratio: float = 1.0) -> np.ndarray:
    """
    水平视场角 → 像素内参 K（假设方形像素，fy = fx）。

    fx 的来历：水平方向跨 width 个像素对应 hfov 的张角，
    半个画幅对应 tan(hfov/2)，故 fx = width / (2·tan(hfov/2))。
    """
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0)) * digital_zoom_ratio
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def build_camera_to_enu_rotation(yaw_deg: float, pitch_deg: float,
                                 roll_deg: float) -> np.ndarray:
    """
    相机坐标系（x 向右 / y 向下 / z 朝前）→ ENU（X 东 / Y 北 / Z 上）。

    用三个基本旋转串联得到：
      1. 绕天轴转 -yaw：把"机头朝北"的基准转到实际航向（DJI：0° 北, 90° 东）；
      2. 绕东轴转 (pitch-90°)：把水平前视压到俯视，pitch=-90° 时正好垂直向下；
      3. 绕光轴转 roll：横滚。

    与"构造 forward/right/down 三个基向量再拼成矩阵"是同一变换的两种写法，
    随机 2000 组姿态下两者差 < 1e-15（见 _geom_regress.py）。
    """
    return (_rot_z(-math.radians(yaw_deg))
            @ _rot_x(math.radians(pitch_deg) - math.pi / 2.0)
            @ _rot_z(math.radians(roll_deg)))


def rotate_about_axis(vector, axis, angle_rad: float) -> np.ndarray:
    """罗德里格斯旋转：把 vector 绕单位轴 axis 转 angle_rad。"""
    v = np.asarray(vector, dtype=np.float64)
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    c, s_ = math.cos(angle_rad), math.sin(angle_rad)
    return v * c + np.cross(k, v) * s_ + k * np.dot(k, v) * (1.0 - c)


def pixel_to_ground_enu(u, v, K: np.ndarray, R_cam_to_enu: np.ndarray,
                        camera_height_m: float, target_z_m: float = 0.0):
    """
    像素 → 地面 ENU 坐标。u / v 支持标量或数组（数组时一次算完）。

    做法：把像素反投影成相机系方向向量（z 分量归一化为 1），旋到 ENU，
    再与水平面 z = target_z_m 求交。相机位于 (0, 0, camera_height_m)。
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    dir_cam = np.stack([(u_arr - cx) / fx,
                        (v_arr - cy) / fy,
                        np.ones_like(u_arr * 1.0)], axis=-1)   # (..., 3)
    dir_enu = dir_cam @ R_cam_to_enu.T

    dz = dir_enu[..., 2]
    bad = np.argwhere(dz >= -1e-9)
    if len(bad):
        raise RuntimeError(
            f"有 {len(bad)} 个像素的射线没有射向地面（打到天空/地平线以上），"
            f"请把目标放在画面中下方。首个异常像素索引：{bad[0].tolist()}"
        )

    t = (target_z_m - camera_height_m) / dz
    return np.array([0.0, 0.0, camera_height_m], dtype=np.float64) + dir_enu * t[..., None]


def polygon_area_xy(points_xy) -> float:
    """鞋带公式：平面多边形面积。叉积求和，等价于 x_i·y_{i+1} - x_{i+1}·y_i。"""
    pts = np.asarray(points_xy, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    cross = x * np.roll(y, -1) - y * np.roll(x, -1)
    return float(abs(cross.sum()) / 2.0)


def distance_xy(p1, p2) -> float:
    """两点水平距离（忽略 Z）。"""
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    return float(np.linalg.norm(b[:2] - a[:2]))


def enu_to_latlon(east: float, north: float, lat0_deg: float, lon0_deg: float):
    """ENU 平面坐标 → 经纬度（局部平面近似，几公里内误差可忽略）。"""
    r = 6378137.0
    lat = lat0_deg + math.degrees(north / r)
    lon = lon0_deg + math.degrees(east / (r * math.cos(math.radians(lat0_deg))))
    return lat, lon


# =========================================================
# 4. 相机封装：一次建好，反复投影
# =========================================================
class Camera:
    """
    把一张照片的相机模型打包，提供像素→地面、面积、经纬度换算。

    高度基准是本方法精度的第一命门：
      - 默认 h = RelativeAltitude（相对起飞点）；
      - target_z 用来平移目标平面（目标地面比起飞点高多少米就填多少）；
      - height_m 可直接给"相机到目标平面的真实高度"，优先级最高（标定/LRF 修正用）。

    height_source 只影响"高度本身有多可信"（误差预算用），不影响几何：
      rel_alt（相机相对高度，最常用） / target_z（手工填了地面高差） /
      lrf（激光测距） / calib（用已知长度反推，最可信） / manual。
    """

    def __init__(self, meta: DjiMeta, target_z: float = 0.0,
                 height_m: float | None = None, height_source: str = ""):
        self.meta = meta
        self.K = build_camera_intrinsic(meta.width, meta.height,
                                        meta.hfov_deg, meta.digital_zoom)
        self.R = build_camera_to_enu_rotation(meta.gimbal_yaw,
                                              meta.gimbal_pitch,
                                              meta.gimbal_roll)
        self.target_z = float(target_z)
        if height_m is not None:
            self.h = float(height_m)
            self.height_source = height_source or "manual"
        else:
            self.h = meta.rel_altitude - self.target_z
            self.height_source = ("rel_alt" if abs(self.target_z) < 1e-9
                                  else "target_z")
        if self.h <= 1.0:
            raise MetadataError(f"相机到目标平面的高度只有 {self.h:.2f} m，无法量算。")

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    def with_height(self, height_m: float) -> "Camera":
        return Camera(self.meta, target_z=0.0, height_m=height_m,
                      height_source="calib")

    def height_note(self) -> str:
        label = {"rel_alt": "相对起飞点", "target_z": "已按地面高差修正",
                 "lrf": "激光测距", "calib": "已知长度标定",
                 "manual": "手工指定"}.get(self.height_source, self.height_source)
        return f"高度基准 {self.h:.2f} m（{label}）"

    def ground_of(self, u, v):
        return pixel_to_ground_enu(u, v, self.K, self.R, self.h, 0.0)

    def pixels_to_ground(self, points_uv) -> np.ndarray:
        pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
        return self.ground_of(pts[:, 0], pts[:, 1])

    def area_of_pixels(self, points_uv) -> float:
        return polygon_area_xy(self.pixels_to_ground(points_uv)[:, :2])

    def distance_of_pixels(self, p1, p2) -> float:
        g = self.pixels_to_ground([p1, p2])
        return distance_xy(g[0], g[1])

    def implied_height(self, p1, p2, known_length_m: float) -> float:
        """用一段已知长度的物体反推"相机到目标地面的真实高度"。"""
        measured = self.distance_of_pixels(p1, p2)
        if measured <= 1e-6:
            raise RuntimeError("两点在地面上重合，无法标定")
        return self.h * float(known_length_m) / measured

    def center_latlon(self, points_uv):
        g = self.pixels_to_ground(points_uv)
        e, n = float(np.mean(g[:, 0])), float(np.mean(g[:, 1]))
        return enu_to_latlon(e, n, self.meta.latitude, self.meta.longitude)

    def depth_range(self, points_uv):
        """
        目标点到相机的三维距离范围（米）。用来评估"目标平面水平"假设的风险：
        斜拍时远场与近场深度比越大，高度误差被放大的程度越大。
        """
        g = self.pixels_to_ground(points_uv)
        cam_pos = np.array([0.0, 0.0, self.h])
        d = np.linalg.norm(g - cam_pos, axis=1)
        return float(d.min()), float(d.max())

    def gsd_at_center(self) -> float:
        """画面中心（沿光轴）的地面分辨率，米/像素，仅作参考。"""
        return self.h / self.fx

    # ---------------- 误差预算 ----------------
    def default_sigma_h(self) -> float:
        """
        未标定时"相机到目标地面高度"的不确定度（1σ，米）。

        两个来源 RSS 合成：
          - 相机相对高度本身：气压/IMU 融合通常 1~2%，取 2%；
          - 目标地面与起飞点的高差没实测：平原也常有 1~3 m 起伏，取 2 m。
        做过标定或 LRF 后高度可信度高一个量级，改用小得多的值。
        """
        if self.height_source in ("calib", "lrf"):
            return math.hypot(0.005 * self.h, 0.3)
        return math.hypot(0.02 * self.h, 2.0)

    def _area_with_angles(self, pts: np.ndarray, yaw: float, pitch: float,
                          roll: float) -> float:
        """换一组姿态角重算面积（只用于求偏导，不改 self）。"""
        R = build_camera_to_enu_rotation(yaw, pitch, roll)
        g = pixel_to_ground_enu(pts[:, 0], pts[:, 1], self.K, R, self.h, 0.0)
        return polygon_area_xy(g[:, :2])

    def area_uncertainty(self, points_uv, sigma_h_m: float | None = None,
                         sigma_pitch_deg: float = 0.5,
                         sigma_roll_deg: float = 0.3,
                         sigma_yaw_deg: float = 1.0,
                         sigma_edge_px: float = 1.0,
                         base_area: float | None = None) -> dict:
        """
        面积误差预算：把"这块地算出来可能差多少"拆成三项，再按平方和合成。

        三项各是什么，怎么算：
          1) 高度（主导项）：A ∝ h²，故一阶相对误差 = 2·σh/h。纯解析，最准。
          2) 姿态：pitch/roll/yaw 各给一个 σ（DJI 云台+IMU 绝对值通常 0.3~1°），
             用中心差分数值求导：|A(θ+σ) - A(θ-σ)| / (2A)，三个角 RSS 合成。
             斜拍越厉害、目标离光轴越远，这项越大 —— 与实测现象一致。
          3) 边界：轮廓每点的定位精度约 ±1 px。一阶保守取 δA ≈ P·σ（把整圈边界的
             误差同向叠加，属上界；随机抖动其实会互相抵消，故真实值通常比它小）。
             相对量 = σ_px · 周长_px / 面积_px。

        返回 dict：rel 为合成后的相对误差（1σ），parts 是三块各自的贡献。
        注意这是"由已知参数推断的精度"，不是与真值比对得到的实测精度。
        """
        pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
        base = float(base_area) if base_area is not None else self.area_of_pixels(pts)
        if not np.isfinite(base) or base <= 0:
            return {"area_m2": 0.0, "rel": float("nan"), "abs_m2": float("nan"),
                    "parts": {}, "sigma_h_m": float("nan")}

        # 1) 高度：A ∝ h²
        sh = float(sigma_h_m) if sigma_h_m is not None else self.default_sigma_h()
        rel_h = 2.0 * sh / self.h

        # 2) 姿态：中心差分
        yaw0, pitch0, roll0 = (self.meta.gimbal_yaw, self.meta.gimbal_pitch,
                               self.meta.gimbal_roll)
        rel_att = {}
        for key, sig, ang0 in (("pitch", sigma_pitch_deg, pitch0),
                               ("roll", sigma_roll_deg, roll0),
                               ("yaw", sigma_yaw_deg, yaw0)):
            kw = {"yaw": yaw0, "pitch": pitch0, "roll": roll0}
            try:
                kw[key] = ang0 + sig
                ap = self._area_with_angles(pts, kw["yaw"], kw["pitch"], kw["roll"])
                kw[key] = ang0 - sig
                am = self._area_with_angles(pts, kw["yaw"], kw["pitch"], kw["roll"])
                rel_att[key] = abs(ap - am) / (2.0 * base)
            except RuntimeError:
                # 扰动后有个别像素射线朝上了，说明本来就在地平线附近，这项不可估
                rel_att[key] = 0.0
        rel_att_total = math.sqrt(sum(v * v for v in rel_att.values()))

        # 3) 边界：δA ≈ P·σ（像素域估，投影是局部线性的，相对量可直接搬）
        per_px = float(np.sum(np.hypot(np.diff(np.r_[pts[:, 0], pts[0, 0]]),
                                       np.diff(np.r_[pts[:, 1], pts[0, 1]]))))
        area_px = polygon_area_xy(pts)
        rel_edge = (float(sigma_edge_px) * per_px / area_px) if area_px > 0 else 0.0

        rel = math.sqrt(rel_h ** 2 + rel_att_total ** 2 + rel_edge ** 2)
        return {
            "area_m2": base,
            "rel": rel,
            "abs_m2": base * rel,
            "sigma_h_m": sh,
            "parts": {"height": rel_h, "attitude": rel_att_total, "edge": rel_edge},
            "attitude_detail": rel_att,
            "height_source": self.height_source,
        }


# =========================================================
# 5. 面积换算
# =========================================================
MU_PER_M2 = 1.0 / 666.6666666666666  # 1 亩 = 666.667 m²


def m2_to_mu(area_m2: float) -> float:
    return area_m2 * MU_PER_M2


def format_area(area_m2: float) -> str:
    return f"{area_m2:.1f} m² / {m2_to_mu(area_m2):.3f} 亩"


# =========================================================
# 6. 命令行：打印单张照片参数
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="读取 DJI 照片参数并打印相机模型（阶段 0 自检工具）")
    ap.add_argument("photo", help="DJI 原始照片路径")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    ap.add_argument("--altitude", type=float, help="手工指定相对高度(m)，覆盖元数据")
    ap.add_argument("--pitch", type=float, help="手工指定云台俯仰(°)")
    ap.add_argument("--yaw", type=float, help="手工指定云台偏航(°)")
    ap.add_argument("--roll", type=float, help="手工指定云台横滚(°)")
    ap.add_argument("--hfov", type=float, help="手工指定水平视场角(°)，覆盖自动推导")
    ap.add_argument("--sensor-width", type=float,
                    help="手工指定传感器物理宽度(mm)，用于精确算内参")
    args = ap.parse_args()

    try:
        meta = read_dji_meta(args.photo, altitude=args.altitude, pitch=args.pitch,
                             yaw=args.yaw, roll=args.roll, hfov=args.hfov,
                             sensor_width=args.sensor_width)
    except MetadataError as exc:
        print("参数读取失败：\n" + str(exc))
        raise SystemExit(2)

    cam = Camera(meta)
    if args.json:
        payload = asdict(meta)
        payload.update({"fx": cam.fx, "vfov_deg": meta.vfov_deg,
                        "gsd_center_m_per_px": cam.gsd_at_center()})
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(meta.summary())
    print(f"相机内参 fx     : {cam.fx:.2f} px")
    print(f"画面中心 GSD    : {cam.gsd_at_center() * 100:.2f} cm/px")
    print(f"画面中心覆盖宽  : {cam.gsd_at_center() * meta.width:.1f} m")


if __name__ == "__main__":
    main()
