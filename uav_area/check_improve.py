# -*- coding: utf-8 -*-
"""
check_improve.py —— 验证"误差预算"与"已知长度标定"两项改进真的成立

全部用可解析的合成场景，不需要 SAM2、不需要真实照片，几秒跑完：

  A 投影无偏    垂直俯拍下 20×30 m 的地面矩形，反投影面积必须等于 600 m²
  B 面积∝h²     航高 +10% → 面积必须 +21%（这是"高度是第一命门"的定量依据）
  C1 误差模型   高度+姿态项：解析 1σ  vs  600 次蒙特卡洛扰动的实测 1σ
  C2 边界项     掩膜整体膨胀/腐蚀 1 px 的真实面积变化  vs  公式 δA≈P·σ
  D 标定纠偏    航高谎报 +10%（面积错 21%），用一段 4.65 m 参照物标定后拉回 <0.5%

用法：
    python check_improve.py            # 只跑合成场景
    python check_improve.py 照片.JPG    # 额外用真实照片参数跑一遍误差预算
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uav_geo as g


# =========================================================
# 合成场景工具
# =========================================================
def synth_meta(rel_alt: float = 100.0, pitch: float = -90.0, yaw: float = 0.0,
               roll: float = 0.0, w: int = 4000, h: int = 3000,
               hfov: float = 73.0) -> g.DjiMeta:
    """造一份和 DJI 等价的最小元数据（参数取值贴近 Mavic 3 广角）。"""
    return g.DjiMeta(
        path="synthetic.jpg", width=w, height=h, latitude=25.0, longitude=102.7,
        rel_altitude=float(rel_alt), gimbal_yaw=float(yaw),
        gimbal_pitch=float(pitch), gimbal_roll=float(roll),
        focal_35=24.0, focal_mm=8.4, digital_zoom=1.0, hfov_deg=float(hfov),
        hfov_source="synthetic", model="synthetic", reader="synthetic",
        absolute_alt=1900.0, lrf_distance=float("nan"),
        lrf_target_abs_alt=float("nan"), zoom_assumed=False, fov_note="")


def ground_to_pixel(cam: g.Camera, G) -> tuple:
    """
    地面 ENU 点 → 像素（pixel_to_ground 的逆运算）。

    相机在 (0,0,h)；地面点相对相机的向量旋到相机系后就是 dir_cam·t，
    归一化第三分量为 1，再用内参投到像素。
    """
    C = np.array([0.0, 0.0, cam.h])
    w = cam.R.T @ (np.asarray(G, dtype=np.float64) - C)
    if abs(w[2]) < 1e-12:
        raise ValueError("该点与相机光心同高，投不到像平面")
    d = w / w[2]
    return (cam.K[0, 0] * d[0] + cam.K[0, 2], cam.K[1, 1] * d[1] + cam.K[1, 2])


def rect_ring(e0, e1, n0, n1, k: int = 24) -> np.ndarray:
    """地面矩形边界 → (N,3) ENU 点环（按边细分，便于算周长/面积）。"""
    es = np.linspace(e0, e1, k)
    ns = np.linspace(n0, n1, k)
    pts = ([(e, n0) for e in es] + [(e1, n) for n in ns]
           + [(e, n1) for e in es[::-1]] + [(e0, n) for n in ns[::-1]])
    return np.array([[e, n, 0.0] for e, n in pts], dtype=np.float64)


def axis_ground_point(cam: g.Camera) -> np.ndarray:
    """光轴与地面的交点 —— 用它当目标中心，保证目标落在画面里。"""
    d = cam.R[:, 2]
    t = (0.0 - cam.h) / d[2]
    return np.array([0.0, 0.0, cam.h]) + d * t


def rect_pixels(cam: g.Camera, dE: float, dN: float, k: int = 24,
                center_uv=None) -> np.ndarray:
    """在指定位置造一个 dE×dN 的地面矩形，返回落到像素上的点环。"""
    c = (axis_ground_point(cam) if center_uv is None
         else cam.ground_of(float(center_uv[0]), float(center_uv[1])))
    ring = rect_ring(c[0] - dE / 2, c[0] + dE / 2,
                     c[1] - dN / 2, c[1] + dN / 2, k)
    return np.array([ground_to_pixel(cam, p) for p in ring], dtype=np.float64)


# =========================================================
# A / B：投影与高度平方律
# =========================================================
def check_projection():
    cam = g.Camera(synth_meta(rel_alt=100.0))
    fx = cam.fx
    # 垂直俯拍：像素 → 地面的尺度就是 h/fx，所以想要 20 m 宽就要 20*fx/h 像素
    a, b = 10.0 * fx / cam.h, 15.0 * fx / cam.h
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    u = np.linspace(cx - a, cx + a, 41)
    v = np.linspace(cy - b, cy + b, 41)
    ring = ([(x, cy - b) for x in u] + [(cx + a, y) for y in v]
            + [(x, cy + b) for x in u[::-1]] + [(cx - a, y) for y in v[::-1]])
    area = cam.area_of_pixels(np.array(ring, dtype=np.float64))
    expect = 20.0 * 30.0
    rel = abs(area - expect) / expect
    return [("A 垂直俯拍 20×30 m 反投影面积", area, expect, rel, 1e-3, "m²")]


def check_h_square():
    cam1 = g.Camera(synth_meta(rel_alt=100.0))
    cam2 = g.Camera(synth_meta(rel_alt=110.0))
    pts = rect_pixels(cam1, 20.0, 30.0)
    a1, a2 = cam1.area_of_pixels(pts), cam2.area_of_pixels(pts)
    ratio, expect = a2 / a1, (110.0 / 100.0) ** 2
    return [("B 航高 +10% → 面积比", ratio, expect, abs(ratio - expect) / expect,
             1e-6, "×")]


# =========================================================
# C1：误差模型 vs 蒙特卡洛
# =========================================================
def check_uncertainty_mc(n: int = 600, seed: int = 20261120):
    meta = synth_meta(rel_alt=100.0, pitch=-75.0, yaw=37.0, roll=3.0)
    cam = g.Camera(meta)
    # 目标故意放在画面右下方（真实使用中很少正好压在光轴上），
    # 这样姿态误差不会被对称性抵消，才有代表性
    cu = (cam.K[0, 2] + 0.30 * meta.width, cam.K[1, 2] + 0.18 * meta.height)
    pts = rect_pixels(cam, 25.0, 18.0, center_uv=cu)
    base = cam.area_of_pixels(pts)

    unc = cam.area_uncertainty(pts)
    pred = math.sqrt(unc["parts"]["height"] ** 2 + unc["parts"]["attitude"] ** 2)

    sh = unc["sigma_h_m"]
    sp, sr, sy = 0.5, 0.3, 1.0                    # 与 area_uncertainty 的默认值一致
    rng = np.random.default_rng(seed)
    areas = np.empty(n)
    for i in range(n):
        h2 = cam.h + rng.normal(0.0, sh)
        if h2 <= 1.0:
            h2 = 1.0
        c2 = g.Camera(meta, height_m=h2)
        c2.R = g.build_camera_to_enu_rotation(meta.gimbal_yaw + rng.normal(0, sy),
                                              meta.gimbal_pitch + rng.normal(0, sp),
                                              meta.gimbal_roll + rng.normal(0, sr))
        areas[i] = c2.area_of_pixels(pts)
    rel_mc = float(np.std(areas) / base)

    return [
        ("C1 误差模型 预测 1σ (含高度+姿态)", pred * 100, rel_mc * 100,
         abs(pred - rel_mc) / rel_mc, 0.30, "%"),
    ], unc, base


# =========================================================
# C2：边界项（像素域，避免投影干扰）
# =========================================================
def check_edge_formula():
    import cv2

    m = np.zeros((1000, 1200), np.uint8)
    cv2.rectangle(m, (300, 300), (900, 800), 1, -1)      # 600 × 500 px 的块
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    a0 = g.polygon_area_xy(cnt)

    k = np.ones((3, 3), np.uint8)
    a_up = g.polygon_area_xy(
        max(cv2.findContours(cv2.dilate(m, k), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)[0],
            key=cv2.contourArea).reshape(-1, 2).astype(np.float64))
    a_dn = g.polygon_area_xy(
        max(cv2.findContours(cv2.erode(m, k), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)[0],
            key=cv2.contourArea).reshape(-1, 2).astype(np.float64))

    actual = (abs(a_up - a0) + abs(a0 - a_dn)) / 2.0 / a0        # 系统性 1 px
    per = float(np.sum(np.hypot(np.diff(np.r_[cnt[:, 0], cnt[0, 0]]),
                                np.diff(np.r_[cnt[:, 1], cnt[0, 1]]))))
    # δA≈P·σ 对凸多边形其实是精确关系（面积对等距偏移的导数就是周长，即 Steiner 公式），
    # 对凹形边界才退化成偏大的近似 —— 这里两者应当几乎相等
    pred = 1.0 * per / a0                                         # 公式 δA≈P·σ
    return [("C2 边界 1px 公式 P·σ/A  vs 膨胀腐蚀实测", pred * 100, actual * 100,
             abs(pred - actual) / max(actual, 1e-9), 1.0, "%")]


# =========================================================
# D：已知长度标定
# =========================================================
def check_calibration():
    # 真值场景：斜拍，真高 100 m
    truth = g.Camera(synth_meta(rel_alt=100.0, pitch=-70.0, yaw=30.0))
    pts = rect_pixels(truth, 25.0, 18.0)
    a_true = truth.area_of_pixels(pts)

    # 现场：照片里写的相对高度错了 10%（=110 m）
    meta_bad = synth_meta(rel_alt=110.0, pitch=-70.0, yaw=30.0)
    bad = g.Camera(meta_bad)

    # 画面里有一段已知 4.65 m 的参照物（车长），两端点在真值几何下的像素
    c = axis_ground_point(truth)
    G1 = np.array([c[0] - 2.0, c[1] - 1.0, 0.0])
    G2 = np.array([c[0] + 2.0, c[1] + 1.0, 0.0])
    G2 = G1 + (G2 - G1) / g.distance_xy(G1, G2) * 4.65          # 拉成精确 4.65 m
    p1, p2 = (ground_to_pixel(truth, G1), ground_to_pixel(truth, G2))

    h_calib = bad.implied_height(p1, p2, 4.65)
    calib = bad.with_height(h_calib)
    a_bad = bad.area_of_pixels(pts)
    a_cal = calib.area_of_pixels(pts)

    # 顺便验证：用真值相机自己标定，应还原出同一个 h（与高度绝对值无关）
    h_self = truth.implied_height(p1, p2, 4.65)

    rows = [
        ("D1 标定前 面积相对真值", abs(a_bad - a_true) / a_true * 100, 21.0,
         abs(abs(a_bad - a_true) / a_true - 0.21), 2.0, "%"),
        ("D2 标定后 面积相对真值", abs(a_cal - a_true) / a_true * 100, 0.0,
         abs(a_cal - a_true) / a_true, 5e-3, "%"),
        ("D3 标定还原的航高", h_calib, 100.0, abs(h_calib - 100.0), 1e-6, "m"),
        ("D4 用正确航高自标定应有 h", h_self, 100.0, abs(h_self - 100.0), 1e-6, "m"),
    ]
    return rows


# =========================================================
# 真实照片：只验"数值不荒谬"
# =========================================================
def check_real_photo(path: str):
    meta = g.read_dji_meta(path)
    cam = g.Camera(meta)
    pts = rect_pixels(cam, 20.0, 20.0)          # 假定画面中间有一块 20×20 m 的地
    unc = cam.area_uncertainty(pts)
    print("-" * 66)
    print(f"真实照片：{os.path.basename(path)}")
    print(f"  {meta.summary().splitlines()[0] if meta.summary() else ''}")
    print(f"  航高 {cam.h:.1f} m（{cam.height_source}）  GSD {cam.gsd_at_center()*100:.2f} cm/px")
    print(f"  20×20 m 参考块面积 {unc['area_m2']:.1f} m²"
          f"  →  误差预算 ±{unc['rel']*100:.1f}%"
          f"（高 {unc['parts']['height']*100:.1f} / 姿 {unc['parts']['attitude']*100:.1f}"
          f" / 边 {unc['parts']['edge']*100:.1f}）")


def check_sam_pipeline(path: str, click=(1810.0, 1810.0)):
    """端到端：真照片 → SAM2 点选 → build_field，确认误差字段一路传到位。"""
    import time

    from measure_area import (DEFAULT_CONFIG, DEFAULT_WEIGHT, Segmenter,
                              build_field)

    meta = g.read_dji_meta(path)
    cam = g.Camera(meta)
    img = g.imread_unicode(path)
    t0 = time.time()
    seg = Segmenter(DEFAULT_WEIGHT, DEFAULT_CONFIG)
    seg.set_image(img)
    t1 = time.time()
    masks, scores = seg.predict([list(click)], [1])
    idx = int(np.argmax(scores))
    f = build_field(cam, masks[idx], 1, 300, score=float(scores[idx]))
    t2 = time.time()

    print("-" * 66)
    print(f"端到端（真实 SAM2）：{os.path.basename(path)}  点 {click}")
    print(f"  模型+编码 {t1-t0:.1f}s　单次分割 {t2-t1:.2f}s　device={seg.device}")
    print(f"  面积 {f.area_m2:.2f} m² / {f.area_mu:.4f} 亩　置信度 {f.score:.3f}")
    p = f.err_parts or {}
    print(f"  误差 ±{f.err_rel*100:.1f}%"
          f"（高度 {p.get('height', 0)*100:.1f} / 姿态 {p.get('attitude', 0)*100:.1f}"
          f" / 边界 {p.get('edge', 0)*100:.1f}）")
    good = np.isfinite(f.err_rel) and 0.0 < f.err_rel < 0.5 and f.area_m2 > 0
    print(f"  {'✓' if good else '✗'} 误差字段已贯通，数值在合理区间（0 < rel < 50%）")
    return good


def check_store_roundtrip():
    """工程数据层：误差与标定能否存进 project.json、再读出来影响重算和导出。"""
    import csv
    import dataclasses
    import json
    import tempfile

    from project_store import FieldRec, ProjectStore

    meta = synth_meta(rel_alt=100.0)
    cam = g.Camera(meta)
    pts = rect_pixels(cam, 20.0, 30.0)
    root = tempfile.mkdtemp(prefix="uavarea_store_")
    open(os.path.join(root, "a.jpg"), "wb").close()     # 只测数据层，内容为空即可

    def builder(st):
        def _b(name, md):
            ch = st.calib_h(name)
            return g.Camera(g.DjiMeta(**md),
                            height_m=ch if ch else None,
                            height_source="calib" if ch else "")
        return _b

    st = ProjectStore(root)
    st.set_meta("a.jpg", dataclasses.asdict(meta))
    st.add_field("a.jpg", FieldRec(fid=1, name="测试块",
                                   contour_px=[list(p) for p in pts],
                                   area_m2=0.0, area_mu=0.0))
    st.recompute(builder(st))
    st.save()

    st2 = ProjectStore(root)                            # 模拟"换台机器重新打开"
    r0 = st2.fields("a.jpg")[0]
    a0, e0 = r0.area_m2, r0.err_rel

    st2.set_calib("a.jpg", 110.0, 20.0, (0, 0), (1, 1))  # 航高 +10%
    st2.recompute(builder(st2))
    a1 = st2.fields("a.jpg")[0].area_m2

    csv_path = st2.export_csv(os.path.join(root, "out.csv"))
    with open(csv_path, encoding="utf-8-sig") as f:
        head = next(csv.reader(f))
    gj = st2.export_geojson(os.path.join(root, "out.geojson"), builder(st2))
    with open(gj[0], encoding="utf-8") as f:
        props = json.load(f)["features"][0]["properties"]

    return [
        ("E1 数据层 面积重算", a0, 600.0, abs(a0 - 600.0) / 600.0, 5e-3, "m²"),
        ("E2 误差字段落盘(重开后仍在)", e0 * 100, e0 * 100, 0.0 if 0 < e0 < 0.5 else 1.0,
         0.5, "%"),
        ("E3 标定 h=110 后面积 ×1.21", a1 / a0, 1.21, abs(a1 / a0 - 1.21) / 1.21,
         5e-3, "×"),
        ("E4 CSV 表头含误差列", 1.0 if "误差_%" in head else 0.0, 1.0,
         0.0 if "误差_%" in head else 1.0, 0.5, ""),
        ("E5 GeoJSON 含 error_pct", 1.0 if "error_pct" in props else 0.0, 1.0,
         0.0 if "error_pct" in props else 1.0, 0.5, ""),
    ]


def main():
    print("=" * 66)
    print("改 进 有 效 性 验 证（误差预算 + 已知长度标定）")
    print("=" * 66)

    rows = []
    rows += check_projection()
    rows += check_h_square()
    mc_rows, unc, base = check_uncertainty_mc()
    rows += mc_rows
    rows += check_edge_formula()
    rows += check_calibration()
    rows += check_store_roundtrip()

    ok = True
    for name, got, expect, err, tol, unit in rows:
        good = err <= tol
        ok = ok and good
        print(f"  {'✓' if good else '✗'} {name}")
        print(f"      实测 {got:.4f}{unit}   期望 {expect:.4f}{unit}"
              f"   差 {err:.2e}（容差 {tol:.0e}）")

    print("-" * 66)
    print(f"场景：斜拍 pitch=-75° 地面 {base:.1f} m² 的一块地")
    print(f"  误差预算分解：高度 ±{unc['parts']['height']*100:.1f}%"
          f"  姿态 ±{unc['parts']['attitude']*100:.1f}%"
          f"  边界 ±{unc['parts']['edge']*100:.1f}%"
          f"  →  合成 ±{unc['rel']*100:.1f}%（{unc['abs_m2']:.1f} m²）")
    ad = unc["attitude_detail"]
    print(f"  姿态明细：pitch ±{ad['pitch']*100:.2f}%  roll ±{ad['roll']*100:.2f}%"
          f"  yaw ±{ad['yaw']*100:.2f}%"
          f"（yaw 是绕天轴的平面内旋转，对面积无一阶影响，恒为 0 属预期）")

    photos = [a for a in sys.argv[1:] if not a.startswith("--") and os.path.isfile(a)]
    if photos:
        check_real_photo(photos[0])
    if "--sam" in sys.argv and photos:
        ok = check_sam_pipeline(photos[0]) and ok

    print("-" * 66)
    print("结论：" + ("两项改进均通过验证" if ok else "存在未通过项，需要修"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
