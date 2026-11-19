# -*- coding: utf-8 -*-
"""
_geom_regress.py —— 几何内核重写前后的等价性回归

做法：把重写前的 uav_geo 当作"旧实现"，与"新实现"逐项比对
  1) 旋转矩阵 R（随机 2000 组 yaw/pitch/roll）
  2) 像素→地面坐标（随机相机位姿 + 随机像素）
  3) 多边形面积、两点距离、ENU→经纬度
要求：最大绝对误差 < 1e-9（浮点等价级别）。

用法：
    python _geom_regress.py <旧实现.py>
"""
import importlib.util
import math
import sys

import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    old_path = sys.argv[1]
    old = load(old_path, "uav_geo_old")
    import uav_geo as new

    rng = np.random.default_rng(20261119)
    n = 2000

    # ---- 1) 旋转矩阵 ----
    dR = 0.0
    for _ in range(n):
        yaw = rng.uniform(-180, 180)
        pitch = rng.uniform(-90, -20)
        roll = rng.uniform(-15, 15)
        Ro = old.build_camera_to_enu_rotation(yaw, pitch, roll)
        Rn = new.build_camera_to_enu_rotation(yaw, pitch, roll)
        dR = max(dR, float(np.abs(Ro - Rn).max()))

    # ---- 2) 内参 ----
    dK = 0.0
    for _ in range(n):
        w = int(rng.integers(1000, 6000))
        h = int(rng.integers(700, 4000))
        hfov = rng.uniform(50, 90)
        zoom = float(rng.choice([1.0, 2.0, 3.0]))
        Ko = old.build_camera_intrinsic(w, h, hfov, zoom)
        Kn = new.build_camera_intrinsic(w, h, hfov, zoom)
        dK = max(dK, float(np.abs(Ko - Kn).max()))

    # ---- 3) 像素→地面 ----
    dP = 0.0
    for _ in range(n):
        w = int(rng.integers(1000, 6000))
        h = int(rng.integers(700, 4000))
        hfov = rng.uniform(50, 90)
        yaw = rng.uniform(-180, 180)
        pitch = rng.uniform(-90, -45)
        roll = rng.uniform(-10, 10)
        H = rng.uniform(20, 200)
        K = new.build_camera_intrinsic(w, h, hfov)
        R = new.build_camera_to_enu_rotation(yaw, pitch, roll)
        for _ in range(3):
            u = rng.uniform(0.1 * w, 0.9 * w)
            v = rng.uniform(0.5 * h, 0.98 * h)   # 只取画面下半部，避免打到天空
            try:
                po = old.pixel_to_ground_enu(u, v, K, R, H)
            except RuntimeError:
                continue
            pn = new.pixel_to_ground_enu(u, v, K, R, H)
            dP = max(dP, float(np.abs(po - pn).max()))

    # ---- 4) 面积 / 距离 / 经纬度 ----
    dA = dD = dL = 0.0
    for _ in range(200):
        m = int(rng.integers(3, 400))
        pts = rng.uniform(-50, 50, size=(m, 2))
        dA = max(dA, abs(old.polygon_area_xy(pts) - new.polygon_area_xy(pts)))
        dD = max(dD, abs(old.distance_xy(pts[0], pts[1]) - new.distance_xy(pts[0], pts[1])))
        e, nn = rng.uniform(-5000, 5000, size=2)
        lat0, lon0 = rng.uniform(20, 45), rng.uniform(100, 125)
        dL = max(dL, float(np.abs(np.array(old.enu_to_latlon(e, nn, lat0, lon0))
                                  - np.array(new.enu_to_latlon(e, nn, lat0, lon0))).max()))

    print("=" * 62)
    print("几 何 内 核 等 价 性 回 归")
    print("=" * 62)
    rows = [("旋转矩阵 R", dR, 1e-9), ("内参 K", dK, 1e-9),
            ("像素→地面(米)", dP, 1e-9), ("多边形面积", dA, 1e-9),
            ("两点距离", dD, 1e-9), ("ENU→经纬度(度)", dL, 1e-12)]
    ok = True
    for name, d, tol in rows:
        good = d <= tol
        ok = ok and good
        print(f"  {'✓' if good else '✗'} {name:16s} 最大差 {d:.3e}  容差 {tol:.0e}")
    print("-" * 62)
    print("结论：" + ("新旧实现数值等价，可以替换" if ok else "存在差异，不应替换"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
