# -*- coding: utf-8 -*-
"""
check_phase0.py —— 阶段 0：环境与几何内核验证（不涉及 SAM2）

做两件事：
1) 批量打印 demo 照片的测量参数（相对高度 / 云台角度 / FOV / 图像尺寸）；
2) 用两个像素点量一段已知长度的物体（默认 car_test 的车长 5m），
   检查几何链路算出来的距离是否合理。

用法
----
    # 批量看参数
    python check_phase0.py --demo

    # 非交互：直接给两个像素点
    python check_phase0.py "照片.jpg" --p1 1075,2204 --p2 1075,2330 --known 5

    # 交互：在窗口里点两个点
    python check_phase0.py "照片.jpg" --interactive --known 5

约定：像素坐标 (u, v) 原点在图像左上角，u 向右、v 向下，单位像素。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_geo import (  # noqa: E402
    Camera, MetadataError, imread_unicode, read_dji_meta, format_area,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(_ROOT, "参考", "UAVGeoMeasure", "demo_data")
DEMO_FILES = ["grass_test.JPG", "car_test.jpg", "road_test.jpg", "solar_panel_test.jpg"]
MAX_DISPLAY_WIDTH = 1500


# =========================================================
# 1. 批量打印参数
# =========================================================
def print_demo_table(demo_dir: str):
    print("=" * 108)
    print("阶 段 0 · 元 数 据 读 取 检 查")
    print("=" * 108)
    header = ("{:22s}{:12s}{:>9s}{:>8s}{:>8s}{:>8s}{:>8s}{:>8s}{:>9s}{:>10s}"
              .format("文件", "尺寸", "高度m", "pitch", "yaw", "roll",
                      "hFOV°", "fx(px)", "GSDcm", "覆盖宽m"))
    print(header)
    print("-" * 108)

    ok, fail = 0, 0
    for name in DEMO_FILES:
        path = os.path.join(demo_dir, name)
        if not os.path.isfile(path):
            print(f"{name:22s}  [缺失] {path}")
            fail += 1
            continue
        try:
            meta = read_dji_meta(path)
        except MetadataError as exc:
            print(f"{name:22s}  [读取失败] {str(exc).splitlines()[0]}")
            fail += 1
            continue

        cam = Camera(meta)
        gsd = cam.gsd_at_center() * 100
        print("{:22s}{:12s}{:>9.2f}{:>8.1f}{:>8.1f}{:>8.1f}{:>8.2f}{:>8.0f}"
              "{:>9.2f}{:>10.1f}".format(
                  name, f"{meta.width}x{meta.height}", meta.rel_altitude,
                  meta.gimbal_pitch, meta.gimbal_yaw, meta.gimbal_roll,
                  meta.hfov_deg, cam.fx, gsd, gsd * meta.width / 100))
        ok += 1

    print("-" * 108)
    print(f"成功 {ok} 张，失败 {fail} 张。")
    print("说明：'GSDcm' 是画面中心的地面分辨率（cm/像素），'覆盖宽m' 是画面中心对应的地面宽度，")
    print("      仅用于直观感受尺度；实际量算走逐点射线投影，不依赖单一 GSD。")
    print()
    print("FOV 口径说明：本项目不用 exiftool 的 Composite:FOV（它可能是对角/水平，机型间不统一），")
    print("      而是由 EXIF 的 FocalLengthIn35mmFilm 按'35mm 等效是对角视场'的定义反推水平 FOV。")
    print("      可用每张图的 物理焦距 与 传感器宽 交叉核对，本 demo 数据两条路结果一致（≈71.6°）。")


# =========================================================
# 2. 交互点选两点
# =========================================================
def pick_two_points(img, title: str = "点两点量距离"):
    import cv2

    h, w = img.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / w)
    disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    points = []
    win = title

    def redraw():
        canvas = disp.copy()
        for i, p in enumerate(points):
            cv2.circle(canvas, p, 6, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if len(points) == 2:
            cv2.line(canvas, points[0], points[1], (0, 0, 255), 2)
        cv2.putText(canvas, "Left click: 2 points | C: clear | Enter: measure | Q: quit",
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and len(points) == 2:
            break
        if key in (ord("c"), ord("C")):
            points.clear()
            redraw()
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)
    return [(x / scale, y / scale) for x, y in points]


# =========================================================
# 3. 两点量距 + 精度诊断
# =========================================================
def report_distance(cam: Camera, p1, p2, known: float | None):
    d = cam.distance_of_pixels(p1, p2)
    g = cam.pixels_to_ground([p1, p2])
    pix = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    print("-" * 78)
    print(f"像素 1: ({p1[0]:.1f}, {p1[1]:.1f})   像素 2: ({p2[0]:.1f}, {p2[1]:.1f})   "
          f"像素间距 {pix:.1f} px")
    print(f"地面 ENU 1: E={g[0][0]:8.2f}  N={g[0][1]:8.2f}")
    print(f"地面 ENU 2: E={g[1][0]:8.2f}  N={g[1][1]:8.2f}")
    print(f"距离: {d:.2f} m")
    print(f"深度（到相机水平距离）: {np.linalg.norm(g[0][:2]):.1f} m → "
          f"{np.linalg.norm(g[1][:2]):.1f} m")

    if known and known > 0:
        ratio = known / d if d > 0 else float("nan")
        implied_alt = cam.meta.rel_altitude * ratio
        print("-" * 78)
        print(f"已知真值 {known:.2f} m  →  量得 {d:.2f} m，误差 {(d - known) / known * 100:+.1f}%")
        print(f"若认为真值可信，则等效高度应为 {implied_alt:.2f} m"
              f"（元数据给的相对高度 {cam.meta.rel_altitude:.2f} m，"
              f"差 {implied_alt - cam.meta.rel_altitude:+.2f} m）")
        print("注意：距离与'相机到目标地面的真实高度'成正比。RelativeAltitude 是相对【起飞点】，")
        print("      若目标所在地面比起飞点高/低，这里就会整体偏大/偏小，误差 ≈ 高差/高度。")
    print("-" * 78)


def main():
    ap = argparse.ArgumentParser(
        description="阶段 0：验证 DJI 元数据读取与单张照片几何投影链路")
    ap.add_argument("photo", nargs="?", help="待测照片；省略时配合 --demo 使用")
    ap.add_argument("--demo", action="store_true", help="批量打印 demo 数据的参数表")
    ap.add_argument("--demo-dir", default=DEMO_DIR, help=f"demo 数据目录（默认 {DEMO_DIR}）")
    ap.add_argument("--p1", help="第一个像素点，格式 u,v（如 1075,2204）")
    ap.add_argument("--p2", help="第二个像素点，格式 u,v")
    ap.add_argument("--interactive", action="store_true", help="在窗口里点两个点")
    ap.add_argument("--known", type=float, default=None, help="两点间的已知真实长度(m)，用于误差评估")
    ap.add_argument("--altitude", type=float, help="手工指定相对高度(m)")
    ap.add_argument("--pitch", type=float, help="手工指定云台俯仰(°)")
    ap.add_argument("--yaw", type=float, help="手工指定云台偏航(°)")
    ap.add_argument("--roll", type=float, help="手工指定云台横滚(°)")
    ap.add_argument("--hfov", type=float, help="手工指定水平 FOV(°)")
    ap.add_argument("--sensor-width", type=float,
                    help="手工指定传感器物理宽度(mm)，用于精确算内参")
    args = ap.parse_args()

    if args.demo:
        print_demo_table(args.demo_dir)

    if not args.photo:
        if not args.demo:
            ap.error("需要给一张照片路径，或加 --demo 批量看参数")
        return

    try:
        meta = read_dji_meta(args.photo, altitude=args.altitude, pitch=args.pitch,
                             yaw=args.yaw, roll=args.roll, hfov=args.hfov,
                             sensor_width=args.sensor_width)
    except MetadataError as exc:
        print("参数读取失败：\n" + str(exc))
        raise SystemExit(2)

    cam = Camera(meta)
    print()
    print(meta.summary())
    print(f"相机内参 fx: {cam.fx:.2f} px    画面中心 GSD: {cam.gsd_at_center() * 100:.2f} cm/px")
    print()

    if args.interactive:
        img = imread_unicode(args.photo)
        pts = pick_two_points(img)
        if not pts:
            print("已取消。")
            return
        report_distance(cam, pts[0], pts[1], args.known)
        return

    if args.p1 and args.p2:
        def parse_pt(text):
            u, v = text.replace("x", ",").split(",")
            return float(u), float(v)

        report_distance(cam, parse_pt(args.p1), parse_pt(args.p2), args.known)
        return

    print("未指定测量点，只打印了参数。")
    print("要量两点距离：加 --interactive，或加 --p1 u,v --p2 u,v [--known 5]")
    print(f"参考：面积换算示例 {format_area(1000.0)}")


if __name__ == "__main__":
    main()
