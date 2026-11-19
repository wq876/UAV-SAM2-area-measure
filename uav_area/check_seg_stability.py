# -*- coding: utf-8 -*-
"""
check_seg_stability.py —— SAM2 分割稳定性自检

回答一个现场很实际的问题：同一个地块，我点偏几个像素，量出来的面积会差多少？

做法：在目标内部随机撒 N 个种子点（都在第一次分割出的掩膜里，保证点的是同一块地），
每个点独立跑一次 SAM2 → 清理 → 算面积，看这 N 个面积的离散程度和掩膜之间的 IoU。

用法
----
    python check_seg_stability.py "照片.JPG" --click 1810,1810
    python check_seg_stability.py "照片.JPG" --click 1810,1810 --n 8 --radius 60
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_geo import Camera, MetadataError, format_area, imread_unicode, m2_to_mu, read_dji_meta  # noqa: E402
from measure_area import (  # noqa: E402
    DEFAULT_CONFIG, DEFAULT_WEIGHT, Segmenter, build_field, contour_to_mask, pick_mask,
)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser(
        description="SAM2 分割稳定性自检：同一地块多个种子点，看面积离散度与掩膜 IoU")
    ap.add_argument("photo", help="DJI 原始照片")
    ap.add_argument("--click", required=True, metavar="u,v", help="目标内部的一个像素点")
    ap.add_argument("--n", type=int, default=6, help="随机种子点个数（默认 6）")
    ap.add_argument("--radius", type=float, default=40.0,
                    help="种子点抖动半径（像素，默认 40）")
    ap.add_argument("--samples", type=int, default=300, help="轮廓采样点数")
    ap.add_argument("--weight", default=DEFAULT_WEIGHT, help="SAM2 权重路径")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="SAM2 模型配置")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    args = ap.parse_args()

    try:
        meta = read_dji_meta(args.photo)
    except MetadataError as exc:
        print("参数读取失败：\n" + str(exc))
        raise SystemExit(2)
    cam = Camera(meta)

    u, v = (float(x) for x in args.click.replace("x", ",").split(","))
    img = imread_unicode(args.photo)

    print(f"照片 {os.path.basename(args.photo)}  h={cam.h:.2f} m  fx={cam.fx:.1f}")
    print("加载 SAM2……")
    seg = Segmenter(args.weight, args.config, args.device)
    seg.set_image(img)
    print("图像编码完成，开始撒点……")

    # 第一次分割，拿到目标内部区域用来撒点
    masks, scores = seg.predict([(u, v)], [1])
    base_idx = pick_mask(masks, scores)
    base_field = build_field(cam, masks[base_idx], 1, args.samples,
                             score=float(scores[base_idx]))
    base_mask = contour_to_mask(base_field.contour_px, masks.shape[-2:])

    # 只在地块"内部"撒点：用距离变换保证距边界 > radius，
    # 否则种子点会落到地块外面，测出来的就不是"点偏一点"而是"点错了"
    import cv2
    dt = cv2.distanceTransform(base_mask.astype(np.uint8), cv2.DIST_L2, 5)
    interior = np.argwhere(dt > args.radius + 3.0)
    if len(interior) < args.n:
        interior = np.argwhere(dt > 3.0)
    if interior.size == 0:
        print("首个点分割出的地块太窄，放不下抖动半径，请调小 --radius。")
        raise SystemExit(1)
    print(f"可撒点区域 {len(interior)} 个像素（距边界 > {args.radius:.0f} px）")

    rng = np.random.default_rng(0)
    areas, ious, radii = [], [], []
    for k in range(args.n):
        # 在掩膜内部随机取一点，再限制在该点周围 radius 内（保证仍是同一块地）
        py, px = interior[rng.integers(0, len(interior))]
        du = float(np.clip(rng.normal(0, args.radius / 2), -args.radius, args.radius))
        dv = float(np.clip(rng.normal(0, args.radius / 2), -args.radius, args.radius))
        su, sv = float(px + du), float(py + dv)

        m, s = seg.predict([(su, sv)], [1])
        idx = pick_mask(m, s)
        f = build_field(cam, m[idx], k + 1, args.samples, score=float(s[idx]))
        areas.append(f.area_m2)
        ious.append(iou(base_mask, contour_to_mask(f.contour_px, masks.shape[-2:])))
        radii.append(float(np.hypot(su - u, sv - v)))
        print(f"  种子 {k + 1}: ({su:7.1f},{sv:7.1f})  面积 {f.area_m2:9.1f} m²  "
              f"score={s[idx]:.3f}  IoU(vs首个)={ious[-1]:.3f}")

    areas = np.array(areas)
    ious = np.array(ious)
    mean = float(areas.mean())
    spread = float((areas.max() - areas.min()) / mean * 100) if mean else float("nan")
    cv = float(areas.std() / mean * 100) if mean else float("nan")
    # 单个离群点（点到了相邻地物）会把标准差和极差拉爆，判读用四分位距更稳
    p25, p75 = np.percentile(areas, [25, 75])
    iqr_pct = float((p75 - p25) / np.median(areas) * 100) if np.median(areas) else float("nan")
    outliers = int((ious < 0.90).sum())

    print("\n" + "=" * 74)
    print("分 割 稳 定 性")
    print("=" * 74)
    print(f"基准面积（首个点）: {base_field.area_m2:.1f} m² / {base_field.area_mu:.3f} 亩")
    print(f"{args.n} 个随机种子点  : 均值 {mean:.1f} m²，标准差 {areas.std():.1f} m²")
    print(f"离散程度          : 变异系数 {cv:.2f}%，四分位距 {iqr_pct:.2f}%"
          f"（后者剔除个别跑偏点，更能代表常态）")
    print(f"极差              : {areas.max() - areas.min():.1f} m²"
          f"（占均值 {spread:.2f}%）")
    print(f"掩膜 IoU          : 中位数 {np.median(ious):.3f}，最低 {ious.min():.3f}"
          f"，IoU<0.90 的离群点 {outliers}/{args.n} 个")
    print(f"种子点偏移距离    : {min(radii):.0f} ~ {max(radii):.0f} px")
    print("-" * 74)

    med_iou = float(np.median(ious))
    if med_iou >= 0.96 and iqr_pct < 1.5:
        verdict = "很稳：点在地块内部任何位置，面积变化 <2%。一次点击的结果可以放心用。"
    elif med_iou >= 0.92 and iqr_pct < 4.0:
        verdict = (f"较稳：常态离散 {iqr_pct:.1f}%（IoU 中位数 {med_iou:.2f}），"
                   f"与 3~5% 的目标相称。建议点在地块正中、避开与相邻地物贴近的那一侧。")
    elif med_iou >= 0.85:
        verdict = ("边界不稳：换个位置点面积会差好几个百分点。建议放大画面、"
                   "点在地块正中、并用右键加背景点把相邻地物剔掉。")
    else:
        verdict = ("很不稳：这个目标不适合单点分割（边界太弱或与周围太像）。"
                   "建议改多点点选，或换更垂直的俯拍/更低高度重拍。")
    print("判读：" + verdict)
    print("=" * 74)


if __name__ == "__main__":
    main()
