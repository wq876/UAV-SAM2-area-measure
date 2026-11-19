# -*- coding: utf-8 -*-
"""
check_accuracy.py —— 精度自检：同一地物在两张重叠照片里各投影一次，比对落点是否重合

原理
----
每张照片都能独立把像素投到地面经纬度（不需要 GCP、不需要正射成果）。
同一块石头，在 A、B 两张重叠照片里投出来的经纬度应当一致。

关键推导：设真实尺度与所用尺度差一个比例因子 k（来自高度、FOV 或姿态任一项），
某地面点相对 A 像底点的位置是 p，相对 B 像底点是 p-b（b 为相机基线），
则两张图各自算出的落点相差 (k-1)·b。
    → 落点差异 ≈ 尺度误差 × 相机基线
所以基线越大越灵敏：70 m 基线下，1 m 的落点差异 ≈ 1.4% 的尺度误差。

用法
----
    python check_accuracy.py --folder "照片目录"
    python check_accuracy.py --folder "照片目录" --pair A.JPG B.JPG
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_geo import (  # noqa: E402
    Camera, MetadataError, imread_unicode, read_dji_meta,
)

WORK_WIDTH = 1500        # 配准用降采样宽度
MIN_INLIERS = 15         # 少于该内点数认为配准不可信
MAX_TRIES = 20           # 最多尝试几对候选照片


# =========================================================
# 1. 目录与元数据
# =========================================================
def scan_folder(folder: str) -> list:
    files = []
    for pat in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        files += glob.glob(os.path.join(folder, pat))
    seen, uniq = set(), []
    for f in sorted(files):
        key = os.path.basename(f).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def load_metas(files: list) -> list:
    out = []
    for f in files:
        try:
            out.append(read_dji_meta(f))
        except MetadataError:
            continue
    return out


def local_offset(m1, m2):
    """相机水平基线(m) 与高度差(m)。"""
    lat0 = np.radians(m1.latitude)
    dx = np.radians(m2.longitude - m1.longitude) * 6378137.0 * np.cos(lat0)
    dy = np.radians(m2.latitude - m1.latitude) * 6378137.0
    return float(np.hypot(dx, dy)), float(m2.rel_altitude - m1.rel_altitude)


def candidate_pairs(metas, min_base=30.0, max_base=60.0, max_dyaw=25.0,
                    look_ahead=40):
    """按"基线够大、朝向差小、高度接近"排序的候选照片对。"""
    scored = []
    for i in range(len(metas)):
        for j in range(i + 1, min(i + 1 + look_ahead, len(metas))):
            base, dz = local_offset(metas[i], metas[j])
            if base < min_base or base > max_base or abs(dz) > 3.0:
                continue
            dyaw = abs((metas[i].gimbal_yaw - metas[j].gimbal_yaw + 180) % 360 - 180)
            if dyaw > max_dyaw:
                continue
            scored.append((dyaw * 2.0 + abs(base - 45.0) * 0.3 + abs(dz) * 3.0, i, j, base))
    scored.sort(key=lambda t: t[0])
    return [(i, j, base) for _, i, j, base in scored]


# =========================================================
# 2. SIFT 配准
# =========================================================
def sift_match(gray_a, gray_b):
    """返回 (pts_a, pts_b) 原图坐标（已按比例还原）、单应性、内点数。"""
    import cv2

    sift = cv2.SIFT_create(nfeatures=6000)
    ka, da = sift.detectAndCompute(gray_a, None)
    kb, db = sift.detectAndCompute(gray_b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None, None, None, 0

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None, None, None, len(good)

    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return None, None, None, 0

    keep = mask.ravel().astype(bool)
    pts_a = src[keep].reshape(-1, 2)
    pts_b = dst[keep].reshape(-1, 2)
    return pts_a, pts_b, H, int(keep.sum())


def homography_plausible(H, m1, m2, aniso_tol=0.06, scale_tol=0.15,
                         rot_tol_deg=12.0):
    """
    校验单应矩阵是否是"同一地平面的两视影像"该有的样子。

    两张等高俯拍照片之间，地平面上的点满足相似变换（各向同性缩放 + 旋转 + 平移）。
    如果 SIFT 在重复纹理（护栏立柱、大棚骨架上）上凑出一个各向异性或尺度离谱的
    单应矩阵，那一定不是真匹配，必须拒掉，否则"精度"测的是错位。

    返回 (是否可信, 诊断信息)。
    """
    A = np.asarray(H, dtype=np.float64)[:2, :2]
    if not np.all(np.isfinite(A)):
        return False, {"reason": "单应矩阵含非有限值"}

    try:
        u, s, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return False, {"reason": "SVD 失败"}

    s1, s2 = float(s[0]), float(s[1])
    if s2 < 1e-9:
        return False, {"reason": "缩放退化"}
    aniso = abs(s1 / s2 - 1.0)

    R = u @ vt
    if np.linalg.det(R) < 0:
        return False, {"reason": "出现镜像变换"}
    theta = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    dyaw = float((m2.gimbal_yaw - m1.gimbal_yaw + 180) % 360 - 180)
    exp_scale = float(m1.rel_altitude / m2.rel_altitude)
    scale_err = abs(s1 - exp_scale) / exp_scale
    rot_err = abs(abs(theta) - abs(dyaw))

    info = {"aniso": aniso, "scale": s1, "exp_scale": exp_scale,
            "rot": theta, "dyaw": dyaw}

    if aniso > aniso_tol:
        info["reason"] = f"各向异性 {aniso * 100:.1f}%，不像同一平面上的两视影像"
        return False, info
    if scale_err > scale_tol:
        info["reason"] = (f"像素尺度 {s1:.3f} 与高度比 {exp_scale:.3f} 不符"
                          f"（偏差 {scale_err * 100:.0f}%）")
        return False, info
    if rot_err > rot_tol_deg:
        info["reason"] = f"图像旋转 {theta:.1f}° 与偏航差 {dyaw:.1f}° 不符"
        return False, info
    return True, info


def spread_ok(pts, shape, min_frac=0.12):
    """内点是否覆盖足够大的画面范围（避免全挤在一个角落）。"""
    if len(pts) < 4:
        return False
    h, w = shape
    span_x = (pts[:, 0].max() - pts[:, 0].min()) / w
    span_y = (pts[:, 1].max() - pts[:, 1].min()) / h
    return span_x > min_frac and span_y > min_frac


# =========================================================
# 3. 主流程
# =========================================================
def evaluate_pair(m1, m2, base):
    """对一对照片做配准 + 双片投影比对，返回结果字典或 None。"""
    import cv2

    img_a = imread_unicode(m1.path)
    img_b = imread_unicode(m2.path)
    full_w = img_a.shape[1]
    s = WORK_WIDTH / full_w

    ga = cv2.resize(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s,
                    interpolation=cv2.INTER_AREA)
    gb = cv2.resize(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s,
                    interpolation=cv2.INTER_AREA)

    pts_a, pts_b, H, n_in = sift_match(ga, gb)
    if pts_a is None or n_in < MIN_INLIERS:
        return {"n_inliers": n_in, "ok": False}
    if not spread_ok(pts_a, ga.shape):
        return {"n_inliers": n_in, "ok": False, "reason": "内点分布过于集中"}

    plausible, diag = homography_plausible(H, m1, m2)
    if not plausible:
        return {"n_inliers": n_in, "ok": False,
                "reason": "伪匹配：" + diag.get("reason", "几何不合理"), "diag": diag}

    cam_a, cam_b = Camera(m1), Camera(m2)
    pa = pts_a / s
    pb = pts_b / s

    # 逐点独立投影，取经纬度差
    ll_a = np.array([cam_a.center_latlon([p]) for p in pa])
    ll_b = np.array([cam_b.center_latlon([p]) for p in pb])

    lat0 = np.radians(m1.latitude)
    de = (ll_a[:, 1] - ll_b[:, 1]) * 111320.0 * np.cos(lat0)
    dn = (ll_a[:, 0] - ll_b[:, 0]) * 111320.0
    dist = np.hypot(de, dn)

    return {
        "ok": True,
        "n_inliers": n_in,
        "n_eval": len(dist),
        "median_m": float(np.median(dist)),
        "p90_m": float(np.percentile(dist, 90)),
        "max_m": float(dist.max()),
        "base": base,
        "scale_err_pct": float(np.median(dist) / base * 100.0),
        "gsd_cm": (cam_a.gsd_at_center() + cam_b.gsd_at_center()) / 2 * 100,
    }


def report(m1, m2, res, base, dz):
    print("\n" + "=" * 78)
    print("双 片 一 致 性 检 验 结 果")
    print("=" * 78)
    print(f"A = {os.path.basename(m1.path)}   h={m1.rel_altitude:.2f} m  "
          f"yaw={m1.gimbal_yaw:.1f}°  pitch={m1.gimbal_pitch:.1f}°")
    print(f"B = {os.path.basename(m2.path)}   h={m2.rel_altitude:.2f} m  "
          f"yaw={m2.gimbal_yaw:.1f}°  pitch={m2.gimbal_pitch:.1f}°")
    print(f"相机基线 {base:.1f} m，高度差 {dz:+.2f} m")
    print(f"SIFT+RANSAC 内点 {res['n_inliers']} 个，参与比对 {res['n_eval']} 个")
    print("-" * 78)
    print(f"同名点落点差异：中位数 {res['median_m']:.2f} m，"
          f"P90 {res['p90_m']:.2f} m，最大 {res['max_m']:.2f} m")
    print(f"平均 GSD {res['gsd_cm']:.2f} cm/px")
    print(f"等效尺度误差 ≈ {res['median_m'] / base * 100:.2f}%  "
          f"（= 落点差异 / 基线；面积误差约为其 2 倍）")
    print(f"像素级噪声下限 ≈ {res['gsd_cm'] / 100 * 2:.2f} m（2 px 匹配误差，可忽略）")

    med = res["median_m"]
    print("-" * 78)
    if med < 1.0:
        verdict = "优：高度/姿态/FOV 三者自洽，面积误差主要只剩地面起伏与边界勾画。"
    elif med < 3.0:
        verdict = "可用：同一地块换视角可能差几个百分点，现场估价够用。"
    elif med < 5.0:
        verdict = "偏差偏大：建议核对高度基准与云台角，或降低飞行高度后复测。"
    else:
        verdict = "不可信：先别用面积，检查参数来源（是否原始照片、是否数码变焦、姿态角是否被改写）。"
    print("判读：" + verdict)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(
        description="精度自检：同一地物在两张重叠照片里的投影是否重合")
    ap.add_argument("--folder", required=True, help="照片目录")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), help="指定两张照片文件名")
    ap.add_argument("--min-base", type=float, default=30.0, help="候选基线下限(m)")
    ap.add_argument("--max-base", type=float, default=60.0, help="候选基线上限(m)")
    args = ap.parse_args()

    files = scan_folder(args.folder)
    if not files:
        print(f"目录里没有照片：{args.folder}")
        raise SystemExit(2)
    metas = load_metas(files)
    print(f"目录 {args.folder}\n可读取参数的照片 {len(metas)}/{len(files)} 张")
    if len(metas) < 2:
        print("可用照片不足 2 张。")
        raise SystemExit(2)

    if args.pair:
        by_name = {os.path.basename(m.path).lower(): m for m in metas}
        try:
            m1, m2 = by_name[args.pair[0].lower()], by_name[args.pair[1].lower()]
        except KeyError as exc:
            print(f"找不到照片：{exc}")
            raise SystemExit(2)
        base, dz = local_offset(m1, m2)
        res = evaluate_pair(m1, m2, base)
        if not res.get("ok"):
            print(f"配准失败（内点 {res.get('n_inliers')}），换一对试试。")
            raise SystemExit(1)
        report(m1, m2, res, base, dz)
        return

    pairs = candidate_pairs(metas, min_base=args.min_base, max_base=args.max_base)
    print(f"候选照片对 {len(pairs)} 组，最多尝试 {MAX_TRIES} 组……")
    best = None
    for i, j, base in pairs[:MAX_TRIES]:
        m1, m2 = metas[i], metas[j]
        res = evaluate_pair(m1, m2, base)
        tag = f"{os.path.basename(m1.path)} ↔ {os.path.basename(m2.path)}  基线 {base:.1f} m"
        if not res.get("ok"):
            print(f"  ✗ {tag}  内点 {res.get('n_inliers')}（{res.get('reason', '配准不足')}）")
            continue
        print(f"  ✓ {tag}  内点 {res['n_inliers']}  落点差异中位数 {res['median_m']:.2f} m")
        if best is None or res["n_inliers"] > best[2]["n_inliers"]:
            best = (m1, m2, res, base, local_offset(m1, m2)[1])

    if best is None:
        print("所有候选都配准失败：这组照片可能重叠不足（航线太稀）或纹理太单一。")
        raise SystemExit(1)

    m1, m2, res, base, dz = best
    report(m1, m2, res, base, dz)


if __name__ == "__main__":
    main()
