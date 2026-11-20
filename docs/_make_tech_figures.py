# -*- coding: utf-8 -*-
"""
生成 README 用的技术原理示意图。

所有几何量都由 uav_area/uav_geo.py 的真实相机模型算出来（不是手绘示意），
所以图上标注的数字与程序跑出来的结果一致。改了几何内核就重跑一遍：

    python docs/_make_tech_figures.py
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "uav_area"))
import uav_geo as g  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#1E3A5F"      # 主文字 / 深蓝
BLUE = "#2563EB"
GREY = "#64748B"
LINE = "#CBD5E1"
BG = "#F8FAFC"
ORANGE = "#D97706"
RED = "#DC2626"
GREEN = "#059669"


def meta(pitch, alt=100.0, yaw=0.0, roll=0.0, w=4000, h=3000):
    return g.DjiMeta(
        path="synthetic", width=w, height=h, latitude=25.0, longitude=102.0,
        rel_altitude=alt, gimbal_yaw=yaw, gimbal_pitch=pitch, gimbal_roll=roll,
        focal_35=None, focal_mm=None, digital_zoom=1.0,
        hfov_deg=73.74, hfov_source="synthetic", model="synthetic", reader="synthetic")


def ground_to_pixel(cam, pts):
    """ENU 地面点 → 像素。相机在 (0,0,h)，与 uav_geo 的约定一致。"""
    P = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    C = np.array([0.0, 0.0, cam.h])
    d_cam = (P - C) @ cam.R              # R^T (P-C)：行向量右乘 R
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    u = fx * d_cam[:, 0] / d_cam[:, 2] + cx
    v = fy * d_cam[:, 1] / d_cam[:, 2] + cy
    return np.stack([u, v], axis=1)


def clean(ax, grid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=GREY, labelsize=9, length=3, width=0.8)
    if grid:
        ax.grid(True, color=LINE, lw=0.7, alpha=0.7)
        ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# 图 1：同一组像素格，落到地面为什么不是同一个尺寸
# --------------------------------------------------------------------------
def fig_gsd_grid(path, pitch_tilt=-55.0):
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2),
                             gridspec_kw={"hspace": 0.34, "wspace": 0.26})
    fig.patch.set_facecolor("white")

    U = np.linspace(0.22, 0.78, 5)
    # 正射从光轴附近到底边；斜拍一直到画面上部，才能看到远近落差
    VP = {(-90.0): 0.52, pitch_tilt: 0.30}

    for row, pitch in enumerate([-90.0, pitch_tilt]):
        V = np.linspace(VP[pitch], 1.00, 4)
        cam = g.Camera(meta(pitch))
        W, H = cam.meta.width, cam.meta.height
        us, vs = U * W, V * H
        uu, vv = np.meshgrid(us, vs)

        # --- 左：像素网格 ---
        ax = axes[row][0]
        ax.set_facecolor(BG)
        ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec=GREY, lw=1.2))
        for u in us:
            ax.plot([u, u], [vs[0], vs[-1]], color=LINE, lw=1.0)
        for v in vs:
            ax.plot([us[0], us[-1]], [v, v], color=LINE, lw=1.0)
        ax.scatter(uu.ravel(), vv.ravel(), s=9, color=BLUE, zorder=3)
        ax.set_xlim(-260, W + 260)
        ax.set_ylim(H + 260, -260)
        ax.set_aspect("equal")
        ax.set_xticks([0, W])
        ax.set_xticklabels(["0", f"{W}"])
        ax.set_yticks([0, H])
        ax.set_yticklabels(["0", f"{H}"])
        ax.set_xlabel("像素 u", fontsize=10, color=GREY)
        ax.set_ylabel("像素 v", fontsize=10, color=GREY)
        tag = "正射 −90°" if pitch == -90 else f"斜拍 {pitch:.0f}°"
        ax.set_title(f"{tag}　照片像素（{W}×{H}）", fontsize=11.5, color=INK, pad=8)
        clean(ax)

        # --- 右：投到地面 ---
        gpts = cam.pixels_to_ground(np.stack([uu.ravel(), vv.ravel()], axis=1))
        G = gpts[:, :2].reshape(uu.shape + (2,))

        ax = axes[row][1]
        col = BLUE if pitch == -90 else ORANGE
        for i in range(len(vs)):                      # 横线（每行像素 → 一条线）
            ax.plot(G[i, :, 0], G[i, :, 1], color=col, lw=1.3, alpha=0.9)
        for j in range(len(us)):                      # 竖线
            ax.plot(G[:, j, 0], G[:, j, 1], color=col, lw=1.3, alpha=0.9)
        ax.scatter(G[:, :, 0].ravel(), G[:, :, 1].ravel(), s=13, color=col, zorder=3)
        ax.margins(0.07)
        ax.set_aspect("equal")
        ax.set_xlabel("E（米）", fontsize=10, color=GREY)
        ax.set_ylabel("N（米）", fontsize=10, color=GREY)

        # 沿视线方向的格距：取中间一列相邻点的距离
        mid = len(us) // 2
        steps = np.array([g.distance_xy(G[i, mid], G[i + 1, mid])
                          for i in range(len(vs) - 1)])
        near, far = steps.min(), steps.max()
        if pitch == -90:
            ax.set_title(f"落到地面：等距方格　{steps.mean():.1f} m / 格",
                         fontsize=11, color=INK, pad=8)
        else:
            ax.set_title(f"落到地面：梯形　远端 {far:.1f} / 近端 {near:.1f} m / 格",
                         fontsize=11, color=RED, pad=8)
        clean(ax)

    fig.suptitle("同一组等间距像素点，落到地面上不再是等间距", fontsize=14,
                 color=INK, y=0.975, weight="bold")
    fig.text(0.5, 0.018,
             "正射时地面格距沿视线方向几乎不变，所以才可以用「像素数 × GSD²」；"
             "斜拍时远端每个像素覆盖的地面长度是近端的 2 倍、面积近 4 倍，一个标量必然顾此失彼。",
             ha="center", fontsize=10.5, color=GREY)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[图] {path}")

    # 顺便把数字打出来，供 README 引用
    for pitch in (-90.0, pitch_tilt):
        cam = g.Camera(meta(pitch))
        V = np.linspace(VP[pitch], 1.00, 4)
        uu, vv = np.meshgrid(U * cam.meta.width, V * cam.meta.height)
        G = cam.pixels_to_ground(np.stack([uu.ravel(), vv.ravel()], 1))[:, :2]
        G = G.reshape(uu.shape + (2,))
        mid = G.shape[1] // 2
        steps = [g.distance_xy(G[i, mid], G[i + 1, mid]) for i in range(G.shape[0] - 1)]
        print(f"    pitch={pitch:>6.1f}  格距 {min(steps):.2f} ~ {max(steps):.2f} m")


# --------------------------------------------------------------------------
# 图 2：倾斜不变性——图像里的形状变了，反投影回地面是同一个矩形
# --------------------------------------------------------------------------
def fig_tilt_invariance(path):
    hw, hd, n0 = 20.0, 10.0, 38.0            # 40 m × 20 m，中心 N=38
    rect = np.array([[-hw, n0 - hd, 0], [hw, n0 - hd, 0],
                     [hw, n0 + hd, 0], [-hw, n0 + hd, 0]], dtype=np.float64)

    fig = plt.figure(figsize=(12.4, 4.3))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.02], wspace=0.22)

    areas = []
    px_areas = []
    gsds = []
    for i, pitch in enumerate([-90.0, -55.0]):
        cam = g.Camera(meta(pitch))
        W, H = cam.meta.width, cam.meta.height
        uv = ground_to_pixel(cam, rect)
        areas.append(cam.area_of_pixels(np.vstack([uv, uv[:1]])))
        px = g.polygon_area_xy(np.vstack([uv, uv[:1]]))
        px_areas.append(px)
        gsds.append(cam.gsd_at_center())

        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(BG)
        ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec=GREY, lw=1.2))
        ax.add_patch(Polygon(uv, closed=True, fc=BLUE, alpha=0.16, ec=BLUE, lw=2))
        for k, (u, v) in enumerate(uv):
            ax.plot(u, v, "o", ms=5, color=BLUE, zorder=4)
        ax.set_xlim(-300, W + 300)
        ax.set_ylim(H + 300, -300)
        ax.set_aspect("equal")
        ax.set_xticks([0, W])
        ax.set_xticklabels(["0", f"{W}"])
        ax.set_yticks([0, H])
        ax.set_yticklabels(["0", f"{H}"])
        ax.set_xlabel("像素 u", fontsize=10, color=GREY)
        if i == 0:
            ax.set_ylabel("像素 v", fontsize=10, color=GREY)
        name = "正射 −90°" if pitch == -90 else "斜拍 −55°"
        ax.set_title(f"照片上看到的：{name}", fontsize=11, color=INK, pad=8)
        clean(ax)

    # 第三张：地面
    ax = fig.add_subplot(gs[0, 2])
    ax.add_patch(Polygon(rect[:, :2], closed=True, fc=GREEN, alpha=0.14,
                         ec=GREEN, lw=2.2))
    ax.plot([-hw, hw], [n0 + hd, n0 + hd], color=GREEN, lw=2.6)
    ax.annotate(f"{2 * hw:.1f} m", (0, n0 + hd), xytext=(0, n0 + hd + 4.0),
                ha="center", fontsize=10.5, color=INK)
    ax.annotate(f"{2 * hd:.1f} m", (hw, n0), xytext=(hw + 3.5, n0), va="center",
                fontsize=10.5, color=INK)
    ax.set_xlim(-hw - 14, hw + 24)
    ax.set_ylim(n0 - hd - 40, n0 + hd + 12)
    ax.set_aspect("equal")
    ax.set_xlabel("E（米）", fontsize=10, color=GREY)
    ax.set_ylabel("N（米）", fontsize=10, color=GREY)
    ax.set_title("反投影回地面：都落在同一个矩形上", fontsize=11, color=INK, pad=8)
    ax.text(0.02, 0.03,
            f"面积 {areas[0]:.1f} ㎡ ＝ {areas[1]:.1f} ㎡\n"
            f"两片投影差 {abs(areas[0] - areas[1]) / areas[0] * 100:.2f}%（浮点噪声级）",
            transform=ax.transAxes, fontsize=10.5, color=INK, va="bottom",
            bbox=dict(fc="white", ec=LINE, boxstyle="round,pad=0.45"))
    clean(ax, grid=True)

    fig.suptitle("斜拍为什么还能算准：像素形状会变，地面位置不变", fontsize=14,
                 color=INK, y=1.02, weight="bold")
    fig.text(0.5, -0.12,
             f"同一块 {2 * hw:.0f}×{2 * hd:.0f} m 的地：正射是矩形，斜拍被压成梯形——形状完全不一样。\n"
             f"把每个顶点反投影回地面，两片都落回同一个矩形：{areas[0]:.2f} ㎡ 与 {areas[1]:.2f} ㎡，"
             f"差 {abs(areas[0] - areas[1]) / areas[0] * 100:.2f}%。\n"
             "前提只有一条：这些点确实在同一个水平面上。",
             ha="center", fontsize=10.5, color=GREY, linespacing=1.8)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[图] {path}  地面面积 {areas[0]:.2f} / {areas[1]:.2f} ㎡")
    for p, a, ap, gsd in zip([-90.0, -55.0], areas, px_areas, gsds):
        est = ap * gsd ** 2
        print(f"    pitch={p:>6.1f}  像素数×GSD² = {est:.1f} ㎡"
              f"（真值 {a:.1f}，偏 {est / a - 1:+.1%}）")


if __name__ == "__main__":
    fig_gsd_grid(os.path.join(HERE, "tech-gsd.png"))
    fig_tilt_invariance(os.path.join(HERE, "tech-tilt.png"))
    print("完成")
