# -*- coding: utf-8 -*-
"""
run_measure.py —— 傻瓜启动入口（配合 启动.bat 双击用）

不带参数：列出几张示例照片让你选。
带参数：直接把照片路径（拖拽到 bat 上）传进来。
额外参数会原样透传给 measure_area.py，例如：
    run_measure.py 照片.JPG --calib 1075,2204,1075,2330,4.65
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(_ROOT, "参考", "UAVGeoMeasure", "demo_data")
SAMPLES = [
    (os.path.join(DEMO_DIR, "无人机正射数据", "DJI_0018.JPG"),
     "坝区 · 垂直俯拍 -90° · 111 m 高"),
    (os.path.join(DEMO_DIR, "练习数据（航拍30张）", "工厂航拍30张", "100_0010_0042.JPG"),
     "温室厂房 · 斜拍 -60° · 100 m 高"),
    (os.path.join(DEMO_DIR, "car_test.jpg"),
     "小区停车 · 斜拍 -45° · 101 m 高"),
    (os.path.join(DEMO_DIR, "solar_panel_test.jpg"),
     "太阳能板 · 斜拍 -41.8° · 85 m 高"),
]
DEFAULT_OUT = os.path.join(_ROOT, "demo_out", "现场测量")


def choose_target() -> tuple[str, str] | None:
    """返回 (kind, path)，kind 是 'photo' 或 'folder'。"""
    print("=" * 78)
    print("无人机照片 · 面积量算")
    print("=" * 78)
    for i, (path, note) in enumerate(SAMPLES, 1):
        tag = "✓" if os.path.isfile(path) else "×"
        print(f"  {i}) [{tag}] {os.path.basename(path):26s} {note}")
    n = len(SAMPLES)
    print(f"  {n + 1}) 输入照片路径（单张）")
    print(f"  {n + 2}) 输入文件夹路径（逐张处理，按 D 下一张 / A 上一张）")
    print()
    print("也可以把照片或文件夹直接拖到「启动测量.bat」图标上。")
    print("-" * 78)

    raw = input("请输入序号后回车（直接回车=1）：").strip() or "1"
    if not raw.isdigit():
        print("输入无效，用第 1 个示例。")
        raw = "1"

    idx = int(raw)
    if 1 <= idx <= n:
        path = SAMPLES[idx - 1][0]
        if not os.path.isfile(path):
            print(f"找不到文件：{path}")
            return None
        return "photo", path

    if idx == n + 1:
        path = input("照片完整路径：").strip().strip('"')
        if not os.path.isfile(path):
            print(f"找不到文件：{path}")
            return None
        return "photo", path

    path = input("文件夹完整路径：").strip().strip('"')
    if not os.path.isdir(path):
        print(f"找不到文件夹：{path}")
        return None
    return "folder", path


def main():
    argv = sys.argv[1:]

    # 直接把 --help 交给主程序，别弹选择菜单
    if "-h" in argv or "--help" in argv:
        import measure_area

        sys.argv = ["measure_area.py"] + argv
        measure_area.main()
        return

    # 把"存在的路径"认作目标，其余原样透传（--click 后面是带值的一对，
    # 不能只挑以 - 开头的，否则值会丢）
    target = None
    extra = []
    for a in argv:
        cand = a.strip('"')
        if target is None and not a.startswith("-") and os.path.exists(cand):
            target = cand
        else:
            extra.append(a)

    if target is not None:
        kind = "folder" if os.path.isdir(target) else "photo"
    else:
        leftovers = [a for a in extra if not a.startswith("-")]
        if leftovers:
            print(f"找不到路径：{leftovers[0]}")
            return
        picked = choose_target()
        if picked is None:
            return
        kind, target = picked

    import measure_area

    if kind == "folder":
        new_argv = ["measure_area.py", "--folder", target]
    else:
        new_argv = ["measure_area.py", target]
    if DEFAULT_OUT:
        new_argv += ["--out", DEFAULT_OUT]
    new_argv += extra

    print("\n执行参数：" + " ".join(new_argv[1:]))
    print("（左键点目标 / 右键排除 / 回车确认 / A 上一张 / D 下一张 / Q 结束）\n")
    sys.argv = new_argv
    measure_area.main()

    print(f"\n结果已存到：{DEFAULT_OUT}")
    print("  result_*.jpg  标注图   fields_*.csv  面积表   summary_全部照片.csv  汇总"
          "   masks/  掩膜")
    try:
        input("\n按回车关闭……")
    except EOFError:
        pass


if __name__ == "__main__":
    main()
