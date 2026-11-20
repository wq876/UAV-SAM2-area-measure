# -*- coding: utf-8 -*-
"""
project_store.py —— 工程数据层（不含任何 Qt / 界面代码）

职责
----
1. 一个照片文件夹 = 一个工程，旁边生成 project.json；
2. 存每张照片的相机参数（DjiMeta 快照）与已标地块（名称 / 面积 / 轮廓像素点）；
3. 只存"轮廓像素点 + 参数"，面积随参数重算 —— 改高度基准后能一键全量重算；
4. 导出 CSV / GeoJSON。

为什么不用数据库：单机、一个文件夹一个工程、几百张照片，JSON 足够，
零依赖、可人工查看、拷走即用。真要到多工程/多人再换 SQLite/PostGIS。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from uav_geo import m2_to_mu

VERSION = 2
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class FieldRec:
    """一个已保存的地块。面积是"当前参数下"的结果，轮廓才是原始数据。"""
    fid: int
    name: str
    contour_px: list                  # [[u, v], ...] 原图像素
    area_m2: float
    area_mu: float
    score: float = float("nan")
    center: list | None = None        # [lat, lon]
    warnings: list = field(default_factory=list)
    h_used: float = float("nan")
    created: str = ""
    err_rel: float = float("nan")     # 面积相对误差（1σ），重新算面积时同步刷新
    err_parts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fid": self.fid,
            "name": self.name,
            "contour_px": [[round(float(u), 2), round(float(v), 2)]
                           for u, v in self.contour_px],
            "area_m2": round(float(self.area_m2), 2),
            "area_mu": round(float(self.area_mu), 4),
            "score": None if not np.isfinite(self.score) else round(float(self.score), 3),
            "center": self.center,
            "warnings": list(self.warnings),
            "h_used": round(float(self.h_used), 2),
            "created": self.created or _now(),
            "err_rel": (round(float(self.err_rel), 4)
                        if np.isfinite(self.err_rel) else None),
            "err_parts": {k: round(float(v), 4)
                          for k, v in (self.err_parts or {}).items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FieldRec":
        return cls(
            fid=int(d["fid"]),
            name=str(d.get("name", f"#{d['fid']}")),
            contour_px=[[float(a), float(b)] for a, b in d.get("contour_px", [])],
            area_m2=float(d.get("area_m2", 0.0)),
            area_mu=float(d.get("area_mu", 0.0)),
            score=float(d["score"]) if d.get("score") is not None else float("nan"),
            center=d.get("center"),
            warnings=list(d.get("warnings") or []),
            h_used=float(d.get("h_used", float("nan"))),
            created=str(d.get("created", "")),
            err_rel=(float(d["err_rel"]) if d.get("err_rel") is not None
                     else float("nan")),
            err_parts=dict(d.get("err_parts") or {}),
        )


class ProjectStore:
    """一个文件夹对应一个工程。"""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"文件夹不存在：{self.root}")
        self.path = os.path.join(self.root, "project.json")
        self.doc = {
            "version": VERSION,
            "root": self.root,
            "created": _now(),
            "defaults": {"target_z": 0.0, "use_lrf": False},
            "images": {},
        }
        self.images = self._scan()
        self._load()

    # ---------- 基础 ----------
    def _scan(self) -> list:
        names = []
        for n in sorted(os.listdir(self.root)):
            if os.path.splitext(n)[1].lower() in IMAGE_EXTS:
                names.append(n)
        return names

    def _load(self):
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                doc = json.load(f)
            if int(doc.get("version", 0)) > VERSION:
                raise ValueError("工程文件版本比程序新")
            self.doc.update({k: v for k, v in doc.items() if k != "images"})
            self.doc["images"] = doc.get("images", {}) or {}
        except Exception as exc:
            # 工程文件坏了不要静默覆盖，先备份再用空工程
            bad = self.path + f".bad-{datetime.now():%Y%m%d-%H%M%S}"
            shutil.copy2(self.path, bad)
            print(f"[工程文件损坏，已备份到 {bad}] {exc}")
            self.doc["images"] = {}

    def save(self):
        """原子写：先写临时文件再替换，避免中途崩溃把工程写坏。"""
        os.makedirs(self.root, exist_ok=True)
        self.doc["version"] = VERSION
        self.doc["saved"] = _now()
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.doc, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ---------- 参数与默认值 ----------
    @property
    def defaults(self) -> dict:
        return self.doc.setdefault("defaults", {"target_z": 0.0, "use_lrf": False})

    def set_defaults(self, target_z: float, use_lrf: bool):
        self.defaults["target_z"] = float(target_z)
        self.defaults["use_lrf"] = bool(use_lrf)

    def meta(self, name: str) -> dict | None:
        return self.doc["images"].get(name, {}).get("meta")

    def set_meta(self, name: str, meta_dict: dict):
        self.doc["images"].setdefault(name, {})["meta"] = meta_dict

    # ---------- 单张标定（已知长度反推航高） ----------
    def calib(self, name: str) -> dict | None:
        """这张照片的标定记录：{h, ref_len_m, p1, p2, note}；没标定过返回 None。"""
        return self.doc["images"].get(name, {}).get("calib")

    def set_calib(self, name: str, h: float, ref_len_m: float,
                  p1, p2, note: str = ""):
        self.doc["images"].setdefault(name, {})["calib"] = {
            "h": round(float(h), 3),
            "ref_len_m": round(float(ref_len_m), 4),
            "p1": [round(float(x), 1) for x in p1],
            "p2": [round(float(x), 1) for x in p2],
            "note": note,
            "at": _now(),
        }

    def clear_calib(self, name: str):
        self.doc["images"].get(name, {}).pop("calib", None)

    def calib_h(self, name: str) -> float | None:
        c = self.calib(name)
        return float(c["h"]) if c and c.get("h") else None

    # ---------- 地块 ----------
    def fields(self, name: str) -> list:
        recs = self.doc["images"].get(name, {}).get("fields", []) or []
        return [FieldRec.from_dict(d) for d in recs]

    def _raw_fields(self, name: str) -> list:
        return self.doc["images"].setdefault(name, {}).setdefault("fields", [])

    def next_fid(self, name: str) -> int:
        existing = [int(d["fid"]) for d in self._raw_fields(name)]
        return (max(existing) + 1) if existing else 1

    def add_field(self, name: str, rec: FieldRec) -> FieldRec:
        self._raw_fields(name).append(rec.to_dict())
        return rec

    def delete_field(self, name: str, fid: int):
        raw = self._raw_fields(name)
        self.doc["images"][name]["fields"] = [d for d in raw if int(d["fid"]) != fid]

    def rename_field(self, name: str, fid: int, new_name: str):
        for d in self._raw_fields(name):
            if int(d["fid"]) == fid:
                d["name"] = new_name
                return

    def replace_fields(self, name: str, recs: list):
        self.doc["images"].setdefault(name, {})["fields"] = [r.to_dict() for r in recs]

    # ---------- 统计 ----------
    def labeled_images(self) -> int:
        return sum(1 for n in self.images if self._raw_fields(n))

    def total_fields(self) -> int:
        return sum(len(self._raw_fields(n)) for n in self.images)

    def total_area(self, name: str | None = None) -> tuple:
        if name is not None:
            recs = self.fields(name)
            m2 = sum(r.area_m2 for r in recs)
            return m2, m2_to_mu(m2)
        m2 = sum(r.area_m2 for n in self.images for r in self.fields(n))
        return m2, m2_to_mu(m2)

    # ---------- 重算（改高度基准后用） ----------
    def recompute(self, cam_builder):
        """
        cam_builder(name, meta_dict) -> Camera
        用存下来的轮廓点重算每块地的面积、误差与中心经纬度。
        返回 (成功张数, 失败张数)。
        """
        ok = bad = 0
        for name in self.images:
            recs = self.fields(name)
            meta = self.meta(name)
            if not recs or not meta:
                continue
            try:
                cam = cam_builder(name, meta)
            except Exception:
                bad += 1
                continue
            out = []
            for r in recs:
                pts = np.asarray(r.contour_px, dtype=np.float64).reshape(-1, 2)
                r.area_m2 = cam.area_of_pixels(pts)
                r.area_mu = m2_to_mu(r.area_m2)
                r.h_used = cam.h
                unc = cam.area_uncertainty(pts, base_area=r.area_m2)
                r.err_rel = float(unc["rel"])
                r.err_parts = dict(unc.get("parts") or {})
                try:
                    lat, lon = cam.center_latlon(pts)
                    r.center = [round(lat, 8), round(lon, 8)]
                except Exception:
                    pass
                out.append(r)
            self.replace_fields(name, out)
            ok += 1
        return ok, bad

    # ---------- 导出 ----------
    def export_csv(self, path: str):
        """两张表合一：先明细，后按照片汇总。"""
        import csv

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["照片", "地块编号", "地块名称", "面积_m2", "面积_亩", "误差_%",
                        "中心纬度", "中心经度", "高度基准_m", "置信度", "提示"])
            for name in self.images:
                for r in self.fields(name):
                    lat = f"{r.center[0]:.8f}" if r.center else ""
                    lon = f"{r.center[1]:.8f}" if r.center else ""
                    w.writerow([name, r.fid, r.name, f"{r.area_m2:.2f}",
                                f"{r.area_mu:.4f}",
                                f"{r.err_rel * 100:.1f}" if np.isfinite(r.err_rel) else "",
                                lat, lon,
                                f"{r.h_used:.2f}" if np.isfinite(r.h_used) else "",
                                f"{r.score:.3f}" if np.isfinite(r.score) else "",
                                "；".join(r.warnings)])
            w.writerow([])
            w.writerow(["照片", "地块数", "合计_m2", "合计_亩"])
            for name in self.images:
                recs = self.fields(name)
                if not recs:
                    continue
                m2 = sum(x.area_m2 for x in recs)
                w.writerow([name, len(recs), f"{m2:.2f}", f"{m2_to_mu(m2):.4f}"])
            m2, mu = self.total_area()
            w.writerow(["全部", self.total_fields(), f"{m2:.2f}", f"{mu:.4f}"])
        return path

    def export_geojson(self, path: str, cam_builder):
        """
        导出每块地为一个 Polygon（经纬度），可直接拖进 QGIS / ArcGIS / 高德。
        cam_builder(name, meta_dict) -> Camera
        """
        feats = []
        for name in self.images:
            recs = self.fields(name)
            meta = self.meta(name)
            if not recs or not meta:
                continue
            try:
                cam = cam_builder(name, meta)
            except Exception:
                continue
            for r in recs:
                pts = np.asarray(r.contour_px, dtype=np.float64).reshape(-1, 2)
                try:
                    g = cam.pixels_to_ground(pts)
                    from uav_geo import enu_to_latlon
                    ring = []
                    for e, n, _ in g:
                        lat, lon = enu_to_latlon(e, n, cam.meta.latitude, cam.meta.longitude)
                        ring.append([round(lon, 9), round(lat, 9)])
                    if ring and ring[0] != ring[-1]:
                        ring.append(ring[0])
                except Exception:
                    continue
                feats.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "photo": name,
                        "fid": r.fid,
                        "name": r.name,
                        "area_m2": round(r.area_m2, 2),
                        "area_mu": round(r.area_mu, 4),
                        "error_pct": (round(r.err_rel * 100, 1)
                                      if np.isfinite(r.err_rel) else None),
                        "height_m": round(r.h_used, 2) if np.isfinite(r.h_used) else None,
                        "score": round(r.score, 3) if np.isfinite(r.score) else None,
                        "note": "单张照片投影估算，非测绘成果",
                    },
                })
        doc = {
            "type": "FeatureCollection",
            "name": os.path.basename(self.root),
            "crs": {"type": "name",
                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": feats,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        return path, len(feats)
