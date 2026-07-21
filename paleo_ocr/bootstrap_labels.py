"""bootstrap_labels
===================

Semi-automatic (bootstrap) bounding-box proposal + export/import utilities.

Why this exists
---------------
Your seal dataset has image-level GT text but *no per-character bounding boxes*.
To train a detector, you need bbox annotations. This module proposes candidate
glyph boxes using classic image processing:

  adaptive threshold -> morphology -> connected components/contours -> bbox candidates

Then it can export those candidates as *pre-annotations* for human review in:

  - Label Studio (predictions/tasks JSON)
  - YOLO (txt labels)
  - COCO (instances)

After you correct/approve annotations (human-in-the-loop), you can import
Label Studio / YOLO / COCO exports back into a simple "gold labels" JSONL,
and optionally write them into your manifest under `gt.bboxes`.

Design goals
------------
- deterministic and debuggable
- minimal dependencies (opencv-python, numpy)
- works with images stored in a directory or inside a zip, as produced by
  dataset_manifest.py

This is a *bootstrap* helper, not a final detector.

Example
-------
Propose boxes and export to Label Studio tasks:

  python bootstrap_labels.py propose \
    --manifest manifest.jsonl \
    --out proposals.jsonl \
    --debug-dir debug_boxes \
    --max-items 300

  python bootstrap_labels.py export-labelstudio \
    --manifest manifest.jsonl \
    --proposals proposals.jsonl \
    --out labelstudio_tasks.json \
    --image-url-prefix "file://" \
    --label-name glyph

Import reviewed annotations (Label Studio export JSON) to gold labels and
update manifest:

  python bootstrap_labels.py import-labelstudio \
    --labelstudio-export export.json \
    --out gold_labels.jsonl

  python bootstrap_labels.py apply-gold \
    --manifest manifest.jsonl \
    --gold gold_labels.jsonl \
    --out manifest_with_bboxes.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import posixpath
import random
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise RuntimeError("opencv-python is required (pip install opencv-python)") from e


# -------------------------
# I/O helpers (manifest + images)
# -------------------------


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _manifest_image_ref(rec: Dict[str, Any]) -> Dict[str, Any]:
    img = rec.get("image")
    if not isinstance(img, dict):
        raise ValueError("manifest record missing 'image' dict")
    return img


def load_image_bgr_from_ref(image_ref: Dict[str, Any]) -> np.ndarray:
    """Load image as BGR uint8, supporting directory paths or zip containers."""
    container = image_ref.get("container")
    if isinstance(container, dict) and container.get("type") == "zip":
        zip_path = container["path"]
        inner = container["inner_path"]
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner) as f:
                data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imdecode failed for {zip_path}:{inner}")
        return img

    path = image_ref.get("path")
    if not path:
        raise ValueError("image_ref missing 'path'")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def apply_meta_transform(img_bgr: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """Apply coarse rotation and optional mirroring according to manifest meta."""
    deg = meta.get("orientation_deg")
    mir = meta.get("mirrored")
    out = img_bgr

    if isinstance(deg, (int, float)):
        d = int(deg) % 360
        if d == 90:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        elif d == 180:
            out = cv2.rotate(out, cv2.ROTATE_180)
        elif d == 270:
            out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if mir is True:
        out = cv2.flip(out, 1)  # horizontal flip

    return out


# -------------------------
# Bootstrap bbox proposal
# -------------------------


@dataclass
class ProposalConfig:
    # thresholding
    block_size: int = 35
    c: int = 7
    # morphology
    open_ks: int = 3
    close_ks: int = 3
    # connected component filtering
    min_area: int = 20
    max_area_frac: float = 0.15
    min_h: int = 6
    min_w: int = 6
    max_aspect: float = 8.0
    min_aspect: float = 0.12
    # density sanity
    fg_frac_min: float = 0.01
    fg_frac_max: float = 0.45
    # merge
    merge_iou: float = 0.25
    # random jitter for visualization order (optional)
    seed: int = 42


def _ensure_odd(x: int) -> int:
    x = int(x)
    if x < 3:
        return 3
    return x if x % 2 == 1 else x + 1


def _adaptive_binarize(gray: np.ndarray, cfg: ProposalConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Try both polarities and choose the one with reasonable foreground density."""
    bs = _ensure_odd(cfg.block_size)
    C = int(cfg.c)

    # OpenCV adaptive threshold expects 8-bit single channel
    g = gray
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    th1 = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, C
    )
    th2 = 255 - th1

    def fg_frac(th: np.ndarray) -> float:
        return float(np.mean(th > 0))  # white fraction

    f1 = fg_frac(th1)
    f2 = fg_frac(th2)

    # We want "ink" to be white (fg) for CC extraction.
    # If f is too high, flip. If too low, maybe flip.
    candidates = [(th1, f1, "binary"), (th2, f2, "inverted")]

    def score(f: float) -> float:
        # penalize outside target range
        if f < cfg.fg_frac_min:
            return abs(cfg.fg_frac_min - f) + 1.0
        if f > cfg.fg_frac_max:
            return abs(f - cfg.fg_frac_max) + 1.0
        # inside range: prefer mid-range
        mid = (cfg.fg_frac_min + cfg.fg_frac_max) / 2.0
        return abs(f - mid)

    best = min(candidates, key=lambda t: score(t[1]))
    th, frac, mode = best
    return th, {"fg_frac": frac, "binarize_mode": mode, "block_size": bs, "C": C}


def _morph(th: np.ndarray, cfg: ProposalConfig) -> np.ndarray:
    out = th
    if cfg.open_ks and cfg.open_ks > 0:
        k = int(cfg.open_ks)
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ker)
    if cfg.close_ks and cfg.close_ks > 0:
        k = int(cfg.close_ks)
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker)
    return out


def _cc_bboxes(mask: np.ndarray, cfg: ProposalConfig) -> List[Tuple[int, int, int, int, float]]:
    """Return bboxes as (x1,y1,x2,y2,score) for connected components."""
    # Ensure binary is 0/255
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape[:2]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)

    max_area = int(cfg.max_area_frac * (h * w))
    out: List[Tuple[int, int, int, int, float]] = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < cfg.min_area:
            continue
        if area > max_area:
            continue
        if bw < cfg.min_w or bh < cfg.min_h:
            continue
        aspect = bw / float(bh + 1e-6)
        if aspect > cfg.max_aspect or aspect < cfg.min_aspect:
            continue
        # simple score: area fraction (bounded)
        score = min(1.0, area / float(max(cfg.min_area, 1)))
        out.append((int(x), int(y), int(x + bw), int(y + bh), float(score)))
    return out


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def _merge_boxes(boxes: List[Tuple[int, int, int, int, float]], iou_thr: float) -> List[Tuple[int, int, int, int, float]]:
    """Greedy merge: if IoU>thr merge into union box."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    merged: List[Tuple[int, int, int, int, float]] = []
    used = [False] * len(boxes)
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        x1, y1, x2, y2, s = b
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            bj = boxes[j]
            if _iou((x1, y1, x2, y2), (bj[0], bj[1], bj[2], bj[3])) >= iou_thr:
                x1 = min(x1, bj[0])
                y1 = min(y1, bj[1])
                x2 = max(x2, bj[2])
                y2 = max(y2, bj[3])
                s = max(s, bj[4])
                used[j] = True
        merged.append((x1, y1, x2, y2, s))
    return merged


def propose_bboxes(img_bgr: np.ndarray, cfg: ProposalConfig) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Propose candidate glyph bounding boxes for a single image."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th, dbg_th = _adaptive_binarize(gray, cfg)
    th = _morph(th, cfg)
    cc = _cc_bboxes(th, cfg)
    cc = _merge_boxes(cc, cfg.merge_iou)

    # Make deterministic, but allow stable ordering by left-to-right then top-to-bottom
    cc = sorted(cc, key=lambda b: (b[1], b[0]))

    boxes: List[Dict[str, Any]] = []
    for (x1, y1, x2, y2, s) in cc:
        boxes.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": "glyph",
                "score": float(s),
            }
        )

    dbg = {
        "image_wh": [int(w), int(h)],
        **dbg_th,
        "n_boxes": len(boxes),
        "cfg": {
            "block_size": _ensure_odd(cfg.block_size),
            "c": int(cfg.c),
            "open_ks": int(cfg.open_ks),
            "close_ks": int(cfg.close_ks),
            "min_area": int(cfg.min_area),
            "max_area_frac": float(cfg.max_area_frac),
            "min_w": int(cfg.min_w),
            "min_h": int(cfg.min_h),
            "min_aspect": float(cfg.min_aspect),
            "max_aspect": float(cfg.max_aspect),
            "merge_iou": float(cfg.merge_iou),
        },
    }
    return boxes, dbg


def draw_boxes(img_bgr: np.ndarray, boxes: Sequence[Dict[str, Any]]) -> np.ndarray:
    out = img_bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


# -------------------------
# Export formats
# -------------------------


def export_yolo(
    manifest: Sequence[Dict[str, Any]],
    proposals: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
    class_names: Sequence[str] = ("glyph",),
) -> None:
    """Export proposals (or gold) into YOLO txt labels.

    out_dir will contain:
      - labels/*.txt (same basename as image rel path, but safe)
      - classes.txt
      - index.jsonl (uid -> image path, label file)
    """
    os.makedirs(out_dir, exist_ok=True)
    labels_dir = Path(out_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # class mapping
    cls_to_id = {c: i for i, c in enumerate(class_names)}
    with open(Path(out_dir) / "classes.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")

    index_rows: List[Dict[str, Any]] = []
    for rec in manifest:
        uid = str(rec["uid"])
        img = _manifest_image_ref(rec)
        if not img.get("exists", True):
            continue
        w = img.get("width")
        h = img.get("height")
        if w is None or h is None:
            # YOLO needs normalization; if missing, skip
            continue

        boxes = proposals.get(uid, [])
        # label filename: uid.txt (stable)
        lbl_path = labels_dir / f"{uid}.txt"
        with open(lbl_path, "w", encoding="utf-8") as f:
            for b in boxes:
                x1, y1, x2, y2 = map(float, b["bbox"])
                label = str(b.get("label", "glyph"))
                if label not in cls_to_id:
                    continue
                cls_id = cls_to_id[label]
                xc = (x1 + x2) / 2.0 / float(w)
                yc = (y1 + y2) / 2.0 / float(h)
                bw = (x2 - x1) / float(w)
                bh = (y2 - y1) / float(h)
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        index_rows.append(
            {
                "uid": uid,
                "image": img.get("path") or img.get("rel_path"),
                "label_file": str(lbl_path),
            }
        )

    write_jsonl(str(Path(out_dir) / "index.jsonl"), index_rows)


def export_coco(
    manifest: Sequence[Dict[str, Any]],
    proposals: Dict[str, List[Dict[str, Any]]],
    out_path: str,
    class_names: Sequence[str] = ("glyph",),
) -> None:
    """Export boxes to COCO instances JSON."""

    cls_to_id = {c: i + 1 for i, c in enumerate(class_names)}  # COCO category ids start at 1
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    uid_to_img_id: Dict[str, int] = {}

    for rec in manifest:
        uid = str(rec["uid"])
        img = _manifest_image_ref(rec)
        if not img.get("exists", True):
            continue
        w = img.get("width")
        h = img.get("height")
        if w is None or h is None:
            continue

        file_name = img.get("path") or img.get("rel_path") or uid
        images.append({"id": img_id, "file_name": file_name, "width": int(w), "height": int(h), "uid": uid})
        uid_to_img_id[uid] = img_id

        boxes = proposals.get(uid, [])
        for b in boxes:
            x1, y1, x2, y2 = map(float, b["bbox"])
            label = str(b.get("label", "glyph"))
            if label not in cls_to_id:
                continue
            cat_id = cls_to_id[label]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            area = float(bw * bh)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(bw), float(bh)],
                    "area": area,
                    "iscrowd": 0,
                    "score": float(b.get("score", 1.0)),
                }
            )
            ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cls_to_id[c], "name": c} for c in class_names],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)


def export_labelstudio(
    manifest: Sequence[Dict[str, Any]],
    proposals: Dict[str, List[Dict[str, Any]]],
    out_path: str,
    image_url_prefix: str = "",
    label_name: str = "glyph",
    from_name: str = "label",
    to_name: str = "image",
) -> None:
    """Export tasks with pre-annotations for Label Studio.

    Creates a JSON list of tasks. Each task includes a `predictions` field.
    Rectangle coords are expressed as percentages.
    """

    tasks = []
    for rec in manifest:
        uid = str(rec["uid"])
        img = _manifest_image_ref(rec)
        if not img.get("exists", True):
            continue
        w = img.get("width")
        h = img.get("height")
        if w is None or h is None:
            continue

        # Label Studio requires an accessible URL. For local use you can use file:// paths.
        path = img.get("path") or img.get("rel_path")
        image_url = image_url_prefix + str(path)

        results = []
        for i, b in enumerate(proposals.get(uid, [])):
            x1, y1, x2, y2 = map(float, b["bbox"])
            x = (x1 / float(w)) * 100.0
            y = (y1 / float(h)) * 100.0
            bw = ((x2 - x1) / float(w)) * 100.0
            bh = ((y2 - y1) / float(h)) * 100.0
            results.append(
                {
                    "id": f"{uid}_{i}",
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "rectanglelabels",
                    "value": {"x": x, "y": y, "width": bw, "height": bh, "rotation": 0},
                    "score": float(b.get("score", 0.5)),
                    "rectanglelabels": [str(b.get("label", label_name))],
                }
            )

        task = {
            "data": {"image": image_url, "uid": uid},
            "predictions": [{"model_version": "bootstrap_v1", "result": results}],
        }
        tasks.append(task)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)


# -------------------------
# Import formats -> gold labels
# -------------------------


def import_labelstudio_export(path: str) -> List[Dict[str, Any]]:
    """Import Label Studio export JSON.

    Expects a list of tasks, each task with `data.uid` and annotation results.
    Works with exports that contain either:
      - `annotations` (human) OR
      - `predictions` (pre-annotations)

    Returns gold labels JSONL rows:
      {"uid": ..., "bboxes": [{"bbox": [...], "label": "glyph"}]}
    """
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("Label Studio export must be a list")

    gold: List[Dict[str, Any]] = []

    for task in tasks:
        data = task.get("data", {}) if isinstance(task, dict) else {}
        uid = data.get("uid")
        if uid is None:
            # try to recover from image name
            uid = task.get("uid")
        if uid is None:
            continue
        uid = str(uid)

        # prefer human annotations
        anns = task.get("annotations") if isinstance(task, dict) else None
        if isinstance(anns, list) and anns:
            sources = anns
            key = "result"
        else:
            sources = task.get("predictions") if isinstance(task, dict) else None
            key = "result"

        bboxes: List[Dict[str, Any]] = []
        if isinstance(sources, list):
            for a in sources:
                res = a.get(key)
                if not isinstance(res, list):
                    continue
                for r in res:
                    if r.get("type") != "rectanglelabels":
                        continue
                    val = r.get("value", {})
                    if not isinstance(val, dict):
                        continue
                    # Keep percent coords; conversion to pixels happens when applying to manifest
                    bboxes.append(
                        {
                            "bbox_pct": [float(val.get("x", 0.0)), float(val.get("y", 0.0)), float(val.get("width", 0.0)), float(val.get("height", 0.0))],
                            "label": (r.get("rectanglelabels") or ["glyph"])[0],
                        }
                    )

        gold.append({"uid": uid, "bboxes": bboxes, "format": "labelstudio_pct"})

    return gold


def import_coco(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco.get("images", [])}
    cats = {c["id"]: c.get("name", "glyph") for c in coco.get("categories", [])}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in coco.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)

    gold: List[Dict[str, Any]] = []
    for img_id, img in images.items():
        uid = img.get("uid") or img.get("file_name") or str(img_id)
        bxs: List[Dict[str, Any]] = []
        for a in anns_by_img.get(img_id, []):
            x, y, w, h = a.get("bbox", [0, 0, 0, 0])
            label = cats.get(a.get("category_id"), "glyph")
            bxs.append({"bbox": [float(x), float(y), float(x + w), float(y + h)], "label": label})
        gold.append({"uid": str(uid), "bboxes": bxs, "format": "xyxy_px"})
    return gold


def import_yolo(labels_dir: str, index_jsonl: Optional[str], class_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Import YOLO txt labels into gold labels.

    Requires image width/height; we obtain it by reading the referenced manifest
    later when applying gold. Therefore we keep the YOLO normalized format here.
    """
    # uid is filename stem by our exporter
    labels_dir_p = Path(labels_dir)
    gold: List[Dict[str, Any]] = []
    id_to_cls = {i: c for i, c in enumerate(class_names)}
    for txt in sorted(labels_dir_p.glob("*.txt")):
        uid = txt.stem
        bxs: List[Dict[str, Any]] = []
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                bxs.append({"bbox_yolo": [xc, yc, bw, bh], "label": id_to_cls.get(cid, "glyph")})
        gold.append({"uid": uid, "bboxes": bxs, "format": "yolo_norm"})
    return gold


# -------------------------
# Apply gold labels back to manifest
# -------------------------


def apply_gold_to_manifest(
    manifest: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
    out_path: str,
    require_image_wh: bool = True,
) -> None:
    gold_map = {str(g["uid"]): g for g in gold}

    updated = []
    missing = 0
    for rec in manifest:
        uid = str(rec["uid"])
        g = gold_map.get(uid)
        if not g:
            updated.append(rec)
            missing += 1
            continue

        img = _manifest_image_ref(rec)
        w = img.get("width")
        h = img.get("height")

        fmt = g.get("format")
        bxs_out: List[Dict[str, Any]] = []
        for b in g.get("bboxes", []):
            if fmt == "xyxy_px":
                bxs_out.append({"bbox": b["bbox"], "label": b.get("label", "glyph")})
            elif fmt == "labelstudio_pct":
                if require_image_wh and (w is None or h is None):
                    continue
                x_pct, y_pct, bw_pct, bh_pct = b["bbox_pct"]
                x1 = (x_pct / 100.0) * float(w)
                y1 = (y_pct / 100.0) * float(h)
                x2 = x1 + (bw_pct / 100.0) * float(w)
                y2 = y1 + (bh_pct / 100.0) * float(h)
                bxs_out.append({"bbox": [x1, y1, x2, y2], "label": b.get("label", "glyph")})
            elif fmt == "yolo_norm":
                if require_image_wh and (w is None or h is None):
                    continue
                xc, yc, bw, bh = b["bbox_yolo"]
                x1 = (xc - bw / 2.0) * float(w)
                x2 = (xc + bw / 2.0) * float(w)
                y1 = (yc - bh / 2.0) * float(h)
                y2 = (yc + bh / 2.0) * float(h)
                bxs_out.append({"bbox": [x1, y1, x2, y2], "label": b.get("label", "glyph")})

        rec = dict(rec)
        rec_gt = dict(rec.get("gt", {}))
        rec_gt["bboxes"] = bxs_out
        rec["gt"] = rec_gt
        updated.append(rec)

    write_jsonl(out_path, updated)


# -------------------------
# CLI
# -------------------------


def cmd_propose(args: argparse.Namespace) -> None:
    manifest = read_jsonl(args.manifest)
    cfg = ProposalConfig(
        block_size=args.block_size,
        c=args.c,
        open_ks=args.open_ks,
        close_ks=args.close_ks,
        min_area=args.min_area,
        max_area_frac=args.max_area_frac,
        min_w=args.min_w,
        min_h=args.min_h,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        merge_iou=args.merge_iou,
        seed=args.seed,
    )

    out_rows: List[Dict[str, Any]] = []
    stats = {"n_images": 0, "n_with_boxes": 0, "n_boxes_total": 0}

    debug_dir = args.debug_dir
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    n = 0
    for rec in manifest:
        if args.max_items is not None and n >= args.max_items:
            break
        img_ref = _manifest_image_ref(rec)
        if not img_ref.get("exists", True):
            continue

        try:
            img = load_image_bgr_from_ref(img_ref)
            if args.apply_meta:
                meta = rec.get("meta", {}) if isinstance(rec.get("meta"), dict) else {}
                img = apply_meta_transform(img, meta)
            boxes, dbg = propose_bboxes(img, cfg)
        except Exception as e:
            boxes, dbg = [], {"error": f"{type(e).__name__}: {e}"}

        uid = str(rec["uid"])
        out_rows.append({"uid": uid, "boxes": boxes, "debug": dbg})

        stats["n_images"] += 1
        stats["n_boxes_total"] += len(boxes)
        if len(boxes) > 0:
            stats["n_with_boxes"] += 1

        if debug_dir:
            try:
                vis = draw_boxes(img, boxes)
                cv2.imwrite(str(Path(debug_dir) / f"{uid}.jpg"), vis)
                with open(Path(debug_dir) / f"{uid}.json", "w", encoding="utf-8") as f:
                    json.dump(dbg, f, ensure_ascii=False)
            except Exception:
                pass

        n += 1

    write_jsonl(args.out, out_rows)

    if args.stats_out:
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)


def _load_proposals_map(path: str) -> Dict[str, List[Dict[str, Any]]]:
    rows = read_jsonl(path)
    m: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        uid = str(r.get("uid"))
        boxes = r.get("boxes")
        if isinstance(boxes, list):
            m[uid] = [dict(b) for b in boxes]
    return m


def cmd_export_labelstudio(args: argparse.Namespace) -> None:
    manifest = read_jsonl(args.manifest)
    props = _load_proposals_map(args.proposals)
    export_labelstudio(
        manifest=manifest,
        proposals=props,
        out_path=args.out,
        image_url_prefix=args.image_url_prefix,
        label_name=args.label_name,
        from_name=args.from_name,
        to_name=args.to_name,
    )


def cmd_export_yolo(args: argparse.Namespace) -> None:
    manifest = read_jsonl(args.manifest)
    props = _load_proposals_map(args.proposals)
    export_yolo(manifest=manifest, proposals=props, out_dir=args.out_dir, class_names=(args.label_name,))


def cmd_export_coco(args: argparse.Namespace) -> None:
    manifest = read_jsonl(args.manifest)
    props = _load_proposals_map(args.proposals)
    export_coco(manifest=manifest, proposals=props, out_path=args.out, class_names=(args.label_name,))


def cmd_import_labelstudio(args: argparse.Namespace) -> None:
    gold = import_labelstudio_export(args.labelstudio_export)
    write_jsonl(args.out, gold)


def cmd_import_coco(args: argparse.Namespace) -> None:
    gold = import_coco(args.coco)
    write_jsonl(args.out, gold)


def cmd_import_yolo(args: argparse.Namespace) -> None:
    gold = import_yolo(args.labels_dir, args.index, class_names=(args.label_name,))
    write_jsonl(args.out, gold)


def cmd_apply_gold(args: argparse.Namespace) -> None:
    manifest = read_jsonl(args.manifest)
    gold = read_jsonl(args.gold)
    apply_gold_to_manifest(manifest, gold, out_path=args.out)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bootstrap bbox proposals + export/import for annotation.")
    sp = p.add_subparsers(dest="cmd", required=True)

    p1 = sp.add_parser("propose", help="Propose glyph bbox candidates using adaptive threshold + CC.")
    p1.add_argument("--manifest", required=True)
    p1.add_argument("--out", required=True, help="JSONL proposals file (uid -> boxes)")
    p1.add_argument("--stats-out", default=None, help="Optional JSON stats output")
    p1.add_argument("--debug-dir", default=None, help="If set, write visualizations + debug json per image")
    p1.add_argument("--max-items", type=int, default=None)
    p1.add_argument("--apply-meta", action="store_true", help="Apply meta orientation/mirror before proposing")

    # threshold/morph params
    p1.add_argument("--block-size", type=int, default=35)
    p1.add_argument("--c", type=int, default=7)
    p1.add_argument("--open-ks", type=int, default=3)
    p1.add_argument("--close-ks", type=int, default=3)
    p1.add_argument("--min-area", type=int, default=20)
    p1.add_argument("--max-area-frac", type=float, default=0.15)
    p1.add_argument("--min-w", type=int, default=6)
    p1.add_argument("--min-h", type=int, default=6)
    p1.add_argument("--min-aspect", type=float, default=0.12)
    p1.add_argument("--max-aspect", type=float, default=8.0)
    p1.add_argument("--merge-iou", type=float, default=0.25)
    p1.add_argument("--seed", type=int, default=42)
    p1.set_defaults(func=cmd_propose)

    p2 = sp.add_parser("export-labelstudio", help="Export proposals to Label Studio tasks JSON")
    p2.add_argument("--manifest", required=True)
    p2.add_argument("--proposals", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--image-url-prefix", default="", help="e.g., file:// or http://server/")
    p2.add_argument("--label-name", default="glyph")
    p2.add_argument("--from-name", default="label")
    p2.add_argument("--to-name", default="image")
    p2.set_defaults(func=cmd_export_labelstudio)

    p3 = sp.add_parser("export-yolo", help="Export proposals to YOLO label txt files")
    p3.add_argument("--manifest", required=True)
    p3.add_argument("--proposals", required=True)
    p3.add_argument("--out-dir", required=True)
    p3.add_argument("--label-name", default="glyph")
    p3.set_defaults(func=cmd_export_yolo)

    p4 = sp.add_parser("export-coco", help="Export proposals to COCO instances JSON")
    p4.add_argument("--manifest", required=True)
    p4.add_argument("--proposals", required=True)
    p4.add_argument("--out", required=True)
    p4.add_argument("--label-name", default="glyph")
    p4.set_defaults(func=cmd_export_coco)

    p5 = sp.add_parser("import-labelstudio", help="Import Label Studio export JSON into gold labels JSONL")
    p5.add_argument("--labelstudio-export", required=True)
    p5.add_argument("--out", required=True)
    p5.set_defaults(func=cmd_import_labelstudio)

    p6 = sp.add_parser("import-coco", help="Import COCO instances JSON into gold labels JSONL")
    p6.add_argument("--coco", required=True)
    p6.add_argument("--out", required=True)
    p6.set_defaults(func=cmd_import_coco)

    p7 = sp.add_parser("import-yolo", help="Import YOLO labels dir into gold labels JSONL")
    p7.add_argument("--labels-dir", required=True)
    p7.add_argument("--index", default=None, help="Optional index.jsonl (not required)")
    p7.add_argument("--label-name", default="glyph")
    p7.add_argument("--out", required=True)
    p7.set_defaults(func=cmd_import_yolo)

    p8 = sp.add_parser("apply-gold", help="Apply gold labels JSONL into manifest.gt.bboxes")
    p8.add_argument("--manifest", required=True)
    p8.add_argument("--gold", required=True)
    p8.add_argument("--out", required=True)
    p8.set_defaults(func=cmd_apply_gold)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
