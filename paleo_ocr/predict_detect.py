"""predict_detect
================

Batch inference for glyph detection (YOLO) with caching.

Input:
- manifest (jsonl) OR a folder of images
- a trained Ultralytics model checkpoint

Output:
- predictions.jsonl: one record per image with list of bboxes + scores

Schema (per image):
{
  "uid": "...",
  "image_rel": "...",
  "image_abs": "...",
  "pred": {
    "bboxes": [[x1,y1,x2,y2], ...],
    "scores": [0.91, ...],
    "cls": [0, ...]
  }
}

Caching:
- If --cache-dir is provided, each uid is cached as a JSON file.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _default_uid_from_path(p: Path) -> str:
    return p.stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch predict glyph bboxes with YOLO.")
    p.add_argument("--model", type=str, required=True, help="Path to trained .pt or Ultralytics model")
    p.add_argument("--manifest", type=str, default=None, help="manifest.jsonl")
    p.add_argument("--images", type=str, default=None, help="Folder with images (if no manifest)")
    p.add_argument("--out", type=str, default="detect_preds.jsonl")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError("Install ultralytics: pip install ultralytics") from e

    model = YOLO(args.model)

    items: List[Dict[str, Any]] = []

    if args.manifest:
        records = _read_jsonl(Path(args.manifest))
        for r in records:
            img = r.get("image", {})
            abs_path = img.get("abs_path")
            rel_path = img.get("rel_path")
            uid = r.get("uid") or _default_uid_from_path(Path(abs_path or rel_path or ""))
            if not abs_path:
                images_root = r.get("images_root")
                if images_root and rel_path:
                    abs_path = str(Path(images_root) / rel_path)
            if not abs_path:
                continue
            items.append({"uid": uid, "image_abs": abs_path, "image_rel": rel_path})
    else:
        if not args.images:
            raise ValueError("Provide --manifest or --images")
        for pth in sorted(Path(args.images).glob("*")):
            if pth.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
                continue
            items.append({"uid": _default_uid_from_path(pth), "image_abs": str(pth), "image_rel": pth.name})

    if args.max_items:
        items = items[: args.max_items]

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, Any]] = []

    for it in items:
        uid = it["uid"]
        img_path = it["image_abs"]

        cache_path = cache_dir / f"{uid}.json" if cache_dir else None
        if cache_path and cache_path.exists():
            pred = json.loads(cache_path.read_text(encoding="utf-8"))
            out_rows.append({"uid": uid, "image_rel": it.get("image_rel"), "image_abs": img_path, "pred": pred})
            continue

        res = model.predict(
            source=img_path,
            conf=float(args.conf),
            imgsz=int(args.imgsz),
            device=args.device,
            verbose=False,
        )
        r0 = res[0]

        bboxes = []
        scores = []
        cls = []

        # xyxy
        if getattr(r0, "boxes", None) is not None and r0.boxes is not None:
            for b in r0.boxes:
                xyxy = b.xyxy[0].tolist()
                bboxes.append([float(x) for x in xyxy])
                scores.append(float(b.conf[0]))
                cls.append(int(b.cls[0]))

        pred = {"bboxes": bboxes, "scores": scores, "cls": cls}
        if cache_path:
            cache_path.write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")

        out_rows.append({"uid": uid, "image_rel": it.get("image_rel"), "image_abs": img_path, "pred": pred})

    _write_jsonl(Path(args.out), out_rows)
    print(f"[predict_detect] wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
