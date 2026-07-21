"""extract_crops
==============

Utility to extract glyph crops from detection predictions.

Input:
- detect_preds.jsonl produced by predict_detect.py
  Each line:
    {
      "uid": "...",
      "image_abs": "/abs/path/img.jpg",
      "image_rel": "...",
      "pred": {"bboxes": [[x1,y1,x2,y2],...], "scores": [...], "cls": [...]}
    }

Output:
- crops_dir/<uid>_det000001.png
- crops_index.jsonl with mapping:
    {
      "id": "<uid>_det000001",
      "uid": "...",
      "image_abs": "...",
      "crop_path": "...",
      "bbox": [x1,y1,x2,y2],
      "det_score": 0.91,
      "det_cls": 0
    }

Notes
-----
- This module does *not* rotate/mirror images. Apply orientation beforehand
  (e.g., bake oriented images or extend this script to read meta from manifest).
- Crops are clamped to image bounds.
- Optional padding enlarges bbox by a fraction of box size.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract glyph crops from detection predictions.")
    p.add_argument("--detect", type=str, required=True, help="detect_preds.jsonl")
    p.add_argument("--crops-dir", type=str, default="crops", help="Output folder for crops")
    p.add_argument("--index", type=str, default="crops_index.jsonl", help="Output index JSONL")
    p.add_argument("--min-score", type=float, default=0.0, help="Filter detections below this score")
    p.add_argument("--max-per-image", type=int, default=None, help="Optional cap per image")
    p.add_argument("--pad", type=float, default=0.05, help="Padding fraction of bbox size")
    p.add_argument("--format", type=str, default="png", choices=["png", "jpg"], help="Crop image format")
    p.add_argument("--skip-existing", action="store_true", help="Skip crop writing if file exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Need pillow. pip install pillow") from e

    det_rows = _read_jsonl(Path(args.detect))
    crops_dir = Path(args.crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    out_index: List[Dict[str, Any]] = []

    for row in det_rows:
        uid = str(row.get("uid"))
        img_path = row.get("image_abs")
        if not img_path:
            continue
        img_path = str(img_path)

        pred = row.get("pred") or {}
        bboxes = pred.get("bboxes") or []
        scores = pred.get("scores") or [None] * len(bboxes)
        clses = pred.get("cls") or [0] * len(bboxes)

        if not bboxes:
            continue

        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        W, H = im.size

        # Optionally sort by score desc so max-per-image keeps best
        order = list(range(len(bboxes)))
        try:
            order.sort(key=lambda i: float(scores[i]) if scores[i] is not None else 0.0, reverse=True)
        except Exception:
            pass

        n_kept = 0
        for j, i in enumerate(order):
            if args.max_per_image is not None and n_kept >= int(args.max_per_image):
                break

            bb = bboxes[i]
            sc = scores[i]
            if sc is not None and float(sc) < float(args.min_score):
                continue

            x1, y1, x2, y2 = map(float, bb)
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            pad_x = bw * float(args.pad)
            pad_y = bh * float(args.pad)

            x1p = max(0, int(x1 - pad_x))
            y1p = max(0, int(y1 - pad_y))
            x2p = min(W, int(x2 + pad_x))
            y2p = min(H, int(y2 + pad_y))

            if x2p <= x1p or y2p <= y1p:
                continue

            crop = im.crop((x1p, y1p, x2p, y2p))

            crop_id = f"{uid}_det{i:06d}"
            crop_path = crops_dir / f"{crop_id}.{args.format}"

            if not (args.skip_existing and crop_path.exists()):
                if args.format == "jpg":
                    crop.save(crop_path, quality=95)
                else:
                    crop.save(crop_path)

            out_index.append({
                "id": crop_id,
                "uid": uid,
                "image_abs": img_path,
                "crop_path": str(crop_path),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_padded": [int(x1p), int(y1p), int(x2p), int(y2p)],
                "det_score": float(sc) if sc is not None else None,
                "det_cls": int(clses[i]) if clses is not None else 0,
            })
            n_kept += 1

    _write_jsonl(Path(args.index), out_index)
    print(f"[extract_crops] wrote {len(out_index)} crops to {crops_dir} and index {args.index}")


if __name__ == "__main__":
    main()
