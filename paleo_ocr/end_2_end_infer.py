from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _argmax(probs: List[float]) -> int:
    if not probs:
        return -1
    best_i = 0
    best_v = probs[0]
    for i, v in enumerate(probs):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def _draw_viz(
    image_path: str,
    items: List[Dict[str, Any]],
    out_path: Path,
    *,
    fontsize: int = 16,
    thickness: int = 2,
    clamp: bool = True,
) -> None:
    """
    Draw YOLO-like bboxes + text on the source image.

    items: [{"bbox":[x1,y1,x2,y2], "text":"א 0.83 | det 0.91", "missing_cls":bool}, ...]
    """
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)

    # Hebrew-friendly font if available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()

    # Colors similar to YOLO-ish look
    color_ok = (0, 128, 255)      # blue-ish
    color_missing = (255, 0, 0)   # red
    text_bg = (0, 0, 0)
    text_fg = (255, 255, 255)

    for it in items:
        bb = it.get("bbox") or []
        if len(bb) != 4:
            continue
        x1, y1, x2, y2 = map(_safe_float, bb)

        if clamp:
            x1 = max(0.0, min(x1, W - 1))
            x2 = max(0.0, min(x2, W - 1))
            y1 = max(0.0, min(y1, H - 1))
            y2 = max(0.0, min(y2, H - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        text = str(it.get("text") or "")
        missing_cls = bool(it.get("missing_cls", False))
        outline = color_missing if missing_cls else color_ok

        # bbox
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=thickness)

        # label box (top-left)
        if text:
            try:
                tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), text, font=font)
                tw, th = (tx1 - tx0), (ty1 - ty0)
            except Exception:
                tw, th = (len(text) * fontsize // 2, fontsize + 4)

            px = x1
            py = max(0.0, y1 - th - 6)

            draw.rectangle([px, py, px + tw + 6, py + th + 6], fill=text_bg)
            draw.text((px + 3, py + 3), text, font=font, fill=text_fg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end OCR pipeline runner: detect -> crop -> classify -> decode -> metrics")

    p.add_argument("--manifest", type=str, required=True, help="manifest_oriented.jsonl")

    # detector
    p.add_argument("--det-model", type=str, required=True, help="YOLO detector checkpoint")
    p.add_argument("--det-conf", type=float, default=0.25)
    p.add_argument("--det-imgsz", type=int, default=1024)
    p.add_argument("--det-device", type=str, default="0")
    p.add_argument("--det-cache", type=str, default="cache_detect")

    # crops
    p.add_argument("--crops-dir", type=str, default="crops")
    p.add_argument("--crops-index", type=str, default="crops_index.jsonl")
    p.add_argument("--crop-pad", type=float, default=0.05)
    p.add_argument("--crop-min-score", type=float, default=0.0)

    # classifier
    p.add_argument("--cls-ckpt", type=str, required=True, help="Classifier best.pt")
    p.add_argument("--cls-model", type=str, default="convnext_base")
    p.add_argument("--cls-imgsz", type=int, default=128)
    p.add_argument("--cls-batch", type=int, default=256)
    p.add_argument("--cls-device", type=str, default="cuda")
    p.add_argument("--cls-cache", type=str, default="cache_cls")
    p.add_argument("--cls-topk", type=int, default=5)

    # decoding
    p.add_argument("--min-char-conf", type=float, default=0.35)
    p.add_argument("--unknown-token", type=str, default="[?]")
    p.add_argument("--postprocess", type=str, default="none", choices=["none", "paleo_basic"])

    # outputs
    p.add_argument("--workdir", type=str, default="e2e_run")
    p.add_argument("--decoded", type=str, default="decoded.jsonl")
    p.add_argument("--metrics-out", type=str, default="metrics.json")
    p.add_argument("--max-items", type=int, default=None)

    # viz
    p.add_argument("--viz", action="store_true", help="Save YOLO-like visualization images")
    p.add_argument("--viz-dir", type=str, default="viz", help="Subfolder under workdir")
    p.add_argument("--viz-max", type=int, default=50, help="Max number of images to visualize")
    p.add_argument("--viz-fontsize", type=int, default=16)
    p.add_argument("--viz-thickness", type=int, default=2)
    p.add_argument("--viz-min-conf", type=float, default=0.0, help="Hide cls labels below this conf (bbox still drawn)")
    p.add_argument("--viz-show-det-score", action="store_true", help="Append detector score to label text")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Lazy imports (local modules)
    from ocr_decoder import decode_detections
    from ocr_metrics import compute_metrics_for_pairs

    # Step 1: detect
    detect_out = workdir / "detect_preds.jsonl"
    if not detect_out.exists():
        from predict_detect import main as predict_detect_main
        import sys

        sys.argv = [
            "predict_detect.py",
            "--model", args.det_model,
            "--manifest", args.manifest,
            "--out", str(detect_out),
            "--conf", str(args.det_conf),
            "--imgsz", str(args.det_imgsz),
            "--device", args.det_device,
            "--cache-dir", args.det_cache,
        ]
        predict_detect_main()

    # Step 2: crops
    crops_index = workdir / args.crops_index
    if not crops_index.exists():
        from extract_crops import main as extract_crops_main
        import sys

        sys.argv = [
            "extract_crops.py",
            "--detect", str(detect_out),
            "--crops-dir", str(workdir / args.crops_dir),
            "--index", str(crops_index),
            "--pad", str(args.crop_pad),
            "--min-score", str(args.crop_min_score),
        ]
        extract_crops_main()

    # Step 3: classify
    classify_out = workdir / "classify_preds.jsonl"
    if not classify_out.exists():
        from predict_classify import main as predict_classify_main
        import sys

        sys.argv = [
            "predict_classify.py",
            "--ckpt", args.cls_ckpt,
            "--model", args.cls_model,
            "--jsonl", str(crops_index),
            "--out", str(classify_out),
            "--imgsz", str(args.cls_imgsz),
            "--batch", str(args.cls_batch),
            "--device", args.cls_device,
            "--topk", str(args.cls_topk),
            "--cache-dir", args.cls_cache,
        ]
        predict_classify_main()

    # Load manifest and predictions
    manifest = _read_jsonl(Path(args.manifest))
    det_rows = _read_jsonl(detect_out)
    cls_rows = _read_jsonl(classify_out)

    # Index helper maps
    det_by_uid = {r["uid"]: r for r in det_rows if "uid" in r}
    cls_by_id = {r["id"]: r for r in cls_rows if "id" in r}

    # classes list for decoding: take from classifier checkpoint
    import torch
    ckpt = torch.load(args.cls_ckpt, map_location="cpu")
    classes: List[str] = ckpt.get("classes") or []
    if not classes:
        raise ValueError("Classifier checkpoint missing 'classes'")

    cls_to_idx = {c: i for i, c in enumerate(classes)}

    def topk_to_probs(topk: List[List[Any]]) -> List[float]:
        probs = [0.0] * len(classes)
        for lab, conf in topk:
            if lab in cls_to_idx:
                probs[cls_to_idx[lab]] = float(conf)
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
        return probs

    def normalize_probs(probs: List[float]) -> List[float]:
        s = float(sum(probs))
        if s <= 0:
            return [1.0 / len(probs)] * len(probs)
        return [float(p) / s for p in probs]

    decoded_rows: List[Dict[str, Any]] = []
    n = 0
    n_viz = 0

    for rec in manifest:
        if args.max_items is not None and n >= int(args.max_items):
            break

        uid = rec.get("uid")
        if not uid or uid not in det_by_uid:
            continue

        meta = rec.get("meta", {})
        reading_dir = meta.get("reading_dir", "rtl")
        sort_primary = meta.get("sort_primary", "x")
        group_mode = meta.get("layout_hint", "auto")

        det_row = det_by_uid[uid]
        det = det_row.get("pred") or {}
        bboxes = det.get("bboxes") or []
        scores = det.get("scores") or [None] * len(bboxes)

        # for viz: find image path
        img_path_for_viz = det_row.get("image_abs") or rec.get("image_abs") or rec.get("image") or None

        detections: List[Dict[str, Any]] = []
        viz_items: List[Dict[str, Any]] = []

        for i, bb in enumerate(bboxes):
            crop_id = f"{uid}_det{i:06d}"
            cls_row = cls_by_id.get(crop_id)

            det_score = float(scores[i]) if scores and i < len(scores) and scores[i] is not None else None

            if not cls_row:
                # keep for viz so you can see detector boxes with missing classification
                if args.viz and n_viz < int(args.viz_max) and img_path_for_viz:
                    txt = "[NO_CLS]"
                    if args.viz_show_det_score and det_score is not None:
                        txt += f" | det {det_score:.2f}"
                    viz_items.append({"bbox": bb, "text": txt, "missing_cls": True})
                continue

            pred = cls_row.get("pred") or {}

            probs = pred.get("probs")
            if isinstance(probs, list) and probs:
                probs = [float(x) for x in probs]
                row_classes = pred.get("classes")
                if isinstance(row_classes, list) and row_classes and row_classes != classes:
                    row_map = {str(c): j for j, c in enumerate(row_classes)}
                    aligned = [0.0] * len(classes)
                    for j, c in enumerate(classes):
                        if c in row_map and row_map[c] < len(probs):
                            aligned[j] = float(probs[row_map[c]])
                    probs = aligned
                probs = normalize_probs(probs)
            else:
                topk = pred.get("topk") or []
                probs = topk_to_probs(topk)

            # viz label/conf
            if args.viz and n_viz < int(args.viz_max) and img_path_for_viz:
                label = pred.get("label")
                conf = pred.get("conf")

                if label is None or conf is None:
                    j = _argmax(probs)
                    if 0 <= j < len(classes):
                        label = classes[j]
                        conf = probs[j]
                    else:
                        label = "[?]"
                        conf = 0.0

                conf_f = _safe_float(conf, 0.0)
                # always draw bbox; optionally hide label text if below threshold
                if conf_f >= float(args.viz_min_conf):
                    txt = f"{label} {conf_f:.2f}"
                else:
                    txt = ""  # bbox only

                if args.viz_show_det_score and det_score is not None:
                    txt = (txt + (" | " if txt else "") + f"det {det_score:.2f}")

                viz_items.append({"bbox": bb, "text": txt, "missing_cls": False})

            detections.append({
                "bbox": bb,
                "probs": probs,
                "classes": classes,
                "score": det_score,
                "id": crop_id,
            })

        decoded = decode_detections(
            detections=detections,
            classes=classes,
            reading_dir=reading_dir,
            group_mode=group_mode,
            sort_primary=sort_primary,
            min_char_conf=float(args.min_char_conf),
            unknown_token=args.unknown_token,
            postprocess=args.postprocess,
        )

        gt = (rec.get("gt") or {}).get("hebrew", {})
        gt_text = gt.get("raw") or gt.get("text") or ""

        out_row = {
            "uid": uid,
            "row_id": rec.get("row_id"),
            "pred_text": decoded.text.replace('\n', ' '),
            "gt_text": gt_text,
            "debug": decoded.debug,
        }

        # save viz per image
        if args.viz and n_viz < int(args.viz_max) and img_path_for_viz and (bboxes or viz_items):
            out_img = workdir / str(args.viz_dir) / f"{uid}.jpg"
            try:
                _draw_viz(
                    str(img_path_for_viz),
                    viz_items,
                    out_img,
                    fontsize=int(args.viz_fontsize),
                    thickness=int(args.viz_thickness),
                )
                out_row["viz_path"] = str(out_img)
                n_viz += 1
            except Exception as e:
                print(f"[viz] failed for uid={uid}: {e}")

        decoded_rows.append(out_row)
        n += 1

    decoded_out = workdir / args.decoded
    _write_jsonl(decoded_out, decoded_rows)

    pairs = [(r.get("gt_text", ""), r.get("pred_text", "")) for r in decoded_rows]
    metrics = compute_metrics_for_pairs(pairs)
    (workdir / args.metrics_out).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[end2end] decoded {len(decoded_rows)} images")
    print(f"[end2end] metrics saved to {workdir / args.metrics_out}")
    if args.viz:
        print(f"[end2end] viz saved to {workdir / str(args.viz_dir)} ({n_viz} images)")


if __name__ == "__main__":
    main()
