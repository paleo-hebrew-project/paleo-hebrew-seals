"""experiment_runner
==================

Experimental harness / runner to compare OCR approaches on a shared test split.

Why
---
You already have CER/WER utilities, and individual pipeline scripts.
This runner adds:
- a single "control panel" for experiments
- unified config schema
- reproducible outputs/artifacts
- OCR metrics (CER/WER) + detection metrics (mAP/Recall)
- worst-cases mining

Supported approaches (configurable)
----------------------------------
(a) End-to-end CTC (plugin)               [ctc]
(b) TrOCR / seq2seq (plugin)              [seq2seq]
(c) Detect + Classify + Decode (native)   [det+cls+dec]
(d) Hybrid (plugin)                        [hybrid]

We implement (c) end-to-end using your local modules:
- predict_detect.py
- extract_crops.py
- predict_classify.py
- ocr_decoder.py

For (a)/(b)/(d), this harness provides *adapter interfaces*.
You can connect existing scripts (your own, surya/doctr/tesseract/kraken baseline, etc.)
without changing the harness.

Detection metrics
-----------------
If manifest includes gold bboxes (per-glyph), we compute:
- precision/recall at IoU thresholds
- AP@0.5 and AP@[0.5:0.95] (COCO-style average) (approx, single-class)

We keep implementation dependency-free (pure python + numpy).

Outputs
-------
run_dir/
  config_used.json
  preds/<approach_name>/pred_text.jsonl
  preds/<approach_name>/detect_preds.jsonl
  reports/<approach_name>/metrics.json
  reports/<approach_name>/worst_cases.jsonl
  reports/<approach_name>/detection_metrics.json

"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# -------------------------
# JSONL helpers
# -------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# OCR metrics
# -------------------------

def compute_ocr_metrics(pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Compute CER/WER using your shared module if available."""
    try:
        from ocr_metrics import compute_metrics_for_pairs

        return compute_metrics_for_pairs(pairs)
    except Exception:
        # Fallback: minimal CER only
        def _lev(a: str, b: str) -> int:
            la, lb = len(a), len(b)
            dp = list(range(lb + 1))
            for i in range(1, la + 1):
                prev = dp[0]
                dp[0] = i
                for j in range(1, lb + 1):
                    cur = dp[j]
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                    prev = cur
            return dp[lb]

        cer = []
        for gt, pr in pairs:
            d = _lev(gt, pr)
            cer.append(d / max(1, len(gt)))
        return {"CER_macro": float(np.mean(cer)), "n": len(cer)}


# -------------------------
# Detection metrics (single-class)
# -------------------------

def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_detections(
    gt: List[List[float]],
    pred: List[List[float]],
    scores: List[float],
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy matching of predictions to GT by score.

    Returns:
      tp: array (len(pred)) 1 if matched
      fp: array (len(pred)) 1 if not matched
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    used = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(pred), dtype=np.float32)
    fp = np.zeros(len(pred), dtype=np.float32)

    for k in order:
        best_i = -1
        best_iou = 0.0
        for i, g in enumerate(gt):
            if used[i]:
                continue
            v = iou_xyxy(g, pred[k])
            if v > best_iou:
                best_iou = v
                best_i = i
        if best_i >= 0 and best_iou >= iou_thr:
            used[best_i] = True
            tp[k] = 1.0
        else:
            fp[k] = 1.0

    return tp, fp


def average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    if tp.size == 0:
        return 0.0
    # sort by score is assumed already in match function; but here tp/fp are aligned
    # We'll compute precision-recall curve in score order.
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    recall = ctp / max(1, n_gt)
    prec = ctp / np.maximum(1e-9, ctp + cfp)

    # 11-point interpolation is simple; COCO uses integration over unique recalls.
    # We'll do numeric integration over recall.
    # Ensure monotonic precision
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    # integrate
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[prec[0]], prec, [0.0]])
    # trapezoid
    ap = float(np.trapz(p, r))
    return ap


def detection_metrics_single_class(
    gt_by_uid: Dict[str, List[List[float]]],
    pred_by_uid: Dict[str, Tuple[List[List[float]], List[float]]],
    iou_thresholds: Sequence[float] = (0.5,),
) -> Dict[str, Any]:
    """Compute AP/Recall/Precision for a single class."""

    out: Dict[str, Any] = {}

    # Flatten across images for AP (COCO-style)
    for thr in iou_thresholds:
        all_tp: List[float] = []
        all_fp: List[float] = []
        all_scores: List[float] = []
        n_gt_total = 0

        for uid, gt in gt_by_uid.items():
            n_gt_total += len(gt)
            pred, scores = pred_by_uid.get(uid, ([], []))
            if len(pred) == 0:
                continue
            tp, fp = match_detections(gt, pred, scores, float(thr))
            # collect in score order
            order = np.argsort(-np.asarray(scores, dtype=np.float32))
            all_tp.extend(tp[order].tolist())
            all_fp.extend(fp[order].tolist())
            all_scores.extend(np.asarray(scores, dtype=np.float32)[order].tolist())

        # Compute AP on concatenated lists (already score-sorted within image;
        # we need global sort by score).
        if len(all_scores) == 0:
            out[f"AP@{thr}"] = 0.0
            out[f"Recall@{thr}"] = 0.0
            out[f"Precision@{thr}"] = 0.0
            continue

        glob_order = np.argsort(-np.asarray(all_scores, dtype=np.float32))
        tp_g = np.asarray(all_tp, dtype=np.float32)[glob_order]
        fp_g = np.asarray(all_fp, dtype=np.float32)[glob_order]

        ap = average_precision(tp_g, fp_g, n_gt_total)
        ctp = float(np.sum(tp_g))
        cfp = float(np.sum(fp_g))
        recall = ctp / max(1.0, float(n_gt_total))
        precision = ctp / max(1.0, ctp + cfp)

        out[f"AP@{thr}"] = float(ap)
        out[f"Recall@{thr}"] = float(recall)
        out[f"Precision@{thr}"] = float(precision)

    # COCO-style mean AP 0.5:0.95 step 0.05
    coco_thrs = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05).tolist()]
    aps = []
    for t in coco_thrs:
        # compute AP for each threshold quickly by reusing the function
        aps.append(out.get(f"AP@{t}", None))
    # if not computed above, compute now
    missing = [t for t, v in zip(coco_thrs, aps) if v is None]
    if missing:
        tmp = detection_metrics_single_class(gt_by_uid, pred_by_uid, iou_thresholds=missing)
        out.update(tmp)
        aps = [out[f"AP@{t}"] for t in coco_thrs]
    out["mAP@0.5:0.95"] = float(np.mean(aps))

    return out


# -------------------------
# Approach adapters
# -------------------------

@dataclass
class ApproachConfig:
    name: str
    kind: str  # ctc, seq2seq, det_cls_dec, hybrid
    params: Dict[str, Any]


def run_det_cls_dec(
    manifest: Path,
    cfg: ApproachConfig,
    run_dir: Path,
) -> Dict[str, Path]:
    """Run detection+classification+decoder approach via end2end_infer.py.

    cfg.params expected keys:
      det_model, det_conf, det_imgsz, det_device
      cls_ckpt, cls_model, cls_imgsz, cls_batch, cls_device
      min_char_conf, postprocess
      workdir (optional)
      classify_output_full (bool) -> if True, reruns predict_classify with --output full
    """

    # Ensure outputs folder
    approach_dir = run_dir / "preds" / cfg.name
    approach_dir.mkdir(parents=True, exist_ok=True)

    # Run end2end
    workdir = Path(cfg.params.get("workdir") or (approach_dir / "work"))
    workdir.mkdir(parents=True, exist_ok=True)

    # Use your end2end runner.
    cmd = [
        "python",
        "end2end_infer.py",
        "--manifest",
        str(manifest),
        "--det-model",
        str(cfg.params["det_model"]),
        "--det-conf",
        str(cfg.params.get("det_conf", 0.25)),
        "--det-imgsz",
        str(cfg.params.get("det_imgsz", 1024)),
        "--det-device",
        str(cfg.params.get("det_device", "0")),
        "--cls-ckpt",
        str(cfg.params["cls_ckpt"]),
        "--cls-model",
        str(cfg.params.get("cls_model", "convnext_base")),
        "--cls-imgsz",
        str(cfg.params.get("cls_imgsz", 128)),
        "--cls-batch",
        str(cfg.params.get("cls_batch", 256)),
        "--cls-device",
        str(cfg.params.get("cls_device", "cuda")),
        "--min-char-conf",
        str(cfg.params.get("min_char_conf", 0.35)),
        "--postprocess",
        str(cfg.params.get("postprocess", "none")),
        "--workdir",
        str(workdir),
        "--decoded",
        "decoded.jsonl",
        "--metrics-out",
        "metrics.json",
    ]

    subprocess.run(cmd, check=True)

    # Copy artifacts into approach_dir
    for fn in ["detect_preds.jsonl", "classify_preds.jsonl", "decoded.jsonl", "metrics.json"]:
        src = workdir / fn
        if src.exists():
            shutil.copy2(src, approach_dir / fn)

    return {
        "decoded": approach_dir / "decoded.jsonl",
        "detect": approach_dir / "detect_preds.jsonl",
        "metrics": approach_dir / "metrics.json",
    }


def run_plugin_external(
    manifest: Path,
    cfg: ApproachConfig,
    run_dir: Path,
) -> Dict[str, Path]:
    """Run an external approach as a black-box script.

    cfg.params expected keys:
      cmd: list[str] or str
      out_decoded: path relative to approach_dir

    The external script should write a decoded JSONL with:
      {"uid":..., "gt_text":..., "pred_text":...}

    Optionally detection preds.
    """

    approach_dir = run_dir / "preds" / cfg.name
    approach_dir.mkdir(parents=True, exist_ok=True)

    cmd = cfg.params.get("cmd")
    if cmd is None:
        raise ValueError(f"Plugin approach {cfg.name} missing params.cmd")

    if isinstance(cmd, str):
        # shell
        subprocess.run(cmd, shell=True, check=True, cwd=str(approach_dir))
    else:
        subprocess.run(list(cmd), check=True, cwd=str(approach_dir))

    out_decoded = cfg.params.get("out_decoded", "decoded.jsonl")
    decoded_path = approach_dir / out_decoded
    if not decoded_path.exists():
        raise FileNotFoundError(f"Plugin output not found: {decoded_path}")

    return {"decoded": decoded_path}


# -------------------------
# Worst cases
# -------------------------

def mine_worst_cases(decoded_jsonl: Path, out_path: Path, topk: int = 50) -> None:
    rows = read_jsonl(decoded_jsonl)
    pairs = [(r.get("gt_text", ""), r.get("pred_text", "")) for r in rows]

    # Use ocr_metrics per-sample if available
    try:
        from ocr_metrics import cer, wer
    except Exception:
        from ocr_metrics import char_error_rate as cer  # fallback name if different
        from ocr_metrics import word_error_rate as wer

    scored = []
    for r in rows:
        gt = r.get("gt_text", "")
        pr = r.get("pred_text", "")
        try:
            c = float(cer(gt, pr))
        except Exception:
            c = 9.0
        try:
            w = float(wer(gt, pr))
        except Exception:
            w = 9.0
        scored.append((c, w, r))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    worst = []
    for c, w, r in scored[: int(topk)]:
        rr = dict(r)
        rr["CER"] = float(c)
        rr["WER"] = float(w)
        worst.append(rr)

    write_jsonl(out_path, worst)


# -------------------------
# Main runner
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment harness for OCR approaches.")
    p.add_argument("--manifest", type=str, required=True, help="Manifest JSONL with split fields")
    p.add_argument("--test-split", type=str, default="test")
    p.add_argument("--config", type=str, required=True, help="Experiments config JSON")
    p.add_argument("--run-dir", type=str, default="runs/experiments")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--worst-k", type=int, default=50)
    p.add_argument("--compute-detection-metrics", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> List[ApproachConfig]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    approaches = []
    for a in cfg.get("approaches", []):
        approaches.append(ApproachConfig(name=a["name"], kind=a["kind"], params=a.get("params", {})))
    return approaches


def filter_manifest(manifest_path: Path, split: str, out_path: Path) -> Path:
    rows = read_jsonl(manifest_path)
    sub = [r for r in rows if (r.get("split") or "") == split]
    write_jsonl(out_path, sub)
    return out_path


def build_gt_text_by_uid(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for r in rows:
        uid = r.get("uid")
        gt = (r.get("gt") or {}).get("hebrew", {})
        gt_text = gt.get("raw") or gt.get("text") or ""
        if uid:
            out[str(uid)] = str(gt_text)
    return out


def build_gt_bboxes_by_uid(rows: List[Dict[str, Any]]) -> Dict[str, List[List[float]]]:
    out: Dict[str, List[List[float]]] = {}
    for r in rows:
        uid = r.get("uid")
        if not uid:
            continue
        gt = r.get("gt") or {}
        bbs = gt.get("bboxes")
        if isinstance(bbs, list) and bbs:
            out[str(uid)] = [list(map(float, bb)) for bb in bbs]
    return out


def build_pred_bboxes_by_uid(detect_jsonl: Path) -> Dict[str, Tuple[List[List[float]], List[float]]]:
    rows = read_jsonl(detect_jsonl)
    out: Dict[str, Tuple[List[List[float]], List[float]]] = {}
    for r in rows:
        uid = r.get("uid")
        if not uid:
            continue
        pred = r.get("pred") or {}
        bbs = pred.get("bboxes") or []
        scs = pred.get("scores") or [0.0] * len(bbs)
        out[str(uid)] = ([list(map(float, bb)) for bb in bbs], [float(s) for s in scs])
    return out


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # freeze config
    cfg_path = Path(args.config)
    shutil.copy2(cfg_path, run_dir / "config_used.json")

    # isolate test manifest
    test_manifest = run_dir / "manifest_test.jsonl"
    filter_manifest(Path(args.manifest), args.test_split, test_manifest)

    test_rows = read_jsonl(test_manifest)
    gt_text_by_uid = build_gt_text_by_uid(test_rows)

    gt_bboxes_by_uid = build_gt_bboxes_by_uid(test_rows) if args.compute_detection_metrics else {}

    approaches = load_config(cfg_path)

    for acfg in approaches:
        print(f"[runner] approach={acfg.name} kind={acfg.kind}")

        if acfg.kind == "det_cls_dec":
            artifacts = run_det_cls_dec(test_manifest, acfg, run_dir)
        else:
            artifacts = run_plugin_external(test_manifest, acfg, run_dir)

        decoded_path = artifacts["decoded"]
        decoded_rows = read_jsonl(decoded_path)

        # Ensure gt_text present; if missing, fill from manifest
        for r in decoded_rows:
            uid = r.get("uid")
            if uid and not r.get("gt_text"):
                r["gt_text"] = gt_text_by_uid.get(str(uid), "")

        # compute OCR metrics
        pairs = [(r.get("gt_text", ""), r.get("pred_text", "")) for r in decoded_rows]
        ocr_m = compute_ocr_metrics(pairs)

        rep_dir = run_dir / "reports" / acfg.name
        rep_dir.mkdir(parents=True, exist_ok=True)
        write_json(rep_dir / "metrics.json", ocr_m)

        # worst cases
        tmp_decoded = rep_dir / "decoded_with_gt.jsonl"
        write_jsonl(tmp_decoded, decoded_rows)
        mine_worst_cases(tmp_decoded, rep_dir / "worst_cases.jsonl", topk=int(args.worst_k))

        # detection metrics if possible
        if args.compute_detection_metrics and "detect" in artifacts and gt_bboxes_by_uid:
            pred_bboxes_by_uid = build_pred_bboxes_by_uid(artifacts["detect"])
            det_m = detection_metrics_single_class(gt_bboxes_by_uid, pred_bboxes_by_uid, iou_thresholds=(0.5,))
            write_json(rep_dir / "detection_metrics.json", det_m)

        print(f"[runner] {acfg.name}: CER/WER saved to {rep_dir}")

    print(f"[runner] done. run_dir={run_dir}")


if __name__ == "__main__":
    main()
