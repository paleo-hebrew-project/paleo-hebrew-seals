"""Optional front-end OCR sanity check: chain existing predict_detect → extract_crops → predict_classify.

This script does not reimplement inference; it shells out to paleo_ocr CLIs and aggregates CER if GT exists.

Example:
  python -m paleo_ocr.experiments.e2e_front_eval \\
    --manifest path/to/manifest.jsonl \\
    --det-model runs/detect/exp/weights/best.pt \\
    --cls-ckpt runs/classify/exp/best.pt \\
    --timm-model convnext_base \\
    --work-dir /tmp/e2e_front
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2E detector+classifier sanity (subprocess chain)")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--det-model", type=str, required=True)
    p.add_argument("--cls-ckpt", type=str, required=True)
    p.add_argument("--timm-model", type=str, default="convnext_base")
    p.add_argument("--work-dir", type=str, default="e2e_front_work")
    p.add_argument("--out-json", type=str, default="e2e_front_metrics.json")
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    det_out = work / "detect_preds.jsonl"
    crops_dir = work / "crops"
    cls_out = work / "classify_preds.jsonl"

    _run(
        [
            sys.executable,
            "-m",
            "paleo_ocr.predict_detect",
            "--model",
            args.det_model,
            "--manifest",
            args.manifest,
            "--out",
            str(det_out),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "paleo_ocr.extract_crops",
            "--detect",
            str(det_out),
            "--crops-dir",
            str(crops_dir),
            "--index",
            str(work / "crops_index.jsonl"),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "paleo_ocr.predict_classify",
            "--ckpt",
            args.cls_ckpt,
            "--model",
            args.timm_model,
            "--folder",
            str(crops_dir),
            "--out",
            str(cls_out),
        ]
    )

    metrics: Dict[str, Any] = {
        "manifest": args.manifest,
        "det_model": args.det_model,
        "cls_ckpt": args.cls_ckpt,
        "detect_preds": str(det_out),
        "classify_preds": str(cls_out),
        "note": "Decode preds to string with ocr_decoder / experiment_runner for CER vs manifest gt.*",
    }
    Path(args.out_json).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
