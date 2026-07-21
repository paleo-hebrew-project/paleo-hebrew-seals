"""Evaluate a trained YOLO detector: mAP, speed, memory, params."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLO detector")
    p.add_argument("--weights", type=str, required=True, help="best.pt from Ultralytics run")
    p.add_argument("--data-yaml", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--out-json", type=str, default="metrics_detector.json")
    p.add_argument("--out-csv", type=str, default="metrics_detector.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("pip install ultralytics") from e

    weights = Path(args.weights)
    model = YOLO(str(weights))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    results = model.val(data=str(args.data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device, plots=False)
    t1 = time.perf_counter()
    if isinstance(results, (list, tuple)) and results:
        results = results[0]

    metrics: Dict[str, Any] = {
        "weights": str(weights),
        "data_yaml": str(args.data_yaml),
        "val_wall_time_s": t1 - t0,
    }

    # Ultralytics results object varies by version — extract common fields
    if hasattr(results, "box"):
        box = results.box
        if hasattr(box, "map50"):
            metrics["mAP50"] = float(box.map50)
        if hasattr(box, "map"):
            metrics["mAP50_95"] = float(box.map)
        if hasattr(box, "mp"):
            metrics["precision"] = float(box.mp)
        if hasattr(box, "mr"):
            metrics["recall"] = float(box.mr)
    if hasattr(results, "speed"):
        sp = results.speed
        if isinstance(sp, dict):
            metrics["speed_ms_preprocess"] = sp.get("preprocess", 0)
            metrics["speed_ms_inference"] = sp.get("inference", 0)
            metrics["speed_ms_postprocess"] = sp.get("postprocess", 0)

    try:
        import numpy as np

        nm = model.model
        params = sum(int(p.numel()) for p in nm.parameters())
        metrics["num_parameters"] = int(params)
    except Exception:
        pass

    if torch.cuda.is_available():
        metrics["cuda_peak_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated())

    Path(args.out_json).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    line = ",".join(str(metrics.get(k, "")) for k in sorted(metrics.keys()))
    Path(args.out_csv).write_text("key,value\n" + "\n".join(f"{k},{v}" for k, v in sorted(metrics.items())), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
